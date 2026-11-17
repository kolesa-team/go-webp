package decoder

import (
	"bytes"
	"context"
	"errors"
	"image"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// countdownCtx is a context that gets cancelled when its Err method has
// been called n times, i.e. after a fixed number of row batches.
type countdownCtx struct {
	context.Context
	cancel context.CancelFunc
	n      atomic.Int64
}

func newCountdown(n int) *countdownCtx {
	ctx, cancel := context.WithCancel(context.Background())
	c := &countdownCtx{Context: ctx, cancel: cancel}
	c.n.Store(int64(n))
	return c
}

func (c *countdownCtx) Err() error {
	if c.n.Add(-1) < 0 {
		c.cancel()
	}
	return c.Context.Err()
}

// waitGoroutines waits until the number of goroutines is back to n (helper
// goroutines may still be returning when the decode does).
func waitGoroutines(t *testing.T, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > n && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	assert.LessOrEqual(t, runtime.NumGoroutine(), n, "goroutine leak")
	assert.Equal(t, 0, workers.InUse(), "helper slots still taken")
}

func checkAbort(t *testing.T, err error, want error, msg string) {
	t.Helper()
	require.Error(t, err, msg)
	assert.True(t, errors.Is(err, want), "%s: %v", msg, err)
	assert.Equal(t, utils.Vp8StatusUserAbort, status.CodeOf(err), msg)
	assert.Equal(t, "cannot decode picture: VP8_STATUS_USER_ABORT: "+want.Error(), err.Error(), msg)
}

var ctxOptions = []*Options{
	nil,
	{UseThreads: true},
	{Scale: image.Rect(0, 0, 300, 200)},
	{Scale: image.Rect(0, 0, 300, 200), UseThreads: true},
	{Crop: image.Rect(10, 10, 300, 400), Flip: true},
	{AlphaDitheringStrength: 50, UseThreads: true},
}

func TestDecodeContextCancel(t *testing.T) {
	g := runtime.NumGoroutine()
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data := readFile(t, file)
		for _, o := range ctxOptions {
			// Already cancelled, or with an expired deadline.
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err := newDec(t, data, o).DecodeContext(ctx)
			checkAbort(t, err, context.Canceled, file)
			dctx, dcancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
			_, err = newDec(t, data, o).DecodeContext(dctx)
			dcancel()
			checkAbort(t, err, context.DeadlineExceeded, file)

			// Cancelled in the middle of the picture.
			for _, n := range []int{1, 2, 5} {
				dst := dirty(1000)
				before := *dst
				img, err := newDec(t, data, o).DecodeIntoContext(newCountdown(n), dst)
				assert.Nil(t, img)
				checkAbort(t, err, context.Canceled, file)
				assert.Equal(t, before, *dst)
			}

			// Not cancelled: same result as Decode.
			if testing.Short() && file != "m4_q75.webp" {
				continue
			}
			want := decode(t, data, o)
			got, err := newDec(t, data, o).DecodeContext(newCountdown(1 << 30))
			require.NoError(t, err)
			assert.Equal(t, want, got)
			got, err = newDec(t, data, o).DecodeContext(context.Background())
			require.NoError(t, err)
			assert.Equal(t, want, got)
		}
	}
	waitGoroutines(t, g)
}

// BenchmarkDecodeContext is BenchmarkDecode with a cancellable context (the
// cost of the cancellation checks).
func BenchmarkDecodeContext(b *testing.B) {
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data := readFile(b, file)
		b.Run(file, func(b *testing.B) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				d, err := NewDecoder(bytes.NewReader(data), nil)
				if err != nil {
					b.Fatal(err)
				}
				if _, err := d.DecodeContext(ctx); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestDecodeContextAlphaHelper cancels lossy+alpha decodes while the alpha
// helper works ahead (tiny bands, every helper budget): the outcome must be
// the sequential one, and the helper must be gone when the call returns.
func TestDecodeContextAlphaHelper(t *testing.T) {
	const w, h = 97, 61
	smallBands(t, 3, w)
	files := alphaFiles(t, w, h)
	g := runtime.NumGoroutine()
	for name, data := range files {
		for _, o := range []*Options{
			{UseThreads: true},
			{UseThreads: true, Crop: image.Rect(0, 10, w, 40), Flip: true},
			{UseThreads: true, Scale: image.Rect(0, 0, 40, 30)},
		} {
			seq := *o
			seq.UseThreads = false
			want := decode(t, data, &seq)
			for _, b := range budgets {
				restore := b.set()
				for _, n := range []int{1, 2, 3} {
					dst := dirty(len(want.Pix))
					before := *dst
					img, err := newDec(t, data, o).DecodeIntoContext(newCountdown(n), dst)
					assert.Nil(t, img)
					checkAbort(t, err, context.Canceled, name)
					assert.Equal(t, before, *dst, "%s %+v budget %s", name, o, b.name)
				}
				got, err := newDec(t, data, o).DecodeIntoContext(newCountdown(1<<30), dirty(len(want.Pix)))
				restore()
				require.NoError(t, err, "%s %+v budget %s", name, o, b.name)
				require.Equal(t, want, got, "%s %+v budget %s", name, o, b.name)
			}
		}
	}
	waitGoroutines(t, g)
}

// TestDecodeContextStopsEarly checks that a cancelled decode does not run to
// the end of the picture: the number of context checks is bounded by the
// batches decoded before the cancellation.
func TestDecodeContextStopsEarly(t *testing.T) {
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data := readFile(t, file)
		for _, o := range []*Options{nil, {UseThreads: true}, {Scale: image.Rect(0, 0, 100, 100)}} {
			full := newCountdown(1 << 30)
			_, err := newDec(t, data, o).DecodeContext(full)
			require.NoError(t, err)
			checks := (1 << 30) - full.n.Load()

			ctx := newCountdown(3)
			_, err = newDec(t, data, o).DecodeContext(ctx)
			checkAbort(t, err, context.Canceled, file)
			used := 3 - ctx.n.Load()
			assert.Less(t, used, checks/2, "%s %+v: %d of %d checks", file, o, used, checks)
		}
	}
}

// TestCtxSinkForwardsPremultiplied checks that the cancellable wrapper keeps
// the optional pipeline.PremultipliedSink of the sink it wraps (the lossless
// decoder premultiplies palettes once instead of every pixel when scaling).
func TestCtxSinkForwardsPremultiplied(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cl := call{ctx: ctx}

	scaled, _, err := pipeline.NewSink(pipeline.Params{SrcW: 64, SrcH: 32, ScaleW: 16, ScaleH: 8, Premultiply: true})
	require.NoError(t, err)
	w := cl.wrap(scaled)
	_, isCtx := w.(*ctxSink)
	require.True(t, isCtx, "a cancellable call must wrap the sink")
	ps, ok := w.(pipeline.PremultipliedSink)
	require.True(t, ok)
	assert.True(t, ps.AcceptPremultiplied(), "scaling sink must accept premultiplied rows through the wrapper")

	direct, _, err := pipeline.NewSink(pipeline.Params{SrcW: 64, SrcH: 32, Premultiply: true})
	require.NoError(t, err)
	ps, ok = cl.wrap(direct).(pipeline.PremultipliedSink)
	require.True(t, ok)
	assert.False(t, ps.AcceptPremultiplied(), "a sink without the interface must refuse")
}

// TestDecodeContextScalePalette: a cancellable scaled decode of a palette
// lossless picture equals the plain decode, with and without UseThreads.
func TestDecodeContextScalePalette(t *testing.T) {
	data := readFile(t, "webp-logo-lossless.webp")
	for _, o := range []*Options{
		{Scale: image.Rect(0, 0, 300, 200)},
		{Scale: image.Rect(0, 0, 777, 333), UseThreads: true},
		{Scale: image.Rect(0, 0, 400, 300), Crop: image.Rect(100, 50, 2100, 1550), UseThreads: true},
	} {
		want := decode(t, data, o)
		ctx, cancel := context.WithCancel(context.Background())
		got, err := newDec(t, data, o).DecodeContext(ctx)
		cancel()
		require.NoError(t, err)
		assert.Equal(t, want, got, "%+v", o)
	}
}
