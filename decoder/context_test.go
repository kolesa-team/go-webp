package decoder

import (
	"context"
	"errors"
	"image"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
