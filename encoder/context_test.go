package encoder

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// pollCtx is a context whose Err reports context.Canceled from its limit-th
// call on (never if limit is 0), counting the calls: the encoders poll it
// through Err, so it cancels an encode at a deterministic point.
type pollCtx struct {
	context.Context
	done  chan struct{}
	limit int64
	polls atomic.Int64
}

func newPollCtx(limit int64) *pollCtx {
	return &pollCtx{Context: context.Background(), done: make(chan struct{}), limit: limit}
}

func (c *pollCtx) Done() <-chan struct{} { return c.done }

func (c *pollCtx) Err() error {
	if n := c.polls.Add(1); c.limit > 0 && n >= c.limit {
		return context.Canceled
	}
	return nil
}

// pollPoints returns the polls (1-based) at which the tests cancel, out of
// n: all of them for short encodes, a sample (including the first and last
// ones) otherwise.
func pollPoints(n int64) []int64 {
	var ks []int64
	if raceEnabled && n > 12 {
		// Under the race detector (slow CI runners): the first and last
		// polls and 8 spread over the encode.
		ks = append(ks, 1, 2)
		for i := int64(1); i <= 8; i++ {
			ks = append(ks, 2+i*(n-4)/9)
		}
		return append(ks, n-1, n)
	}
	if n <= 48 {
		for k := int64(1); k <= n; k++ {
			ks = append(ks, k)
		}
		return ks
	}
	for k := int64(1); k <= 8; k++ {
		ks = append(ks, k)
	}
	for i := int64(1); i < 32; i++ {
		ks = append(ks, 8+i*(n-16)/32)
	}
	for k := n - 7; k <= n; k++ {
		ks = append(ks, k)
	}
	return ks
}

type contextCase struct {
	name string
	img  image.Image
	o    *Options
}

func contextCases(t *testing.T) []contextCase {
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	multiPass, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	multiPass.Pass, multiPass.TargetSize = 4, 3000
	ll6, err := NewLosslessEncoderOptions(PresetDefault, 6)
	require.NoError(t, err)
	ll9, err := NewLosslessEncoderOptions(PresetDefault, 9)
	require.NoError(t, err)
	srcs := sourceImages(67, 45, 11)
	return []contextCase{
		{"lossy", srcs[1], lossy},
		{"lossy YCbCr", srcs[3], lossy},
		{"lossy multi-pass", srcs[2], multiPass},
		{"lossy alpha", alphaImage(67, 45, 2), lossy},
		{"lossless", srcs[1], ll6},
		{"lossless z9", alphaImage(40, 30, 3), ll9},
		{"lossless 1x1", sourceImages(1, 1, 1)[0], ll6},
	}
}

// TestEncodeContextPolls cancels each encode at every point where it polls
// the context: EncodeContext must then return exactly ctx.Err(), write
// nothing, and stop polling at once.
func TestEncodeContextPolls(t *testing.T) {
	checkNoLeak(t)
	for _, tc := range contextCases(t) {
		e, err := NewEncoder(tc.img, tc.o)
		require.NoError(t, err)
		var want bytes.Buffer
		require.NoError(t, e.Encode(&want))

		never := newPollCtx(0)
		var got bytes.Buffer
		require.NoError(t, e.EncodeContext(never, &got), tc.name)
		require.Equal(t, want.Bytes(), got.Bytes(), tc.name)
		polls := never.polls.Load()
		// Before, after, and at least once during the encode.
		require.Greater(t, polls, int64(2), tc.name)

		for _, k := range pollPoints(polls) {
			ctx := newPollCtx(k)
			var cw countWriter
			err := e.EncodeContext(ctx, &cw)
			require.Equal(t, context.Canceled, err, "%s: canceled at poll %d of %d", tc.name, k, polls)
			require.Zero(t, cw.calls, tc.name)
			extra := ctx.polls.Load() - k
			require.LessOrEqual(t, extra, int64(16), "%s: kept polling after poll %d", tc.name, k)
		}
		e.Release()
	}
}

// TestEncodeContextThreads cancels lossy+alpha encodes with ThreadLevel at
// various points, for every helper budget: the helper must always be joined.
func TestEncodeContextThreads(t *testing.T) {
	checkNoLeak(t)
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	o.ThreadLevel = true
	img := alphaImage(300, 260, 4)
	want := encodeBytes(t, img, o)
	for _, b := range budgets {
		withBudget(t, b, func() {
			e, err := NewEncoder(img, o)
			require.NoError(t, err)
			defer e.Release()
			var got bytes.Buffer
			require.NoError(t, e.EncodeContext(newPollCtx(0), &got))
			require.Equal(t, want, got.Bytes())
			for _, k := range []int64{1, 2, 3, 5, 8, 13, 21, 34} {
				var cw countWriter
				err := e.EncodeContext(newPollCtx(k), &cw)
				if err == nil {
					continue // finished before the k-th poll
				}
				require.Equal(t, context.Canceled, err, "budget %s, poll %d", b, k)
				require.Zero(t, cw.calls)
			}
		})
		require.Zero(t, workers.InUse())
	}
}

func TestEncodeContextNotCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	type key struct{}
	for _, tc := range contextCases(t) {
		e, err := NewEncoder(tc.img, tc.o)
		require.NoError(t, err)
		var want bytes.Buffer
		require.NoError(t, e.Encode(&want))
		for _, c := range []context.Context{context.Background(), context.WithValue(context.Background(), key{}, 1), ctx} {
			var got bytes.Buffer
			require.NoError(t, e.EncodeContext(c, &got))
			require.Equal(t, want.Bytes(), got.Bytes(), tc.name)
		}
		e.Release()
	}
}

func TestEncodeContextDone(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, cancel2 := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel2()
	cause, cancel3 := context.WithCancelCause(context.Background())
	cancel3(fmt.Errorf("custom cause"))
	img := sourceImages(16, 16, 1)[0]
	e, err := NewEncoder(img, nil)
	require.NoError(t, err)
	for _, c := range []context.Context{canceled, expired, cause} {
		var cw countWriter
		err := e.EncodeContext(c, &cw)
		require.Equal(t, c.Err(), err)
		require.Zero(t, cw.calls)
	}
}

// TestEncodeContextOtherErrors checks that errors unrelated to the context
// are reported as by Encode.
func TestEncodeContextOtherErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e, err := NewEncoder(image.NewNRGBA(image.Rect(0, 0, 16384, 1)), nil)
	require.NoError(t, err)
	err = e.EncodeContext(ctx, &bytes.Buffer{})
	require.EqualError(t, err, "cannot encode webppicture")

	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	o.AlphaCompression = 2
	e, err = NewEncoder(img, o)
	require.NoError(t, err)
	require.EqualError(t, e.EncodeContext(ctx, &bytes.Buffer{}), "cannot encode webppicture")

	require.EqualError(t, e.EncodeContext(ctx, failWriter{}), "cannot encode webppicture")
	e, err = NewEncoder(sourceImages(8, 8, 1)[0], nil)
	require.NoError(t, err)
	require.EqualError(t, e.EncodeContext(ctx, failWriter{}), "boom")
}
