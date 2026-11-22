// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package encoder

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

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
	t.Helper()
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	multiPass, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	multiPass.Pass, multiPass.TargetSize = 4, 3000
	ll6, err := NewLosslessEncoderOptions(PresetDefault, 6)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions(6): %v", err)
	}
	ll9, err := NewLosslessEncoderOptions(PresetDefault, 9)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions(9): %v", err)
	}
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
		if err != nil {
			t.Fatalf("%s: NewEncoder: %v", tc.name, err)
		}
		var want bytes.Buffer
		if err := e.Encode(&want); err != nil {
			t.Fatalf("%s: Encode: %v", tc.name, err)
		}

		never := newPollCtx(0)
		var got bytes.Buffer
		if err := e.EncodeContext(never, &got); err != nil {
			t.Fatalf("%s: EncodeContext: %v", tc.name, err)
		}
		if !bytes.Equal(got.Bytes(), want.Bytes()) {
			t.Fatalf("%s: EncodeContext output differs from Encode (%d bytes, want %d)", tc.name, got.Len(), want.Len())
		}
		polls := never.polls.Load()
		// Before, after, and at least once during the encode.
		if polls <= 2 {
			t.Fatalf("%s: %d polls, want more than 2", tc.name, polls)
		}

		for _, k := range pollPoints(polls) {
			ctx := newPollCtx(k)
			var cw countWriter
			err := e.EncodeContext(ctx, &cw)
			if !reflect.DeepEqual(err, context.Canceled) {
				t.Fatalf("%s: canceled at poll %d of %d: EncodeContext = %v, want %v", tc.name, k, polls, err, context.Canceled)
			}
			if cw.calls != 0 {
				t.Fatalf("%s: canceled at poll %d: %d Write calls, want 0", tc.name, k, cw.calls)
			}
			extra := ctx.polls.Load() - k
			if extra > 16 {
				t.Fatalf("%s: kept polling after poll %d: %d more polls, want at most 16", tc.name, k, extra)
			}
		}
		e.Release()
	}
}

// TestEncodeContextThreads cancels lossy+alpha encodes with ThreadLevel at
// various points, for every helper budget: the helper must always be joined.
func TestEncodeContextThreads(t *testing.T) {
	checkNoLeak(t)
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	o.ThreadLevel = true
	img := alphaImage(300, 260, 4)
	want := encodeBytes(t, img, o)
	for _, b := range budgets {
		withBudget(t, b, func() {
			e, err := NewEncoder(img, o)
			if err != nil {
				t.Fatalf("budget %s: NewEncoder: %v", b, err)
			}
			defer e.Release()
			var got bytes.Buffer
			if err := e.EncodeContext(newPollCtx(0), &got); err != nil {
				t.Fatalf("budget %s: EncodeContext: %v", b, err)
			}
			if !bytes.Equal(got.Bytes(), want) {
				t.Fatalf("budget %s: EncodeContext output differs from the sequential one (%d bytes, want %d)", b, got.Len(), len(want))
			}
			for _, k := range []int64{1, 2, 3, 5, 8, 13, 21, 34} {
				var cw countWriter
				err := e.EncodeContext(newPollCtx(k), &cw)
				if err == nil {
					continue // finished before the k-th poll
				}
				if !reflect.DeepEqual(err, context.Canceled) {
					t.Fatalf("budget %s, poll %d: EncodeContext = %v, want %v", b, k, err, context.Canceled)
				}
				if cw.calls != 0 {
					t.Fatalf("budget %s, poll %d: %d Write calls, want 0", b, k, cw.calls)
				}
			}
		})
		if n := workers.InUse(); n != 0 {
			t.Fatalf("budget %s: workers.InUse() = %d, want 0", b, n)
		}
	}
}

func TestEncodeContextNotCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	type key struct{}
	for _, tc := range contextCases(t) {
		e, err := NewEncoder(tc.img, tc.o)
		if err != nil {
			t.Fatalf("%s: NewEncoder: %v", tc.name, err)
		}
		var want bytes.Buffer
		if err := e.Encode(&want); err != nil {
			t.Fatalf("%s: Encode: %v", tc.name, err)
		}
		for _, c := range []context.Context{context.Background(), context.WithValue(context.Background(), key{}, 1), ctx} {
			var got bytes.Buffer
			if err := e.EncodeContext(c, &got); err != nil {
				t.Fatalf("%s: EncodeContext(%v): %v", tc.name, c, err)
			}
			if !bytes.Equal(got.Bytes(), want.Bytes()) {
				t.Fatalf("%s: EncodeContext(%v) output differs from Encode (%d bytes, want %d)", tc.name, c, got.Len(), want.Len())
			}
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
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	for _, c := range []context.Context{canceled, expired, cause} {
		var cw countWriter
		err := e.EncodeContext(c, &cw)
		if want := c.Err(); !reflect.DeepEqual(want, err) {
			t.Fatalf("EncodeContext(%v) = %v, want %v", c, err, want)
		}
		if cw.calls != 0 {
			t.Fatalf("EncodeContext(%v): %d Write calls, want 0", c, cw.calls)
		}
	}
}

// TestEncodeContextOtherErrors checks that errors unrelated to the context
// are reported as by Encode.
func TestEncodeContextOtherErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e, err := NewEncoder(image.NewNRGBA(image.Rect(0, 0, 16384, 1)), nil)
	if err != nil {
		t.Fatalf("NewEncoder(16384x1): %v", err)
	}
	err = e.EncodeContext(ctx, &bytes.Buffer{})
	if err == nil || err.Error() != "cannot encode webppicture" {
		t.Fatalf("EncodeContext(16384x1) = %v, want %q", err, "cannot encode webppicture")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	o.AlphaCompression = 2
	e, err = NewEncoder(img, o)
	if err != nil {
		t.Fatalf("NewEncoder(AlphaCompression 2): %v", err)
	}
	if err := e.EncodeContext(ctx, &bytes.Buffer{}); err == nil || err.Error() != "cannot encode webppicture" {
		t.Fatalf("EncodeContext(AlphaCompression 2) = %v, want %q", err, "cannot encode webppicture")
	}

	if err := e.EncodeContext(ctx, failWriter{}); err == nil || err.Error() != "cannot encode webppicture" {
		t.Fatalf("EncodeContext(AlphaCompression 2, failWriter) = %v, want %q", err, "cannot encode webppicture")
	}
	e, err = NewEncoder(sourceImages(8, 8, 1)[0], nil)
	if err != nil {
		t.Fatalf("NewEncoder(8x8): %v", err)
	}
	if err := e.EncodeContext(ctx, failWriter{}); err == nil || err.Error() != "boom" {
		t.Fatalf("EncodeContext(failWriter) = %v, want %q", err, "boom")
	}
}
