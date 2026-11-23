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
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// wavefrontImage is large enough for the VP8 coding pass to run as a
// wavefront under ThreadLevel (64x48 macroblocks), opaque or with a
// transparent pattern (the alpha plane is then coded on a helper too).
func wavefrontImage(alpha bool) *image.NRGBA {
	img := alphaImage(1024, 768, 7)
	if !alpha {
		for i := 3; i < len(img.Pix); i += 4 {
			img.Pix[i] = 255
		}
	}
	return img
}

// TestWavefrontIdentical checks that lossy ThreadLevel encodes, where the
// alpha helper and the wavefront rows of the VP8 frame share the helper
// budget, give the bytes of the sequential encode for any budget.
func TestWavefrontIdentical(t *testing.T) {
	checkNoLeak(t)
	methods := []int{4, 6}
	if testing.Short() {
		methods = []int{4}
	}
	for _, alpha := range []bool{false, true} {
		img := wavefrontImage(alpha)
		for _, m := range methods {
			if alpha && m == 6 {
				continue // slow alpha plane, nothing more to check
			}
			o, err := NewLossyEncoderOptions(PresetDefault, 75)
			if err != nil {
				t.Fatalf("NewLossyEncoderOptions: %v", err)
			}
			o.Method = m
			want := encodeBytes(t, img, o)
			o.ThreadLevel = true
			for _, limit := range []int{1, 3, 8} {
				name := fmt.Sprintf("alpha=%v m%d budget %d", alpha, m, limit)
				prev := workers.SetLimitForTest(limit)
				if got := encodeBytes(t, img, o); !bytes.Equal(got, want) {
					t.Fatalf("%s: ThreadLevel output differs from the sequential one (%d bytes, want %d)", name, len(got), len(want))
				}
				workers.SetLimitForTest(prev)
				if n := workers.InUse(); n != 0 {
					t.Fatalf("%s: workers.InUse() = %d, want 0", name, n)
				}
			}
			withBudget(t, "exhausted", func() {
				if got := encodeBytes(t, img, o); !bytes.Equal(got, want) {
					t.Fatalf("alpha=%v m%d no budget: ThreadLevel output differs from the sequential one (%d bytes, want %d)", alpha, m, len(got), len(want))
				}
			})
		}
	}
}

// TestWavefrontContextCancel cancels ThreadLevel encodes at polls spread
// over the analysis bands and the wavefront rows: EncodeContext returns
// the context's error with every helper joined and nothing written.
func TestWavefrontContextCancel(t *testing.T) {
	checkNoLeak(t)
	for _, alpha := range []bool{false, true} {
		img := wavefrontImage(alpha)
		o, err := NewLossyEncoderOptions(PresetDefault, 75)
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions: %v", err)
		}
		o.ThreadLevel = true
		want := encodeBytes(t, img, o)
		for _, limit := range []int{1, 4} {
			prev := workers.SetLimitForTest(limit)
			e, err := NewEncoder(img, o)
			if err != nil {
				t.Fatalf("alpha=%v budget %d: NewEncoder: %v", alpha, limit, err)
			}
			ctx := newPollCtx(0)
			var got countWriter
			if err := e.EncodeContext(ctx, &got); err != nil {
				t.Fatalf("alpha=%v budget %d: EncodeContext: %v", alpha, limit, err)
			}
			n := ctx.polls.Load()
			for _, k := range pollPoints(n) {
				var cw countWriter
				err := e.EncodeContext(newPollCtx(k), &cw)
				if inUse := workers.InUse(); inUse != 0 {
					t.Fatalf("alpha=%v budget %d poll %d/%d: helpers running after EncodeContext returned: workers.InUse() = %d, want 0", alpha, limit, k, n, inUse)
				}
				if err == nil && k == n {
					continue // the last poll may come after the frame is coded
				}
				if !reflect.DeepEqual(err, context.Canceled) {
					t.Fatalf("alpha=%v budget %d poll %d/%d: EncodeContext = %v, want %v", alpha, limit, k, n, err, context.Canceled)
				}
				if cw.calls != 0 {
					t.Fatalf("alpha=%v budget %d poll %d/%d: %d Write calls, want 0", alpha, limit, k, n, cw.calls)
				}
			}
			// The encoder is reusable after a cancellation.
			if out := encodeBytesWith(t, e); !bytes.Equal(out, want) {
				t.Fatalf("alpha=%v budget %d: Encode after a cancellation differs from the first encode (%d bytes, want %d)", alpha, limit, len(out), len(want))
			}
			e.Release()
			workers.SetLimitForTest(prev)
		}
	}
}

func encodeBytesWith(t *testing.T, e *Encoder) []byte {
	t.Helper()
	var buf sliceWriter
	if err := e.Encode(&buf); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return buf.b
}
