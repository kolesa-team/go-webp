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
	"context"
	"fmt"
	"image"
	"testing"

	"github.com/stretchr/testify/require"

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
			require.NoError(t, err)
			o.Method = m
			want := encodeBytes(t, img, o)
			o.ThreadLevel = true
			for _, limit := range []int{1, 3, 8} {
				name := fmt.Sprintf("alpha=%v m%d budget %d", alpha, m, limit)
				prev := workers.SetLimitForTest(limit)
				require.Equal(t, want, encodeBytes(t, img, o), name)
				workers.SetLimitForTest(prev)
				require.Zero(t, workers.InUse(), name)
			}
			withBudget(t, "exhausted", func() {
				require.Equal(t, want, encodeBytes(t, img, o), "alpha=%v m%d no budget", alpha, m)
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
		require.NoError(t, err)
		o.ThreadLevel = true
		want := encodeBytes(t, img, o)
		for _, limit := range []int{1, 4} {
			prev := workers.SetLimitForTest(limit)
			e, err := NewEncoder(img, o)
			require.NoError(t, err)
			ctx := newPollCtx(0)
			var got countWriter
			require.NoError(t, e.EncodeContext(ctx, &got))
			n := ctx.polls.Load()
			for _, k := range pollPoints(n) {
				var cw countWriter
				err := e.EncodeContext(newPollCtx(k), &cw)
				require.Zero(t, workers.InUse(), "helpers running after EncodeContext returned")
				if err == nil && k == n {
					continue // the last poll may come after the frame is coded
				}
				require.Equal(t, context.Canceled, err, "alpha=%v budget %d poll %d/%d", alpha, limit, k, n)
				require.Zero(t, cw.calls)
			}
			// The encoder is reusable after a cancellation.
			require.Equal(t, want, encodeBytesWith(t, e))
			e.Release()
			workers.SetLimitForTest(prev)
		}
	}
}

func encodeBytesWith(t *testing.T, e *Encoder) []byte {
	t.Helper()
	var buf sliceWriter
	require.NoError(t, e.Encode(&buf))
	return buf.b
}
