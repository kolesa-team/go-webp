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
	"errors"
	"fmt"
	"image"
	"image/draw"
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// TestParallelAlphaSubImage codes the alpha plane of sub-images (non-zero
// origin, stride wider than the picture) on the helper: the output must
// equal the sequential one for every helper budget, for a shared *image.NRGBA
// and for a pooled snapshot of an *image.RGBA.
func TestParallelAlphaSubImage(t *testing.T) {
	checkNoLeak(t)
	full := alphaImage(420, 330, 9)
	sub := full.SubImage(image.Rect(37, 21, 37+301, 21+251)).(*image.NRGBA) // 75551 px
	rgba := image.NewRGBA(full.Rect)
	draw.Draw(rgba, rgba.Rect, full, image.Point{}, draw.Src)
	srcs := []image.Image{rgba.SubImage(sub.Rect), sub}
	filters := []int{1, 0, 2}
	if testing.Short() {
		srcs, filters = srcs[:1], filters[:1]
	}
	for _, src := range srcs {
		for _, af := range filters {
			o, err := NewLossyEncoderOptions(PresetDefault, 75)
			require.NoError(t, err)
			o.AlphaFiltering = af
			want := encodeBytes(t, src, o)
			o.ThreadLevel = true
			for _, b := range budgets {
				withBudget(t, b, func() {
					require.Equal(t, want, encodeBytes(t, src, o), "%T filter %d budget %s", src, af, b)
				})
				require.Zero(t, workers.InUse())
			}
		}
	}
}

// TestReleaseWaitsForParallelAlpha releases Encoders (pooled snapshots)
// while ThreadLevel encodes with an alpha helper run on other goroutines,
// and scribbles over every buffer taken from the snapshot pool meanwhile:
// a snapshot recycled under the helper would change the output (or be
// reported by -race).
func TestReleaseWaitsForParallelAlpha(t *testing.T) {
	checkNoLeak(t)
	img := alphaImage(300, 260, 12)
	rgba := image.NewRGBA(img.Rect)
	draw.Draw(rgba, rgba.Rect, img, image.Point{}, draw.Src)
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	o.Method = 0
	want := encodeBytes(t, rgba, o)
	o.ThreadLevel = true
	prev := workers.SetLimitForTest(8)
	defer workers.SetLimitForTest(prev)

	stop := make(chan struct{})
	var scribbler sync.WaitGroup
	scribbler.Go(func() { scribble(4*img.Rect.Dx()*img.Rect.Dy(), 0x5a, stop, false) })
	defer func() {
		close(stop)
		scribbler.Wait()
	}()

	rounds := 6
	if testing.Short() {
		rounds = 2
	}
	for i := 0; i < rounds; i++ {
		e, err := NewEncoder(rgba, o)
		require.NoError(t, err)
		var wg sync.WaitGroup
		results := make([]error, 2)
		outs := make([][]byte, 2)
		for g := range results {
			wg.Go(func() {
				var buf bytes.Buffer
				results[g] = e.Encode(&buf)
				outs[g] = buf.Bytes()
			})
		}
		runtime.Gosched()
		e.Release()
		wg.Wait()
		for g, err := range results {
			if errors.Is(err, ErrReleased) {
				continue
			}
			require.NoError(t, err)
			require.Equal(t, want, outs[g])
		}
	}
}

// TestEncodeContextThreadsAll cancels ThreadLevel encodes (lossy with and
// without alpha, lossless, whose encoders run their own helpers) at many
// points under every helper budget: each must either finish with the
// sequential bytes or return exactly context.Canceled without writing,
// and leave no helper behind.
func TestEncodeContextThreadsAll(t *testing.T) {
	checkNoLeak(t)
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	bestFilter, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	bestFilter.AlphaFiltering = 2
	ll, err := NewLosslessEncoderOptions(PresetDefault, 6)
	require.NoError(t, err)
	// 65780 pixels: the alpha plane is coded on a helper when one is free.
	transparent := alphaImage(260, 253, 14)
	cases := []struct {
		name string
		img  image.Image
		o    *Options
	}{
		{"lossy YCbCr", sourceImages(260, 253, 13)[3], lossy},
		{"lossy alpha", transparent, lossy},
		{"lossy alpha best filter", transparent, bestFilter},
		{"lossless", alphaImage(150, 110, 15), ll},
	}
	points := 16
	if testing.Short() {
		cases, points = cases[1:3], 6
	}
	// A sample of the polls: the first and last ones and some in between.
	sample := func(n int64) []int64 {
		ks := pollPoints(n)
		if len(ks) <= points {
			return ks
		}
		var out []int64
		for i := 0; i < points; i++ {
			out = append(out, ks[i*(len(ks)-1)/(points-1)])
		}
		return out
	}
	for _, tc := range cases {
		want := encodeBytes(t, tc.img, tc.o)
		o := *tc.o
		o.ThreadLevel = true
		for _, b := range budgets {
			withBudget(t, b, func() {
				e, err := NewEncoder(tc.img, &o)
				require.NoError(t, err)
				defer e.Release()
				never := newPollCtx(0)
				var got bytes.Buffer
				require.NoError(t, e.EncodeContext(never, &got))
				require.Equal(t, want, got.Bytes(), "%s budget %s", tc.name, b)
				n := never.polls.Load()
				for _, k := range sample(n) {
					name := fmt.Sprintf("%s budget %s poll %d of %d", tc.name, b, k, n)
					var cw countWriter
					err := e.EncodeContext(newPollCtx(k), &cw)
					if err == nil { // the helpers polled less often this time
						continue
					}
					require.Equal(t, context.Canceled, err, name)
					require.Zero(t, cw.calls, name)
				}
			})
			require.Zero(t, workers.InUse(), tc.name)
		}
	}
}

// TestPooledNRGBAAnySize checks that snapshots of any size, 2 MP and more
// included, go back to the pool and are reused, with the size and stride of
// the request.
func TestPooledNRGBAAnySize(t *testing.T) {
	// The 64-bit default budget (32-bit platforms default to 64 MiB, which
	// keeps no single buffer above 8 MiB).
	defer bufpool.SetLimit(bufpool.SetLimit(256 << 20))
	bufpool.Trim()
	defer bufpool.Trim()
	img, pix := pooledNRGBA(1449, 1449) // above v2.0's 2 MP pooling limit
	require.Len(t, img.Pix, 4*1449*1449)
	putPix(pix)
	require.GreaterOrEqual(t, bufpool.Retained(), int64(4*1449*1449), "the large buffer was not pooled")
	img2, pix2 := pooledNRGBA(1449, 1449)
	require.Same(t, &pix[0], &pix2[0], "the pooled buffer was not reused")
	require.Equal(t, 4*1449, img2.Stride)
	putPix(pix2)

	// A smaller request reuses a large enough buffer of its size class.
	small, pix3 := pooledNRGBA(63, 64)
	require.Len(t, small.Pix, 4*63*64)
	require.Equal(t, 4*63, small.Stride)
	require.Equal(t, image.Rect(0, 0, 63, 64), small.Rect)
	putPix(pix3)
	putPix(nil) // no-op
}
