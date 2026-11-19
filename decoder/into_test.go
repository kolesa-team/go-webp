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

package decoder

import (
	"bytes"
	"image"
	"math"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// intoOptions are the option sets the output-reuse tests run with: every
// sink (direct, flip, crop, crop+flip, scale) and the decoder variants.
var intoOptions = []*Options{
	nil,
	{UseThreads: true},
	{NoFancyUpsampling: true},
	{BypassFiltering: true, DitheringStrength: 100},
	{AlphaDitheringStrength: 100},
	{Flip: true},
	{Crop: image.Rect(31, 17, 200, 100)},
	{Crop: image.Rect(0, 5, 675, 300)}, // full width
	{Crop: image.Rect(30, 16, 64, 48), Flip: true},
	{Scale: image.Rect(0, 0, 123, 45)},
	{Scale: image.Rect(0, 0, 320, 240), UseThreads: true},
	{Crop: image.Rect(10, 10, 200, 100), Scale: image.Rect(0, 0, 400, 50), Flip: true},
}

var intoFiles = []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp", "100x150_lossy.webp", "100x150_lossless.webp"}

// dirty returns an image whose Pix has capacity c and is filled with a
// pattern, with a header that does not describe it.
func dirty(c int) *image.NRGBA {
	pix := make([]byte, c)
	for i := range pix {
		pix[i] = byte(0x5a ^ i*7)
	}
	return &image.NRGBA{Pix: pix[:c/3], Stride: 12, Rect: image.Rect(1, 2, 4, 5)}
}

func newDec(t testing.TB, data []byte, o *Options) *Decoder {
	t.Helper()
	d, err := NewDecoder(bytes.NewReader(data), o)
	require.NoError(t, err)
	return d
}

// TestDecodeIntoMatchesDecode decodes into dirty buffers (large enough, and
// too small) and compares with Decode.
func TestDecodeIntoMatchesDecode(t *testing.T) {
	files, sizes := intoFiles, []int{0, 4096, -1}
	if testing.Short() {
		// The 3000x2000 pictures are covered by TestDecodeIntoReuse.
		files, sizes = []string{"m4_q75.webp", "100x150_lossy.webp", "100x150_lossless.webp"}, []int{0, -1}
	}
	for _, file := range files {
		data := readFile(t, file)
		for _, o := range intoOptions {
			if o != nil && !o.Crop.Empty() {
				f := newDec(t, data, nil).GetFeatures()
				if o.Crop.Min.X+o.Crop.Max.X > f.Width || o.Crop.Min.Y+o.Crop.Max.Y > f.Height {
					continue
				}
			}
			want := decode(t, data, o)
			n := len(want.Pix)
			for _, extra := range sizes {
				c := n + extra
				dst := dirty(c)
				got, err := newDec(t, data, o).DecodeInto(dst)
				require.NoError(t, err, "%s %+v", file, o)
				require.Same(t, dst, got)
				require.Equal(t, want.Rect, got.Rect, "%s %+v", file, o)
				require.Equal(t, want.Stride, got.Stride)
				require.True(t, bytes.Equal(want.Pix, got.Pix), "%s %+v cap %d", file, o, c)
				if c >= n {
					assert.Equal(t, c, cap(got.Pix), "%s %+v: buffer not reused", file, o)
				}
			}
		}
	}
}

// TestDecodeIntoReuse decodes pictures of different sizes into one buffer:
// once it has grown to the largest output, nothing proportional to the
// picture is allocated any more.
func TestDecodeIntoReuse(t *testing.T) {
	big := readFile(t, "webp-logo-lossless.webp") // 3000x2000
	small := readFile(t, "m4_q75.webp")           // 675x900
	alpha := readFile(t, "webp-logo-lossy.webp")  // 3000x2000 with alpha
	var dst *image.NRGBA
	var err error
	dst, err = newDec(t, small, nil).DecodeInto(dst)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, 675, 900), dst.Rect)
	dst, err = newDec(t, big, nil).DecodeInto(dst)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, 3000, 2000), dst.Rect)
	p0 := &dst.Pix[0]
	for _, c := range []struct {
		data []byte
		o    *Options
		rect image.Rectangle
	}{
		{small, nil, image.Rect(0, 0, 675, 900)},
		{big, &Options{Flip: true}, image.Rect(0, 0, 3000, 2000)},
		{alpha, nil, image.Rect(0, 0, 3000, 2000)},
		{small, &Options{Crop: image.Rect(10, 10, 100, 100)}, image.Rect(0, 0, 100, 100)},
		{alpha, &Options{Scale: image.Rect(0, 0, 300, 200)}, image.Rect(0, 0, 300, 200)},
		{big, &Options{UseThreads: true}, image.Rect(0, 0, 3000, 2000)},
	} {
		want := decode(t, c.data, c.o)
		// The internal pools are sync.Pools: a decode whose goroutine moved
		// to another P since the previous one, or that a garbage collection
		// separated from it, misses them, which happens often on a loaded
		// machine. So the decode is repeated, after warming the pools each
		// time, and the smallest allocation counts: a decode that allocates
		// in proportion to the picture fails every time.
		attempts := 5
		if raceEnabled {
			attempts = 1
		}
		var least uint64 = math.MaxUint64
		for attempt := 0; attempt < attempts && least >= reuseAllocLimit; attempt++ {
			_, err := newDec(t, c.data, c.o).DecodeInto(dst) // warm the internal pools
			require.NoError(t, err)
			d := newDec(t, c.data, c.o)
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			got, err := d.DecodeInto(dst)
			runtime.ReadMemStats(&m1)
			require.NoError(t, err)
			require.Equal(t, c.rect, got.Rect)
			require.Same(t, p0, &got.Pix[0])
			require.Equal(t, 3000*2000*4, cap(got.Pix))
			require.True(t, bytes.Equal(want.Pix, got.Pix))
			least = min(least, m1.TotalAlloc-m0.TotalAlloc)
		}
		// The output alone would be 2.4-24 MB. (With the race detector
		// sync.Pool drops objects at random.)
		if !raceEnabled {
			assert.Less(t, least, uint64(reuseAllocLimit), "%+v", c.o)
		}
	}
}

// reuseAllocLimit is the most TestDecodeIntoReuse lets a decode into a
// large enough buffer allocate.
const reuseAllocLimit = 2 << 20

// TestDecodeIntoNil checks that a nil dst allocates like Decode.
func TestDecodeIntoNil(t *testing.T) {
	data := readFile(t, "100x150_lossy.webp")
	got, err := newDec(t, data, nil).DecodeInto(nil)
	require.NoError(t, err)
	assert.Equal(t, decode(t, data, nil), got)
}

// TestDecodeIntoErrors checks that a failed decode reports the same error
// as Decode and leaves the dst fields unchanged.
func TestDecodeIntoErrors(t *testing.T) {
	m4 := readFile(t, "m4_q75.webp")
	lossless := readFile(t, "webp-logo-lossless.webp")
	alpha := readFile(t, "webp-logo-lossy.webp")
	for _, c := range []struct {
		data []byte
		o    *Options
		code utils.VP8StatusCode
	}{
		{m4[:len(m4)/2], nil, utils.Vp8StatusNotEnoughData},
		{lossless[:len(lossless)/2], nil, utils.Vp8StatusNotEnoughData},
		{alpha[:len(alpha)/2], &Options{UseThreads: true}, utils.Vp8StatusNotEnoughData},
		{alpha[:len(alpha)/2], &Options{Scale: image.Rect(0, 0, 30, 20)}, utils.Vp8StatusNotEnoughData},
		{m4, &Options{Crop: image.Rect(670, 0, 100, 100)}, utils.Vp8StatusInvalidParam},
		{lossless, &Options{Crop: image.Rect(2990, 0, 100, 100)}, utils.Vp8StatusInvalidParam},
	} {
		for _, size := range []int{0, 100, 3000 * 2000 * 4} {
			dst := dirty(size)
			before := *dst
			_, wantErr := newDec(t, c.data, c.o).Decode()
			require.Error(t, wantErr)
			got, err := newDec(t, c.data, c.o).DecodeInto(dst)
			assert.Nil(t, got)
			require.EqualError(t, err, wantErr.Error())
			assert.Equal(t, c.code, status.CodeOf(err))
			assert.Equal(t, before, *dst, "dst fields must be unchanged on error")
		}
	}
}

func BenchmarkDecodeInto(b *testing.B) {
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data := readFile(b, file)
		for _, o := range []struct {
			name string
			opts *Options
		}{
			{"default", nil},
			{"threads", &Options{UseThreads: true}},
			{"scale", &Options{Scale: image.Rectangle{Max: image.Pt(320, 240)}}},
		} {
			b.Run(file+"/"+o.name, func(b *testing.B) {
				b.ReportAllocs()
				var dst image.NRGBA
				for i := 0; i < b.N; i++ {
					d, err := NewDecoder(bytes.NewReader(data), o.opts)
					if err != nil {
						b.Fatal(err)
					}
					if _, err := d.DecodeInto(&dst); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
