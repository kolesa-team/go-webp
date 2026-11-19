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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// clip8 and yuvToRGB are libwebp's VP8YuvToRgb (src/dsp/yuv.h).
func clip8(v int) uint8 {
	if v&^16383 == 0 {
		return uint8(v >> 6)
	}
	if v < 0 {
		return 0
	}
	return 255
}

func yuvToRGB(y, u, v uint8) (r, g, b uint8) {
	yy := int(y) * 19077 >> 8
	r = clip8(yy + int(v)*26149>>8 - 14234)
	g = clip8(yy - int(u)*6419>>8 - int(v)*13320>>8 + 8708)
	b = clip8(yy + int(u)*33050>>8 - 17685)
	return
}

// checkYCbCrMatchesRGB checks that converting the planes with point-sampled
// chroma gives exactly the RGBA of Decode with NoFancyUpsampling (libwebp's
// "-nofancy" output, the same conversion), and that the alpha plane is the
// alpha channel.
func checkYCbCrMatchesRGB(t *testing.T, ycc *image.YCbCr, a []byte, want *image.NRGBA, msg string) {
	t.Helper()
	b := want.Rect
	require.Equal(t, b, ycc.Rect, msg)
	require.Equal(t, image.YCbCrSubsampleRatio420, ycc.SubsampleRatio)
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, bb := yuvToRGB(ycc.Y[y*ycc.YStride+x], ycc.Cb[(y/2)*ycc.CStride+x/2], ycc.Cr[(y/2)*ycc.CStride+x/2])
			p := want.Pix[y*want.Stride+4*x:]
			alpha := uint8(0xff)
			if a != nil {
				alpha = a[y*b.Dx()+x]
			}
			if p[0] != r || p[1] != g || p[2] != bb || p[3] != alpha {
				t.Fatalf("%s: pixel (%d,%d) is %v, planes give %v", msg, x, y, p[:4], []uint8{r, g, bb, alpha})
			}
		}
	}
}

var ycbcrOptions = []*Options{
	nil,
	{UseThreads: true},
	{BypassFiltering: true},
	{DitheringStrength: 100},
	{AlphaDitheringStrength: 100},
	{Crop: image.Rect(31, 17, 200, 100)},
	{Crop: image.Rect(30, 16, 63, 47), UseThreads: true},
	{Crop: image.Rect(0, 0, 1, 1)},
}

func withNoFancy(o *Options) *Options {
	var c Options
	if o != nil {
		c = *o
	}
	c.NoFancyUpsampling = true
	return &c
}

func TestDecodeYCbCr(t *testing.T) {
	files := []string{"m4_q75.webp", "webp-logo-lossy.webp", "100x150_lossy.webp"}
	if testing.Short() {
		files = []string{"m4_q75.webp", "100x150_lossy.webp"}
	}
	for _, file := range files {
		data := readFile(t, file)
		f := newDec(t, data, nil).GetFeatures()
		hasAlpha := f.HasAlpha
		for _, o := range ycbcrOptions {
			if o != nil && (o.Crop.Min.X+o.Crop.Max.X > f.Width || o.Crop.Min.Y+o.Crop.Max.Y > f.Height) {
				continue
			}
			want := decode(t, data, withNoFancy(o))
			na, err := newDec(t, data, o).DecodeNYCbCrA()
			require.NoError(t, err)
			checkYCbCrMatchesRGB(t, &na.YCbCr, na.A, want, file)
			require.Equal(t, na.Rect.Dx(), na.AStride)
			if hasAlpha {
				_, err := newDec(t, data, o).DecodeYCbCr()
				assert.Equal(t, utils.Vp8StatusUnsupportedFeature, status.CodeOf(err))
				continue
			}
			ycc, err := newDec(t, data, o).DecodeYCbCr()
			require.NoError(t, err)
			checkYCbCrMatchesRGB(t, ycc, nil, want, file)
			assert.Equal(t, &na.YCbCr, ycc)
		}
	}
}

// flipPlane mirrors a plane of h rows of stride bytes.
func flipPlane(p []byte, stride, h int) []byte {
	out := make([]byte, len(p))
	for y := 0; y < h; y++ {
		copy(out[y*stride:(y+1)*stride], p[(h-1-y)*stride:])
	}
	return out
}

// TestDecodeYCbCrFlip checks that Flip mirrors each plane, also for odd
// heights (where the chroma rows no longer line up with the luma pairs, as
// in libwebp's flipped Y'CbCr output).
func TestDecodeYCbCrFlip(t *testing.T) {
	data := readFile(t, "webp-logo-lossy.webp")
	for _, crop := range []image.Rectangle{{}, image.Rect(10, 10, 101, 77), image.Rect(0, 0, 3000, 1)} {
		ref, err := newDec(t, data, &Options{Crop: crop}).DecodeNYCbCrA()
		require.NoError(t, err)
		got, err := newDec(t, data, &Options{Crop: crop, Flip: true}).DecodeNYCbCrA()
		require.NoError(t, err)
		h, ch := ref.Rect.Dy(), (ref.Rect.Dy()+1)/2
		assert.Equal(t, flipPlane(ref.Y, ref.YStride, h), got.Y)
		assert.Equal(t, flipPlane(ref.Cb, ref.CStride, ch), got.Cb)
		assert.Equal(t, flipPlane(ref.Cr, ref.CStride, ch), got.Cr)
		assert.Equal(t, flipPlane(ref.A, ref.AStride, h), got.A)
	}
}

// TestDecodeYCbCrOddSizes covers 1xN, Nx1 and sizes that are not multiples
// of the macroblock size, with and without alpha.
func TestDecodeYCbCrOddSizes(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {1, 37}, {37, 1}, {17, 33}, {33, 18}} {
		w, h := sz[0], sz[1]
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = uint8(i * 37 % 251)
		}
		for _, alpha := range []bool{false, true} {
			if !alpha {
				for i := 3; i < len(img.Pix); i += 4 {
					img.Pix[i] = 0xff
				}
			}
			o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 80)
			require.NoError(t, err)
			buf := &bytes.Buffer{}
			require.NoError(t, encodeTo(buf, img, o))
			data := buf.Bytes()
			for _, opts := range []*Options{nil, {UseThreads: true}, {Flip: true}} {
				na, err := newDec(t, data, opts).DecodeNYCbCrA()
				require.NoError(t, err)
				if opts == nil || !opts.Flip || h%2 == 0 {
					checkYCbCrMatchesRGB(t, &na.YCbCr, na.A, decode(t, data, withNoFancy(opts)), "odd size")
				}
				assert.Len(t, na.Cb, ((w+1)/2)*((h+1)/2))
				ycc, err := newDec(t, data, opts).DecodeYCbCr()
				if alpha {
					assert.Error(t, err)
				} else {
					require.NoError(t, err)
					assert.Equal(t, &na.YCbCr, ycc)
				}
			}
			img.Pix[3] = 0 // has alpha from now on
		}
	}
}

func encodeTo(buf *bytes.Buffer, img image.Image, o *encoder.Options) error {
	e, err := encoder.NewEncoder(img, o)
	if err != nil {
		return err
	}
	return e.Encode(buf)
}

func TestDecodeYCbCrErrors(t *testing.T) {
	m4 := readFile(t, "m4_q75.webp")
	alpha := readFile(t, "webp-logo-lossy.webp")
	lossless := readFile(t, "webp-logo-lossless.webp")
	for _, c := range []struct {
		data []byte
		o    *Options
		code utils.VP8StatusCode
	}{
		{lossless, nil, utils.Vp8StatusUnsupportedFeature},
		{m4, &Options{Scale: image.Rect(0, 0, 10, 10)}, utils.Vp8StatusInvalidParam},
		{m4, &Options{Crop: image.Rect(670, 0, 100, 100)}, utils.Vp8StatusInvalidParam},
		{m4[:len(m4)/2], nil, utils.Vp8StatusNotEnoughData},
		{alpha[:len(alpha)/2], &Options{UseThreads: true}, utils.Vp8StatusNotEnoughData},
		{modifyChunk(t, alpha, "ALPH", func(p []byte) []byte { p[0] = 0xc1; return p }), nil, utils.Vp8StatusOutOfMemory},
	} {
		d := newDec(t, c.data, c.o)
		na, err := d.DecodeNYCbCrA()
		assert.Nil(t, na)
		assert.Equal(t, c.code, status.CodeOf(err), "%v", err)
		assert.Contains(t, err.Error(), "cannot decode picture: ")
		if c.code != utils.Vp8StatusOutOfMemory && !bytes.Equal(c.data, alpha[:len(alpha)/2]) {
			ycc, err := d.DecodeYCbCr()
			assert.Nil(t, ycc)
			assert.Equal(t, c.code, status.CodeOf(err), "%v", err)
		}
	}
}

func BenchmarkDecodeYCbCr(b *testing.B) {
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp"} {
		data := readFile(b, file)
		for _, o := range []struct {
			name string
			opts *Options
		}{
			{"default", nil},
			{"threads", &Options{UseThreads: true}},
		} {
			b.Run(file+"/"+o.name, func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					d, err := NewDecoder(bytes.NewReader(data), o.opts)
					if err != nil {
						b.Fatal(err)
					}
					if d.GetFeatures().HasAlpha {
						_, err = d.DecodeNYCbCrA()
					} else {
						_, err = d.DecodeYCbCr()
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
