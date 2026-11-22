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

package webp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/vp8enc"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// toNRGBA converts any image to a fresh *image.NRGBA with color.NRGBAModel.
func toNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x-b.Min.X, y-b.Min.Y, src.At(x, y))
		}
	}
	return dst
}

// psnr returns the RGB PSNR of b against a (alpha is ignored).
func psnr(a, b *image.NRGBA) float64 {
	w, h := a.Rect.Dx(), a.Rect.Dy()
	var sse float64
	for y := 0; y < h; y++ {
		ra := a.Pix[a.PixOffset(a.Rect.Min.X, a.Rect.Min.Y+y):][:4*w]
		rb := b.Pix[b.PixOffset(b.Rect.Min.X, b.Rect.Min.Y+y):][:4*w]
		for i := 0; i < 4*w; i++ {
			if i&3 == 3 {
				continue
			}
			d := float64(ra[i]) - float64(rb[i])
			sse += d * d
		}
	}
	if sse == 0 {
		return 99
	}
	return 10 * math.Log10(255*255*3*float64(w*h)/sse)
}

type synth struct {
	name string
	img  *image.NRGBA
}

// synthImages returns synthetic test pictures of the given size.
func synthImages(w, h int) []synth {
	rnd := rand.New(rand.NewSource(int64(w*7919 + h)))
	newImg := func() *image.NRGBA { return image.NewNRGBA(image.Rect(0, 0, w, h)) }

	grad := newImg()
	noise := newImg()
	withAlpha := newImg()
	pal := newImg()
	palette := make([]color.NRGBA, 1+rnd.Intn(16))
	for i := range palette {
		palette[i] = color.NRGBA{uint8(rnd.Intn(256)), uint8(rnd.Intn(256)), uint8(rnd.Intn(256)), 255}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gx, gy := x*255/max(w-1, 1), y*255/max(h-1, 1)
			grad.SetNRGBA(x, y, color.NRGBA{uint8(gx), uint8(gy), uint8((gx + gy) / 2), 255})
			noise.SetNRGBA(x, y, color.NRGBA{uint8(rnd.Intn(256)), uint8(rnd.Intn(256)), uint8(rnd.Intn(256)), 255})
			a := uint8((x + 2*y) * 9)
			if (x/4+y/4)%3 == 0 {
				a = 0
			} else if (x/4+y/4)%3 == 1 {
				a = 255
			}
			withAlpha.SetNRGBA(x, y, color.NRGBA{uint8(gx), 100, uint8(gy), a})
			pal.SetNRGBA(x, y, palette[(x/3+y/2)%len(palette)])
		}
	}
	return []synth{{"gradient", grad}, {"noise", noise}, {"alpha", withAlpha}, {"palette", pal}}
}

var roundtripSizes = [][2]int{{1, 1}, {3, 7}, {17, 33}, {1000, 1}, {1, 300}, {64, 64}}

// testSizes returns the picture sizes for the round-trip tests (fewer with
// -short).
func testSizes() [][2]int {
	if testing.Short() {
		return [][2]int{{1, 1}, {17, 33}}
	}
	return roundtripSizes
}

func encodeDecode(t *testing.T, img image.Image, o *encoder.Options) *image.NRGBA {
	t.Helper()
	buf := &bytes.Buffer{}
	if err := Encode(buf, img, o); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	out, err := Decode(bytes.NewReader(buf.Bytes()), nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	n := out.(*image.NRGBA)
	if got, want := n.Bounds().Size(), img.Bounds().Size(); got != want {
		t.Fatalf("decoded size %v, want %v", got, want)
	}
	return n
}

func TestRoundtripLossless(t *testing.T) {
	for _, size := range testSizes() {
		for _, s := range synthImages(size[0], size[1]) {
			for level := 0; level <= 9; level++ {
				for _, exact := range []int{0, 1} {
					if level == 9 && size[0]*size[1] > 1000 {
						continue // slow
					}
					name := fmt.Sprintf("%dx%d/%s/z%d/exact%d", size[0], size[1], s.name, level, exact)
					o, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, level)
					if err != nil {
						t.Fatalf("%s: NewLosslessEncoderOptions: %v", name, err)
					}
					o.Exact = exact
					got := encodeDecode(t, s.img, o)
					for i := 0; i < len(got.Pix); i += 4 {
						want := s.img.Pix[i : i+4]
						if exact == 0 && want[3] == 0 {
							if got.Pix[i+3] != want[3] {
								t.Fatalf("%s pixel %d: alpha %d, want %d", name, i/4, got.Pix[i+3], want[3])
							}
							continue
						}
						if !bytes.Equal(want, got.Pix[i:i+4]) {
							t.Fatalf("%s pixel %d = %v, want %v", name, i/4, got.Pix[i:i+4], want)
						}
					}
				}
			}
		}
	}
}

func TestRoundtripLossy(t *testing.T) {
	presets := []encoder.EncodingPreset{encoder.PresetDefault, encoder.PresetPicture, encoder.PresetPhoto,
		encoder.PresetDrawing, encoder.PresetIcon, encoder.PresetText}
	for _, size := range testSizes() {
		for _, s := range synthImages(size[0], size[1]) {
			for _, preset := range presets {
				for _, q := range []float32{0, 50, 90, 100} {
					for method := 0; method <= 6; method++ {
						if preset != encoder.PresetDefault && method != 4 {
							continue
						}
						name := fmt.Sprintf("%dx%d/%s/preset%d/q%g/m%d", size[0], size[1], s.name, preset, q, method)
						o, err := encoder.NewLossyEncoderOptions(preset, q)
						if err != nil {
							t.Fatalf("%s: NewLossyEncoderOptions: %v", name, err)
						}
						o.Method = method
						o.Exact = 1
						got := encodeDecode(t, s.img, o)
						// alpha_quality is 100: alpha is coded losslessly.
						for i := 3; i < len(got.Pix); i += 4 {
							if got.Pix[i] != s.img.Pix[i] {
								t.Fatalf("%s alpha %d = %d, want %d", name, i/4, got.Pix[i], s.img.Pix[i])
							}
						}
						if s.name == "gradient" && q >= 50 && size[0] >= 17 {
							if p := psnr(s.img, got); !(p > 30.0) {
								t.Fatalf("%s: PSNR %.2f dB, want > 30", name, p)
							}
						}
						if q >= 90 && s.name == "gradient" {
							if p := psnr(s.img, got); !(p > 20.0) {
								t.Fatalf("%s: PSNR %.2f dB, want > 20", name, p)
							}
						}
					}
				}
			}
		}
	}
}

func TestRoundtripLossyOptions(t *testing.T) {
	img := synthImages(97, 61)[0].img
	mutations := map[string]func(o *encoder.Options){
		"segments1":      func(o *encoder.Options) { o.Segments = 1 },
		"partitions3":    func(o *encoder.Options) { o.Partitions = 3 },
		"simple filter":  func(o *encoder.Options) { o.FilterType = 0 },
		"autofilter":     func(o *encoder.Options) { o.Autofilter = true },
		"sharp yuv":      func(o *encoder.Options) { o.UseSharpYuv = true },
		"preprocessing":  func(o *encoder.Options) { o.Preprocessing = 7 },
		"target size":    func(o *encoder.Options) { o.TargetSize = 2000 },
		"target psnr":    func(o *encoder.Options) { o.TargetPsnr = 40 },
		"pass 10":        func(o *encoder.Options) { o.Pass = 10 },
		"threads":        func(o *encoder.Options) { o.ThreadLevel = true },
		"low memory":     func(o *encoder.Options) { o.LowMemory = true },
		"jpeg emulation": func(o *encoder.Options) { o.EmulateJpegSize = true },
		"partition lim":  func(o *encoder.Options) { o.PartitionLimit = 100 },
	}
	for name, mut := range mutations {
		o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			t.Fatalf("%s: NewLossyEncoderOptions: %v", name, err)
		}
		mut(o)
		got := encodeDecode(t, img, o)
		if p := psnr(img, got); !(p > 25.0) {
			t.Fatalf("%s: PSNR %.2f dB, want > 25", name, p)
		}
	}
}

func TestRoundtripSourceTypes(t *testing.T) {
	r := image.Rect(5, 3, 70, 50)
	rgba := image.NewRGBA(r)
	gray := image.NewGray(r)
	ycc := image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			rgba.SetRGBA(x, y, color.RGBA{uint8(x * 3), uint8(y * 4), uint8(x + y), 255})
			gray.SetGray(x, y, color.Gray{uint8(x * y)})
		}
	}
	for i := range ycc.Y {
		ycc.Y[i] = uint8(i)
	}
	for i := range ycc.Cb {
		ycc.Cb[i] = uint8(128 + i%16)
		ycc.Cr[i] = uint8(128 - i%16)
	}
	for _, src := range []image.Image{rgba, gray, ycc} {
		ref := toNRGBA(src)
		lossless, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 6)
		if err != nil {
			t.Fatalf("NewLosslessEncoderOptions: %v", err)
		}
		got := encodeDecode(t, src, lossless)
		if _, isYCC := src.(*image.YCbCr); isYCC {
			if p := psnr(ref, got); !(p > 45.0) { // YCbCr->RGB rounding
				t.Fatalf("%T lossless: PSNR %.2f dB, want > 45", src, p)
			}
		} else if !bytes.Equal(ref.Pix, got.Pix) {
			t.Fatalf("%T lossless: decoded pixels differ from the source (%d bytes, want %d)", src, len(got.Pix), len(ref.Pix))
		}
		got = encodeDecode(t, src, nil)
		if p := psnr(ref, got); !(p > 30.0) {
			t.Fatalf("%T lossy: PSNR %.2f dB, want > 30", src, p)
		}
	}
}

// TestAlphaDithering checks the AlphaDitheringStrength option on a file
// whose alpha plane was quantized (ALPH pre-processing = 1).
func TestAlphaDithering(t *testing.T) {
	const w, h = 64, 48
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	plane := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8(x * 4)
			img.SetNRGBA(x, y, color.NRGBA{200, 100, 50, a})
			plane[y*w+x] = a
		}
	}
	if err := alpha.QuantizeLevels(plane, w, h, w, 4); err != nil {
		t.Fatalf("QuantizeLevels: %v", err)
	}
	quantized := bytes.Clone(plane)
	enc, err := vp8lenc.EncodeAlpha(plane, w, h, w, 4, 50)
	if err != nil {
		t.Fatalf("EncodeAlpha: %v", err)
	}
	alph := append([]byte{alpha.Header{Compression: alpha.CompressionLossless, PreProcessing: alpha.PreprocessingLevels}.Byte()}, enc...)
	pic := vp8enc.FromNRGBA(img, false, true)
	vp8, err := vp8enc.Encode(pic, nil)
	if err != nil {
		t.Fatalf("vp8enc.Encode: %v", err)
	}
	buf := &bytes.Buffer{}
	if err := container.WriteLossy(buf, w, h, vp8, alph); err != nil {
		t.Fatalf("WriteLossy: %v", err)
	}

	for _, strength := range []int{0, 50, 100, 1000} {
		out, err := Decode(bytes.NewReader(buf.Bytes()), &decoder.Options{AlphaDitheringStrength: strength})
		if err != nil {
			t.Fatalf("strength %d: Decode: %v", strength, err)
		}
		want := bytes.Clone(quantized)
		if strength > 0 {
			if err := alpha.Dequantize(want, w, h, w, min(strength, 100)); err != nil {
				t.Fatalf("strength %d: Dequantize: %v", strength, err)
			}
		}
		n := out.(*image.NRGBA)
		for i := range want {
			if n.Pix[4*i+3] != want[i] {
				t.Fatalf("strength %d pixel %d: alpha %d, want %d", strength, i, n.Pix[4*i+3], want[i])
			}
		}
	}

	f, err := GetFeatures(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if f.Format != utils.FormatLossy {
		t.Fatalf("Format = %v, want %v (lossy)", f.Format, utils.FormatLossy)
	}
	if !f.HasAlpha {
		t.Fatal("HasAlpha = false, want true")
	}
}

// TestDecodeScaledAlpha decodes a lossy picture with alpha through the YUV
// scaling path.
func TestDecodeScaledAlpha(t *testing.T) {
	img := synthImages(80, 60)[2].img
	buf := &bytes.Buffer{}
	if err := Encode(buf, img, nil); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for _, o := range []*decoder.Options{
		{Scale: image.Rectangle{Max: image.Pt(40, 30)}},
		{Scale: image.Rectangle{Max: image.Pt(160, 90)}, Flip: true},
		{Scale: image.Rectangle{Max: image.Pt(33, 21)}, Crop: image.Rectangle{Min: image.Pt(9, 5), Max: image.Pt(50, 40)}},
	} {
		out, err := Decode(bytes.NewReader(buf.Bytes()), o)
		if err != nil {
			t.Fatalf("Decode(%+v): %v", o, err)
		}
		if got := out.Bounds().Max; got != o.Scale.Max {
			t.Fatalf("Decode(%+v): Bounds().Max = %v, want %v", o, got, o.Scale.Max)
		}
	}
}
