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
	"errors"
	"image"
	"image/color"
	"io"
	"math"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
)

// The v1 tests compared the output byte for byte with files produced by
// libwebp. A different encoder produces different (valid) bytes, so these
// tests check the structure of the output instead; pixel round trips are
// tested in package webp.

func TestNewEncoder(t *testing.T) {
	t.Run("encode nrgba image with lossy preset", func(t *testing.T) {
		out := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rect(0, 0, 100, 150))

		options, err := NewLossyEncoderOptions(PresetDefault, 0.75)
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions: %v", err)
		}

		e, err := NewEncoder(img, options)
		if err != nil {
			t.Fatalf("NewEncoder: %v", err)
		}
		if err := e.Encode(out); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		parsed, err := container.Parse(out.Bytes())
		if err != nil {
			t.Fatalf("container.Parse: %v", err)
		}
		if parsed.Width != 100 {
			t.Errorf("Width = %d, want 100", parsed.Width)
		}
		if parsed.Height != 150 {
			t.Errorf("Height = %d, want 150", parsed.Height)
		}
		if !parsed.HasAlpha {
			t.Error("HasAlpha = false, want true: a fully transparent image needs an ALPH chunk")
		}
		if parsed.VP8 == nil {
			t.Error("VP8 = nil, want a lossy bitstream")
		}
		if parsed.ALPH == nil {
			t.Error("ALPH = nil, want an ALPH chunk")
		}
	})
	t.Run("encode nrgba image with lossless preset", func(t *testing.T) {
		out := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rect(0, 0, 100, 150))

		options, err := NewLosslessEncoderOptions(PresetDefault, 1)
		if err != nil {
			t.Fatalf("NewLosslessEncoderOptions: %v", err)
		}

		e, err := NewEncoder(img, options)
		if err != nil {
			t.Fatalf("NewEncoder: %v", err)
		}
		if err := e.Encode(out); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		parsed, err := container.Parse(out.Bytes())
		if err != nil {
			t.Fatalf("container.Parse: %v", err)
		}
		if parsed.Width != 100 {
			t.Errorf("Width = %d, want 100", parsed.Width)
		}
		if parsed.Height != 150 {
			t.Errorf("Height = %d, want 150", parsed.Height)
		}
		if parsed.VP8L == nil {
			t.Error("VP8L = nil, want a lossless bitstream")
		}
	})
}

func TestNewEncoder_NilOptions(t *testing.T) {
	t.Run("encode image without passing options should not panic", func(t *testing.T) {
		out := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rect(0, 0, 100, 150))

		e, err := NewEncoder(img, nil)
		if err != nil {
			t.Fatalf("NewEncoder: %v", err)
		}
		if err := e.Encode(out); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		parsed, err := container.Parse(out.Bytes())
		if err != nil {
			t.Fatalf("container.Parse: %v", err)
		}
		if parsed.VP8 == nil {
			t.Error("VP8 = nil: nil options must encode lossy")
		}
	})
}

func TestEncoder_BadDimensions(t *testing.T) {
	options, err := NewLosslessEncoderOptions(PresetDefault, 4)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions: %v", err)
	}
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 16384, 1),
		image.Rect(0, 0, 1, 20000),
	} {
		e, err := NewEncoder(image.NewNRGBA(r), options)
		if err != nil {
			t.Fatalf("NewEncoder(%v): %v", r, err)
		}
		err = e.Encode(io.Discard)
		if err == nil || err.Error() != "cannot encode webppicture" { // exactly v1's message
			t.Fatalf("Encode(%v) = %v, want %q", r, err, "cannot encode webppicture")
		}
		if errors.Unwrap(err) == nil {
			t.Fatalf("Encode(%v): errors.Unwrap(err) = nil, want the detailed cause wrapped", r)
		}
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("boom") }

type countWriter struct{ calls int }

func (c *countWriter) Write(p []byte) (int, error) { c.calls++; return len(p), nil }

func TestEncoder_Writer(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	img.Pix[3] = 0x80 // forces VP8X + ALPH + VP8
	e, err := NewEncoder(img, nil)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}

	cw := &countWriter{}
	if err := e.Encode(cw); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if cw.calls != 1 {
		t.Errorf("Encode made %d Write calls, want 1: the file must be written with a single Write call", cw.calls)
	}

	if err := e.Encode(failWriter{}); err == nil || err.Error() != "boom" {
		t.Errorf("Encode(failWriter) = %v, want %q", err, "boom")
	}
}

func TestEncoder_InvalidAlphaCompression(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	options, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	options.AlphaCompression = 2
	e, err := NewEncoder(img, options)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	if err := e.Encode(io.Discard); err == nil || err.Error() != "cannot encode webppicture" {
		t.Fatalf("Encode = %v, want %q", err, "cannot encode webppicture")
	}
}

// decodeALPH decodes the ALPH chunk of a lossy file.
func decodeALPH(t *testing.T, data []byte, w, h int) (alpha.Header, []byte) {
	t.Helper()
	parsed, err := container.Parse(data)
	if err != nil {
		t.Fatalf("container.Parse: %v", err)
	}
	if parsed.ALPH == nil {
		t.Fatal("ALPH = nil, want an ALPH chunk")
	}
	hdr, err := alpha.ParseHeader(parsed.ALPH[0])
	if err != nil {
		t.Fatalf("alpha.ParseHeader: %v", err)
	}
	plane := make([]byte, w*h)
	if hdr.Compression == alpha.CompressionNone {
		copy(plane, parsed.ALPH[1:])
	} else if err := vp8l.DecodeAlpha(parsed.ALPH[1:], w, h, plane); err != nil {
		t.Fatalf("vp8l.DecodeAlpha: %v", err)
	}
	alpha.Unfilter(hdr.Filter, plane, w, h, w)
	return hdr, plane
}

func TestEncoder_Alpha(t *testing.T) {
	const w, h = 37, 23
	rnd := rand.New(rand.NewSource(1))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		img.Pix[4*i+0] = uint8(rnd.Intn(256))
		img.Pix[4*i+1] = uint8(i)
		img.Pix[4*i+2] = uint8(i * 3)
		img.Pix[4*i+3] = uint8((i % w) * 7)
	}
	want := make([]byte, w*h)
	for i := range want {
		want[i] = img.Pix[4*i+3]
	}

	for _, compression := range []int{0, 1} {
		for _, filtering := range []int{0, 1, 2, 5} {
			for _, method := range []int{0, 4, 6} {
				options, err := NewLossyEncoderOptions(PresetDefault, 80)
				if err != nil {
					t.Fatalf("NewLossyEncoderOptions: %v", err)
				}
				options.AlphaCompression = compression
				options.AlphaFiltering = filtering
				options.Method = method
				buf := &bytes.Buffer{}
				if err := encodeTo(t, img, options, buf); err != nil {
					t.Fatalf("compression %d filtering %d method %d: encode: %v", compression, filtering, method, err)
				}
				hdr, plane := decodeALPH(t, buf.Bytes(), w, h)
				if hdr.PreProcessing != 0 {
					t.Errorf("compression %d filtering %d method %d: PreProcessing = %d, want 0", compression, filtering, method, hdr.PreProcessing)
				}
				if compression == 0 {
					if hdr.Filter != alpha.FilterNone {
						t.Errorf("compression 0 filtering %d method %d: Filter = %d, want %d (FilterNone)", filtering, method, hdr.Filter, alpha.FilterNone)
					}
				}
				if !bytes.Equal(plane, want) {
					t.Errorf("compression %d filtering %d method %d: the decoded alpha plane (%d bytes) differs from the source alpha (%d bytes)", compression, filtering, method, len(plane), len(want))
				}
			}
		}
	}

	t.Run("quantized alpha", func(t *testing.T) {
		options, err := NewLossyEncoderOptions(PresetDefault, 80)
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions: %v", err)
		}
		options.alphaQuality = 10 // 4 levels
		buf := &bytes.Buffer{}
		if err := encodeTo(t, img, options, buf); err != nil {
			t.Fatalf("encode: %v", err)
		}
		hdr, plane := decodeALPH(t, buf.Bytes(), w, h)
		if hdr.PreProcessing != alpha.PreprocessingLevels {
			t.Errorf("PreProcessing = %d, want %d (PreprocessingLevels)", hdr.PreProcessing, alpha.PreprocessingLevels)
		}
		levels := map[byte]bool{}
		for _, v := range plane {
			levels[v] = true
		}
		if limit := alpha.AlphaLevels(10); len(levels) > limit {
			t.Errorf("%d alpha levels, want at most %d", len(levels), limit)
		}
	})
}

// encodeTo is a test helper running NewEncoder + Encode.
func encodeTo(t *testing.T, img image.Image, options *Options, w io.Writer) error {
	t.Helper()
	e, err := NewEncoder(img, options)
	if err != nil {
		return err
	}
	return e.Encode(w)
}

func TestToNRGBA(t *testing.T) {
	r := image.Rect(3, 5, 20, 17)
	rnd := rand.New(rand.NewSource(2))

	rgba := image.NewRGBA(r)
	rnd.Read(rgba.Pix)
	for i := 0; i < len(rgba.Pix); i += 4 { // valid premultiplied colors
		a := rgba.Pix[i+3]
		for c := 0; c < 3; c++ {
			if rgba.Pix[i+c] > a {
				rgba.Pix[i+c] = a
			}
		}
	}
	gray := image.NewGray(r)
	rnd.Read(gray.Pix)
	ycc := image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
	rnd.Read(ycc.Y)
	rnd.Read(ycc.Cb)
	rnd.Read(ycc.Cr)
	nycca := image.NewNYCbCrA(r, image.YCbCrSubsampleRatio422)
	rnd.Read(nycca.Y)
	rnd.Read(nycca.Cb)
	rnd.Read(nycca.Cr)
	rnd.Read(nycca.A)
	pal := image.NewPaletted(r, color.Palette{color.Black, color.RGBA{10, 20, 30, 40}, color.White})
	for i := range pal.Pix {
		pal.Pix[i] = uint8(rnd.Intn(3))
	}
	gray16 := image.NewGray16(r)
	rnd.Read(gray16.Pix)

	for _, src := range []image.Image{rgba, gray, ycc, nycca, pal, gray16} {
		got := toNRGBA(src)
		if got.Rect.Dx() != r.Dx() {
			t.Fatalf("%T: width = %d, want %d", src, got.Rect.Dx(), r.Dx())
		}
		if got.Rect.Dy() != r.Dy() {
			t.Fatalf("%T: height = %d, want %d", src, got.Rect.Dy(), r.Dy())
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				want := color.NRGBAModel.Convert(src.At(x, y)).(color.NRGBA)
				have := got.NRGBAAt(x-r.Min.X, y-r.Min.Y)
				if src == nycca {
					// Non-premultiplied source: no premultiplication round
					// trip (color.NRGBAModel would lose precision).
					c := nycca.NYCbCrAAt(x, y)
					rr, gg, bb := color.YCbCrToRGB(c.Y, c.Cb, c.Cr)
					want = color.NRGBA{rr, gg, bb, c.A}
				}
				if src == ycc || src == nycca {
					// color.YCbCrToRGB vs the 16-bit YCbCr.RGBA path may
					// differ by one.
					if d := int(want.R) - int(have.R); d < -1 || d > 1 {
						t.Fatalf("%T at %d,%d: R = %d, want %d ± 1", src, x, y, have.R, want.R)
					}
					if d := int(want.G) - int(have.G); d < -1 || d > 1 {
						t.Fatalf("%T at %d,%d: G = %d, want %d ± 1", src, x, y, have.G, want.G)
					}
					if d := int(want.B) - int(have.B); d < -1 || d > 1 {
						t.Fatalf("%T at %d,%d: B = %d, want %d ± 1", src, x, y, have.B, want.B)
					}
					if have.A != want.A {
						t.Fatalf("%T at %d,%d: A = %d, want %d", src, x, y, have.A, want.A)
					}
					continue
				}
				if have != want {
					t.Fatalf("%T at %d,%d: got %v, want %v", src, x, y, have, want)
				}
			}
		}
	}
}

func BenchmarkToNRGBA_RGBA(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 1024, 768))
	rand.New(rand.NewSource(1)).Read(img.Pix)
	b.ReportAllocs()
	b.SetBytes(int64(len(img.Pix)))
	for i := 0; i < b.N; i++ {
		toNRGBA(img)
	}
}

// TestToNRGBAOpaqueRGBACopies checks that an opaque *image.RGBA is copied
// (NewEncoder takes a snapshot, like v1) and encodes like the equivalent
// NRGBA.
func TestToNRGBAOpaqueRGBACopies(t *testing.T) {
	r := image.Rect(3, 5, 40, 33)
	rgba := image.NewRGBA(r)
	rand.New(rand.NewSource(4)).Read(rgba.Pix)
	for i := 3; i < len(rgba.Pix); i += 4 {
		rgba.Pix[i] = 0xff
	}
	got := toNRGBA(rgba)
	if &rgba.Pix[0] == &got.Pix[0] {
		t.Fatal("toNRGBA shares the pixels of an opaque *image.RGBA, want a copy")
	}
	if got.Rect.Size() != r.Size() {
		t.Fatalf("size = %v, want %v", got.Rect.Size(), r.Size())
	}

	nrgba := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := 0; y < r.Dy(); y++ {
		copy(nrgba.Pix[y*nrgba.Stride:][:4*r.Dx()], rgba.Pix[rgba.PixOffset(r.Min.X, r.Min.Y+y):][:4*r.Dx()])
	}
	for _, lossless := range []bool{false, true} {
		var o *Options
		var err error
		if lossless {
			o, err = NewLosslessEncoderOptions(PresetDefault, 6)
		} else {
			o, err = NewLossyEncoderOptions(PresetDefault, 75)
		}
		if err != nil {
			t.Fatalf("lossless=%v: options: %v", lossless, err)
		}
		var a, b bytes.Buffer
		e1, err := NewEncoder(rgba, o)
		if err != nil {
			t.Fatalf("lossless=%v: NewEncoder(RGBA): %v", lossless, err)
		}
		if err := e1.Encode(&a); err != nil {
			t.Fatalf("lossless=%v: Encode(RGBA): %v", lossless, err)
		}
		e2, err := NewEncoder(nrgba, o)
		if err != nil {
			t.Fatalf("lossless=%v: NewEncoder(NRGBA): %v", lossless, err)
		}
		if err := e2.Encode(&b); err != nil {
			t.Fatalf("lossless=%v: Encode(NRGBA): %v", lossless, err)
		}
		if !bytes.Equal(a.Bytes(), b.Bytes()) {
			t.Fatalf("lossless=%v: the opaque *image.RGBA encodes differently from the equivalent *image.NRGBA (%d bytes, want %d)", lossless, a.Len(), b.Len())
		}
	}
}

// TestNewEncoderSnapshot checks v1's semantics: NewEncoder takes a snapshot
// of the source (v1 imported it into a WebPPicture), so changes made to an
// *image.RGBA, *image.YCbCr or other source between NewEncoder and Encode
// do not affect the output. An *image.NRGBA is used in place, as in v1.
func TestNewEncoderSnapshot(t *testing.T) {
	const w, h = 48, 40
	rng := rand.New(rand.NewSource(9))
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	rng.Read(rgba.Pix)
	for i := 3; i < len(rgba.Pix); i += 4 {
		rgba.Pix[i] = 0xff
	}
	translucent := image.NewRGBA(image.Rect(0, 0, w, h))
	rng.Read(translucent.Pix)
	for i := 0; i < len(translucent.Pix); i += 4 {
		a := translucent.Pix[i+3]
		for c := 0; c < 3; c++ {
			translucent.Pix[i+c] = uint8(uint16(translucent.Pix[i+c]) * uint16(a) / 255)
		}
	}
	ycc := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	rng.Read(ycc.Y)
	rng.Read(ycc.Cb)
	rng.Read(ycc.Cr)
	gray := image.NewGray(image.Rect(0, 0, w, h))
	rng.Read(gray.Pix)

	sources := []struct {
		name   string
		img    image.Image
		mutate func()
	}{
		{"RGBA", rgba, func() { clear(rgba.Pix) }},
		{"RGBA translucent", translucent, func() { clear(translucent.Pix) }},
		{"YCbCr", ycc, func() { clear(ycc.Y); clear(ycc.Cb) }},
		{"Gray", gray, func() { clear(gray.Pix) }},
	}
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	lossless, err := NewLosslessEncoderOptions(PresetDefault, 6)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions: %v", err)
	}
	for _, src := range sources {
		for _, o := range []*Options{lossy, lossless} {
			var want bytes.Buffer
			e0, err := NewEncoder(src.img, o)
			if err != nil {
				t.Fatalf("%s lossless=%v: NewEncoder: %v", src.name, o.Lossless, err)
			}
			if err := e0.Encode(&want); err != nil {
				t.Fatalf("%s lossless=%v: Encode: %v", src.name, o.Lossless, err)
			}

			e, err := NewEncoder(src.img, o)
			if err != nil {
				t.Fatalf("%s lossless=%v: NewEncoder: %v", src.name, o.Lossless, err)
			}
			saved := snapshotPix(src.img)
			src.mutate()
			var got bytes.Buffer
			if err := e.Encode(&got); err != nil {
				t.Fatalf("%s lossless=%v: Encode after the source changed: %v", src.name, o.Lossless, err)
			}
			if !bytes.Equal(got.Bytes(), want.Bytes()) {
				t.Errorf("%s lossless=%v: source changed after NewEncoder: output differs (%d bytes, want %d)", src.name, o.Lossless, got.Len(), want.Len())
			}
			// Encode can be called again (the snapshot is not released).
			got.Reset()
			if err := e.Encode(&got); err != nil {
				t.Fatalf("%s lossless=%v: second Encode: %v", src.name, o.Lossless, err)
			}
			if !bytes.Equal(got.Bytes(), want.Bytes()) {
				t.Errorf("%s lossless=%v: second Encode: output differs (%d bytes, want %d)", src.name, o.Lossless, got.Len(), want.Len())
			}
			restorePix(src.img, saved)
		}
	}

	t.Run("NRGBA is shared", func(t *testing.T) {
		n := image.NewNRGBA(image.Rect(0, 0, w, h))
		rng.Read(n.Pix)
		e, err := NewEncoder(n, lossless)
		if err != nil {
			t.Fatalf("NewEncoder: %v", err)
		}
		var before bytes.Buffer
		if err := e.Encode(&before); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		clear(n.Pix)
		var after bytes.Buffer
		if err := e.Encode(&after); err != nil {
			t.Fatalf("Encode after clearing the source: %v", err)
		}
		if bytes.Equal(before.Bytes(), after.Bytes()) {
			t.Error("clearing the *image.NRGBA source did not change the output, want its pixels used in place")
		}
	})
}

func snapshotPix(img image.Image) [][]byte {
	switch v := img.(type) {
	case *image.RGBA:
		return [][]byte{append([]byte(nil), v.Pix...)}
	case *image.Gray:
		return [][]byte{append([]byte(nil), v.Pix...)}
	case *image.YCbCr:
		return [][]byte{append([]byte(nil), v.Y...), append([]byte(nil), v.Cb...), append([]byte(nil), v.Cr...)}
	}
	panic("unexpected image type")
}

func restorePix(img image.Image, saved [][]byte) {
	switch v := img.(type) {
	case *image.RGBA:
		copy(v.Pix, saved[0])
	case *image.Gray:
		copy(v.Pix, saved[0])
	case *image.YCbCr:
		copy(v.Y, saved[0])
		copy(v.Cb, saved[1])
		copy(v.Cr, saved[2])
	}
}

// TestEncodeNaN checks that a NaN Quality or TargetPsnr, which libwebp's
// WebPValidateConfig (and so v1) accepts, encodes without error and like
// libwebp: a NaN quality maps to quantizer 0 (the same output as quality
// 100 at the default method, as with libwebp 1.6.0 through v1), and a NaN
// PSNR target is ignored.
func TestEncodeNaN(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 64, 48))
	rand.New(rand.NewSource(3)).Read(img.Pix)
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 0xff
	}
	nan := float32(math.NaN())
	encode := func(f func(o *Options), lossless bool) []byte {
		t.Helper()
		var o *Options
		var err error
		if lossless {
			o, err = NewLosslessEncoderOptions(PresetDefault, 6)
		} else {
			o, err = NewLossyEncoderOptions(PresetDefault, 75)
		}
		if err != nil {
			t.Fatalf("lossless=%v: options: %v", lossless, err)
		}
		f(o)
		var buf bytes.Buffer
		if err := encodeTo(t, img, o, &buf); err != nil {
			t.Fatalf("lossless=%v: encode: %v", lossless, err)
		}
		_, err = container.Parse(buf.Bytes())
		if err != nil {
			t.Fatalf("lossless=%v: container.Parse: %v", lossless, err)
		}
		return buf.Bytes()
	}
	for m := 0; m <= 6; m++ {
		got := encode(func(o *Options) { o.Quality, o.Method = nan, m }, false)
		if m >= 2 {
			// The quality only enters the quantizers (and the error diffusion
			// switch, off for both); methods 0-1 also use (int)quality in
			// the analysis.
			want := encode(func(o *Options) { o.Quality, o.Method = 100, m }, false)
			if !bytes.Equal(got, want) {
				t.Errorf("method %d: NaN quality: output differs from quality 100 (%d bytes, want %d)", m, len(got), len(want))
			}
		}
	}
	for _, pass := range []int{1, 3} {
		want := encode(func(o *Options) { o.Pass = pass }, false)
		got := encode(func(o *Options) { o.TargetPsnr, o.Pass = nan, pass }, false)
		if !bytes.Equal(got, want) {
			t.Errorf("pass %d: NaN target PSNR: output differs from no target (%d bytes, want %d)", pass, len(got), len(want))
		}
	}
	// Searching with a NaN quality must terminate.
	encode(func(o *Options) { o.Quality, o.TargetSize, o.Pass = nan, 500, 10 }, false)
	encode(func(o *Options) { o.Quality = nan }, true)
}
