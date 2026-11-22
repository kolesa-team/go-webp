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
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"os"
	"reflect"
	"testing"
	"testing/iotest"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func loadJPEG(t testing.TB, name string) image.Image {
	t.Helper()
	r, err := os.Open("../test_data/images/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	img, err := jpeg.Decode(r)
	if err != nil {
		t.Fatalf("jpeg.Decode(%s): %v", name, err)
	}
	return img
}

func TestEncode(t *testing.T) {
	t.Run("encode lossy", func(t *testing.T) {
		img := loadJPEG(t, "source.jpg")

		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions: %v", err)
		}

		buf := &bytes.Buffer{}
		if err := Encode(buf, img, options); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		out, err := Decode(bytes.NewReader(buf.Bytes()), nil)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if got, want := out.Bounds().Size(), img.Bounds().Size(); got != want {
			t.Fatalf("decoded size %v, want %v", got, want)
		}
		if p := psnr(toNRGBA(img), out.(*image.NRGBA)); !(p > 30.0) {
			t.Errorf("PSNR %.2f dB, want > 30", p)
		}
	})
	t.Run("encode lossless", func(t *testing.T) {
		img := loadJPEG(t, "source.jpg")

		options, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 4)
		if err != nil {
			t.Fatalf("NewLosslessEncoderOptions: %v", err)
		}

		buf := &bytes.Buffer{}
		if err := Encode(buf, img, options); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		out, err := Decode(bytes.NewReader(buf.Bytes()), nil)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if want, got := toNRGBA(img).Pix, out.(*image.NRGBA).Pix; !bytes.Equal(want, got) {
			t.Fatalf("decoded pixels differ from the source (%d bytes, want %d)", len(got), len(want))
		}
	})
	t.Run("failed encoding with large size", func(t *testing.T) {
		img := image.NewNRGBA(image.Rectangle{
			Max: image.Point{X: 20000, Y: 1},
		})

		options, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 4)
		if err != nil {
			t.Fatal(err)
		}

		if err = Encode(io.Discard, img, options); err == nil {
			t.Fatal("an error was expected")
		}
	})
}

func TestDecode(t *testing.T) {
	t.Run("decode lossy webp picture with many options", func(t *testing.T) {
		r, err := os.Open("../test_data/images/webp-logo-lossy.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if img, err := Decode(r, &decoder.Options{
			BypassFiltering:   true,
			NoFancyUpsampling: true,

			Crop: image.Rectangle{},
			Scale: image.Rectangle{
				Max: image.Point{
					X: 400,
					Y: 300,
				},
			},

			UseThreads:             true,
			Flip:                   true,
			DitheringStrength:      1,
			AlphaDitheringStrength: 1,
		}); err != nil {
			t.Fatal(err)
		} else if img == nil {
			t.Fatal("img is empty")
		} else if img.Bounds().Max.X != 400 || img.Bounds().Max.Y != 300 {
			t.Fatal("img is invalid")
		}
	})
	t.Run("decode lossless webp picture", func(t *testing.T) {
		r, err := os.Open("../test_data/images/webp-logo-lossless.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if img, err := Decode(r, &decoder.Options{}); err != nil {
			t.Fatal(err)
		} else if img == nil {
			t.Fatal("img is empty")
		}
	})
	t.Run("decode invalid webp file", func(t *testing.T) {
		r, err := os.Open("../test_data/images/invalid.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if _, err := Decode(r, &decoder.Options{}); err == nil {
			t.Fatal("an error was expected")
		}
	})
	t.Run("decode lossy image via standard image.Decode", func(t *testing.T) {
		r, err := os.Open("../test_data/images/webp-logo-lossy.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		img, format, err := image.Decode(r)
		if err != nil {
			t.Fatalf("image.Decode: %v", err)
		}

		if format != "webp" {
			t.Fatalf("image.Decode format = %q, want %q", format, "webp")
		}
		if !reflect.DeepEqual(color.NRGBAModel, img.ColorModel()) {
			t.Fatalf("ColorModel = %v, want color.NRGBAModel (%v)", img.ColorModel(), color.NRGBAModel)
		}
	})
	t.Run("decode lossless image via standard image.Decode", func(t *testing.T) {
		r, err := os.Open("../test_data/images/webp-logo-lossless.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		img, format, err := image.Decode(r)
		if err != nil {
			t.Fatalf("image.Decode: %v", err)
		}

		if format != "webp" {
			t.Fatalf("image.Decode format = %q, want %q", format, "webp")
		}
		if !reflect.DeepEqual(color.NRGBAModel, img.ColorModel()) {
			t.Fatalf("ColorModel = %v, want color.NRGBAModel (%v)", img.ColorModel(), color.NRGBAModel)
		}
	})
}

func TestDecodeConfig(t *testing.T) {
	tests := []struct {
		file string
		w, h int
		// maxRead bounds the bytes read: the headers up to the frame
		// header (here behind a 26 KB ALPH chunk for the lossy logo) are
		// needed to validate the file like NewDecoder does.
		maxRead int
	}{
		{"m4_q75.webp", 675, 900, 1024},
		{"webp-logo-lossy.webp", 3000, 2000, 2 * 26900},
		{"webp-logo-lossless.webp", 3000, 2000, 1024},
		{"100x150_lossy.webp", 100, 150, 1024},
		{"100x150_lossless.webp", 100, 150, 1024},
	}
	for _, tt := range tests {
		data, err := os.ReadFile("../test_data/images/" + tt.file)
		if err != nil {
			t.Fatal(err)
		}

		cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: image.DecodeConfig: %v", tt.file, err)
		}
		if format != "webp" {
			t.Errorf("%s: image.DecodeConfig format = %q, want %q", tt.file, format, "webp")
		}
		if cfg.Width != tt.w {
			t.Errorf("%s: image.DecodeConfig Width = %d, want %d", tt.file, cfg.Width, tt.w)
		}
		if cfg.Height != tt.h {
			t.Errorf("%s: image.DecodeConfig Height = %d, want %d", tt.file, cfg.Height, tt.h)
		}
		if !reflect.DeepEqual(color.RGBAModel, cfg.ColorModel) { // as in v1
			t.Errorf("%s: image.DecodeConfig ColorModel = %v, want color.RGBAModel (%v)", tt.file, cfg.ColorModel, color.RGBAModel)
		}

		// Only the headers are needed.
		r := &countingReader{r: bytes.NewReader(data)}
		cfg, err = DecodeConfig(r, nil)
		if err != nil {
			t.Fatalf("%s: DecodeConfig: %v", tt.file, err)
		}
		if cfg.Width != tt.w {
			t.Errorf("%s: DecodeConfig Width = %d, want %d", tt.file, cfg.Width, tt.w)
		}
		if r.n > tt.maxRead {
			t.Errorf("%s: DecodeConfig read %d bytes, want <= %d", tt.file, r.n, tt.maxRead)
		}
	}

	_, err := DecodeConfig(bytes.NewReader(nil), nil)
	if want := "data is empty"; err == nil || err.Error() != want {
		t.Fatalf("DecodeConfig(empty) error = %v, want %q", err, want)
	}
	_, err = DecodeConfig(bytes.NewReader([]byte("RIFF\x16\x00\x00\x00WEBPVP8 \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")), nil)
	if want := "cannot fetch features: VP8_STATUS_BITSTREAM_ERROR"; err == nil || err.Error() != want {
		t.Fatalf("DecodeConfig(zeroed VP8 frame) error = %v, want %q", err, want)
	}
	_, err = DecodeConfig(bytes.NewReader([]byte("RIFF\x10\x00\x00\x00WEBP")), nil)
	if want := "cannot fetch features: VP8_STATUS_NOT_ENOUGH_DATA"; err == nil || err.Error() != want {
		t.Fatalf("DecodeConfig(RIFF header only) error = %v, want %q", err, want)
	}
}

func TestGetFeatures(t *testing.T) {
	data, err := os.ReadFile("../test_data/images/webp-logo-lossy.webp")
	if err != nil {
		t.Fatal(err)
	}
	f, err := GetFeatures(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if want := (utils.BitstreamFeatures{Width: 3000, Height: 2000, HasAlpha: true, Format: utils.FormatLossy}); f != want {
		t.Errorf("GetFeatures = %+v, want %+v", f, want)
	}
}

type countingReader struct {
	r io.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func benchmarkDecode(b *testing.B, file string, viaImageDecode bool) {
	data, err := os.ReadFile("../test_data/images/" + file)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if viaImageDecode {
			if _, _, err := image.Decode(bytes.NewReader(data)); err != nil {
				b.Fatal(err)
			}
			continue
		}
		dec, err := decoder.NewDecoder(bytes.NewBuffer(data), &decoder.Options{})
		if err != nil {
			b.Fatal(err)
		}
		if _, err := dec.Decode(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLossy(b *testing.B) { benchmarkDecode(b, "webp-logo-lossy.webp", false) }

// BenchmarkImageDecodeLossy decodes through the standard library's
// image.Decode (format registry, buffered reader).
func BenchmarkImageDecodeLossy(b *testing.B) { benchmarkDecode(b, "webp-logo-lossy.webp", true) }

func BenchmarkDecodeLossyPhoto(b *testing.B) { benchmarkDecode(b, "m4_q75.webp", false) }

func BenchmarkDecodeLossless(b *testing.B) { benchmarkDecode(b, "webp-logo-lossless.webp", false) }

// BenchmarkImageDecodeLossless decodes through the standard library's
// image.Decode (format registry, buffered reader).
func BenchmarkImageDecodeLossless(b *testing.B) {
	benchmarkDecode(b, "webp-logo-lossless.webp", true)
}

func benchmarkEncode(b *testing.B, newOptions func() (*encoder.Options, error)) {
	img := loadJPEG(b, "source.jpg")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		options, err := newOptions()
		if err != nil {
			b.Fatal(err)
		}

		if err = Encode(io.Discard, img, options); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	benchmarkEncode(b, func() (*encoder.Options, error) {
		return encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	})
}

func BenchmarkEncodeLossless(b *testing.B) {
	benchmarkEncode(b, func() (*encoder.Options, error) {
		return encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 6)
	})
}

func BenchmarkEncodeLosslessFast(b *testing.B) {
	benchmarkEncode(b, func() (*encoder.Options, error) {
		return encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 0)
	})
}

// TestDecodeConfigMatchesNewDecoder checks on header-mutated files that
// DecodeConfig accepts exactly what decoder.NewDecoder accepts and reports
// the same dimensions (it used to stop at the VP8X chunk).
func TestDecodeConfigMatchesNewDecoder(t *testing.T) {
	for _, name := range []string{"webp-logo-lossy.webp", "100x150_lossy.webp", "100x150_lossless.webp", "m4_q75.webp"} {
		data, err := os.ReadFile("../test_data/images/" + name)
		if err != nil {
			t.Fatal(err)
		}
		var muts [][]byte
		for off := 0; off < 48 && off < len(data); off++ {
			for b := 0; b < 8; b++ {
				m := append([]byte(nil), data...)
				m[off] ^= 1 << b
				muts = append(muts, m)
			}
		}
		for _, n := range []int{8, 12, 20, 29, 30, 31, 40} {
			muts = append(muts, data[:n])
		}
		for i, m := range muts {
			cfg, cerr := DecodeConfig(bytes.NewReader(m), nil)
			d, derr := decoder.NewDecoder(bytes.NewReader(m), nil)
			if (cerr == nil) != (derr == nil) {
				t.Fatalf("%s mutant %d: DecodeConfig err %v, NewDecoder err %v", name, i, cerr, derr)
			}
			if cerr != nil {
				if cerr.Error() != derr.Error() {
					t.Errorf("%s mutant %d: DecodeConfig error %q, want NewDecoder's %q", name, i, cerr.Error(), derr.Error())
				}
				continue
			}
			f := d.GetFeatures()
			if want, got := [2]int{f.Width, f.Height}, [2]int{cfg.Width, cfg.Height}; got != want {
				t.Errorf("%s mutant %d: DecodeConfig size %v, want NewDecoder's %v", name, i, got, want)
			}
		}
	}
}

// rawPayload returns the payload of the first chunk with the given tag.
func rawPayload(t *testing.T, data []byte, tag string) []byte {
	t.Helper()
	i := bytes.Index(data, []byte(tag))
	if i < 0 {
		t.Fatalf("no %q chunk", tag)
	}
	n := int(binary.LittleEndian.Uint32(data[i+4:]))
	return data[i+8 : i+8+n]
}

// TestDecodeConfigRawBitstream checks raw VP8 / VP8L bitstreams (no RIFF
// header), which libwebp, v1 and decoder.NewDecoder accept: DecodeConfig
// and GetFeatures must too, although the first partition of a raw VP8
// stream (its size check needs the whole input) extends far past the first
// bytes read. image.DecodeConfig does not recognize them (the registered
// magic starts with "RIFF"), exactly as with v1.
func TestDecodeConfigRawBitstream(t *testing.T) {
	tests := []struct {
		file, tag string
		want      utils.BitstreamFeatures
	}{
		{"m4_q75.webp", "VP8 ", utils.BitstreamFeatures{Width: 675, Height: 900, Format: utils.FormatLossy}},
		{"100x150_lossy.webp", "VP8 ", utils.BitstreamFeatures{Width: 100, Height: 150, Format: utils.FormatLossy}},
		{"webp-logo-lossless.webp", "VP8L", utils.BitstreamFeatures{Width: 3000, Height: 2000, HasAlpha: true, Format: utils.FormatLossless}},
		{"100x150_lossless.webp", "VP8L", utils.BitstreamFeatures{Width: 100, Height: 150, HasAlpha: true, Format: utils.FormatLossless}},
	}
	for _, tt := range tests {
		data, err := os.ReadFile("../test_data/images/" + tt.file)
		if err != nil {
			t.Fatal(err)
		}
		raw := rawPayload(t, data, tt.tag)

		d, err := decoder.NewDecoder(bytes.NewReader(raw), nil)
		if err != nil {
			t.Fatalf("%s: NewDecoder: %v", tt.file, err)
		}
		if got := d.GetFeatures().Width; got != tt.want.Width {
			t.Fatalf("%s: NewDecoder Width = %d, want %d", tt.file, got, tt.want.Width)
		}
		if got := d.GetFeatures().Format; got != tt.want.Format {
			t.Fatalf("%s: NewDecoder Format = %v, want %v", tt.file, got, tt.want.Format)
		}

		for _, r := range []io.Reader{bytes.NewReader(raw), iotest.OneByteReader(bytes.NewReader(raw))} {
			cfg, err := DecodeConfig(r, nil)
			if err != nil {
				t.Fatalf("%s: DecodeConfig(%T): %v", tt.file, r, err)
			}
			if want := (image.Config{ColorModel: color.RGBAModel, Width: tt.want.Width, Height: tt.want.Height}); !reflect.DeepEqual(want, cfg) {
				t.Errorf("%s: DecodeConfig(%T) = %+v, want %+v", tt.file, r, cfg, want)
			}
		}
		f, err := GetFeatures(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("%s: GetFeatures: %v", tt.file, err)
		}
		if want := d.GetFeatures(); f != want {
			t.Errorf("%s: GetFeatures = %+v, want NewDecoder's %+v", tt.file, f, want)
		}
		if f.HasAlpha != tt.want.HasAlpha {
			t.Errorf("%s: GetFeatures HasAlpha = %v, want %v", tt.file, f.HasAlpha, tt.want.HasAlpha)
		}

		_, _, err = image.DecodeConfig(bytes.NewReader(raw))
		if !errors.Is(err, image.ErrFormat) {
			t.Errorf("%s: image.DecodeConfig error = %v, want %v", tt.file, err, image.ErrFormat)
		}

		// Truncations (shorter than the first partition for VP8) are
		// reported like NewDecoder does.
		for _, n := range []int{10, 11, 64, 100, len(raw) / 2, len(raw) - 1} {
			if n <= 0 || n > len(raw) {
				continue
			}
			_, cerr := DecodeConfig(bytes.NewReader(raw[:n]), nil)
			_, derr := decoder.NewDecoder(bytes.NewReader(raw[:n]), nil)
			if derr == nil {
				if cerr != nil {
					t.Errorf("%s[:%d]: DecodeConfig: %v, want no error as NewDecoder", tt.file, n, cerr)
				}
			} else if cerr == nil || cerr.Error() != derr.Error() {
				t.Errorf("%s[:%d]: DecodeConfig error = %v, want NewDecoder's %q", tt.file, n, cerr, derr.Error())
			}
		}
	}
}
