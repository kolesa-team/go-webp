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
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"os"
	"testing"
	"testing/iotest"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func loadJPEG(t testing.TB, name string) image.Image {
	t.Helper()
	r, err := os.Open("../test_data/images/" + name)
	require.NoError(t, err)
	defer r.Close()
	img, err := jpeg.Decode(r)
	require.NoError(t, err)
	return img
}

func TestEncode(t *testing.T) {
	t.Run("encode lossy", func(t *testing.T) {
		img := loadJPEG(t, "source.jpg")

		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		require.NoError(t, err)

		buf := &bytes.Buffer{}
		require.NoError(t, Encode(buf, img, options))

		out, err := Decode(bytes.NewReader(buf.Bytes()), nil)
		require.NoError(t, err)
		require.Equal(t, img.Bounds().Size(), out.Bounds().Size())
		assert.Greater(t, psnr(toNRGBA(img), out.(*image.NRGBA)), 30.0)
	})
	t.Run("encode lossless", func(t *testing.T) {
		img := loadJPEG(t, "source.jpg")

		options, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 4)
		require.NoError(t, err)

		buf := &bytes.Buffer{}
		require.NoError(t, Encode(buf, img, options))

		out, err := Decode(bytes.NewReader(buf.Bytes()), nil)
		require.NoError(t, err)
		require.Equal(t, toNRGBA(img).Pix, out.(*image.NRGBA).Pix)
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
		require.NoError(t, err)
		defer r.Close()
		img, format, err := image.Decode(r)
		require.NoError(t, err)

		require.Equal(t, "webp", format)
		require.Equal(t, color.NRGBAModel, img.ColorModel())
	})
	t.Run("decode lossless image via standard image.Decode", func(t *testing.T) {
		r, err := os.Open("../test_data/images/webp-logo-lossless.webp")
		require.NoError(t, err)
		defer r.Close()
		img, format, err := image.Decode(r)
		require.NoError(t, err)

		require.Equal(t, "webp", format)
		require.Equal(t, color.NRGBAModel, img.ColorModel())
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
		require.NoError(t, err)

		cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
		require.NoError(t, err)
		assert.Equal(t, "webp", format)
		assert.Equal(t, tt.w, cfg.Width)
		assert.Equal(t, tt.h, cfg.Height)
		assert.Equal(t, color.RGBAModel, cfg.ColorModel) // as in v1

		// Only the headers are needed.
		r := &countingReader{r: bytes.NewReader(data)}
		cfg, err = DecodeConfig(r, nil)
		require.NoError(t, err)
		assert.Equal(t, tt.w, cfg.Width)
		assert.LessOrEqual(t, r.n, tt.maxRead, tt.file)
	}

	_, err := DecodeConfig(bytes.NewReader(nil), nil)
	require.EqualError(t, err, "data is empty")
	_, err = DecodeConfig(bytes.NewReader([]byte("RIFF\x16\x00\x00\x00WEBPVP8 \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")), nil)
	require.EqualError(t, err, "cannot fetch features: VP8_STATUS_BITSTREAM_ERROR")
	_, err = DecodeConfig(bytes.NewReader([]byte("RIFF\x10\x00\x00\x00WEBP")), nil)
	require.EqualError(t, err, "cannot fetch features: VP8_STATUS_NOT_ENOUGH_DATA")
}

func TestGetFeatures(t *testing.T) {
	data, err := os.ReadFile("../test_data/images/webp-logo-lossy.webp")
	require.NoError(t, err)
	f, err := GetFeatures(bytes.NewReader(data))
	require.NoError(t, err)
	assert.Equal(t, utils.BitstreamFeatures{Width: 3000, Height: 2000, HasAlpha: true, Format: utils.FormatLossy}, f)
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
		require.NoError(t, err)
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
				assert.Equal(t, derr.Error(), cerr.Error(), "%s mutant %d", name, i)
				continue
			}
			f := d.GetFeatures()
			assert.Equal(t, [2]int{f.Width, f.Height}, [2]int{cfg.Width, cfg.Height}, "%s mutant %d", name, i)
		}
	}
}

// rawPayload returns the payload of the first chunk with the given tag.
func rawPayload(t *testing.T, data []byte, tag string) []byte {
	t.Helper()
	i := bytes.Index(data, []byte(tag))
	require.GreaterOrEqual(t, i, 0, "no %q chunk", tag)
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
		require.NoError(t, err)
		raw := rawPayload(t, data, tt.tag)

		d, err := decoder.NewDecoder(bytes.NewReader(raw), nil)
		require.NoError(t, err, tt.file)
		require.Equal(t, tt.want.Width, d.GetFeatures().Width, tt.file)
		require.Equal(t, tt.want.Format, d.GetFeatures().Format, tt.file)

		for _, r := range []io.Reader{bytes.NewReader(raw), iotest.OneByteReader(bytes.NewReader(raw))} {
			cfg, err := DecodeConfig(r, nil)
			require.NoError(t, err, tt.file)
			assert.Equal(t, image.Config{ColorModel: color.RGBAModel, Width: tt.want.Width, Height: tt.want.Height}, cfg, tt.file)
		}
		f, err := GetFeatures(bytes.NewReader(raw))
		require.NoError(t, err, tt.file)
		assert.Equal(t, d.GetFeatures(), f, tt.file)
		assert.Equal(t, tt.want.HasAlpha, f.HasAlpha, tt.file)

		_, _, err = image.DecodeConfig(bytes.NewReader(raw))
		assert.ErrorIs(t, err, image.ErrFormat, tt.file)

		// Truncations (shorter than the first partition for VP8) are
		// reported like NewDecoder does.
		for _, n := range []int{10, 11, 64, 100, len(raw) / 2, len(raw) - 1} {
			if n <= 0 || n > len(raw) {
				continue
			}
			_, cerr := DecodeConfig(bytes.NewReader(raw[:n]), nil)
			_, derr := decoder.NewDecoder(bytes.NewReader(raw[:n]), nil)
			if derr == nil {
				assert.NoError(t, cerr, "%s[:%d]", tt.file, n)
			} else {
				assert.EqualError(t, cerr, derr.Error(), "%s[:%d]", tt.file, n)
			}
		}
	}
}
