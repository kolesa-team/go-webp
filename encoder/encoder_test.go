package encoder

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"io"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
		require.NoError(t, err)

		e, err := NewEncoder(img, options)
		require.NoError(t, err)
		require.NoError(t, e.Encode(out))

		parsed, err := container.Parse(out.Bytes())
		require.NoError(t, err)
		assert.Equal(t, 100, parsed.Width)
		assert.Equal(t, 150, parsed.Height)
		assert.True(t, parsed.HasAlpha, "a fully transparent image needs an ALPH chunk")
		assert.NotNil(t, parsed.VP8)
		assert.NotNil(t, parsed.ALPH)
	})
	t.Run("encode nrgba image with lossless preset", func(t *testing.T) {
		out := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rect(0, 0, 100, 150))

		options, err := NewLosslessEncoderOptions(PresetDefault, 1)
		require.NoError(t, err)

		e, err := NewEncoder(img, options)
		require.NoError(t, err)
		require.NoError(t, e.Encode(out))

		parsed, err := container.Parse(out.Bytes())
		require.NoError(t, err)
		assert.Equal(t, 100, parsed.Width)
		assert.Equal(t, 150, parsed.Height)
		assert.NotNil(t, parsed.VP8L)
	})
}

func TestNewEncoder_NilOptions(t *testing.T) {
	t.Run("encode image without passing options should not panic", func(t *testing.T) {
		out := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rect(0, 0, 100, 150))

		e, err := NewEncoder(img, nil)
		require.NoError(t, err)
		require.NoError(t, e.Encode(out))

		parsed, err := container.Parse(out.Bytes())
		require.NoError(t, err)
		assert.NotNil(t, parsed.VP8, "nil options must encode lossy")
	})
}

func TestEncoder_BadDimensions(t *testing.T) {
	options, err := NewLosslessEncoderOptions(PresetDefault, 4)
	require.NoError(t, err)
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 16384, 1),
		image.Rect(0, 0, 1, 20000),
	} {
		e, err := NewEncoder(image.NewNRGBA(r), options)
		require.NoError(t, err)
		err = e.Encode(io.Discard)
		require.EqualError(t, err, "cannot encode webppicture") // exactly v1's message
		require.Error(t, errors.Unwrap(err), "the detailed cause is wrapped")
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
	require.NoError(t, err)

	cw := &countWriter{}
	require.NoError(t, e.Encode(cw))
	assert.Equal(t, 1, cw.calls, "the file must be written with a single Write call")

	assert.EqualError(t, e.Encode(failWriter{}), "boom")
}

func TestEncoder_InvalidAlphaCompression(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	options, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	options.AlphaCompression = 2
	e, err := NewEncoder(img, options)
	require.NoError(t, err)
	require.EqualError(t, e.Encode(io.Discard), "cannot encode webppicture")
}

// decodeALPH decodes the ALPH chunk of a lossy file.
func decodeALPH(t *testing.T, data []byte, w, h int) (alpha.Header, []byte) {
	t.Helper()
	parsed, err := container.Parse(data)
	require.NoError(t, err)
	require.NotNil(t, parsed.ALPH)
	hdr, err := alpha.ParseHeader(parsed.ALPH[0])
	require.NoError(t, err)
	plane := make([]byte, w*h)
	if hdr.Compression == alpha.CompressionNone {
		copy(plane, parsed.ALPH[1:])
	} else {
		require.NoError(t, vp8l.DecodeAlpha(parsed.ALPH[1:], w, h, plane))
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
				require.NoError(t, err)
				options.AlphaCompression = compression
				options.AlphaFiltering = filtering
				options.Method = method
				buf := &bytes.Buffer{}
				require.NoError(t, encodeTo(t, img, options, buf))
				hdr, plane := decodeALPH(t, buf.Bytes(), w, h)
				assert.Equal(t, 0, hdr.PreProcessing)
				if compression == 0 {
					assert.Equal(t, alpha.FilterNone, hdr.Filter)
				}
				assert.Equal(t, want, plane, "compression %d filtering %d method %d", compression, filtering, method)
			}
		}
	}

	t.Run("quantized alpha", func(t *testing.T) {
		options, err := NewLossyEncoderOptions(PresetDefault, 80)
		require.NoError(t, err)
		options.alphaQuality = 10 // 4 levels
		buf := &bytes.Buffer{}
		require.NoError(t, encodeTo(t, img, options, buf))
		hdr, plane := decodeALPH(t, buf.Bytes(), w, h)
		assert.Equal(t, alpha.PreprocessingLevels, hdr.PreProcessing)
		levels := map[byte]bool{}
		for _, v := range plane {
			levels[v] = true
		}
		assert.LessOrEqual(t, len(levels), alpha.AlphaLevels(10))
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
		require.Equal(t, r.Dx(), got.Rect.Dx())
		require.Equal(t, r.Dy(), got.Rect.Dy())
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
					require.InDelta(t, want.R, have.R, 1)
					require.InDelta(t, want.G, have.G, 1)
					require.InDelta(t, want.B, have.B, 1)
					require.Equal(t, want.A, have.A)
					continue
				}
				require.Equal(t, want, have, "%T at %d,%d", src, x, y)
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

// TestToNRGBAOpaqueRGBAAliases checks that an opaque *image.RGBA is used in
// place (no 4 bytes/pixel copy) and encodes like the equivalent NRGBA.
func TestToNRGBAOpaqueRGBAAliases(t *testing.T) {
	r := image.Rect(3, 5, 40, 33)
	rgba := image.NewRGBA(r)
	rand.New(rand.NewSource(4)).Read(rgba.Pix)
	for i := 3; i < len(rgba.Pix); i += 4 {
		rgba.Pix[i] = 0xff
	}
	got := toNRGBA(rgba)
	require.Same(t, &rgba.Pix[0], &got.Pix[0])
	require.Equal(t, r, got.Rect)

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
		require.NoError(t, err)
		var a, b bytes.Buffer
		e1, err := NewEncoder(rgba, o)
		require.NoError(t, err)
		require.NoError(t, e1.Encode(&a))
		e2, err := NewEncoder(nrgba, o)
		require.NoError(t, err)
		require.NoError(t, e2.Encode(&b))
		require.Equal(t, b.Bytes(), a.Bytes(), "lossless=%v", lossless)
	}
}
