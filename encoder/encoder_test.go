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
	require.NotSame(t, &rgba.Pix[0], &got.Pix[0])
	require.Equal(t, r.Size(), got.Rect.Size())

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
	require.NoError(t, err)
	lossless, err := NewLosslessEncoderOptions(PresetDefault, 6)
	require.NoError(t, err)
	for _, src := range sources {
		for _, o := range []*Options{lossy, lossless} {
			var want bytes.Buffer
			e0, err := NewEncoder(src.img, o)
			require.NoError(t, err)
			require.NoError(t, e0.Encode(&want))

			e, err := NewEncoder(src.img, o)
			require.NoError(t, err)
			saved := snapshotPix(src.img)
			src.mutate()
			var got bytes.Buffer
			require.NoError(t, e.Encode(&got))
			assert.Equal(t, want.Bytes(), got.Bytes(), "%s lossless=%v: source changed after NewEncoder", src.name, o.Lossless)
			// Encode can be called again (the snapshot is not released).
			got.Reset()
			require.NoError(t, e.Encode(&got))
			assert.Equal(t, want.Bytes(), got.Bytes(), "%s lossless=%v: second Encode", src.name, o.Lossless)
			restorePix(src.img, saved)
		}
	}

	t.Run("NRGBA is shared", func(t *testing.T) {
		n := image.NewNRGBA(image.Rect(0, 0, w, h))
		rng.Read(n.Pix)
		e, err := NewEncoder(n, lossless)
		require.NoError(t, err)
		var before bytes.Buffer
		require.NoError(t, e.Encode(&before))
		clear(n.Pix)
		var after bytes.Buffer
		require.NoError(t, e.Encode(&after))
		assert.NotEqual(t, before.Bytes(), after.Bytes())
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
		require.NoError(t, err)
		f(o)
		var buf bytes.Buffer
		require.NoError(t, encodeTo(t, img, o, &buf))
		_, err = container.Parse(buf.Bytes())
		require.NoError(t, err)
		return buf.Bytes()
	}
	for m := 0; m <= 6; m++ {
		got := encode(func(o *Options) { o.Quality, o.Method = nan, m }, false)
		if m >= 2 {
			// The quality only enters the quantizers (and the error diffusion
			// switch, off for both); methods 0-1 also use (int)quality in
			// the analysis.
			want := encode(func(o *Options) { o.Quality, o.Method = 100, m }, false)
			assert.Equal(t, want, got, "method %d: NaN quality", m)
		}
	}
	for _, pass := range []int{1, 3} {
		want := encode(func(o *Options) { o.Pass = pass }, false)
		got := encode(func(o *Options) { o.TargetPsnr, o.Pass = nan, pass }, false)
		assert.Equal(t, want, got, "pass %d: NaN target PSNR", pass)
	}
	// Searching with a NaN quality must terminate.
	encode(func(o *Options) { o.Quality, o.TargetSize, o.Pass = nan, 500, 10 }, false)
	encode(func(o *Options) { o.Quality = nan }, true)
}
