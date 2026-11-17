package webp

import (
	"bytes"
	"image"
	"image/draw"
	"io"
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/v2/encoder"
)

// loadLogo decodes the 3000x2000 transparent logo (lossy encodes of it carry
// an ALPH chunk).
func loadLogo(b testing.TB) *image.NRGBA {
	b.Helper()
	data, err := os.ReadFile("../test_data/images/webp-logo-lossless.webp")
	if err != nil {
		b.Fatal(err)
	}
	img, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		b.Fatal(err)
	}
	return img.(*image.NRGBA)
}

func benchmarkEncodeImage(b *testing.B, img image.Image, o *encoder.Options) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Encode(io.Discard, img, o); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeLossyAlpha encodes a transparent picture lossy: VP8 frame
// plus compressed ALPH chunk, sequentially.
func BenchmarkEncodeLossyAlpha(b *testing.B) {
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	benchmarkEncodeImage(b, loadLogo(b), o)
}

// BenchmarkEncodeLossyAlphaThreads is BenchmarkEncodeLossyAlpha with
// ThreadLevel, which codes the alpha plane concurrently with the VP8 frame.
func BenchmarkEncodeLossyAlphaThreads(b *testing.B) {
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	o.ThreadLevel = true
	benchmarkEncodeImage(b, loadLogo(b), o)
}

// BenchmarkEncodeRGBA encodes an opaque *image.RGBA (what png.Decode returns
// for RGB files) lossy through the one-shot Encode.
func BenchmarkEncodeRGBA(b *testing.B) {
	src := loadJPEG(b, "source.jpg")
	img := image.NewRGBA(src.Bounds())
	draw.Draw(img, img.Rect, src, src.Bounds().Min, draw.Src)
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	benchmarkEncodeImage(b, img, o)
}

// BenchmarkEncoderNoRelease uses NewEncoder and Encode without Release, as
// code written for v1 does.
func BenchmarkEncoderNoRelease(b *testing.B) {
	img := loadJPEG(b, "source.jpg")
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e, err := encoder.NewEncoder(img, o)
		if err != nil {
			b.Fatal(err)
		}
		if err := e.Encode(io.Discard); err != nil {
			b.Fatal(err)
		}
	}
}
