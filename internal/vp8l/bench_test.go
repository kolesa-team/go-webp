package vp8l

import (
	"image"
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

func benchDecode(b *testing.B, path string) {
	file, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	payload := findChunk(file, "VP8L")
	w, h, _, err := DecodeHeader(payload)
	if err != nil {
		b.Fatal(err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	sink := pipeline.NewDirect(img)
	b.SetBytes(int64(4 * w * h))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Decode(payload, sink); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLogo(b *testing.B) {
	benchDecode(b, "../../test_data/images/webp-logo-lossless.webp")
}

func BenchmarkDecode100x150(b *testing.B) {
	benchDecode(b, "../../test_data/images/100x150_lossless.webp")
}

func BenchmarkDecodeNoise64(b *testing.B) {
	benchDecode(b, "testdata/lossless/noise_64x64_z3.webp")
}

func BenchmarkDecodePalette(b *testing.B) {
	benchDecode(b, "testdata/lossless/pal16_500x375_z2.webp")
}

func BenchmarkDecodeAlpha(b *testing.B) {
	file, err := os.ReadFile("testdata/alpha/alpha_257x129_aq30.webp")
	if err != nil {
		b.Fatal(err)
	}
	_, payload, w, h, _ := alphaFromFile(file)
	dst := make([]byte, w*h)
	b.SetBytes(int64(w * h))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := DecodeAlpha(payload, w, h, dst); err != nil {
			b.Fatal(err)
		}
	}
}
