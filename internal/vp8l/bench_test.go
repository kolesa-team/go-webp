package vp8l

import (
	"image"
	"image/draw"
	"image/jpeg"
	"os"
	"sync"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

func benchDecode(b *testing.B, path string) {
	file, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	benchDecodePayload(b, findChunk(file, "VP8L"))
}

// photoPayload is test_data/images/source.jpg (960x1280) encoded losslessly
// (predictor, cross-color and subtract-green transforms, 10-bit color
// cache), built once.
var photoPayload = sync.OnceValues(func() ([]byte, error) {
	f, err := os.Open("../../test_data/images/source.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	src, err := jpeg.Decode(f)
	if err != nil {
		return nil, err
	}
	img := image.NewNRGBA(src.Bounds())
	draw.Draw(img, img.Rect, src, src.Bounds().Min, draw.Src)
	file, err := vp8lenc.Encode(img, &vp8lenc.Config{Quality: 75, Method: 4})
	if err != nil {
		return nil, err
	}
	if p := findChunk(file, "VP8L"); p != nil {
		return p, nil
	}
	return file, nil
})

func BenchmarkDecodePhoto(b *testing.B) {
	payload, err := photoPayload()
	if err != nil {
		b.Skip(err)
	}
	benchDecodePayload(b, payload)
}

func benchDecodePayload(b *testing.B, payload []byte) {
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
