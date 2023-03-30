package decoder

import (
	"bytes"
	"image"
	"os"
	"testing"
)

func loadImage(b *testing.B) []byte {
	filename := "../test_data/images/100x150_lossless.webp"
	data, err := os.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

func BenchmarkDecodePooled(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	data := loadImage(b)

	imagePool := NewImagePool()
	bufferPool := NewBufferPool()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := bufferPool.Get()
			decoder, err := NewDecoder(bytes.NewReader(data), &Options{imageFactory: imagePool, buffer: buf})
			img, err := decoder.Decode()
			if err != nil {
				b.Fatal(err)
			}

			// put everything back
			imagePool.Put(img.(*image.NRGBA))
			bufferPool.Put(buf)
		}
	})
}

func BenchmarkDecodeUnPooled(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	data := loadImage(b)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			decoder, err := NewDecoder(bytes.NewReader(data), &Options{})
			img, err := decoder.Decode()
			if err != nil {
				b.Fatal(err)
			}
			_ = img
		}
	})
}
