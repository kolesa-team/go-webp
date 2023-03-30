package decoder

import (
	"bytes"
	"image"
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/pool"
)

func BenchmarkDecodeOneSizeOnly(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	data := loadImage(b)

	nrgbaPool := pool.NewNRGBAOneSizePool(512, 512)
	bufferPool := pool.NewBufferPool(512 * 512)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := bufferPool.Get()
			decoder, err := NewDecoder(bytes.NewReader(data), &Options{imageFactory: nrgbaPool, buffer: buf})
			img, err := decoder.Decode()
			if err != nil {
				b.Fatal(err)
			}
			nrgbaPool.Put(img.(*image.NRGBA))
			bufferPool.Put(buf)
		}
	})
}

func loadImage(b *testing.B) []byte {
	filename := "../test_data/images/100x150_lossless.webp"
	// filename := "../test_data/images/composite.webp"
	data, err := os.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

func BenchmarkDecodeAnySize(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	data := loadImage(b)

	nrgbaPool := pool.NewNRGBAMultiPool()
	bufferPool := pool.NewBufferPool(512 * 512)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := bufferPool.Get()
			decoder, err := NewDecoder(bytes.NewReader(data), &Options{imageFactory: nrgbaPool, buffer: buf})
			img, err := decoder.Decode()
			if err != nil {
				b.Fatal(err)
			}
			nrgbaPool.Put(img.(*image.NRGBA))
			bufferPool.Put(buf)
		}
	})
}

func BenchmarkDecodeOriginal(b *testing.B) {
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
