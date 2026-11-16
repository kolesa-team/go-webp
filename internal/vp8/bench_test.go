package vp8

import (
	"image"
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

func benchDecode(b *testing.B, path string, opts *Options) {
	data, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	ri, err := extractVP8(data)
	if err != nil {
		b.Fatal(err)
	}
	w, h, err := DecodeHeader(ri.vp8)
	if err != nil {
		b.Fatal(err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	sink := pipeline.NewDirect(img)
	b.SetBytes(int64(w * h * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Decode(ri.vp8, nil, opts, sink); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeM4(b *testing.B) {
	benchDecode(b, "../../test_data/images/m4_q75.webp", nil)
}

func BenchmarkDecodeM4NoFancy(b *testing.B) {
	benchDecode(b, "../../test_data/images/m4_q75.webp", &Options{NoFancyUpsampling: true})
}

func BenchmarkDecodeM4Threads(b *testing.B) {
	benchDecode(b, "../../test_data/images/m4_q75.webp", &Options{UseThreads: true})
}

func BenchmarkDecodeLogo(b *testing.B) {
	benchDecode(b, "../../test_data/images/webp-logo-lossy.webp", nil)
}

func BenchmarkDecodeLogoThreads(b *testing.B) {
	benchDecode(b, "../../test_data/images/webp-logo-lossy.webp", &Options{UseThreads: true})
}

func BenchmarkUpsamplePair(b *testing.B) {
	const w = 1024
	ys := make([]byte, w)
	u0, v0, u1, v1 := make([]byte, w/2), make([]byte, w/2), make([]byte, w/2), make([]byte, w/2)
	for i := range ys {
		ys[i] = byte(i * 7)
	}
	for i := range u0 {
		u0[i], v0[i], u1[i], v1[i] = byte(i*3), byte(i*5), byte(i*11), byte(i*13)
	}
	dst := make([]byte, 4*w)
	uvT, uvB := make([]uint32, w), make([]uint32, w)
	b.SetBytes(2 * w)
	for i := 0; i < b.N; i++ {
		upsamplePair(ys, ys, u0, v0, u1, v1, dst, dst, nil, nil, uvT, uvB, w)
	}
}

func BenchmarkFilterComplex(b *testing.B) {
	const stride = 64
	buf := make([]byte, stride*24)
	for i := range buf {
		buf[i] = byte(128 + (i*7)%9 - 4)
	}
	work := make([]byte, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		o := 4*stride + 8
		loop26H(work, o, stride, 16, 2*(30+4)+1, 10, 1)
		loop24H(work, o+4, stride, 16, 2*30+1, 10, 1)
		loop24H(work, o+8, stride, 16, 2*30+1, 10, 1)
		loop24H(work, o+12, stride, 16, 2*30+1, 10, 1)
		loop26V(work, o, stride, 16, 2*(30+4)+1, 10, 1)
		loop24V(work, o+4*stride, stride, 16, 2*30+1, 10, 1)
		loop24V(work, o+8*stride, stride, 16, 2*30+1, 10, 1)
		loop24V(work, o+12*stride, stride, 16, 2*30+1, 10, 1)
	}
}
