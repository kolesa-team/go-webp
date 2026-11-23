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

package alpha

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func TestHeader(t *testing.T) {
	for b := 0; b < 256; b++ {
		h, err := ParseHeader(byte(b))
		valid := b&3 <= 1 && (b>>4)&3 <= 1 && b>>6 == 0
		if !valid {
			if err == nil {
				t.Errorf("ParseHeader(%#x) succeeded, want error", b)
			}
			if got := status.CodeOf(err); got != utils.Vp8StatusBitstreamError {
				t.Errorf("ParseHeader(%#x): status = %v, want %v", b, got, utils.Vp8StatusBitstreamError)
			}
			continue
		}
		if err != nil {
			t.Fatalf("ParseHeader(%#x): %v", b, err)
		}
		if got := h.Byte(); got != byte(b) {
			t.Errorf("ParseHeader(%#x).Byte() = %#x, want %#x", b, got, b)
		}
		if h.Compression != b&3 {
			t.Errorf("ParseHeader(%#x).Compression = %d, want %d", b, h.Compression, b&3)
		}
		if h.Filter != (b>>2)&3 {
			t.Errorf("ParseHeader(%#x).Filter = %d, want %d", b, h.Filter, (b>>2)&3)
		}
		if h.PreProcessing != (b>>4)&3 {
			t.Errorf("ParseHeader(%#x).PreProcessing = %d, want %d", b, h.PreProcessing, (b>>4)&3)
		}
	}
	if got := (Header{Compression: 1, Filter: FilterGradient, PreProcessing: 1}).Byte(); got != 0x1d {
		t.Errorf("Header{Compression: 1, Filter: FilterGradient, PreProcessing: 1}.Byte() = %#x, want 0x1d", got)
	}
}

func TestAlphaLevels(t *testing.T) {
	if got := AlphaLevels(0); got != 2 {
		t.Errorf("AlphaLevels(0) = %d, want 2", got)
	}
	if got := AlphaLevels(70); got != 16 {
		t.Errorf("AlphaLevels(70) = %d, want 16", got)
	}
	if got := AlphaLevels(71); got != 24 {
		t.Errorf("AlphaLevels(71) = %d, want 24", got)
	}
	if got := AlphaLevels(100); got != 256 {
		t.Errorf("AlphaLevels(100) = %d, want 256", got)
	}
}

func randPlane(r *rand.Rand, w, h, stride int) []byte {
	p := make([]byte, (h-1)*stride+w)
	r.Read(p)
	return p
}

func TestFilterRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 500; i++ {
		w, h := 1+r.Intn(50), 1+r.Intn(50)
		srcStride, dstStride := w+r.Intn(5), w+r.Intn(5)
		src := randPlane(r, w, h, srcStride)
		for f := FilterNone; f <= FilterGradient; f++ {
			dst := make([]byte, (h-1)*dstStride+w)
			Filter(f, src, w, h, srcStride, dst, dstStride)
			Unfilter(f, dst, w, h, dstStride)
			for y := 0; y < h; y++ {
				if !bytes.Equal(dst[y*dstStride:y*dstStride+w], src[y*srcStride:y*srcStride+w]) {
					t.Fatalf("filter %d %dx%d: row %d differs after Filter + Unfilter", f, w, h, y)
				}
			}
		}
	}
}

func TestUnfilterRowNotInPlace(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	w, h := 37, 9
	src := randPlane(r, w, h, w)
	for f := FilterNone; f <= FilterGradient; f++ {
		res := make([]byte, w*h)
		Filter(f, src, w, h, w, res, w)
		out := make([]byte, w*h)
		var prev []byte
		for y := 0; y < h; y++ {
			UnfilterRow(f, prev, res[y*w:(y+1)*w], out[y*w:(y+1)*w])
			prev = out[y*w : (y+1)*w]
		}
		if !bytes.Equal(out, src) {
			t.Errorf("filter %d: UnfilterRow output differs from the source plane", f)
		}
	}
}

func TestEstimateBestFilter(t *testing.T) {
	w, h := 64, 64
	p := make([]byte, w*h)
	if got := EstimateBestFilter(p, w, h, w); got != FilterNone {
		t.Errorf("EstimateBestFilter(zero plane) = %d, want %d (FilterNone)", got, FilterNone)
	}
	// Constant vertically, steep changes horizontally.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p[y*w+x] = byte(x * x)
		}
	}
	if got := EstimateBestFilter(p, w, h, w); got != FilterVertical {
		t.Errorf("EstimateBestFilter(vertically constant plane) = %d, want %d (FilterVertical)", got, FilterVertical)
	}
	// Tiny planes are never sampled.
	if got := EstimateBestFilter([]byte{1, 2, 3}, 3, 1, 3); got != FilterNone {
		t.Errorf("EstimateBestFilter(3x1 plane) = %d, want %d (FilterNone)", got, FilterNone)
	}
}

func TestQuantizeLevels(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	w, h, stride := 40, 30, 45
	p := randPlane(r, w, h, stride)
	for _, n := range []int{2, 3, 7, 16, 100} {
		q := append([]byte(nil), p...)
		if err := QuantizeLevels(q, w, h, stride, n); err != nil {
			t.Fatalf("QuantizeLevels(%d levels): %v", n, err)
		}
		seen := map[byte]bool{}
		for y := 0; y < h; y++ {
			for _, v := range q[y*stride : y*stride+w] {
				seen[v] = true
			}
			// Padding bytes are untouched.
			if y < h-1 {
				if got, want := q[y*stride+w:(y+1)*stride], p[y*stride+w:(y+1)*stride]; !bytes.Equal(got, want) {
					t.Errorf("QuantizeLevels(%d levels): padding after row %d = %v, want %v", n, y, got, want)
				}
			}
		}
		if len(seen) > n {
			t.Errorf("QuantizeLevels(%d levels): %d distinct values, want <= %d", n, len(seen), n)
		}
	}
	if err := QuantizeLevels(p, w, h, stride, 1); err == nil {
		t.Error("QuantizeLevels(1 level) succeeded, want error")
	}
	if err := QuantizeLevels(p, w, h, stride, 257); err == nil {
		t.Error("QuantizeLevels(257 levels) succeeded, want error")
	}
	if err := QuantizeLevels(p, 0, h, stride, 4); err == nil {
		t.Error("QuantizeLevels(width 0) succeeded, want error")
	}
	if err := QuantizeLevels(p[:10], w, h, stride, 4); err == nil {
		t.Error("QuantizeLevels(10-byte plane) succeeded, want error")
	}
	// Already few levels: untouched.
	two := []byte{0, 255, 0, 255}
	if err := QuantizeLevels(two, 2, 2, 2, 2); err != nil {
		t.Fatalf("QuantizeLevels(2x2, 2 levels): %v", err)
	}
	if want := []byte{0, 255, 0, 255}; !bytes.Equal(two, want) {
		t.Errorf("QuantizeLevels(2x2, 2 levels) = %v, want %v", two, want)
	}
}

func TestDequantize(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	w, h, stride := 64, 48, 70
	p := make([]byte, (h-1)*stride+w)
	for i := range p {
		p[i] = byte(r.Intn(4) * 85)
	}
	q := append([]byte(nil), p...)
	if err := Dequantize(q, w, h, stride, 0); err != nil {
		t.Fatalf("Dequantize(strength 0): %v", err)
	}
	if !bytes.Equal(q, p) {
		t.Error("Dequantize(strength 0) changed the plane; strength 0 is a no-op")
	}
	if err := Dequantize(q, w, h, stride, 100); err != nil {
		t.Fatalf("Dequantize(strength 100): %v", err)
	}
	if bytes.Equal(q, p) {
		t.Error("Dequantize(strength 100) left the plane unchanged")
	}
	for y := 0; y < h-1; y++ {
		if got, want := q[y*stride+w:(y+1)*stride], p[y*stride+w:(y+1)*stride]; !bytes.Equal(got, want) {
			t.Errorf("Dequantize(strength 100): padding after row %d = %v, want %v", y, got, want)
		}
	}
	// Strided result equals the packed result.
	packed := make([]byte, w*h)
	for y := 0; y < h; y++ {
		copy(packed[y*w:], p[y*stride:y*stride+w])
	}
	if err := Dequantize(packed, w, h, w, 100); err != nil {
		t.Fatalf("Dequantize(packed, strength 100): %v", err)
	}
	for y := 0; y < h; y++ {
		if !bytes.Equal(q[y*stride:y*stride+w], packed[y*w:(y+1)*w]) {
			t.Errorf("row %d: strided Dequantize result differs from the packed result", y)
		}
	}
	if err := Dequantize(p, w, h, stride, 101); err == nil {
		t.Error("Dequantize(strength 101) succeeded, want error")
	}
	if err := Dequantize(p, w, h, stride, -1); err == nil {
		t.Error("Dequantize(strength -1) succeeded, want error")
	}
	if err := Dequantize(p, w, h, w-1, 50); err == nil {
		t.Errorf("Dequantize(stride %d < width %d) succeeded, want error", w-1, w)
	}
	if err := Dequantize(p[:5], w, h, stride, 50); err == nil {
		t.Error("Dequantize(5-byte plane) succeeded, want error")
	}
	if err := Dequantize([]byte{1}, 1, 1, 1, 100); err != nil {
		t.Fatalf("Dequantize(1x1, strength 100): %v", err)
	}
}

func FuzzAlpha(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8}, uint8(3), uint8(3), uint8(50), uint8(4))
	f.Fuzz(func(t *testing.T, data []byte, w, h, strength, levels uint8) {
		width, height := int(w%64)+1, int(h%64)+1
		if len(data) < width*height {
			return
		}
		for flt := FilterNone; flt <= FilterGradient; flt++ {
			p := append([]byte(nil), data...)
			out := make([]byte, len(p))
			Filter(flt, p, width, height, width, out, width)
			Unfilter(flt, out, width, height, width)
			if string(out[:width*height]) != string(p[:width*height]) {
				t.Fatalf("round trip failed for filter %d", flt)
			}
		}
		_ = EstimateBestFilter(data, width, height, width)
		p := append([]byte(nil), data...)
		if err := QuantizeLevels(p, width, height, width, int(levels)); err != nil && levels >= 2 {
			t.Fatal(err)
		}
		if err := Dequantize(p, width, height, width, int(strength%101)); err != nil {
			t.Fatal(err)
		}
	})
}

func benchPlane(w, h int) []byte {
	r := rand.New(rand.NewSource(5))
	p := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p[y*w+x] = byte(x + y + r.Intn(8))
		}
	}
	return p
}

func BenchmarkUnfilter(b *testing.B) {
	const w, h = 1024, 768
	src := benchPlane(w, h)
	for _, f := range []struct {
		name string
		f    int
	}{{"Horizontal", FilterHorizontal}, {"Vertical", FilterVertical}, {"Gradient", FilterGradient}} {
		b.Run(f.name, func(b *testing.B) {
			res := make([]byte, w*h)
			Filter(f.f, src, w, h, w, res, w)
			p := make([]byte, w*h)
			b.SetBytes(w * h)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(p, res)
				Unfilter(f.f, p, w, h, w)
			}
		})
	}
}

func BenchmarkFilter(b *testing.B) {
	const w, h = 1024, 768
	src := benchPlane(w, h)
	dst := make([]byte, w*h)
	for _, f := range []int{FilterHorizontal, FilterVertical, FilterGradient} {
		b.Run([]string{"", "Horizontal", "Vertical", "Gradient"}[f], func(b *testing.B) {
			b.SetBytes(w * h)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Filter(f, src, w, h, w, dst, w)
			}
		})
	}
}

func BenchmarkQuantizeLevels(b *testing.B) {
	const w, h = 1024, 768
	src := benchPlane(w, h)
	p := make([]byte, w*h)
	b.SetBytes(w * h)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		copy(p, src)
		_ = QuantizeLevels(p, w, h, w, 16)
	}
}

func BenchmarkDequantize(b *testing.B) {
	const w, h = 1024, 768
	src := benchPlane(w, h)
	_ = QuantizeLevels(src, w, h, w, 8)
	p := make([]byte, w*h)
	b.SetBytes(w * h)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		copy(p, src)
		_ = Dequantize(p, w, h, w, 100)
	}
}

func BenchmarkEstimateBestFilter(b *testing.B) {
	const w, h = 1024, 768
	src := benchPlane(w, h)
	b.SetBytes(w * h)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = EstimateBestFilter(src, w, h, w)
	}
}
