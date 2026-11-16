package alpha

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func TestHeader(t *testing.T) {
	for b := 0; b < 256; b++ {
		h, err := ParseHeader(byte(b))
		valid := b&3 <= 1 && (b>>4)&3 <= 1 && b>>6 == 0
		if !valid {
			assert.Error(t, err, "byte %#x", b)
			assert.Equal(t, utils.Vp8StatusBitstreamError, status.CodeOf(err))
			continue
		}
		require.NoError(t, err, "byte %#x", b)
		assert.Equal(t, byte(b), h.Byte())
		assert.Equal(t, b&3, h.Compression)
		assert.Equal(t, (b>>2)&3, h.Filter)
		assert.Equal(t, (b>>4)&3, h.PreProcessing)
	}
	assert.Equal(t, byte(0x1d), Header{Compression: 1, Filter: FilterGradient, PreProcessing: 1}.Byte())
}

func TestAlphaLevels(t *testing.T) {
	assert.Equal(t, 2, AlphaLevels(0))
	assert.Equal(t, 16, AlphaLevels(70))
	assert.Equal(t, 24, AlphaLevels(71))
	assert.Equal(t, 256, AlphaLevels(100))
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
				require.Equal(t, src[y*srcStride:y*srcStride+w], dst[y*dstStride:y*dstStride+w], "filter %d %dx%d", f, w, h)
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
		assert.Equal(t, src, out, "filter %d", f)
	}
}

func TestEstimateBestFilter(t *testing.T) {
	w, h := 64, 64
	p := make([]byte, w*h)
	assert.Equal(t, FilterNone, EstimateBestFilter(p, w, h, w))
	// Constant vertically, steep changes horizontally.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p[y*w+x] = byte(x * x)
		}
	}
	assert.Equal(t, FilterVertical, EstimateBestFilter(p, w, h, w))
	// Tiny planes are never sampled.
	assert.Equal(t, FilterNone, EstimateBestFilter([]byte{1, 2, 3}, 3, 1, 3))
}

func TestQuantizeLevels(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	w, h, stride := 40, 30, 45
	p := randPlane(r, w, h, stride)
	for _, n := range []int{2, 3, 7, 16, 100} {
		q := append([]byte(nil), p...)
		require.NoError(t, QuantizeLevels(q, w, h, stride, n))
		seen := map[byte]bool{}
		for y := 0; y < h; y++ {
			for _, v := range q[y*stride : y*stride+w] {
				seen[v] = true
			}
			// Padding bytes are untouched.
			if y < h-1 {
				assert.Equal(t, p[y*stride+w:(y+1)*stride], q[y*stride+w:(y+1)*stride])
			}
		}
		assert.LessOrEqual(t, len(seen), n)
	}
	assert.Error(t, QuantizeLevels(p, w, h, stride, 1))
	assert.Error(t, QuantizeLevels(p, w, h, stride, 257))
	assert.Error(t, QuantizeLevels(p, 0, h, stride, 4))
	assert.Error(t, QuantizeLevels(p[:10], w, h, stride, 4))
	// Already few levels: untouched.
	two := []byte{0, 255, 0, 255}
	require.NoError(t, QuantizeLevels(two, 2, 2, 2, 2))
	assert.Equal(t, []byte{0, 255, 0, 255}, two)
}

func TestDequantize(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	w, h, stride := 64, 48, 70
	p := make([]byte, (h-1)*stride+w)
	for i := range p {
		p[i] = byte(r.Intn(4) * 85)
	}
	q := append([]byte(nil), p...)
	require.NoError(t, Dequantize(q, w, h, stride, 0))
	assert.Equal(t, p, q, "strength 0 is a no-op")
	require.NoError(t, Dequantize(q, w, h, stride, 100))
	assert.NotEqual(t, p, q)
	for y := 0; y < h-1; y++ {
		assert.Equal(t, p[y*stride+w:(y+1)*stride], q[y*stride+w:(y+1)*stride])
	}
	// Strided result equals the packed result.
	packed := make([]byte, w*h)
	for y := 0; y < h; y++ {
		copy(packed[y*w:], p[y*stride:y*stride+w])
	}
	require.NoError(t, Dequantize(packed, w, h, w, 100))
	for y := 0; y < h; y++ {
		assert.Equal(t, packed[y*w:(y+1)*w], q[y*stride:y*stride+w])
	}
	assert.Error(t, Dequantize(p, w, h, stride, 101))
	assert.Error(t, Dequantize(p, w, h, stride, -1))
	assert.Error(t, Dequantize(p, w, h, w-1, 50))
	assert.Error(t, Dequantize(p[:5], w, h, stride, 50))
	require.NoError(t, Dequantize([]byte{1}, 1, 1, 1, 100))
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
