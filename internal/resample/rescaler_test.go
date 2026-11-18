package resample

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestScaledDimensions(t *testing.T) {
	tests := []struct{ sw, sh, w, h, ew, eh int }{
		{100, 50, 40, 0, 40, 20},
		{100, 50, 0, 25, 50, 25},
		{100, 50, 33, 0, 33, 17}, // rounds up
		{3, 7, 0, 1, 1, 1},
		{100, 50, 7, 9, 7, 9},
	}
	for _, tt := range tests {
		w, h, ok := ScaledDimensions(tt.sw, tt.sh, tt.w, tt.h)
		assert.True(t, ok)
		assert.Equal(t, [2]int{tt.ew, tt.eh}, [2]int{w, h}, "%+v", tt)
	}
	for _, bad := range [][4]int{{10, 10, 0, 0}, {10, 10, -1, 5}, {0, 10, 5, 5}, {10, 10, 1 << 30, 5}} {
		_, _, ok := ScaledDimensions(bad[0], bad[1], bad[2], bad[3])
		assert.False(t, ok, "%v", bad)
	}
}

func TestInitInvalid(t *testing.T) {
	for _, d := range [][5]int{{0, 1, 1, 1, 4}, {1, 0, 1, 1, 4}, {1, 1, 0, 1, 4}, {1, 1, 1, 0, 4}, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 5}} {
		_, err := New(d[0], d[1], d[2], d[3], d[4])
		assert.ErrorIs(t, err, ErrInvalid, "%v", d)
	}
}

func TestConstantImage(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		// Any rescale of a constant image is constant.
		r := rand.New(rand.NewSource(1))
		for i := 0; i < 200; i++ {
			sw, sh, dw, dh := 1+r.Intn(40), 1+r.Intn(40), 1+r.Intn(40), 1+r.Intn(40)
			src := make([]byte, sw*sh*4)
			for j := range src {
				src[j] = []byte{17, 200, 0, 255}[j%4]
			}
			out := rescaleAll(t, src, sw, sh, dw, dh, 4)
			for j, v := range out {
				require.Equal(t, []byte{17, 200, 0, 255}[j%4], v, "%dx%d -> %dx%d", sw, sh, dw, dh)
			}
		}
	})
}

func TestReuseWork(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		src := make([]byte, 30*20)
		for i := range src {
			src[i] = byte(i)
		}
		want := rescaleAll(t, src, 30, 20, 7, 5, 1)
		var rs Rescaler
		work := make([]uint32, 100)
		require.NoError(t, rs.Init(30, 20, 7, 5, 1, work))
		out := make([]byte, 7*5)
		for y := 0; y < 20; y++ {
			rs.ImportRow(src[y*30:])
			for rs.HasPendingOutput() {
				rs.ExportRow(out[rs.DstY()*7:])
			}
		}
		assert.Equal(t, want, out)
		assert.False(t, rs.ExportRow(out))
		assert.True(t, rs.InputDone())
		assert.True(t, rs.OutputDone())
		assert.Equal(t, 7, rs.DstWidth())
		assert.Equal(t, 5, rs.DstHeight())
	})
}

func TestPremultiply(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		row := []byte{200, 100, 50, 255, 200, 100, 50, 0, 200, 100, 50, 128}
		PremultiplyRGBA(row)
		assert.Equal(t, []byte{200, 100, 50, 255, 0, 0, 0, 0, 100, 50, 25, 128}, row)
		UnpremultiplyRGBA(row)
		assert.Equal(t, []byte{200, 100, 50, 255, 0, 0, 0, 0, 199, 100, 50, 128}, row)
	})
}

func FuzzRescale(f *testing.F) {
	f.Add(uint16(10), uint16(10), uint16(3), uint16(4), uint8(4), int64(1))
	f.Fuzz(func(t *testing.T, sw, sh, dw, dh uint16, ch uint8, seed int64) {
		eachLevel(func() {
			w, h, ow, oh := int(sw%300)+1, int(sh%300)+1, int(dw%300)+1, int(dh%300)+1
			c := int(ch%4) + 1
			src := make([]byte, w*h*c)
			rand.New(rand.NewSource(seed)).Read(src)
			_ = rescaleAll(t, src, w, h, ow, oh, c)
		})
	})
}

func benchImage(w, h int) []byte {
	r := rand.New(rand.NewSource(2))
	b := make([]byte, w*h*4)
	r.Read(b)
	return b
}

func benchRescale(b *testing.B, sw, sh, dw, dh int) {
	src := benchImage(sw, sh)
	out := make([]byte, dw*4)
	var rs Rescaler
	work := make([]uint32, BufferSize(dw, 4))
	b.SetBytes(int64(sw * sh * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rs.Init(sw, sh, dw, dh, 4, work)
		for y := 0; y < sh; y++ {
			rs.ImportRow(src[y*sw*4:])
			for rs.ExportRow(out) {
			}
		}
	}
}

func BenchmarkShrink1024to300(b *testing.B) { benchRescale(b, 1024, 768, 300, 225) }
func BenchmarkExpand300to1024(b *testing.B) { benchRescale(b, 300, 225, 1024, 768) }

func BenchmarkPremultiply(b *testing.B) {
	row := benchImage(4096, 1)
	b.SetBytes(int64(len(row)))
	for i := 0; i < b.N; i++ {
		PremultiplyRGBA(row)
		row[3] = byte(i)
	}
}
