package vp8lenc

import (
	"bytes"
	"image"
	"image/color"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
)

func TestOptimizeSampling(t *testing.T) {
	// A 40x24 image sampled with bits 2 (10x6 tiles) made of 4x4-tile
	// squares (bits 4) is reduced to 3x2 tiles.
	const fw, fh = 40, 24
	img := make([]uint32, 10*6)
	for y := 0; y < 6; y++ {
		for x := 0; x < 10; x++ {
			img[y*10+x] = uint32((y/4)*3 + x/4)
		}
	}
	if bits := optimizeSampling(img, fw, fh, 2, maxTransformBits); bits != 4 {
		t.Fatalf("bits %d, want 4", bits)
	}
	for i, v := range img[:3*2] {
		if v != uint32(i) {
			t.Fatalf("subsampled image %v", img[:6])
		}
	}
	// The reduction is bounded by maxBits.
	flat := make([]uint32, 10*6)
	if bits := optimizeSampling(flat, fw, fh, 2, 3); bits != 3 {
		t.Fatalf("flat: bits %d, want 3", bits)
	}
	// Non-uniform squares are kept.
	noise := make([]uint32, 10*6)
	for i := range noise {
		noise[i] = uint32(i)
	}
	want := append([]uint32(nil), noise...)
	if bits := optimizeSampling(noise, fw, fh, 2, maxTransformBits); bits != 2 || !equalSlice(noise, want) {
		t.Fatalf("noise: bits %d", bits)
	}
	// Rows equal but columns not: no reduction.
	rows := make([]uint16, 10*6)
	for i := range rows {
		rows[i] = uint16(i % 10)
	}
	if bits := optimizeSampling(rows, fw, fh, 2, maxTransformBits); bits != 2 {
		t.Fatalf("rows: bits %d, want 2", bits)
	}
}

func TestClampBits(t *testing.T) {
	for _, c := range []struct{ w, h, bits, min, max, size, want int }{
		{1000, 1, 3, 2, 9, 2600, 3},
		{4, 4, 5, 2, 9, 2600, 2},          // 1 tile: smallest bits giving 1 tile
		{3000, 2000, 2, 2, 9, 2600, 6},    // too many tiles
		{3000, 2000, 2, 2, 9, 1 << 14, 5}, // predictor image limit
		{16383, 16383, 2, 2, 5, 2600, 5},  // bounded by max
	} {
		if got := clampBits(c.w, c.h, c.bits, c.min, c.max, c.size); got != c.want {
			t.Errorf("clampBits(%d, %d, %d, %d, %d, %d) = %d, want %d", c.w, c.h, c.bits, c.min, c.max, c.size, got, c.want)
		}
	}
}

func TestFixedLog(t *testing.T) {
	// fastSLog2 and fastLog2 approximate v*log2(v) and log2(v) in fixed
	// point; the libwebp tables are exact roundings.
	for _, v := range []uint32{1, 2, 3, 255, 256, 1000, 4095, 4096, 65535, 65536, 1 << 20} {
		l := float64(fastLog2(v)) / (1 << log2PrecisionBits)
		sl := float64(fastSLog2(v)) / (1 << log2PrecisionBits)
		want := 0.0
		for x := v; x > 1; x >>= 1 {
			want++
		}
		if d := l - want; d < -0.01 || d > 1.01 {
			t.Errorf("fastLog2(%d) = %v", v, l)
		}
		if d := sl/float64(v) - l; d < -0.01 || d > 0.01 {
			t.Errorf("fastSLog2(%d)/%d = %v, fastLog2 %v", v, v, sl/float64(v), l)
		}
	}
	if divRound(7, 2) != 4 || divRound(-7, 2) != -4 || divRound(5, 10) != 1 {
		t.Error("divRound")
	}
}

// decodeVP8L decodes a VP8L payload (signature included) with the package's
// decoder.
func decodeVP8L(t *testing.T, payload []byte, w, h int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if err := vp8l.Decode(payload, pipeline.NewDirect(img)); err != nil {
		t.Fatal(err)
	}
	return img.Pix
}

// TestSubsampledRoundTrip encodes images whose transform and entropy images
// can be sub-sampled (smooth gradients, uniform blocks), few-color images
// (box LZ77 parse) and alpha planes, and checks that they decode exactly.
func TestSubsampledRoundTrip(t *testing.T) {
	imgs := []testImage{
		{"grad1000x1", newNRGBA(1000, 1, func(x, y int) color.NRGBA {
			v := uint8(x * 255 / 999)
			return color.NRGBA{v, v, v, 255}
		})},
		{"grad512x64", newNRGBA(512, 64, func(x, y int) color.NRGBA {
			return color.NRGBA{uint8(x / 2), uint8(y * 4), uint8(x/2 + y), 255}
		})},
		{"blocks", newNRGBA(300, 200, func(x, y int) color.NRGBA {
			b := (x/64 + y/64) % 4
			return color.NRGBA{uint8(b * 60), uint8(x*b + y), uint8(255 - b*50), uint8(255 - (b&1)*100)}
		})},
		{"mask", newNRGBA(97, 61, func(x, y int) color.NRGBA {
			if (x-48)*(x-48)+(y-30)*(y-30) < 600 {
				return color.NRGBA{200, 10, 10, 255}
			}
			return color.NRGBA{0, 0, 0, 0}
		})},
		{"constant", newNRGBA(100, 150, func(x, y int) color.NRGBA { return color.NRGBA{0, 0, 0, 0} })},
	}
	for _, ti := range imgs {
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		for _, lv := range [][2]int{{0, 0}, {2, 25}, {4, 75}, {5, 90}, {6, 75}, {6, 100}} {
			out, err := Encode(ti.img, &Config{Quality: float32(lv[1]), Method: lv[0], Exact: true, NearLossless: 100})
			if err != nil {
				t.Fatal(err)
			}
			if got := decodeVP8L(t, out, w, h); !bytes.Equal(got, ti.img.Pix) {
				t.Fatalf("%s m%d q%d: decoded image differs", ti.name, lv[0], lv[1])
			}
		}
		plane := make([]byte, w*h)
		for i := range plane {
			plane[i] = ti.img.Pix[4*i+3] ^ ti.img.Pix[4*i]
		}
		for _, m := range []int{0, 3, 4, 6} {
			for _, q := range []float32{50, 100} {
				stream, err := EncodeAlpha(plane, w, h, w, m, q)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]byte, w*h)
				if err := vp8l.DecodeAlpha(stream, w, h, got); err != nil {
					t.Fatalf("%s alpha m%d q%v: %v", ti.name, m, q, err)
				}
				if !bytes.Equal(got, plane) {
					t.Fatalf("%s alpha m%d q%v: decoded plane differs", ti.name, m, q)
				}
			}
		}
	}
}
