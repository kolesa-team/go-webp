package pipeline

import (
	"bytes"
	"math/rand/v2"
	"testing"
)

// Differential tests of yuvToRGBARow (NEON on arm64) against the portable
// reference yuvToRGBARowGo.

// yuvRefPixel is libwebp's VP8YuvToRgb written out independently.
func yuvRefPixel(y, u, v int) (r, g, b byte) {
	clip := func(x int) byte { return byte(min(max(x>>6, 0), 255)) }
	yh := (y * 19077) >> 8
	return clip(yh + (v*26149)>>8 - 14234),
		clip(yh - (u*6419)>>8 - (v*13320)>>8 + 8708),
		clip(yh + (u*33050)>>8 - 17685)
}

// TestYUVToRGBARowExhaustive converts all 2^24 (y, u, v) triples: row
// (u, v>>4) holds the 16 v values x 256 luma values; a second pass shifts
// the rows by 8 pixels so every triple also lands in the other half of a
// 16-pixel NEON block.
func TestYUVToRGBARowExhaustive(t *testing.T) {
	const w = 16 * 256
	for _, shift := range []int{0, 8} {
		ys, us, vs := make([]byte, w+shift), make([]byte, w+shift), make([]byte, w+shift)
		out := make([]byte, 4*(w+shift))
		for x := 0; x < w; x++ {
			ys[shift+x] = byte(x)
		}
		for u := 0; u < 256; u++ {
			for vh := 0; vh < 16; vh++ {
				for x := 0; x < w; x++ {
					us[shift+x] = byte(u)
					vs[shift+x] = byte(vh<<4 | x>>8)
				}
				yuvToRGBARow(ys, us, vs, nil, out)
				for x := 0; x < w; x++ {
					y, v := x&255, vh<<4|x>>8
					r, g, b := yuvRefPixel(y, u, v)
					o := out[4*(shift+x):]
					if o[0] != r || o[1] != g || o[2] != b || o[3] != 255 {
						t.Fatalf("y=%d u=%d v=%d: got %v, want [%d %d %d 255]", y, u, v, o[:4], r, g, b)
					}
				}
			}
		}
	}
}

func TestYUVToRGBARowMatchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	widths := []int{}
	for w := 1; w <= 64; w++ {
		widths = append(widths, w)
	}
	widths = append(widths, 100, 255, 256, 257, 1023, 1024, 4097)
	for _, w := range widths {
		for i := 0; i < 8; i++ {
			mk := func() []byte {
				b := make([]byte, w)
				for k := range b {
					b[k] = byte(r.IntN(256))
				}
				return b
			}
			ys, us, vs := mk(), mk(), mk()
			var as []byte
			if i&1 != 0 {
				as = mk()
			}
			want := make([]byte, 4*w)
			yuvToRGBARowGo(ys, us, vs, as, want)
			buf := bytes.Repeat([]byte{0xa5}, 4*w+128)
			got := buf[64 : 64+4*w]
			yuvToRGBARow(ys, us, vs, as, got)
			if !bytes.Equal(got, want) {
				t.Fatalf("w=%d alpha=%v: rows differ", w, as != nil)
			}
			for k, b := range append(buf[:64:64], buf[64+4*w:]...) {
				if b != 0xa5 {
					t.Fatalf("w=%d: write outside the row (guard byte %d)", w, k)
				}
			}
		}
	}
}

func FuzzYUVToRGBARow(f *testing.F) {
	f.Add([]byte{1, 2, 3, 200, 0, 255}, uint16(17), true)
	f.Add([]byte{255}, uint16(64), false)
	f.Fuzz(func(t *testing.T, seed []byte, w16 uint16, alpha bool) {
		w := 1 + int(w16)%600
		if len(seed) == 0 {
			seed = []byte{0}
		}
		k := 0
		mk := func() []byte {
			b := make([]byte, w)
			for i := range b {
				b[i] = seed[k%len(seed)] + byte(k/len(seed))
				k++
			}
			return b
		}
		ys, us, vs := mk(), mk(), mk()
		var as []byte
		if alpha {
			as = mk()
		}
		want, got := make([]byte, 4*w), make([]byte, 4*w)
		yuvToRGBARowGo(ys, us, vs, as, want)
		yuvToRGBARow(ys, us, vs, as, got)
		if !bytes.Equal(got, want) {
			t.Fatalf("w=%d alpha=%v: rows differ", w, alpha)
		}
	})
}

func BenchmarkYUVToRGBARow(b *testing.B) {
	const w = 1024
	r := rand.New(rand.NewPCG(23, 24))
	ys, us, vs, out := make([]byte, w), make([]byte, w), make([]byte, w), make([]byte, 4*w)
	for i := range ys {
		ys[i], us[i], vs[i] = byte(r.IntN(256)), byte(r.IntN(256)), byte(r.IntN(256))
	}
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(w)
		for b.Loop() {
			yuvToRGBARow(ys, us, vs, nil, out)
		}
	})
	b.Run("go", func(b *testing.B) {
		b.SetBytes(w)
		for b.Loop() {
			yuvToRGBARowGo(ys, us, vs, nil, out)
		}
	})
}
