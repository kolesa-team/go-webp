package pipeline

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of yuvToRGBARow (NEON on arm64, SSE4.1/AVX2 on amd64
// at every level the machine supports) against the portable reference
// yuvToRGBARowGo.

// withYUVLevels runs fn as a subtest once per SIMD level this machine
// supports, with yuvLevel set to it (on non-x86 machines only "go" exists
// and fn runs once on the platform's kernel).
func withYUVLevels(t *testing.T, fn func(t *testing.T)) {
	saved := yuvLevel
	defer func() { yuvLevel = saved }()
	for _, l := range cpuinfo.Levels() {
		yuvLevel = l
		t.Run(l.String(), fn)
	}
}

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
// 16-pixel NEON block (and both halves of a 32-pixel AVX2 block).
func TestYUVToRGBARowExhaustive(t *testing.T) {
	withYUVLevels(t, testYUVToRGBARowExhaustive)
}

func testYUVToRGBARowExhaustive(t *testing.T) {
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
	withYUVLevels(t, testYUVToRGBARowMatchesGo)
}

func testYUVToRGBARowMatchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	widths := []int{}
	for w := 1; w <= 64; w++ {
		widths = append(widths, w)
	}
	widths = append(widths, 100, 255, 256, 257, 1023, 1024, 4097)
	for _, w := range widths {
		for i := 0; i < 8; i++ {
			mk := func() []byte { // at a random offset: unaligned rows
				off := r.IntN(16)
				b := make([]byte, off+w+r.IntN(16))
				for k := range b {
					b[k] = byte(r.IntN(256))
				}
				return b[off : off+w]
			}
			ys, us, vs := mk(), mk(), mk()
			var as []byte
			if i&1 != 0 {
				as = mk()
			}
			want := make([]byte, 4*w)
			yuvToRGBARowGo(ys, us, vs, as, want)
			off := 64 + r.IntN(16)
			buf := bytes.Repeat([]byte{0xa5}, 4*w+off+64)
			got := buf[off : off+4*w]
			yuvToRGBARow(ys, us, vs, as, got)
			if !bytes.Equal(got, want) {
				t.Fatalf("%v w=%d alpha=%v: rows differ", yuvLevel, w, as != nil)
			}
			for k, b := range append(buf[:off:off], buf[off+4*w:]...) {
				if b != 0xa5 {
					t.Fatalf("w=%d: write outside the row (guard byte %d)", w, k)
				}
			}
		}
	}
}

// TestYUVToRGBARowGuardPages runs the converter on rows that end right
// before (or start right after) an inaccessible page: an out-of-bounds
// access faults.
func TestYUVToRGBARowGuardPages(t *testing.T) {
	withYUVLevels(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(25, 26))
		for _, w := range []int{15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 1023, 1024} {
			for _, atEnd := range []bool{true, false} {
				for _, alpha := range []bool{false, true} {
					g := func() []byte {
						b := guardedBytes(t, w, atEnd)
						for i := range b {
							b[i] = byte(r.IntN(256))
						}
						return b
					}
					ys, us, vs := g(), g(), g()
					var as []byte
					if alpha {
						as = g()
					}
					want := make([]byte, 4*w)
					yuvToRGBARowGo(ys, us, vs, as, want)
					got := guardedBytes(t, 4*w, atEnd)
					yuvToRGBARow(ys, us, vs, as, got)
					if !bytes.Equal(got, want) {
						t.Fatalf("w=%d atEnd=%v alpha=%v: rows differ", w, atEnd, alpha)
					}
				}
			}
		}
	})
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
		want := make([]byte, 4*w)
		yuvToRGBARowGo(ys, us, vs, as, want)
		saved := yuvLevel
		defer func() { yuvLevel = saved }()
		for _, l := range cpuinfo.Levels() {
			yuvLevel = l
			got := make([]byte, 4*w)
			yuvToRGBARow(ys, us, vs, as, got)
			if !bytes.Equal(got, want) {
				t.Fatalf("%v w=%d alpha=%v: rows differ", l, w, alpha)
			}
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
	saved := yuvLevel
	defer func() { yuvLevel = saved }()
	for _, l := range cpuinfo.Levels() {
		yuvLevel = l
		b.Run("kernel/"+l.String(), func(b *testing.B) {
			b.SetBytes(w)
			for b.Loop() {
				yuvToRGBARow(ys, us, vs, nil, out)
			}
		})
	}
	b.Run("go", func(b *testing.B) {
		b.SetBytes(w)
		for b.Loop() {
			yuvToRGBARowGo(ys, us, vs, nil, out)
		}
	})
}
