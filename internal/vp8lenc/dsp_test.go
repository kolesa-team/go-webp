package vp8lenc

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of the kernel entry points (dsp_amd64.go,
// dsp_generic.go) against their Go references, at every kernel level of
// the machine (only the Go level outside amd64).

// forEachLevel runs f as a subtest at every level of testLevels.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	for _, l := range testLevels() {
		t.Run(l.String(), func(t *testing.T) {
			defer setLevel(l)()
			f(t)
		})
	}
}

// forEachLevelIters is forEachLevel for the differential tests of kernels
// that exist at the levels own: f runs n iterations at those levels and
// n/10 at the others (the levels above run the same kernels, and the Go
// level compares the reference with itself).
func forEachLevelIters(t *testing.T, n int, own []cpuinfo.Level, f func(t *testing.T, iters int)) {
	t.Helper()
	for _, l := range testLevels() {
		iters := max(n/10, 1)
		if slices.Contains(own, l) {
			iters = n
		}
		t.Run(l.String(), func(t *testing.T) {
			defer setLevel(l)()
			f(t, iters)
		})
	}
}

// atEveryLevel calls f at every level of testLevels.
func atEveryLevel(f func(l cpuinfo.Level)) {
	for _, l := range testLevels() {
		restore := setLevel(l)
		f(l)
		restore()
	}
}

// randPixels returns n pixels whose bytes are drawn from a mix of uniform
// noise, the extremes 0 and 255 (where the clamps and the averages round)
// and small values around a base (smooth content).
func randPixels(rng *rand.Rand, n int) []uint32 {
	p := make([]uint32, n)
	kind := rng.Intn(4)
	base := rng.Uint32()
	for i := range p {
		switch kind {
		case 0:
			p[i] = rng.Uint32()
		case 1:
			var v uint32
			for k := 0; k < 4; k++ {
				v |= []uint32{0, 0xff, 1, 0xfe}[rng.Intn(4)] << (8 * k)
			}
			p[i] = v
		case 2:
			p[i] = base + uint32(rng.Intn(5))*0x01010101 - uint32(rng.Intn(5))*0x00010001
		default:
			if rng.Intn(3) == 0 {
				p[i] = rng.Uint32()
			} else {
				p[i] = base
			}
		}
	}
	return p
}

func TestPredictRowLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	widths := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 1000}
	forEachLevel(t, func(t *testing.T) {
		for _, w := range widths {
			for iter := 0; iter < 40; iter++ {
				x0 := 1 + rng.Intn(16)
				width := x0 + w + rng.Intn(3)
				cur := randPixels(rng, width)
				up := randPixels(rng, width+1)
				x1 := x0 + w
				for mode := 0; mode < numPredModes; mode++ {
					want := make([]uint32, w+2)
					got := make([]uint32, w+2)
					want[w], want[w+1] = 0xdeadbeef, 0xfeedface
					got[w], got[w+1] = 0xdeadbeef, 0xfeedface
					predictRowGo(mode, cur, up, x0, x1, want)
					predictRow(mode, cur, up, x0, x1, got)
					if !slices.Equal(got, want) {
						t.Fatalf("mode %d x0 %d x1 %d: got %08x, want %08x", mode, x0, x1, got, want)
					}
				}
			}
		}
	})
}

// TestPredictRowRandomBlocks checks 8-pixel rows (one AVX2 block, two
// SSE4.1 blocks) of independent random bytes, 200k per mode and level.
func TestPredictRowRandomBlocks(t *testing.T) {
	n := 200000
	if testing.Short() {
		n = 20000
	}
	rng := rand.New(rand.NewSource(2))
	cur := make([]uint32, 9)
	up := make([]uint32, 10)
	want := make([]uint32, 8)
	got := make([]uint32, 8)
	forEachLevel(t, func(t *testing.T) {
		for i := 0; i < n; i++ {
			for k := range cur {
				cur[k] = rng.Uint32()
			}
			for k := range up {
				up[k] = rng.Uint32()
			}
			mode := i % numPredModes
			predictRowGo(mode, cur, up, 1, 9, want)
			predictRow(mode, cur, up, 1, 9, got)
			if !slices.Equal(got, want) {
				t.Fatalf("mode %d cur %08x up %08x: got %08x, want %08x", mode, cur, up, got, want)
			}
		}
	})
}

// deBruijn256 returns a cyclic sequence of 65536 bytes in which every pair
// of bytes appears once as consecutive elements (de Bruijn B(256, 2)).
func deBruijn256() []byte {
	const k, n = 256, 2
	a := make([]int, k*n)
	var seq []byte
	var db func(t, p int)
	db = func(t, p int) {
		if t > n {
			if n%p == 0 {
				for _, v := range a[1 : p+1] {
					seq = append(seq, byte(v))
				}
			}
			return
		}
		a[t] = a[t-p]
		db(t+1, p)
		for j := a[t-p] + 1; j < k; j++ {
			a[t] = j
			db(t+1, t)
		}
	}
	db(1, 1)
	return seq
}

// TestPredictRowExhaustiveTriples runs every (L, T, TL) byte triple of
// every channel through every mode at every level: up is a de Bruijn
// sequence (every (TL, T) pair) and the rows of cur are constant (L), one
// row per value of L; each channel uses a different rotation of the
// sequence.
func TestPredictRowExhaustiveTriples(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	seq := deBruijn256()
	const w = 1 << 16
	up := make([]uint32, w+2)
	for x := range up {
		var v uint32
		for c := 0; c < 4; c++ {
			v |= uint32(seq[(x+12345*c)%w]) << (8 * c)
		}
		up[x] = v
	}
	cur := make([]uint32, w+1)
	want := make([]uint32, w)
	got := make([]uint32, w)
	for l := 0; l < 256; l++ {
		for x := range cur {
			cur[x] = uint32(l) | uint32(l^0x5a)<<8 | uint32(255-l)<<16 | uint32(l*7&0xff)<<24
		}
		for mode := 0; mode < numPredModes; mode++ {
			predictRowGo(mode, cur, up, 1, w+1, want)
			atEveryLevel(func(lv cpuinfo.Level) {
				clear(got)
				predictRow(mode, cur, up, 1, w+1, got)
				if !slices.Equal(got, want) {
					i := 0
					for got[i] == want[i] {
						i++
					}
					t.Fatalf("level %v l %d mode %d: pixel %d: got %08x, want %08x", lv, l, mode, i, got[i], want[i])
				}
			})
		}
	}
}

func FuzzPredictRow(f *testing.F) {
	f.Add(uint8(11), uint8(1), []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44})
	f.Add(uint8(13), uint8(3), make([]byte, 200))
	f.Fuzz(func(t *testing.T, mode, x0 uint8, data []byte) {
		m := int(mode) % numPredModes
		s := 1 + int(x0)%16
		n := len(data) / 8
		if n <= s {
			return
		}
		cur := make([]uint32, n)
		up := make([]uint32, n+1)
		for i := range cur {
			cur[i] = binary.LittleEndian.Uint32(data[4*i:])
			up[i] = binary.LittleEndian.Uint32(data[4*n+4*i:])
		}
		up[n] = cur[0]
		want := make([]uint32, n-s)
		predictRowGo(m, cur, up, s, n, want)
		atEveryLevel(func(l cpuinfo.Level) {
			got := make([]uint32, n-s)
			predictRow(m, cur, up, s, n, got)
			if !slices.Equal(got, want) {
				t.Fatalf("level %v mode %d x0 %d n %d: got %08x, want %08x", l, m, s, n, got, want)
			}
		})
	})
}

// BenchmarkPredictRow runs the 14 modes on one row segment per op.
func BenchmarkPredictRow(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	for _, w := range []int{16, 32, 960} {
		cur := randPixels(rng, w+1)
		up := randPixels(rng, w+2)
		out := make([]uint32, w)
		for _, l := range testLevels() {
			b.Run(fmt.Sprintf("w%d/%v", w, l), func(b *testing.B) {
				defer setLevel(l)()
				for i := 0; i < b.N; i++ {
					for mode := 0; mode < numPredModes; mode++ {
						predictRow(mode, cur, up, 1, w+1, out)
					}
				}
			})
		}
	}
}

// TestKernelChunks runs the differential tests of the kernels that walk a
// whole image or parse in chunks (one assembly call per chunk, see hcChunk
// in dsp_amd64.go) with chunks of 1 and 3, so that every call boundary
// falls inside the data.
func TestKernelChunks(t *testing.T) {
	if len(kernelLevels()) == 1 {
		t.Skip("no kernels on this machine")
	}
	tests := []struct {
		name string
		f    func(*testing.T)
	}{
		{"HashChainRunLevels", TestHashChainRunLevels},
		{"HashChainRunStripes", TestHashChainRunStripes},
		{"BackwardRefsRleLevels", TestBackwardRefsRleLevels},
		{"BestCacheBitsLevels", TestBestCacheBitsLevels},
		{"BestCacheBitsBadRefs", TestBestCacheBitsBadRefs},
		{"SubtractGreenLevels", TestSubtractGreenLevels},
		{"AddRefsLevels", TestAddRefsLevels},
		{"AddRefsBadRefs", TestAddRefsBadRefs},
		{"AddTileRefsLevels", TestAddTileRefsLevels},
		{"AddTileRefsBadRefs", TestAddTileRefsBadRefs},
		{"WriteImageDataLevels", TestWriteImageDataLevels},
		{"WriteImageDataBadRefs", TestWriteImageDataBadRefs},
	}
	for _, c := range []int{1, 3} {
		for _, tc := range tests {
			t.Run(fmt.Sprintf("%s/chunk%d", tc.name, c), func(t *testing.T) {
				defer setChunks(c)()
				tc.f(t)
			})
		}
	}
}
