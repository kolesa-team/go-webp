package vp8lenc

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of the cross-color histograms (ccHistoRed,
// ccHistoBlue) against their Go references, at every kernel level.

// ccBoth adds the transformed red and blue values of the entries to
// histograms prefilled with base, with the kernels (at the current level)
// and with the Go references, and fails if they differ.
func ccBoth(t *testing.T, b, g, r []int8, c []int32, mg, mr int8, base *[256]int32) {
	t.Helper()
	wantR, gotR := *base, *base
	ccHistoRedGo(&wantR, r, g, c, mg)
	ccHistoRed(&gotR, r, g, c, mg)
	if gotR != wantR {
		t.Fatalf("red n %d m %d: got %v, want %v", len(r), mg, gotR, wantR)
	}
	wantB, gotB := *base, *base
	ccHistoBlueGo(&wantB, b, g, r, c, mg, mr)
	ccHistoBlue(&gotB, b, g, r, c, mg, mr)
	if gotB != wantB {
		t.Fatalf("blue n %d mg %d mr %d: got %v, want %v", len(b), mg, mr, gotB, wantB)
	}
}

// randCC returns n entries (int8 channels, counts) at offset off of their
// buffers: random bytes, extremes (-128, 127, 0, -1) or few distinct
// values, and counts that are small, large or any int32 (the sums wrap
// as in the Go code).
func randCC(rng *rand.Rand, n, off int) (b, g, r []int8, c []int32) {
	mk := func() []int8 {
		s := make([]int8, n+off)[off:]
		kind := rng.Intn(3)
		for i := range s {
			switch kind {
			case 0:
				s[i] = int8(rng.Intn(256))
			case 1:
				s[i] = []int8{-128, 127, 0, -1}[rng.Intn(4)]
			default:
				s[i] = int8(rng.Intn(3) - 1)
			}
		}
		return s
	}
	b, g, r = mk(), mk(), mk()
	c = make([]int32, n+off)[off:]
	hi := []int32{1, 1000, 1 << 30}[rng.Intn(3)]
	for i := range c {
		c[i] = rng.Int31n(hi)
		if hi == 1<<30 && rng.Intn(2) == 0 {
			c[i] = -c[i]
		}
	}
	return b, g, r, c
}

func TestCCHistoLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	sizes := []int{}
	for n := 1; n <= 64; n++ {
		sizes = append(sizes, n)
	}
	sizes = append(sizes, 100, 255, 1000, 4096)
	iters := 3
	if testing.Short() {
		iters = 1
	}
	forEachLevel(t, func(t *testing.T) {
		var base [256]int32
		for _, n := range sizes {
			for off := 0; off < 16; off++ {
				for k := 0; k < iters; k++ {
					b, g, r, c := randCC(rng, n, off)
					for i := range base {
						base[i] = rng.Int31n(1000)
					}
					mg := int8(rng.Intn(256))
					mr := int8(rng.Intn(256))
					if rng.Intn(4) == 0 {
						mg = []int8{-128, 127, 0, -1}[rng.Intn(4)]
						mr = []int8{-128, 127, 0, -1}[rng.Intn(4)]
					}
					ccBoth(t, b, g, r, c, mg, mr, &base)
				}
			}
		}
		// Empty entries.
		ccBoth(t, nil, nil, nil, nil, 3, 5, &base)
	})
}

// TestCCHistoExhaustive checks every (value, green, multiplier) of the red
// transform and every (blue, green, red) with extreme and random
// multipliers of the blue one: each value in its own histogram entry with
// a count unique to it (its index), so that a wrong bin shows.
func TestCCHistoExhaustive(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	rng := rand.New(rand.NewSource(17))
	d, s := make([]int8, 1<<16), make([]int8, 1<<16)
	c := make([]int32, 1<<16)
	for i := range d {
		d[i], s[i], c[i] = int8(i), int8(i>>8), int32(i)
	}
	for _, l := range kernelLevels() {
		if l == cpuinfo.Go {
			continue
		}
		t.Run(l.String(), func(t *testing.T) {
			defer setLevel(l)()
			var zero [256]int32
			for m := -128; m < 128; m++ {
				ccBoth(t, d, s, s, c, int8(m), 0, &zero)
				ccBoth(t, d, s, d, c, int8(m), int8(rng.Intn(256)), &zero)
				ccBoth(t, d, d, s, c, int8(rng.Intn(256)), int8(m), &zero)
			}
		})
	}
}

func FuzzCCHisto(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, int8(-3), int8(100), uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, mg, mr int8, off uint8) {
		n := len(data) / 3
		o := int(off % 16)
		b, g, r := make([]int8, n+o)[o:], make([]int8, n+o)[o:], make([]int8, n+o)[o:]
		c := make([]int32, n+o)[o:]
		for i := 0; i < n; i++ {
			b[i], g[i], r[i] = int8(data[3*i]), int8(data[3*i+1]), int8(data[3*i+2])
			c[i] = int32(binary.LittleEndian.Uint16(data[i:])) - 1000
		}
		var base [256]int32
		forEachLevel(t, func(t *testing.T) { ccBoth(t, b, g, r, c, mg, mr, &base) })
	})
}

// BenchmarkCCHisto runs the cost histograms of the cross-color search on
// the distinct pairs and triples of the 32x32 tiles of the photo (as at
// level 6), for 8 multiplier candidates per tile.
func BenchmarkCCHisto(b *testing.B) {
	w, argb, _, _ := photoParse(b)
	h := len(argb) / w
	var tiles []tilePixels
	for y := 0; y+32 <= h && len(tiles) < 64; y += 96 {
		for x := 0; x+32 <= w && len(tiles) < 64; x += 96 {
			var tp tilePixels
			tp.load(argb, w, x, y, 32, 32)
			tiles = append(tiles, tp)
		}
	}
	ms := []int8{0, 16, -16, 32, -8, 4, -2, 2}
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			var histo [256]int32
			for i := 0; i < b.N; i++ {
				tp := &tiles[i%len(tiles)]
				for k, m := range ms {
					histo = [256]int32{}
					ccHistoRed(&histo, tp.pr, tp.pg, tp.pc, m)
					histo = [256]int32{}
					ccHistoBlue(&histo, tp.tb, tp.tg, tp.tr, tp.tc, m, ms[(k+3)&7])
				}
			}
		})
	}
}
