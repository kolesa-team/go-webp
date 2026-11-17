package vp8lenc

import (
	"math/rand"
	"slices"
	"testing"
)

// TestClusterCopyOnWrite checks the copy-on-write working set of the
// clustering: the tile histograms (shared with the working set) keep their
// populations and costs through every combination step, the final
// histograms are the sums of their tiles, and a reused clusterer gives the
// same result as a new one.
func TestClusterCopyOnWrite(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const w, h = 200, 150
	n := w * h
	// Blocks of different statistics, so that some tiles merge and others
	// do not; literals, copies and cache hits.
	var refs []pixOrCopy
	for i := 0; i < n; {
		x, y := i%w, i/w
		block := (x/40 + y/30) % 5
		switch {
		case block == 0 && i > w && x+8 <= w:
			refs = append(refs, mkCopy(uint32(w), 8))
			i += 8
		case block == 1:
			refs = append(refs, mkCache(uint32(rng.Intn(16))))
			i++
		default:
			c := 0xff000000 | uint32(block*40+rng.Intn(4+block*8))<<8 | uint32(rng.Intn(1+block))
			refs = append(refs, mkLiteral(c))
			i++
		}
	}
	tileHistos := func(bits, cacheBits int) histoSet {
		var s histoSet
		hx := subSampleSize(w, bits)
		s.init(hx*subSampleSize(h, bits), cacheBits)
		x, y := 0, 0
		for _, r := range refs {
			s.addRef(s.h[(y>>bits)*hx+(x>>bits)].d, r, w)
			x += r.length()
			for x >= w {
				x -= w
				y++
			}
		}
		for i := range s.h {
			s.computeCost(&s.h[i])
		}
		return s
	}
	reused := new(clusterer)
	for _, bits := range []int{2, 3, 4} {
		for _, quality := range []int{0, 25, 75, 100} {
			for _, lowEffort := range []bool{false, true} {
				const cacheBits = 4
				var c clusterer
				var out histoSet
				ng := c.buildHistoImage(w, h, refs, quality, lowEffort, bits, cacheBits, &out, 0)
				want := tileHistos(bits, cacheBits)
				if !slices.Equal(c.orig.data, want.data) {
					t.Fatalf("bits %d q %d low %v: tile histograms changed", bits, quality, lowEffort)
				}
				for i := range want.h {
					a, b := &c.orig.h[i], &want.h[i]
					if a.bitCost != b.bitCost || a.costs != b.costs || a.trivial != b.trivial || a.isUsed != b.isUsed {
						t.Fatalf("bits %d q %d low %v: costs of tile %d changed", bits, quality, lowEffort, i)
					}
				}
				var sum histoSet
				sum.init(ng, cacheBits)
				for i := range want.h {
					addInto(&sum.h[c.symbols[i]], &want.h[i])
				}
				if !slices.Equal(out.data, sum.data) {
					t.Fatalf("bits %d q %d low %v: final histograms are not the sums of their tiles", bits, quality, lowEffort)
				}
				if ng < 2 && quality > 0 && bits == 2 && !lowEffort {
					t.Errorf("bits %d q %d: %d cluster(s), want several", bits, quality, ng)
				}
				var out2 histoSet
				ng2 := reused.buildHistoImage(w, h, refs, quality, lowEffort, bits, cacheBits, &out2, 0)
				if ng2 != ng || !slices.Equal(reused.symbols, c.symbols) || !slices.Equal(out2.data, out.data) {
					t.Fatalf("bits %d q %d low %v: reused clusterer differs", bits, quality, lowEffort)
				}
			}
		}
	}
}
