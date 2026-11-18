//go:build amd64 && !purego

package vp8lenc

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// cacheCountsRef is the counting loop of bestCacheBitsGo on the layout of
// cacheHistos (lits and caches as slices).
func cacheCountsRef(argb []uint32, refs []pixOrCopy, maxBits int, h *cacheHistos, lits, caches [][]uint32) {
	shift := uint(32 - maxBits)
	pos := 0
	for _, r := range refs {
		if r.mode() == modeLiteral {
			pix := argb[pos]
			pos++
			a, rr, g, b := pix>>24, (pix>>16)&0xff, (pix>>8)&0xff, pix&0xff
			key := colorCacheHash(pix, shift)
			h.alpha[0][a]++
			h.red[0][rr]++
			lits[0][g]++
			h.blue[0][b]++
			for i := maxBits; i >= 1; i-- {
				if caches[i][key] == pix {
					lits[i][numLiteralCodes+numLengthCodes+int(key)]++
				} else {
					caches[i][key] = pix
					h.alpha[i][a]++
					h.red[i][rr]++
					lits[i][g]++
					h.blue[i][b]++
				}
				key >>= 1
			}
		} else {
			l := r.length()
			code := prefixCode(l)
			for i := 0; i <= maxBits; i++ {
				lits[i][numLiteralCodes+code]++
			}
			prev := argb[pos] ^ 0xffffffff
			for _, p := range argb[pos : pos+l] {
				if p != prev {
					key := colorCacheHash(p, shift)
					for i := maxBits; i >= 1; i-- {
						caches[i][key] = p
						key >>= 1
					}
					prev = p
				}
			}
			pos += l
		}
	}
}

// newCacheHistos returns zeroed histograms for maxBits with their slices.
func newCacheHistos(maxBits int) (*cacheHistos, [][]uint32, [][]uint32) {
	h := new(cacheHistos)
	lits := make([][]uint32, maxBits+1)
	caches := make([][]uint32, maxBits+1)
	for i := 0; i <= maxBits; i++ {
		n := numLiteralCodes + numLengthCodes
		if i > 0 {
			n += 1 << uint(i)
			caches[i] = make([]uint32, 1<<uint(i))
			h.cache[i] = &caches[i][0]
		}
		lits[i] = make([]uint32, n)
		h.lit[i] = &lits[i][0]
	}
	return h, lits, caches
}

// TestCacheCountsExact compares every histogram and cache of the kernel
// with the Go loop.
func TestCacheCountsExact(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
	rng := rand.New(rand.NewSource(13))
	for i := 0; i < 300; i++ {
		n := 1 + rng.Intn(20000)
		argb := randHashImage(rng, n, 1+rng.Intn(100))
		if rng.Intn(2) == 0 {
			for k := range argb {
				argb[k] &= 0x01030507
			}
		}
		refs := randRefs(rng, n, rng.Intn(4) == 0)
		maxBits := 1 + rng.Intn(maxColorCacheBits)
		hw, lw, cw := newCacheHistos(maxBits)
		cacheCountsRef(argb, refs, maxBits, hw, lw, cw)
		hg, lg, cg := newCacheHistos(maxBits)
		if got := cacheCountsSSE41(&argb[0], n, &refs[0], len(refs), maxBits, hg); got != n {
			t.Fatalf("n %d: kernel covered %d pixels", n, got)
		}
		if hg.red != hw.red || hg.blue != hw.blue || hg.alpha != hw.alpha {
			t.Fatalf("n %d maxBits %d: channel histograms differ", n, maxBits)
		}
		for k := range lw {
			if !slices.Equal(lg[k], lw[k]) || !slices.Equal(cg[k], cw[k]) {
				t.Fatalf("n %d maxBits %d: literal histogram or cache %d differs", n, maxBits, k)
			}
		}
	}
}

func TestCacheCountsRefuses(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
	argb := make([]uint32, 10)
	for _, refs := range [][]pixOrCopy{
		{mkCopy(1, 11)},
		{mkLiteral(0), mkCopy(1, 9), mkLiteral(0)},
		{mkCopy(1, 0)},
		{mkCopy(1, 4097)},
	} {
		h, _, _ := newCacheHistos(3)
		if got := cacheCountsSSE41(&argb[0], len(argb), &refs[0], len(refs), 3, h); got != -1 {
			t.Errorf("refs %v: got %d, want -1", refs, got)
		}
	}
}
