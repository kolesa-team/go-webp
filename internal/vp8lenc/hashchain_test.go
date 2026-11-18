package vp8lenc

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// hcInputs returns match-search inputs of various kinds: noise, few colors
// with runs, runs longer than maxLength, repeated rows and a palette-index
// like image.
func hcInputs() []struct {
	name string
	argb []uint32
	w, h int
} {
	rng := rand.New(rand.NewSource(11))
	type in = struct {
		name string
		argb []uint32
		w, h int
	}
	var out []in
	add := func(name string, w, h int, f func(i int, a []uint32) uint32) {
		a := make([]uint32, w*h)
		for i := range a {
			a[i] = f(i, a)
		}
		out = append(out, in{name, a, w, h})
	}
	add("noise", 97, 61, func(i int, a []uint32) uint32 { return rng.Uint32() })
	add("fewcolors", 211, 83, func(i int, a []uint32) uint32 {
		if i > 0 && rng.Intn(4) != 0 {
			return a[i-1]
		}
		return uint32(rng.Intn(5))
	})
	add("longruns", 300, 70, func(i int, a []uint32) uint32 { return uint32(i / 5000) })
	add("rows", 129, 90, func(i int, a []uint32) uint32 {
		if i >= 129 && (i/129)%3 != 0 {
			return a[i-129]
		}
		return uint32(rng.Intn(64))
	})
	add("repeats", 64, 64, func(i int, a []uint32) uint32 {
		if i >= 37 && rng.Intn(8) != 0 {
			return a[i-37]
		}
		return rng.Uint32() & 0x0f0f0f0f
	})
	add("tiny", 3, 1, func(i int, a []uint32) uint32 { return uint32(i & 1) })
	add("column", 1, 700, func(i int, a []uint32) uint32 { return uint32(rng.Intn(3)) })
	return out
}

// TestHashChainStripes checks that the striped match search gives exactly
// the sequential results for any number of stripes and helpers.
func TestHashChainStripes(t *testing.T) {
	defer func(v int) { minHCStripe = v }(minHCStripe)
	for _, in := range hcInputs() {
		for _, q := range []int{0, 30, 75, 100} {
			for _, low := range []bool{false, true} {
				for _, wcap := range []int{0, 100} {
					var ref hashChain
					ref.fill(in.argb, in.w, in.h, q, low, wcap)
					want := slices.Clone(ref.ol)
					for _, stripe := range []int{64, 128, 1024} {
						minHCStripe = stripe
						for _, helpers := range []int{1, 3, 7} {
							for _, limit := range []int{1, 8} {
								func() {
									defer workers.SetLimitForTest(workers.SetLimitForTest(limit))
									var hc hashChain
									sc := new(scratch)
									hc.fillPar(in.argb, in.w, in.h, q, low, wcap, helpers, sc)
									sc.crew.stop()
									if !slices.Equal(hc.ol, want) {
										i := 0
										for hc.ol[i] == want[i] {
											i++
										}
										t.Fatalf("%s q%d low=%v cap=%d stripe %d helpers %d: ol[%d] = %x, want %x",
											in.name, q, low, wcap, stripe, helpers, i, hc.ol[i], want[i])
									}
								}()
							}
						}
					}
				}
			}
		}
	}
}

// TestHashChainStripesExhausted runs the striped search with no helper slot
// free: the caller searches every stripe itself.
func TestHashChainStripesExhausted(t *testing.T) {
	defer func(v int) { minHCStripe = v }(minHCStripe)
	minHCStripe = 64
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	defer takeBudget(t)()
	for _, in := range hcInputs() {
		var ref, hc hashChain
		ref.fill(in.argb, in.w, in.h, 75, false, 0)
		sc := new(scratch)
		hc.fillPar(in.argb, in.w, in.h, 75, false, 0, 7, sc)
		sc.crew.stop()
		if !slices.Equal(hc.ol, ref.ol) {
			t.Fatalf("%s: striped search differs", in.name)
		}
	}
}

func BenchmarkHashChainStripes(b *testing.B) {
	in := hcInputs()[1]
	big := make([]uint32, 0, 1<<20)
	for len(big) < 1<<20 {
		big = append(big, in.argb...)
	}
	w := 1024
	h := len(big) / w
	for _, helpers := range []int{0, 3} {
		b.Run(fmt.Sprint("helpers", helpers), func(b *testing.B) {
			var hc hashChain
			sc := new(scratch)
			for i := 0; i < b.N; i++ {
				hc.fillPar(big, w, h, 75, false, 0, helpers, sc)
			}
			sc.crew.stop()
		})
	}
}
