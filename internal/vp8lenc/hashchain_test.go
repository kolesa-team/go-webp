// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

// buildChainRef is buildChain with a table of -1 for no position (as
// libwebp), allocated per call.
func buildChainRef(argb []uint32, chain []uint32) {
	size := len(argb)
	first := make([]int32, hashSize)
	for i := range first {
		first[i] = -1
	}
	argbComp := argb[0] == argb[1]
	pos := 0
	for pos < size-2 {
		argbCompNext := argb[pos+1] == argb[pos+2]
		if argbComp && argbCompNext {
			length := 1
			c := argb[pos]
			for pos+length+2 < size && argb[pos+length+2] == c {
				length++
			}
			if length > maxLength {
				for k := 0; k < length-maxLength; k++ {
					chain[pos+k] = 0xffffffff
				}
				pos += length - maxLength
				length = maxLength
			}
			for length > 0 {
				h := pixPairHash(c, uint32(length))
				length--
				chain[pos] = uint32(first[h])
				first[h] = int32(pos)
				pos++
			}
			argbComp = false
		} else {
			h := pixPairHash(argb[pos], argb[pos+1])
			chain[pos] = uint32(first[h])
			first[h] = int32(pos)
			pos++
			argbComp = argbCompNext
		}
	}
	chain[pos] = uint32(first[pixPairHash(argb[pos], argb[pos+1])])
}

// TestBuildChain compares buildChain (whose pooled table is reused from
// call to call) with buildChainRef.
func TestBuildChain(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	iters := 300
	if testing.Short() {
		iters = 60
	}
	for i := 0; i < iters; i++ {
		n := 2 + rng.Intn(3000)
		if i%10 == 0 {
			n = 2 + rng.Intn(100000)
		}
		xsize := 1 + rng.Intn(min(n, 512))
		argb := randHashImage(rng, n, xsize)
		got, want := make([]uint32, n), make([]uint32, n)
		buildChain(argb, got)
		buildChainRef(argb, want)
		if !slices.Equal(got, want) {
			t.Fatalf("n %d xsize %d: chains differ", n, xsize)
		}
	}
}
