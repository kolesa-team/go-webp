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
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of the entropy kernels against their Go references,
// at every kernel level: the results must be bit-identical.

// randHisto fills h with a random histogram: a random fraction of non-zero
// bins, with counts in a random range (small, around the 4096 limit of the
// slog2 table, large, or any int32 including negative ones).
func randHisto(rng *rand.Rand, h []int32) {
	density := rng.Float64()
	if rng.Intn(4) == 0 {
		density = 0
		if rng.Intn(2) == 0 {
			density = 1
		}
	}
	var lo, hi int64
	switch rng.Intn(5) {
	case 0:
		lo, hi = 1, 16
	case 1:
		lo, hi = 1, 1000
	case 2:
		lo, hi = 3000, 5000
	case 3:
		lo, hi = 1, 1<<24
	default:
		lo, hi = -1<<31, 1<<31-1
	}
	for i := range h {
		h[i] = 0
		if rng.Float64() < density {
			h[i] = int32(lo + rng.Int63n(hi-lo+1))
		}
	}
}

// randAcc returns the accEntropy of y, or one with random fields (the
// kernel must not depend on their consistency).
func randAcc(rng *rand.Rand, y *[256]int32) *accEntropy {
	var a accEntropy
	a.init(y)
	if rng.Intn(4) == 0 {
		for i := range a.s {
			a.s[i] = rng.NormFloat64() * 1e6
		}
		a.total = rng.NormFloat64() * 1e9
		a.sum = rng.Uint32()
	}
	return &a
}

func TestCombinedShannonEntropyLevels(t *testing.T) {
	n := 200000
	if testing.Short() {
		n = 20000
	}
	rng := rand.New(rand.NewSource(5))
	var x, y [256]int32
	forEachLevelIters(t, n, []cpuinfo.Level{cpuinfo.SSE41}, func(t *testing.T, n int) {
		for i := 0; i < n; i++ {
			randHisto(rng, x[:])
			randHisto(rng, y[:])
			acc := randAcc(rng, &y)
			want := combinedShannonEntropyGo(&x, &y, acc)
			got := combinedShannonEntropy(&x, &y, acc)
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("x %v y %v: got %v, want %v", x, y, got, want)
			}
		}
	})
}

// TestSlog2Exhaustive checks the kernels' v*log2(v) for every v below 2^24
// and random larger ones, isolated: with a zero acc and a single non-zero
// bin x[i] = 1, y[i] = v-1, the result is exactly -slog2(v). The levels
// above SSE4.1 run the same kernel: they only check every 251st v (and the
// Go level would compare the reference with itself).
func TestSlog2Exhaustive(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	var x, y [256]int32
	var acc accEntropy
	rng := rand.New(rand.NewSource(6))
	for _, l := range testLevels() {
		if l == cpuinfo.Go {
			continue
		}
		step := uint32(1)
		if l != cpuinfo.SSE41 {
			step = 251
		}
		t.Run(l.String(), func(t *testing.T) {
			defer setLevel(l)()
			slog2Levels(t, &x, &y, &acc, rng, step)
		})
	}
}

// slog2Levels is TestSlog2Exhaustive at the current level, for every
// step-th v.
func slog2Levels(t *testing.T, x, y *[256]int32, acc *accEntropy, rng *rand.Rand, step uint32) {
	check := func(v uint32, i int) {
		x[i], y[i] = 1, int32(v-1)
		got := combinedShannonEntropy(x, y, acc)
		want := combinedShannonEntropyGo(x, y, acc)
		x[i], y[i] = 0, 0
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("v %d: got %v, want %v", v, got, want)
		}
	}
	for v := uint32(1); v < 1<<24; v += step {
		check(v, int(v&255))
	}
	for k := 0; k < 1<<20; k += int(step) {
		check(rng.Uint32()|1<<24, k&255)
	}
	check(1<<32-1, 7)
	check(0, 9) // x+y wraps to 0
}

func FuzzCombinedShannonEntropy(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0}, []byte{0, 16, 0, 0}, false)
	f.Fuzz(func(t *testing.T, xb, yb []byte, junk bool) {
		var x, y [256]int32
		for i := 0; i < 256 && 4*i+4 <= len(xb); i++ {
			x[i] = int32(binary.LittleEndian.Uint32(xb[4*i:]))
		}
		for i := 0; i < 256 && 4*i+4 <= len(yb); i++ {
			y[i] = int32(binary.LittleEndian.Uint32(yb[4*i:]))
		}
		var acc accEntropy
		acc.init(&y)
		if junk {
			acc.total = float64(len(xb)) * 1.5
			acc.sum = uint32(len(yb)) << 20
		}
		want := combinedShannonEntropyGo(&x, &y, &acc)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := combinedShannonEntropy(&x, &y, &acc); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("level %v: got %v, want %v", l, got, want)
			}
		})
	})
}

// randStreaky returns n counts made of runs of random lengths (1..12) of
// zeros or of values drawn like randHisto's, so that both short and long
// streaks of both kinds occur.
func randStreaky(rng *rand.Rand, n int) []uint32 {
	p := make([]uint32, n)
	vals := make([]int32, 8)
	randHisto(rng, vals)
	for i := 0; i < n; {
		l := 1 + rng.Intn(12)
		v := uint32(vals[rng.Intn(len(vals))])
		if rng.Intn(3) == 0 {
			v = 0
		}
		for ; l > 0 && i < n; l-- {
			p[i] = v
			i++
		}
	}
	return p
}

func TestCombinedPopulationCostLevels(t *testing.T) {
	iters := 100000
	if testing.Short() {
		iters = 10000
	}
	rng := rand.New(rand.NewSource(8))
	sizes := []int{0, 1, 2, 15, 16, 17, 18, 31, 32, 33, 40, 47, 48, 49, 256, 280, 281, 282, 284, 288, 296, 312, 344, 408, 536, 792, 1304}
	forEachLevelIters(t, iters, []cpuinfo.Level{cpuinfo.SSE41}, func(t *testing.T, iters int) {
		for i := 0; i < iters; i++ {
			n := sizes[i%len(sizes)]
			if i%3 == 0 {
				n = rng.Intn(400)
			}
			var a, b []uint32
			switch rng.Intn(3) {
			case 0:
				a, b = randStreaky(rng, n), randStreaky(rng, n)
			case 1:
				a, b = randStreaky(rng, n), make([]uint32, n)
			default:
				ah, bh := make([]int32, n), make([]int32, n)
				randHisto(rng, ah)
				randHisto(rng, bh)
				a, b = make([]uint32, n), make([]uint32, n)
				for k := range a {
					a[k], b[k] = uint32(ah[k]), uint32(bh[k])
				}
			}
			want := combinedPopulationCostGo(a, b)
			got := combinedPopulationCost(a, b)
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("a %v b %v: got %v, want %v", a, b, got, want)
			}
		}
	})
}

func FuzzCombinedPopulationCost(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0}, uint16(40))
	f.Fuzz(func(t *testing.T, data []byte, n uint16) {
		size := int(n) % 600
		a, b := make([]uint32, size), make([]uint32, size)
		for i := range a {
			if len(data) >= 8 {
				a[i] = uint32(data[(2*i)%len(data)]) << (data[(2*i+1)%len(data)] & 31)
				b[i] = uint32(data[(3*i+5)%len(data)] & data[(i+7)%len(data)] >> 3)
			}
		}
		want := combinedPopulationCostGo(a, b)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := combinedPopulationCost(a, b); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("level %v: got %v, want %v", l, got, want)
			}
		})
	})
}

// BenchmarkCombinedPopulationCost runs the kernel on the literal
// histograms of photo-like data (dense around the middle, sparse tails).
func BenchmarkCombinedPopulationCost(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const n = 280 + 1<<6
	var hs [8][]uint32
	for k := range hs {
		hs[k] = make([]uint32, n)
		for i := 0; i < 4000; i++ {
			v := int(128 + rng.NormFloat64()*float64(10+5*k))
			if v >= 0 && v < n {
				hs[k][v]++
			}
		}
	}
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			var s float64
			for i := 0; i < b.N; i++ {
				s += combinedPopulationCost(hs[i&7], hs[(i+3)&7])
			}
			sinkF = s
		})
	}
}

var sinkF float64

// BenchmarkCombinedShannonEntropy runs the kernel on residual-like
// histograms (a peak around 0, sparse tails).
func BenchmarkCombinedShannonEntropy(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	var xs [16][256]int32
	var y [256]int32
	for k := range xs {
		for i := 0; i < 1024; i++ {
			xs[k][uint8(rng.NormFloat64()*float64(2+k))]++
		}
	}
	for i := 0; i < 1<<18; i++ {
		y[uint8(rng.NormFloat64()*6)]++
	}
	var acc accEntropy
	acc.init(&y)
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			var s float64
			for i := 0; i < b.N; i++ {
				s += combinedShannonEntropy(&xs[i&15], &y, &acc)
			}
			sinkF = s
		})
	}
}

// junkAcc fills a with values derived from seed, which init must
// overwrite.
func junkAcc(seed int, a *accEntropy) {
	for i := range a.s {
		a.s[i] = float64(seed+i) * 0.37
	}
	a.total = float64(seed) * 1.5
	a.sum = uint32(seed) * 2654435761
}

// accDiff returns a description of the first difference between a and b
// (bit for bit), or "".
func accDiff(a, b *accEntropy) string {
	for i := range a.s {
		if math.Float64bits(a.s[i]) != math.Float64bits(b.s[i]) {
			return fmt.Sprintf("s[%d] %v != %v", i, a.s[i], b.s[i])
		}
	}
	if math.Float64bits(a.total) != math.Float64bits(b.total) {
		return fmt.Sprintf("total %v != %v", a.total, b.total)
	}
	if a.sum != b.sum {
		return fmt.Sprintf("sum %d != %d", a.sum, b.sum)
	}
	return ""
}

func TestAccEntropyInitLevels(t *testing.T) {
	n := 200000
	if testing.Short() {
		n = 20000
	}
	rng := rand.New(rand.NewSource(12))
	var h [256]int32
	var got, want accEntropy
	forEachLevelIters(t, n, []cpuinfo.Level{cpuinfo.SSE41}, func(t *testing.T, n int) {
		for i := 0; i < n; i++ {
			randHisto(rng, h[:])
			if i%8 == 0 {
				// (Otherwise got holds the previous result: a bin the
				// kernel did not write would differ from want.)
				junkAcc(i, &got)
				junkAcc(i+1, &want)
			}
			want.initGo(&h)
			got.init(&h)
			if d := accDiff(&got, &want); d != "" {
				t.Fatalf("h %v: %s", h, d)
			}
		}
	})
}

// TestAccEntropyInitSlog2 checks the kernel's v*log2(v) for every v below
// 2^16, every 7th below 2^24 (every 251st and 1757th above the SSE4.1
// level, which runs the same kernel; TestSlog2Exhaustive checks every v
// of the shared SLOG2 code) and random larger ones, isolated in bin v&255
// of an otherwise zero histogram: the bin's term and the total are then
// exactly slog2(v).
func TestAccEntropyInitSlog2(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	rng := rand.New(rand.NewSource(13))
	for _, l := range testLevels() {
		if l == cpuinfo.Go {
			continue
		}
		step := uint32(1)
		if l != cpuinfo.SSE41 {
			step = 251
		}
		t.Run(l.String(), func(t *testing.T) {
			defer setLevel(l)()
			var h [256]int32
			var a accEntropy
			check := func(v uint32) {
				i := int(v & 255)
				h[i] = int32(v)
				a.init(&h)
				h[i] = 0
				want := math.Float64bits(slog2(v))
				if math.Float64bits(a.s[i]) != want || math.Float64bits(a.total) != want || a.sum != v {
					t.Fatalf("v %d: s %v total %v sum %d, want %v", v, a.s[i], a.total, a.sum, slog2(v))
				}
			}
			for v := uint32(1); v < 1<<16; v += step {
				check(v)
			}
			for v := uint32(1 << 16); v < 1<<24; v += 7 * step {
				check(v)
			}
			for k := 0; k < 1<<18; k += int(step) {
				check(rng.Uint32() | 1<<24)
			}
			check(1<<32 - 1)
		})
	}
}

func FuzzAccEntropyInit(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0, 0, 16, 0, 0})
	f.Fuzz(func(t *testing.T, hb []byte) {
		var h [256]int32
		for i := 0; i < 256 && 4*i+4 <= len(hb); i++ {
			h[i] = int32(binary.LittleEndian.Uint32(hb[4*i:]))
		}
		var want accEntropy
		want.initGo(&h)
		atEveryLevel(func(l cpuinfo.Level) {
			var got accEntropy
			got.total, got.sum = 1, 1
			got.init(&h)
			if d := accDiff(&got, &want); d != "" {
				t.Fatalf("level %v: %s", l, d)
			}
		})
	})
}

func TestPopulationCostLevels(t *testing.T) {
	iters := 100000
	if testing.Short() {
		iters = 10000
	}
	rng := rand.New(rand.NewSource(14))
	sizes := []int{0, 1, 2, 15, 16, 17, 18, 31, 32, 33, 40, 47, 48, 49, 256, 280, 281, 282, 284, 288, 296, 312, 344, 408, 536, 792, 1304, 1305, 2000}
	forEachLevelIters(t, iters, []cpuinfo.Level{cpuinfo.SSE41}, func(t *testing.T, iters int) {
		for i := 0; i < iters; i++ {
			n := sizes[i%len(sizes)]
			if i%3 == 0 {
				n = rng.Intn(400)
			}
			var p []uint32
			if rng.Intn(2) == 0 {
				p = randStreaky(rng, n)
			} else {
				h := make([]int32, n)
				randHisto(rng, h)
				p = make([]uint32, n)
				for k := range p {
					p[k] = uint32(h[k])
				}
			}
			want := populationCostGo(p)
			got := populationCost(p)
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("p %v: got %v, want %v", p, got, want)
			}
		}
	})
}

func FuzzPopulationCost(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0}, uint16(40))
	f.Fuzz(func(t *testing.T, data []byte, n uint16) {
		size := int(n) % 1400
		p := make([]uint32, size)
		for i := range p {
			if len(data) >= 4 {
				p[i] = uint32(data[(2*i)%len(data)]) << (data[(2*i+1)%len(data)] & 31)
				if data[(i+3)%len(data)]&1 == 0 {
					p[i] = 0
				}
			}
		}
		want := populationCostGo(p)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := populationCost(p); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("level %v: got %v, want %v", l, got, want)
			}
		})
	})
}

// BenchmarkAccEntropyInit runs init on accumulated residual histograms:
// sparse (a graphic: a few bins around 0) and dense (a photo).
func BenchmarkAccEntropyInit(b *testing.B) {
	rng := rand.New(rand.NewSource(15))
	for _, kind := range []struct {
		name  string
		sigma float64
	}{{"sparse", 0.7}, {"dense", 12}} {
		var hs [8][256]int32
		for k := range hs {
			for i := 0; i < 1<<16; i++ {
				hs[k][uint8(int(rng.NormFloat64()*kind.sigma*float64(4+k)/4))]++
			}
		}
		for _, l := range testLevels() {
			b.Run(kind.name+"/"+l.String(), func(b *testing.B) {
				defer setLevel(l)()
				var a accEntropy
				for i := 0; i < b.N; i++ {
					a.init(&hs[i&7])
				}
				sinkF = a.total
			})
		}
	}
}

// BenchmarkPopulationCost runs populationCost on the literal histograms of
// BenchmarkCombinedPopulationCost.
func BenchmarkPopulationCost(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const n = 280 + 1<<6
	var hs [8][]uint32
	for k := range hs {
		hs[k] = make([]uint32, n)
		for i := 0; i < 4000; i++ {
			v := int(128 + rng.NormFloat64()*float64(10+5*k))
			if v >= 0 && v < n {
				hs[k][v]++
			}
		}
	}
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			var s float64
			for i := 0; i < b.N; i++ {
				s += populationCost(hs[i&7])
			}
			sinkF = s
		})
	}
}
