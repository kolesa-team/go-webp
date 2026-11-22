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
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of the hash chain search (hcSearcher.run) against
// runGo at every kernel level.

// randHashImage returns n pixels of one of several kinds: noise, few
// colors (many candidates), runs (long matches, maximal lengths), a
// repeated row pattern (matches at distance xsize) or a smooth photo-like
// gradient.
func randHashImage(rng *rand.Rand, n, xsize int) []uint32 {
	p := make([]uint32, n)
	switch rng.Intn(5) {
	case 0:
		for i := range p {
			p[i] = rng.Uint32()
		}
	case 1:
		k := 1 + rng.Intn(6)
		for i := range p {
			p[i] = uint32(rng.Intn(k)) * 0x01020304
		}
	case 2:
		for i := 0; i < n; {
			l := 1 + rng.Intn(6000)
			if rng.Intn(4) != 0 {
				l = 1 + rng.Intn(40)
			}
			c := uint32(rng.Intn(3))
			for ; l > 0 && i < n; l-- {
				p[i] = c
				i++
			}
		}
	case 3:
		row := make([]uint32, xsize)
		for i := range row {
			row[i] = rng.Uint32() & 0x03030303
		}
		for i := range p {
			p[i] = row[i%xsize]
			if rng.Intn(50) == 0 {
				p[i] = rng.Uint32()
			}
		}
	default:
		for i := range p {
			x, y := i%xsize, i/xsize
			p[i] = 0xff000000 | uint32(x/3+rng.Intn(2))<<16 | uint32(y/3)<<8 | uint32((x+y)/5)
		}
	}
	return p
}

// hcBoth runs the sequential search (as fill does) with runGo and with run
// at every level, and fails if the matches differ.
func hcBoth(t *testing.T, argb []uint32, xsize, quality int, lowEffort bool, windowCap int) {
	t.Helper()
	size := len(argb)
	if size <= 2 {
		return
	}
	window := windowSizeForHashChain(quality, xsize)
	if windowCap > 0 && window > windowCap {
		window = windowCap
	}
	search := func(useGo bool) ([]uint32, int, int) {
		ol := make([]uint32, size)
		s := hcSearcher{argb: argb, ol: ol, chain: ol, xsize: xsize, size: size,
			iterMax: maxItersForQuality(quality), window: window, lowEffort: lowEffort}
		buildChain(argb, ol)
		ol[0], ol[size-1] = 0, 0
		var last, next int
		if useGo {
			last, next = s.runGo(size-2, 0, 1, nil)
		} else {
			last, next = s.run(size-2, 0, 1, nil)
		}
		return ol, last, next
	}
	want, wl, wn := search(true)
	atEveryLevel(func(l cpuinfo.Level) {
		got, gl, gn := search(false)
		if gl != wl || gn != wn {
			t.Fatalf("level %v size %d xsize %d q %d low %v: last, next %d, %d, want %d, %d", l, size, xsize, quality, lowEffort, gl, gn, wl, wn)
		}
		if !slices.Equal(got, want) {
			i := 0
			for got[i] == want[i] {
				i++
			}
			t.Fatalf("level %v size %d xsize %d q %d low %v: ol[%d] = %#x, want %#x", l, size, xsize, quality, lowEffort, i, got[i], want[i])
		}
	})
}

func TestHashChainRunLevels(t *testing.T) {
	iters := 600
	if testing.Short() {
		iters = 60
	}
	rng := rand.New(rand.NewSource(10))
	for i := 0; i < iters; i++ {
		xsize := 1 + rng.Intn(300)
		n := 3 + rng.Intn(20000)
		if i%10 == 0 {
			n = 3 + rng.Intn(20)
		}
		argb := randHashImage(rng, n, xsize)
		q := []int{0, 20, 25, 30, 50, 75, 90, 100}[rng.Intn(8)]
		hcBoth(t, argb, xsize, q, rng.Intn(3) == 0, []int{0, 0, 7, 100}[rng.Intn(4)])
	}
}

// TestHashChainRunStripes checks run as fillStripes uses it: a separate
// chain, searches bounded below by lo, marks, and the fix-up searches.
func TestHashChainRunStripes(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 200; i++ {
		xsize := 1 + rng.Intn(200)
		size := 300 + rng.Intn(10000)
		argb := randHashImage(rng, size, xsize)
		q := []int{0, 50, 75, 100}[rng.Intn(4)]
		chain := make([]uint32, size)
		buildChain(argb, chain)
		hi := 1 + rng.Intn(size-2)
		lo := 1 + rng.Intn(hi)
		stop := lo + rng.Intn(hi-lo+1)
		search := func(useGo bool) ([]uint32, []uint64, int, int) {
			ol := make([]uint32, size)
			for k := range ol {
				ol[k] = 0xdeadbeef
			}
			marks := make([]uint64, (size+63)/64)
			s := hcSearcher{argb: argb, ol: ol, chain: chain, xsize: xsize, size: size,
				iterMax: maxItersForQuality(q), window: windowSizeForHashChain(q, xsize), lowEffort: rng.Intn(2) == 0}
			if useGo {
				l, n := s.runGo(hi, lo, stop, marks)
				return ol, marks, l, n
			}
			l, n := s.run(hi, lo, stop, marks)
			return ol, marks, l, n
		}
		seed := rng.Int63()
		rng.Seed(seed)
		want, wm, wl, wn := search(true)
		atEveryLevel(func(l cpuinfo.Level) {
			rng.Seed(seed)
			got, gm, gl, gn := search(false)
			if gl != wl || gn != wn || !slices.Equal(got, want) || !slices.Equal(gm, wm) {
				t.Fatalf("level %v size %d hi %d lo %d stop %d: results differ (last %d/%d next %d/%d)", l, size, hi, lo, stop, gl, wl, gn, wn)
			}
		})
	}
}

// TestHashChainRunPhoto checks the photo at the qualities of the lossless
// levels.
func TestHashChainRunPhoto(t *testing.T) {
	img := downscale(loadTestJPEG(t, "source.jpg"), 1)
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb := make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	for _, lv := range losslessLevels {
		hcBoth(t, argb, w, lv[1], lv[0] == 0, 0)
	}
}

func FuzzHashChainRun(f *testing.F) {
	f.Add([]byte{1, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2, 1, 2}, uint8(3), uint8(75), false)
	f.Fuzz(func(t *testing.T, data []byte, xs, q uint8, low bool) {
		n := len(data) / 2
		if n < 3 {
			return
		}
		argb := make([]uint32, n)
		for i := range argb {
			// Few distinct values: many candidates and long matches.
			argb[i] = uint32(binary.LittleEndian.Uint16(data[2*i:])) & 0x0303
		}
		hcBoth(t, argb, 1+int(xs)%64, int(q)%101, low, 0)
	})
}

// rleBoth compares backwardRefsRle with backwardRefsRleGo at every level,
// from refs of the given capacity (the growth must match too).
func rleBoth(t *testing.T, argb []uint32, xsize, capacity int) {
	t.Helper()
	n := len(argb)
	want := backwardRefsRleGo(xsize, n, argb, make([]pixOrCopy, 0, capacity))
	atEveryLevel(func(l cpuinfo.Level) {
		got := backwardRefsRle(xsize, n, argb, make([]pixOrCopy, 0, capacity))
		if !slices.Equal(got, want) || cap(got) != cap(want) {
			t.Fatalf("level %v n %d xsize %d cap %d: %d refs (cap %d), want %d (cap %d)", l, n, xsize, capacity, len(got), cap(got), len(want), cap(want))
		}
	})
}

func TestBackwardRefsRleLevels(t *testing.T) {
	iters := 600
	if testing.Short() {
		iters = 60
	}
	rng := rand.New(rand.NewSource(14))
	for i := 0; i < iters; i++ {
		xsize := 1 + rng.Intn(300)
		n := 1 + rng.Intn(20000)
		if i%10 == 0 {
			n = 1 + rng.Intn(20)
		}
		rleBoth(t, randHashImage(rng, n, xsize), xsize, []int{0, 1, 7, n}[rng.Intn(4)])
	}
}

func FuzzBackwardRefsRle(f *testing.F) {
	f.Add([]byte{1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1}, uint8(5), uint8(3))
	f.Fuzz(func(t *testing.T, data []byte, xs, c uint8) {
		if len(data) == 0 {
			return
		}
		argb := make([]uint32, len(data))
		for i, b := range data {
			argb[i] = uint32(b & 3)
		}
		rleBoth(t, argb, 1+int(xs)%64, int(c))
	})
}

func BenchmarkBackwardRefsRle(b *testing.B) {
	img := loadTestJPEG(b, "source.jpg")
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb := make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	refs := make([]pixOrCopy, 0, len(argb))
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				refs = backwardRefsRle(w, len(argb), argb, refs)
			}
		})
	}
}

// BenchmarkHashChainRun runs the sequential search on the photo at the
// qualities of levels 0 and 6, with and without the prefetch.
func BenchmarkHashChainRun(b *testing.B) {
	img := loadTestJPEG(b, "source.jpg")
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb := make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	size := len(argb)
	ol := make([]uint32, size)
	chain := make([]uint32, size)
	buildChain(argb, chain)
	for _, lv := range []struct {
		name string
		q    int
		low  bool
	}{{"z0", 0, true}, {"z6", 75, false}} {
		s := hcSearcher{argb: argb, ol: ol, chain: chain, xsize: w, size: size,
			iterMax: maxItersForQuality(lv.q), window: windowSizeForHashChain(lv.q, w), lowEffort: lv.low}
		for _, l := range testLevels() {
			for _, pf := range []int{0, 4, 8, 16} {
				if l == cpuinfo.Go && pf > 0 {
					continue
				}
				b.Run(fmt.Sprintf("%s/%v/pf%d", lv.name, l, pf), func(b *testing.B) {
					defer setLevel(l)()
					defer setPrefetch(pf)()
					for i := 0; i < b.N; i++ {
						s.run(size-2, 0, 1, nil)
					}
				})
			}
		}
	}
}
