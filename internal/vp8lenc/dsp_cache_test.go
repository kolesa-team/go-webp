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
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// randRefs returns a random parse of argb: literals, short and long copies
// (lengths 1 to 4095), with cache indices when cache is set (bestCacheBits
// treats them as copies of their length).
func randRefs(rng *rand.Rand, n int, cache bool) []pixOrCopy {
	var refs []pixOrCopy
	for pos := 0; pos < n; {
		switch k := rng.Intn(10); {
		case k < 6:
			refs = append(refs, mkLiteral(0))
			pos++
		case k == 6 && cache:
			refs = append(refs, mkCache(uint32(rng.Intn(1024))))
			pos++
		default:
			l := 1 + rng.Intn(min(n-pos, []int{8, 64, 4095}[rng.Intn(3)]))
			refs = append(refs, mkCopy(uint32(1+rng.Intn(100)), l))
			pos += l
		}
	}
	return refs
}

func TestBestCacheBitsLevels(t *testing.T) {
	iters := 400
	if testing.Short() {
		iters = 40
	}
	rng := rand.New(rand.NewSource(12))
	sc := getScratch(0)
	defer putScratch(sc)
	for i := 0; i < iters; i++ {
		n := 1 + rng.Intn(30000)
		argb := randHashImage(rng, n, 1+rng.Intn(100))
		if rng.Intn(2) == 0 {
			// Few colors: many cache hits.
			for k := range argb {
				argb[k] &= 0x01030507
			}
		}
		refs := randRefs(rng, n, rng.Intn(4) == 0)
		q := 26 + rng.Intn(75)
		maxBits := 1 + rng.Intn(maxColorCacheBits)
		want := bestCacheBitsGo(argb, q, refs, maxBits, sc)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := bestCacheBits(argb, q, refs, maxBits, sc); got != want {
				t.Fatalf("level %v n %d maxBits %d: got %d, want %d", l, n, maxBits, got, want)
			}
		})
	}
}

// TestBestCacheBitsBadRefs checks that a parse that does not fit the image
// panics as in the Go code (the kernel refuses it).
func TestBestCacheBitsBadRefs(t *testing.T) {
	sc := getScratch(0)
	defer putScratch(sc)
	argb := make([]uint32, 100)
	for _, refs := range [][]pixOrCopy{
		{mkCopy(1, 101)},
		{mkLiteral(0), mkCopy(1, 99), mkLiteral(0)},
		{mkCopy(1, 0)},
	} {
		atEveryLevel(func(l cpuinfo.Level) {
			defer func() {
				if recover() == nil {
					t.Errorf("level %v refs %v: no panic", l, refs)
				}
			}()
			bestCacheBits(argb, 75, refs, 4, sc)
		})
	}
}

func FuzzBestCacheBits(f *testing.F) {
	f.Add([]byte{1, 2, 3, 1, 2, 3, 9, 9, 9, 9}, int64(1), uint8(4))
	f.Fuzz(func(t *testing.T, data []byte, seed int64, bits uint8) {
		if len(data) == 0 {
			return
		}
		rng := rand.New(rand.NewSource(seed))
		argb := make([]uint32, len(data))
		for i, b := range data {
			argb[i] = uint32(b&7) * 0x01010101
		}
		refs := randRefs(rng, len(argb), true)
		maxBits := 1 + int(bits)%maxColorCacheBits
		sc := getScratch(0)
		defer putScratch(sc)
		want := bestCacheBitsGo(argb, 75, refs, maxBits, sc)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := bestCacheBits(argb, 75, refs, maxBits, sc); got != want {
				t.Fatalf("level %v: got %d, want %d", l, got, want)
			}
		})
	})
}

// BenchmarkBestCacheBits runs the cache size search on the RLE parse of
// the photo.
func BenchmarkBestCacheBits(b *testing.B) {
	img := downscale(loadTestJPEG(b, "source.jpg"), 1)
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb := make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	refs := backwardRefsRle(w, len(argb), argb, nil)
	sc := getScratch(0)
	defer putScratch(sc)
	for _, l := range testLevels() {
		b.Run(l.String(), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				bestCacheBits(argb, 75, refs, maxColorCacheBits, sc)
			}
		})
	}
}
