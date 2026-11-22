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

package vp8l

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// predictLeftRef is predictor mode 1 pixel by pixel.
func predictLeftRef(in, out []uint32, x0, x1 int) {
	l := out[x0-1]
	for x := x0; x < x1; x++ {
		l = addPixels(in[x], l)
		out[x] = l
	}
}

func TestPredictLeft(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 2000; iter++ {
		n := 2 + rng.Intn(80)
		x0 := 1 + rng.Intn(n-1)
		x1 := x0 + rng.Intn(n-x0+1)
		in := make([]uint32, n)
		for i := range in {
			in[i] = rng.Uint32()
		}
		want := make([]uint32, n)
		want[x0-1] = rng.Uint32()
		got := slices.Clone(want)
		predictLeftRef(in, want, x0, x1)
		predictLeft(in, got, x0, x1)
		if !slices.Equal(want, got) {
			t.Fatalf("n %d [%d, %d): output differs", n, x0, x1)
		}
		// In place (in aliases out), as inversePredictor does for a
		// predictor after another transform.
		inPlace := slices.Clone(in)
		inPlace[x0-1] = want[x0-1]
		predictLeft(inPlace, inPlace, x0, x1)
		if !slices.Equal(want[x0-1:x1], inPlace[x0-1:x1]) {
			t.Fatalf("n %d [%d, %d): in-place output differs", n, x0, x1)
		}
	}
}

func BenchmarkPredictLeft(b *testing.B) {
	const w = 3000
	in, out := make([]uint32, w), make([]uint32, w)
	for i := range in {
		in[i] = uint32(i) * 0x9e3779b9
	}
	b.Run("ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predictLeftRef(in, out, 1, w)
		}
	})
	b.Run("new", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predictLeft(in, out, 1, w)
		}
	})
}

// emitRef converts ARGB pixels to RGBA bytes one pixel at a time, adding
// green to red and blue first with addGreen.
func emitRef(src []uint32, dst []byte, addGreen bool) {
	for _, c := range src {
		if len(dst) < 4 {
			return
		}
		if addGreen {
			g := c >> 8 & 0xff
			c = c&0xff00ff00 | (c&0x00ff00ff+(g<<16|g))&0x00ff00ff
		}
		dst[0], dst[1], dst[2], dst[3] = byte(c>>16), byte(c>>8), byte(c), byte(c>>24)
		dst = dst[4:]
	}
}

func TestEmit(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 1000; iter++ {
		src := make([]uint32, rng.Intn(40))
		for i := range src {
			src[i] = rng.Uint32()
		}
		// dst may be shorter or longer than the row.
		n := 4 * len(src)
		if iter%3 == 0 {
			n = max(n+rng.Intn(9)-4, 0)
		}
		for _, ag := range []bool{false, true} {
			want := make([]byte, n)
			rng.Read(want)
			got := slices.Clone(want)
			emitRef(src, want, ag)
			if ag {
				emitAddGreen(src, got)
			} else {
				emitRGBA(src, got)
			}
			if !slices.Equal(want, got) {
				t.Fatalf("%d pixels, %d bytes, add green %v: output differs", len(src), n, ag)
			}
		}
	}
}

func BenchmarkEmit(b *testing.B) {
	const w = 3000
	src, dst := make([]uint32, w), make([]byte, 4*w)
	for i := range src {
		src[i] = uint32(i) * 0x9e3779b9
	}
	for _, c := range []struct {
		name string
		f    func([]uint32, []byte)
	}{
		{"rgba/ref", func(s []uint32, d []byte) { emitRef(s, d, false) }},
		{"rgba/new", emitRGBA},
		{"addgreen/ref", func(s []uint32, d []byte) { emitRef(s, d, true) }},
		{"addgreen/new", emitAddGreen},
	} {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(4 * w)
			for i := 0; i < b.N; i++ {
				c.f(src, dst)
			}
		})
	}
}

// predictorRef is libwebp's predictor of the given mode for one pixel, from
// its left, top, top-left and top-right neighbors, channel by channel.
func predictorRef(mode uint32, l, t, tl, tr uint32) uint32 {
	ch := func(c uint32, k uint) int32 { return int32(c >> k & 0xff) }
	avg := func(a, b uint32) uint32 {
		var r uint32
		for k := uint(0); k < 32; k += 8 {
			r |= uint32((ch(a, k)+ch(b, k))>>1) << k
		}
		return r
	}
	clip := func(v int32) uint32 { return uint32(min(max(v, 0), 255)) }
	abs := func(v int32) int32 { return max(v, -v) }
	var r uint32
	switch mode {
	case 1:
		return l
	case 2:
		return t
	case 3:
		return tr
	case 4:
		return tl
	case 5:
		return avg(avg(l, tr), t)
	case 6:
		return avg(l, tl)
	case 7:
		return avg(l, t)
	case 8:
		return avg(tl, t)
	case 9:
		return avg(t, tr)
	case 10:
		return avg(avg(l, tl), avg(t, tr))
	case 11:
		s := int32(0)
		for k := uint(0); k < 32; k += 8 {
			s += abs(ch(l, k)-ch(tl, k)) - abs(ch(t, k)-ch(tl, k))
		}
		if s <= 0 {
			return t
		}
		return l
	case 12:
		for k := uint(0); k < 32; k += 8 {
			r |= clip(ch(l, k)+ch(t, k)-ch(tl, k)) << k
		}
		return r
	case 13:
		a := avg(l, t)
		for k := uint(0); k < 32; k += 8 {
			r |= clip(ch(a, k)+(ch(a, k)-ch(tl, k))/2) << k
		}
		return r
	}
	return 0xff000000
}

// TestPredictRun checks every predictor mode against predictorRef, for
// runs at any position of rows of any width, in place or not.
func TestPredictRun(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for iter := 0; iter < 3000; iter++ {
		mode := uint32(rng.Intn(16))
		n := 2 + rng.Intn(70)
		x0 := 1 + rng.Intn(n-1)
		x1 := x0 + rng.Intn(n-x0+1)
		in := make([]uint32, n)
		up := make([]uint32, n+1)
		for i := range in {
			in[i] = rng.Uint32()
		}
		for i := range up {
			up[i] = rng.Uint32()
			if rng.Intn(3) == 0 {
				up[i] = in[min(i, n-1)] // close values exercise the clamps
			}
		}
		want := make([]uint32, n)
		want[x0-1] = rng.Uint32()
		got := slices.Clone(want)
		for x := x0; x < x1; x++ {
			want[x] = addPixels(in[x], predictorRef(mode, want[x-1], up[x], up[x-1], up[x+1]))
		}
		predictRun(mode, in, got, up, x0, x1)
		if !slices.Equal(want, got) {
			t.Fatalf("mode %d, n %d [%d, %d): output differs", mode, n, x0, x1)
		}
		inPlace := slices.Clone(in)
		inPlace[x0-1] = want[x0-1]
		predictRun(mode, inPlace, inPlace, up, x0, x1)
		if !slices.Equal(want[x0-1:x1], inPlace[x0-1:x1]) {
			t.Fatalf("mode %d, n %d [%d, %d): in-place output differs", mode, n, x0, x1)
		}
	}
}

func BenchmarkPredictRun(b *testing.B) {
	const w = 3000
	rng := rand.New(rand.NewSource(1))
	in, out, up := make([]uint32, w), make([]uint32, w), make([]uint32, w+1)
	for i := range in {
		in[i] = rng.Uint32() & 0x0f0f0f0f // small residuals
		up[i] = rng.Uint32()
	}
	for mode := uint32(0); mode < 14; mode++ {
		b.Run(fmt.Sprintf("mode%d", mode), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				// Tiles of 32 pixels, as in a photo.
				for x := 1; x < w; x += 32 {
					predictRun(mode, in, out, up, x, min(x+32, w))
				}
			}
		})
	}
}
