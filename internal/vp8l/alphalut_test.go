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
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// inverseColorIndexAlphaRef is the straightforward per-pixel palette
// expansion, the reference for inverseColorIndexAlpha.
func inverseColorIndexAlphaRef(t *transform, height int, in, out []byte) {
	width := t.xsize
	var pal8 [256]byte
	for i, c := range t.data {
		pal8[i] = byte(c >> 8)
	}
	if t.bits == 0 {
		for i, p := range in[:width*height] {
			out[i] = pal8[p]
		}
		return
	}
	bpp := uint(8 >> t.bits)
	ppb := 1 << t.bits
	mask := byte(1)<<bpp - 1
	pw := subSampleSize(width, t.bits)
	i, o := 0, 0
	for y := 0; y < height; y++ {
		x := 0
		for k := 0; k < pw; k++ {
			packed := in[i+k]
			n := min(ppb, width-x)
			for j := 0; j < n; j++ {
				out[o+x] = pal8[packed&mask]
				packed >>= bpp
				x++
			}
		}
		i += pw
		o += width
	}
}

// randomPaletteTransform returns a color-indexing transform with random
// palette colors for an image of the given width.
func randomPaletteTransform(rng *rand.Rand, width, bits int) *transform {
	t := &transform{kind: colorIndexingTransform, xsize: width, bits: bits, data: make([]uint32, 1<<(8>>bits))}
	for i := range t.data {
		t.data[i] = rng.Uint32()
	}
	return t
}

func TestInverseColorIndexAlpha(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 3000; iter++ {
		bits := rng.Intn(4)
		width, height := 1+rng.Intn(70), 1+rng.Intn(4)
		if iter%100 == 0 {
			width = 1 + rng.Intn(3000)
		}
		tr := randomPaletteTransform(rng, width, bits)
		in := make([]byte, subSampleSize(width, bits)*height)
		rng.Read(in)
		want := make([]byte, width*height+9)
		got := make([]byte, width*height+9)
		rng.Read(want)
		copy(got, want) // bytes past the plane must stay untouched
		inverseColorIndexAlphaRef(tr, height, in, want)
		var tab alphaTables
		tab.init(tr)
		inverseColorIndexAlpha(tr, &tab, height, in, got[:width*height])
		if !bytes.Equal(want, got) {
			t.Fatalf("bits %d, %dx%d: output differs", bits, width, height)
		}
	}
}

func BenchmarkInverseColorIndexAlpha(b *testing.B) {
	const width, height = 3000, 16
	for bits := 0; bits < 4; bits++ {
		rng := rand.New(rand.NewSource(1))
		tr := randomPaletteTransform(rng, width, bits)
		in := make([]byte, subSampleSize(width, bits)*height)
		rng.Read(in)
		out := make([]byte, width*height)
		b.Run(fmt.Sprintf("bits%d/ref", bits), func(b *testing.B) {
			b.SetBytes(width * height)
			for i := 0; i < b.N; i++ {
				inverseColorIndexAlphaRef(tr, height, in, out)
			}
		})
		b.Run(fmt.Sprintf("bits%d/new", bits), func(b *testing.B) {
			b.SetBytes(width * height)
			var tab alphaTables
			tab.init(tr)
			for i := 0; i < b.N; i++ {
				inverseColorIndexAlpha(tr, &tab, height, in, out)
			}
		})
	}
}
