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
	"slices"
	"testing"
)

// refTile is the straightforward cross-color search (every pixel of the
// tile, every candidate evaluated), kept as the reference of the
// deduplicated one.
type refTile struct{ r, g, b []int8 }

func (t *refTile) load(argb []uint32, stride, x0, y0, w, h int) {
	t.r, t.g, t.b = t.r[:0], t.g[:0], t.b[:0]
	for y := 0; y < h; y++ {
		for _, p := range argb[(y0+y)*stride+x0 : (y0+y)*stride+x0+w] {
			t.r = append(t.r, int8(p>>16))
			t.g = append(t.g, int8(p>>8))
			t.b = append(t.b, int8(p))
		}
	}
}

func (t *refTile) costRed(g2r int, prevX, prevY multipliers, acc *[256]int32, ae *accEntropy) float64 {
	var histo [256]int32
	for i, rv := range t.r {
		histo[uint8(int32(rv)-colorTransformDelta(int8(g2r), t.g[i]))]++
	}
	d := predictionCostCrossColor(acc, &histo, ae)
	if uint8(g2r) == prevX.greenToRed {
		d -= 3
	}
	if uint8(g2r) == prevY.greenToRed {
		d -= 3
	}
	if g2r == 0 {
		d -= 3
	}
	return d
}

func (t *refTile) costBlue(g2b, r2b int, prevX, prevY multipliers, acc *[256]int32, ae *accEntropy) float64 {
	var histo [256]int32
	for i, bv := range t.b {
		nb := int32(bv) - colorTransformDelta(int8(g2b), t.g[i]) - colorTransformDelta(int8(r2b), t.r[i])
		histo[uint8(nb)]++
	}
	d := predictionCostCrossColor(acc, &histo, ae)
	for _, c := range []bool{uint8(g2b) == prevX.greenToBlue, uint8(g2b) == prevY.greenToBlue,
		uint8(r2b) == prevX.redToBlue, uint8(r2b) == prevY.redToBlue, g2b == 0, r2b == 0} {
		if c {
			d -= 3
		}
	}
	return d
}

func (t *refTile) bestTransform(prevX, prevY multipliers, quality int, accRed, accBlue *[256]int32) multipliers {
	var best multipliers
	var aeR, aeB accEntropy
	aeR.init(accRed)
	aeB.init(accBlue)
	maxIters := 4 + (7*quality)>>8
	g2rBest := 0
	bestDiff := t.costRed(0, prevX, prevY, accRed, &aeR)
	for iter := 0; iter < maxIters; iter++ {
		delta := 32 >> uint(iter)
		for off := -delta; off <= delta; off += 2 * delta {
			cur := off + g2rBest
			if d := t.costRed(cur, prevX, prevY, accRed, &aeR); d < bestDiff {
				bestDiff, g2rBest = d, cur
			}
		}
	}
	best.greenToRed = uint8(g2rBest)
	offsets := [8][2]int{{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	deltaLut := [7]int{16, 16, 8, 4, 2, 2, 2}
	iters := 4
	if quality < 25 {
		iters = 1
	} else if quality > 50 {
		iters = 7
	}
	g2b, r2b := 0, 0
	bestDiff = t.costBlue(0, 0, prevX, prevY, accBlue, &aeB)
	for iter := 0; iter < iters; iter++ {
		delta := deltaLut[iter]
		for axis := 0; axis < 8; axis++ {
			cg := offsets[axis][0]*delta + g2b
			cr := offsets[axis][1]*delta + r2b
			if d := t.costBlue(cg, cr, prevX, prevY, accBlue, &aeB); d < bestDiff {
				bestDiff, g2b, r2b = d, cg, cr
			}
		}
		if delta == 2 && g2b == 0 && r2b == 0 {
			break
		}
	}
	best.greenToBlue, best.redToBlue = uint8(g2b), uint8(r2b)
	return best
}

func refColorSpaceTransform(width, height, bits, quality int, argb []uint32, image []uint32) int {
	tile := 1 << uint(bits)
	tilesX := subSampleSize(width, bits)
	tilesY := subSampleSize(height, bits)
	var accRed, accBlue [256]int32
	var prevX, prevY multipliers
	var tp refTile
	for ty := 0; ty < tilesY; ty++ {
		for tx := 0; tx < tilesX; tx++ {
			x0, y0 := tx*tile, ty*tile
			xMax, yMax := min(x0+tile, width), min(y0+tile, height)
			off := ty*tilesX + tx
			if ty != 0 {
				prevY = multipliersFromCode(image[off-tilesX])
			}
			tp.load(argb, width, x0, y0, xMax-x0, yMax-y0)
			prevX = tp.bestTransform(prevX, prevY, quality, &accRed, &accBlue)
			image[off] = prevX.code()
			for y := y0; y < yMax; y++ {
				row := argb[y*width+x0 : y*width+xMax]
				for i, p := range row {
					row[i] = transformColor(prevX, p)
				}
			}
			for y := y0; y < yMax; y++ {
				for ix := y*width + x0; ix < y*width+xMax; ix++ {
					pix := argb[ix]
					if ix >= 2 && pix == argb[ix-2] && pix == argb[ix-1] {
						continue
					}
					if ix >= width+2 && argb[ix-2] == argb[ix-width-2] &&
						argb[ix-1] == argb[ix-width-1] && pix == argb[ix-width] {
						continue
					}
					accRed[(pix>>16)&0xff]++
					accBlue[pix&0xff]++
				}
			}
		}
	}
	return optimizeSampling(image, width, height, bits, maxTransformBits)
}

// TestCrossColorDedup checks that the deduplicated cross-color search gives
// the same multipliers and pixels as the reference on images with few and
// many distinct colors, odd sizes, all qualities and tiles larger than the
// deduplication tables.
func TestCrossColorDedup(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	type img struct {
		name string
		w, h int
		f    func(x, y int) uint32
	}
	smooth := func(x, y int) uint32 {
		g := uint32(x*3+y) & 0xff
		r := (g + uint32(rng.Intn(9))) & 0xff
		b := (g*2 + uint32(rng.Intn(5)) + uint32(y)) & 0xff
		return 0xff000000 | r<<16 | g<<8 | b
	}
	imgs := []img{
		{"smooth", 100, 77, smooth},
		{"noise", 67, 45, func(x, y int) uint32 { return rng.Uint32() }},
		{"few", 130, 70, func(x, y int) uint32 { return 0xff000000 | uint32((x/9+y/5)%4)*0x203040 }},
		{"flat", 40, 40, func(x, y int) uint32 { return 0xff102030 }},
		{"1xN", 1, 300, smooth},
		{"Nx1", 300, 1, smooth},
		{"tiny", 3, 2, smooth},
	}
	sc := new(scratch)
	for _, im := range imgs {
		src := make([]uint32, im.w*im.h)
		for y := 0; y < im.h; y++ {
			for x := 0; x < im.w; x++ {
				src[y*im.w+x] = im.f(x, y)
			}
		}
		for _, bits := range []int{2, 3, 5, 6, 7} {
			for _, q := range []int{0, 24, 25, 50, 51, 75, 100} {
				n := subSampleSize(im.w, bits) * subSampleSize(im.h, bits)
				want, wantImg := slices.Clone(src), make([]uint32, n)
				wantBits := refColorSpaceTransform(im.w, im.h, bits, q, want, wantImg)
				got, gotImg := slices.Clone(src), make([]uint32, n)
				gotBits := colorSpaceTransform(im.w, im.h, bits, q, got, gotImg, sc)
				if gotBits != wantBits || !slices.Equal(got, want) || !slices.Equal(gotImg, wantImg) {
					t.Fatalf("%s bits %d q %d: result differs from the reference", im.name, bits, q)
				}
			}
		}
	}
}

// TestCrossColorTableWrap checks the generation counter of the tables: more
// than 255 tiles force a table reset, after which stale entries must not be
// seen.
func TestCrossColorTableWrap(t *testing.T) {
	var tp tilePixels
	argb := make([]uint32, 16)
	for k := 0; k < 600; k++ {
		for i := range argb {
			argb[i] = uint32(k*7+i%3) * 0x010101
		}
		tp.load(argb, 4, 0, 0, 4, 4)
		total := int32(0)
		for _, c := range tp.tc {
			total += c
		}
		if len(tp.tc) != 3 || total != 16 {
			t.Fatalf("tile %d: %d triples covering %d pixels", k, len(tp.tc), total)
		}
	}
}
