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

// Cross-color (color space) transform.
// Algorithm adapted from libwebp (BSD-3-Clause).

type multipliers struct {
	greenToRed, greenToBlue, redToBlue uint8
}

func (m multipliers) code() uint32 {
	return argbBlack | uint32(m.redToBlue)<<16 | uint32(m.greenToBlue)<<8 | uint32(m.greenToRed)
}

func multipliersFromCode(c uint32) multipliers {
	return multipliers{greenToRed: uint8(c), greenToBlue: uint8(c >> 8), redToBlue: uint8(c >> 16)}
}

func colorTransformDelta(pred, color int8) int32 {
	return (int32(pred) * int32(color)) >> 5
}

// The cost of a multiplier candidate only depends on the histogram of the
// transformed red (or blue) channel of the tile, which only depends on the
// distinct (green, red) pairs (or (green, red, blue) triples) of the tile and
// their counts. Tiles of photos have 25-40% distinct triples and graphics far
// fewer, so the candidates are evaluated on the distinct values, weighted by
// their counts (exact: the histograms are integer counts).
const (
	ccTabBits = 13
	ccTabSize = 1 << ccTabBits
	// ccMaxTilePixels bounds the tiles that are deduplicated (keeping the
	// tables at most half full); larger tiles (not produced by the encoder:
	// its transform bits are at most 6) list every pixel.
	ccMaxTilePixels = ccTabSize / 2
)

// ccTable is an open-addressing set of keys (at most 24 bits) mapping to the
// index of their entry. A slot is in use for the current generation gen
// when slot&0xff == gen, so that the table is cleared only every 255 tiles.
type ccTable struct {
	slot [ccTabSize]uint32 // key<<8 | generation
	idx  [ccTabSize]uint16
}

// ccTables are the deduplication tables of a tilePixels (allocated on first
// use: about 96 KB).
type ccTables struct {
	pairs, triples ccTable
	gen            uint32 // 1..255
}

// tilePixels holds the distinct (green, red) pairs and (green, red, blue)
// triples of one tile, as signed bytes, with their pixel counts.
type tilePixels struct {
	pg, pr     []int8 // distinct (green, red) pairs
	pc         []int32
	tg, tr, tb []int8 // distinct (green, red, blue) triples
	tc         []int32
	tabs       *ccTables
}

func (t *tilePixels) reset(n int) {
	if cap(t.pg) < n {
		t.pg, t.pr, t.pc = make([]int8, 0, n), make([]int8, 0, n), make([]int32, 0, n)
		t.tg, t.tr, t.tb, t.tc = make([]int8, 0, n), make([]int8, 0, n), make([]int8, 0, n), make([]int32, 0, n)
	}
	t.pg, t.pr, t.pc = t.pg[:0], t.pr[:0], t.pc[:0]
	t.tg, t.tr, t.tb, t.tc = t.tg[:0], t.tr[:0], t.tb[:0], t.tc[:0]
}

// load collects the distinct pairs and triples of the w x h tile at (x0, y0)
// of argb (whose rows are stride pixels apart).
func (t *tilePixels) load(argb []uint32, stride, x0, y0, w, h int) {
	n := w * h
	t.reset(min(n, ccMaxTilePixels))
	if n > ccMaxTilePixels {
		// Not deduplicated: one entry per pixel.
		for y := 0; y < h; y++ {
			for _, p := range argb[(y0+y)*stride+x0 : (y0+y)*stride+x0+w] {
				g, r := int8(p>>8), int8(p>>16)
				t.pg, t.pr, t.pc = append(t.pg, g), append(t.pr, r), append(t.pc, 1)
				t.tg, t.tr, t.tb, t.tc = append(t.tg, g), append(t.tr, r), append(t.tb, int8(p)), append(t.tc, 1)
			}
		}
		return
	}
	if t.tabs == nil {
		t.tabs = new(ccTables)
	}
	tabs := t.tabs
	tabs.gen++
	if tabs.gen > 0xff {
		clear(tabs.pairs.slot[:])
		clear(tabs.triples.slot[:])
		tabs.gen = 1
	}
	gen := tabs.gen
	pairs, triples := &tabs.pairs, &tabs.triples
	// The entries of the previous pixel (neighbors are often equal).
	last := ^uint32(0)
	lastPair, lastTriple := 0, 0
	for y := 0; y < h; y++ {
		for _, p := range argb[(y0+y)*stride+x0 : (y0+y)*stride+x0+w] {
			p &= 0xffffff // r<<16 | g<<8 | b
			if p == last {
				t.pc[lastPair]++
				t.tc[lastTriple]++
				continue
			}
			if p>>8 != last>>8 {
				key := p >> 8 // r<<8 | g
				h := (key * 0x9e3779b1) >> (32 - ccTabBits)
				for {
					s := pairs.slot[h]
					if s&0xff != gen {
						pairs.slot[h] = key<<8 | gen
						lastPair = len(t.pc)
						pairs.idx[h] = uint16(lastPair)
						t.pg, t.pr, t.pc = append(t.pg, int8(p>>8)), append(t.pr, int8(p>>16)), append(t.pc, 1)
						break
					}
					if s>>8 == key {
						lastPair = int(pairs.idx[h])
						t.pc[lastPair]++
						break
					}
					h = (h + 1) & (ccTabSize - 1)
				}
			} else {
				t.pc[lastPair]++
			}
			h := (p * 0x9e3779b1) >> (32 - ccTabBits)
			for {
				s := triples.slot[h]
				if s&0xff != gen {
					triples.slot[h] = p<<8 | gen
					lastTriple = len(t.tc)
					triples.idx[h] = uint16(lastTriple)
					t.tg, t.tr, t.tb, t.tc = append(t.tg, int8(p>>8)), append(t.tr, int8(p>>16)), append(t.tb, int8(p)), append(t.tc, 1)
					break
				}
				if s>>8 == p {
					lastTriple = int(triples.idx[h])
					t.tc[lastTriple]++
					break
				}
				h = (h + 1) & (ccTabSize - 1)
			}
			last = p
		}
	}
}

func predictionCostCrossColor(acc, counts *[256]int32, ae *accEntropy) float64 {
	return combinedShannonEntropy(counts, acc, ae) + predictionCostSpatial(counts, 3, 2.4)
}

// ccHistoRedGo adds to histo the counts c of the red values r transformed
// with the green-to-red multiplier m (g holds their greens). It is the Go
// reference of ccHistoRed (dsp_amd64.go, dsp_generic.go).
func ccHistoRedGo(histo *[256]int32, r, g []int8, c []int32, m int8) {
	g = g[:len(r)]
	c = c[:len(r)]
	for i, rv := range r {
		nr := int32(rv) - colorTransformDelta(m, g[i])
		histo[uint8(nr)] += c[i]
	}
}

// ccHistoBlueGo adds to histo the counts c of the blue values b transformed
// with the green-to-blue and red-to-blue multipliers mg and mr (g and r
// hold their greens and reds). It is the Go reference of ccHistoBlue
// (dsp_amd64.go, dsp_generic.go).
func ccHistoBlueGo(histo *[256]int32, b, g, r []int8, c []int32, mg, mr int8) {
	g = g[:len(b)]
	r = r[:len(b)]
	c = c[:len(b)]
	for i, bv := range b {
		nb := int32(bv) - colorTransformDelta(mg, g[i]) - colorTransformDelta(mr, r[i])
		histo[uint8(nb)] += c[i]
	}
}

func (t *tilePixels) costRed(g2r int, prevX, prevY multipliers, acc *[256]int32, ae *accEntropy, histo *[256]int32) float64 {
	*histo = [256]int32{}
	ccHistoRed(histo, t.pr, t.pg, t.pc, int8(g2r))
	d := predictionCostCrossColor(acc, histo, ae)
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

func (t *tilePixels) costBlue(g2b, r2b int, prevX, prevY multipliers, acc *[256]int32, ae *accEntropy, histo *[256]int32) float64 {
	*histo = [256]int32{}
	ccHistoBlue(histo, t.tb, t.tg, t.tr, t.tc, int8(g2b), int8(r2b))
	d := predictionCostCrossColor(acc, histo, ae)
	if uint8(g2b) == prevX.greenToBlue {
		d -= 3
	}
	if uint8(g2b) == prevY.greenToBlue {
		d -= 3
	}
	if uint8(r2b) == prevX.redToBlue {
		d -= 3
	}
	if uint8(r2b) == prevY.redToBlue {
		d -= 3
	}
	if g2b == 0 {
		d -= 3
	}
	if r2b == 0 {
		d -= 3
	}
	return d
}

// visitedSet records the candidates already evaluated by a search. The cost
// of a candidate is a pure function of it, and a candidate replaces the best
// one only if its cost is strictly lower, so a candidate seen before can never
// win again and is skipped. The candidates stay within [-63, 63] (sums of the
// search steps), so their byte keys do not alias.
type visitedSet struct {
	n    int
	keys [64]uint16
}

// seen reports whether k was added before, and adds it otherwise (when the
// set is full, k is not added: it would only be evaluated again).
func (v *visitedSet) seen(k uint16) bool {
	for _, x := range v.keys[:v.n] {
		if x == k {
			return true
		}
	}
	if v.n < len(v.keys) {
		v.keys[v.n] = k
		v.n++
	}
	return false
}

func (t *tilePixels) bestTransform(prevX, prevY multipliers, quality int, accRed, accBlue *[256]int32) multipliers {
	var histo [256]int32
	var best multipliers
	var aeR, aeB accEntropy
	aeR.init(accRed)
	aeB.init(accBlue)
	// Green to red.
	{
		var vis visitedSet
		maxIters := 4 + (7*quality)>>8
		g2rBest := 0
		bestDiff := t.costRed(0, prevX, prevY, accRed, &aeR, &histo)
		vis.seen(0)
		for iter := 0; iter < maxIters; iter++ {
			delta := 32 >> uint(iter)
			for off := -delta; off <= delta; off += 2 * delta {
				cur := off + g2rBest
				if vis.seen(uint16(uint8(cur))) {
					continue
				}
				d := t.costRed(cur, prevX, prevY, accRed, &aeR, &histo)
				if d < bestDiff {
					bestDiff = d
					g2rBest = cur
				}
			}
		}
		best.greenToRed = uint8(g2rBest)
	}
	// Green and red to blue.
	{
		offsets := [8][2]int{{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
		deltaLut := [7]int{16, 16, 8, 4, 2, 2, 2}
		iters := 4
		if quality < 25 {
			iters = 1
		} else if quality > 50 {
			iters = 7
		}
		var vis visitedSet
		g2b, r2b := 0, 0
		bestDiff := t.costBlue(0, 0, prevX, prevY, accBlue, &aeB, &histo)
		vis.seen(0)
		for iter := 0; iter < iters; iter++ {
			delta := deltaLut[iter]
			for axis := 0; axis < 8; axis++ {
				cg := offsets[axis][0]*delta + g2b
				cr := offsets[axis][1]*delta + r2b
				if !vis.seen(uint16(uint8(cg)) | uint16(uint8(cr))<<8) {
					d := t.costBlue(cg, cr, prevX, prevY, accBlue, &aeB, &histo)
					if d < bestDiff {
						bestDiff = d
						g2b, r2b = cg, cr
					}
				}
				if quality < 25 && iter == 4 {
					break
				}
			}
			if delta == 2 && g2b == 0 && r2b == 0 {
				break
			}
		}
		best.greenToBlue = uint8(g2b)
		best.redToBlue = uint8(r2b)
	}
	return best
}

// transformColor applies the forward cross-color transform to one pixel.
func transformColor(m multipliers, p uint32) uint32 {
	g := int8(p >> 8)
	r := int8(p >> 16)
	nr := int32(p>>16) & 0xff
	nb := int32(p) & 0xff
	nr -= colorTransformDelta(int8(m.greenToRed), g)
	nr &= 0xff
	nb -= colorTransformDelta(int8(m.greenToBlue), g)
	nb -= colorTransformDelta(int8(m.redToBlue), r)
	nb &= 0xff
	return p&0xff00ff00 | uint32(nr)<<16 | uint32(nb)
}

// colorSpaceTransform selects per-tile multipliers and applies them to argb.
// image receives the multipliers; like libwebp 1.6, its sampling is then
// reduced when it is made of uniform squares, and the resulting bits are
// returned.
func colorSpaceTransform(width, height, bits, quality int, argb []uint32, image []uint32, sc *scratch) int {
	tile := 1 << uint(bits)
	tilesX := subSampleSize(width, bits)
	tilesY := subSampleSize(height, bits)
	var accRed, accBlue [256]int32
	var prevX, prevY multipliers
	tp := &sc.tile
	for ty := 0; ty < tilesY && !sc.interrupted(); ty++ {
		for tx := 0; tx < tilesX; tx++ {
			x0 := tx * tile
			y0 := ty * tile
			xMax := min(x0+tile, width)
			yMax := min(y0+tile, height)
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
				ix := y*width + x0
				end := y*width + xMax
				for ; ix < end; ix++ {
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
