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

// tilePixels holds the channels of one tile as signed bytes.
type tilePixels struct {
	r, g, b []int8
}

func (t *tilePixels) load(argb []uint32, stride, x0, y0, w, h int) {
	n := w * h
	if cap(t.r) < n {
		t.r = make([]int8, n)
		t.g = make([]int8, n)
		t.b = make([]int8, n)
	}
	t.r, t.g, t.b = t.r[:n], t.g[:n], t.b[:n]
	k := 0
	for y := 0; y < h; y++ {
		row := argb[(y0+y)*stride+x0 : (y0+y)*stride+x0+w]
		for _, p := range row {
			t.r[k] = int8(p >> 16)
			t.g[k] = int8(p >> 8)
			t.b[k] = int8(p)
			k++
		}
	}
}

func predictionCostCrossColor(acc, counts *[256]int32, ae *accEntropy) float64 {
	return combinedShannonEntropy(counts, acc, ae) + predictionCostSpatial(counts, 3, 2.4)
}

func (t *tilePixels) costRed(g2r int, prevX, prevY multipliers, acc *[256]int32, ae *accEntropy, histo *[256]int32) float64 {
	*histo = [256]int32{}
	m := int8(g2r)
	r := t.r
	g := t.g[:len(r)]
	for i, rv := range r {
		nr := int32(rv) - colorTransformDelta(m, g[i])
		histo[uint8(nr)]++
	}
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
	mg := int8(g2b)
	mr := int8(r2b)
	b := t.b
	g := t.g[:len(b)]
	r := t.r[:len(b)]
	for i, bv := range b {
		nb := int32(bv) - colorTransformDelta(mg, g[i]) - colorTransformDelta(mr, r[i])
		histo[uint8(nb)]++
	}
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

func (t *tilePixels) bestTransform(prevX, prevY multipliers, quality int, accRed, accBlue *[256]int32) multipliers {
	var histo [256]int32
	var best multipliers
	var aeR, aeB accEntropy
	aeR.init(accRed)
	aeB.init(accBlue)
	// Green to red.
	{
		maxIters := 4 + (7*quality)>>8
		g2rBest := 0
		bestDiff := t.costRed(0, prevX, prevY, accRed, &aeR, &histo)
		for iter := 0; iter < maxIters; iter++ {
			delta := 32 >> uint(iter)
			for off := -delta; off <= delta; off += 2 * delta {
				cur := off + g2rBest
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
		g2b, r2b := 0, 0
		bestDiff := t.costBlue(0, 0, prevX, prevY, accBlue, &aeB, &histo)
		for iter := 0; iter < iters; iter++ {
			delta := deltaLut[iter]
			for axis := 0; axis < 8; axis++ {
				cg := offsets[axis][0]*delta + g2b
				cr := offsets[axis][1]*delta + r2b
				d := t.costBlue(cg, cr, prevX, prevY, accBlue, &aeB, &histo)
				if d < bestDiff {
					bestDiff = d
					g2b, r2b = cg, cr
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
	for ty := 0; ty < tilesY; ty++ {
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
