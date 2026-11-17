package vp8lenc

// Spatial prediction transform.
// Algorithm adapted from libwebp (BSD-3-Clause).

import "math"

const (
	numPredModes         = 14
	spatialPredictorBias = 15.0
	predLowEffort        = 11
)

func average2(a, b uint32) uint32 {
	return (((a ^ b) & 0xfefefefe) >> 1) + (a & b)
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

func sub3(a, b, c uint32) int32 {
	pb := int32(b) - int32(c)
	pa := int32(a) - int32(c)
	return abs32(pb) - abs32(pa)
}

// selectPred implements predictor 11 (a = top, b = left, c = top-left).
func selectPred(a, b, c uint32) uint32 {
	d := sub3(a>>24, b>>24, c>>24) +
		sub3((a>>16)&0xff, (b>>16)&0xff, (c>>16)&0xff) +
		sub3((a>>8)&0xff, (b>>8)&0xff, (c>>8)&0xff) +
		sub3(a&0xff, b&0xff, c&0xff)
	if d <= 0 {
		return a
	}
	return b
}

func clip255(v int32) uint32 {
	if uint32(v) < 256 {
		return uint32(v)
	}
	if v < 0 {
		return 0
	}
	return 255
}

func clampedAddSubtractFull(c0, c1, c2 uint32) uint32 {
	a := clip255(int32(c0>>24) + int32(c1>>24) - int32(c2>>24))
	r := clip255(int32((c0>>16)&0xff) + int32((c1>>16)&0xff) - int32((c2>>16)&0xff))
	g := clip255(int32((c0>>8)&0xff) + int32((c1>>8)&0xff) - int32((c2>>8)&0xff))
	b := clip255(int32(c0&0xff) + int32(c1&0xff) - int32(c2&0xff))
	return a<<24 | r<<16 | g<<8 | b
}

func addSubHalf(a, b int32) uint32 { return clip255(a + (a-b)/2) }

func clampedAddSubtractHalf(c0, c1, c2 uint32) uint32 {
	ave := average2(c0, c1)
	a := addSubHalf(int32(ave>>24), int32(c2>>24))
	r := addSubHalf(int32((ave>>16)&0xff), int32((c2>>16)&0xff))
	g := addSubHalf(int32((ave>>8)&0xff), int32((c2>>8)&0xff))
	b := addSubHalf(int32(ave&0xff), int32(c2&0xff))
	return a<<24 | r<<16 | g<<8 | b
}

// predictRow writes residuals for x in [x0,x1) of a row with y >= 1 and
// x0 >= 1. cur is the current row, up the row above extended by one pixel
// (up[width] is the first pixel of cur).
func predictRow(mode int, cur, up []uint32, x0, x1 int, out []uint32) {
	_ = up[x1]
	_ = cur[x1-1]
	out = out[:x1-x0]
	switch mode {
	case 0:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], argbBlack)
		}
	case 1:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], cur[x-1])
		}
	case 2:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], up[x])
		}
	case 3:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], up[x+1])
		}
	case 4:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], up[x-1])
		}
	case 5:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(average2(cur[x-1], up[x+1]), up[x]))
		}
	case 6:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(cur[x-1], up[x-1]))
		}
	case 7:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(cur[x-1], up[x]))
		}
	case 8:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(up[x-1], up[x]))
		}
	case 9:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(up[x], up[x+1]))
		}
	case 10:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], average2(average2(cur[x-1], up[x-1]), average2(up[x], up[x+1])))
		}
	case 11:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], selectPred(up[x], cur[x-1], up[x-1]))
		}
	case 12:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], clampedAddSubtractFull(cur[x-1], up[x], up[x-1]))
		}
	case 13:
		for x := x0; x < x1; x++ {
			out[x-x0] = subPixels(cur[x], clampedAddSubtractHalf(cur[x-1], up[x], up[x-1]))
		}
	}
}

// residualRow computes the residuals of row y for x in [x0,x1), handling the
// image borders (first row predicts from the left, first column from the top).
func residualRow(mode int, cur, up []uint32, y, x0, x1 int, out []uint32) {
	if y == 0 {
		for x := x0; x < x1; x++ {
			if x == 0 {
				out[x-x0] = subPixels(cur[0], argbBlack)
			} else {
				out[x-x0] = subPixels(cur[x], cur[x-1])
			}
		}
		return
	}
	xs := x0
	if xs == 0 {
		out[0] = subPixels(cur[0], up[0])
		xs = 1
	}
	if xs < x1 {
		predictRow(mode, cur, up, xs, x1, out[xs-x0:])
	}
}

// predictPixel returns the prediction of mode for pixel x (x >= 1, y >= 1).
func predictPixel(mode int, cur, up []uint32, x int) uint32 {
	switch mode {
	case 0:
		return argbBlack
	case 1:
		return cur[x-1]
	case 2:
		return up[x]
	case 3:
		return up[x+1]
	case 4:
		return up[x-1]
	case 5:
		return average2(average2(cur[x-1], up[x+1]), up[x])
	case 6:
		return average2(cur[x-1], up[x-1])
	case 7:
		return average2(cur[x-1], up[x])
	case 8:
		return average2(up[x-1], up[x])
	case 9:
		return average2(up[x], up[x+1])
	case 10:
		return average2(average2(cur[x-1], up[x-1]), average2(up[x], up[x+1]))
	case 11:
		return selectPred(up[x], cur[x-1], up[x-1])
	case 12:
		return clampedAddSubtractFull(cur[x-1], up[x], up[x-1])
	default:
		return clampedAddSubtractHalf(cur[x-1], up[x], up[x-1])
	}
}

// residualRowNL is residualRow with near-lossless quantization (maxQuant >
// 1) and/or the exact=0 handling of transparent pixels (clean), like
// libwebp's GetResidual. It updates cur with the reconstructed pixels (and
// up[width] when the first pixel of the row changes). md holds the max
// diffs of the row (only used with near-lossless).
func residualRowNL(mode int, cur, up []uint32, y, width, height, x0, x1 int, md []uint8,
	maxQuant int, usedSubGreen, clean bool, out []uint32) {
	for x := x0; x < x1; x++ {
		var pred uint32
		switch {
		case y == 0:
			if x == 0 {
				pred = argbBlack
			} else {
				pred = cur[x-1]
			}
		case x == 0:
			pred = up[0]
		default:
			pred = predictPixel(mode, cur, up, x)
		}
		var res uint32
		if maxQuant <= 1 || mode == 0 || y == 0 || y == height-1 || x == 0 || x == width-1 {
			res = subPixels(cur[x], pred)
		} else {
			res = nearLosslessResidual(cur[x], pred, maxQuant, int(md[x]), usedSubGreen)
			cur[x] = addPixels(pred, res)
		}
		if clean && cur[x] < 0x01000000 {
			// Alpha is 0: the RGB of the residual can be chosen freely;
			// make it 0 and update the source accordingly. The prediction
			// of alpha itself must be kept.
			res &= 0xff000000
			cur[x] = pred & 0x00ffffff
			if x == 0 && y != 0 {
				// The rightmost pixel of a row uses the leftmost pixel of
				// the same row as its top-right context.
				up[width] = cur[0]
			}
		}
		out[x-x0] = res
	}
}

type histo4 [4][256]int32

func (h *histo4) add(p uint32) {
	h[0][p>>24]++
	h[1][(p>>16)&0xff]++
	h[2][(p>>8)&0xff]++
	h[3][p&0xff]++
}

func predictionCostSpatial(counts *[256]int32, weight0 int, expVal float64) float64 {
	const significant = 256 >> 4
	const decay = 0.6
	bits := float64(weight0) * float64(counts[0])
	for i := 1; i < significant; i++ {
		bits += expVal * float64(counts[i]+counts[256-i])
		expVal *= decay
	}
	return -0.1 * bits
}

func predictionCostSpatialHistogram(acc, tile *histo4, ae *[4]accEntropy) float64 {
	var r float64
	for i := 0; i < 4; i++ {
		r += predictionCostSpatial(&tile[i], 1, 0.94)
		r += combinedShannonEntropy(&tile[i], &acc[i], &ae[i])
	}
	return r
}

// predictorImage selects the predictor modes and replaces argb by the
// prediction residuals. Like libwebp 1.6 (VP8LResidualImage), the modes are
// searched for tiles of every size from 1<<minBits to 1<<maxBits pixels and
// the sampling with the lowest estimated cost is kept, then reduced further
// when the chosen mode image is made of uniform squares
// (VP8LOptimizeSampling). It returns the transform bits; modes (which must
// hold one entry per tile of 1<<minBits pixels) receives the mode image as
// ARGB pixels. maxQuant > 1 enables near-lossless quantization of the
// residuals.
//
// clean enables libwebp's exact=0 handling of fully transparent pixels
// (ignored with lowEffort, as in libwebp).
func predictorImage(width, height, minBits, maxBits int, lowEffort bool, maxQuant int, usedSubGreen, clean bool,
	argb []uint32, modes []uint32, sc *scratch) int {
	n := width * height
	var md []uint8
	nl := maxQuant > 1 && !lowEffort && height >= 3 && width >= 3
	if !nl {
		maxQuant = 1
	}
	clean = clean && !lowEffort
	// slow selects the per-pixel GetResidual path working on row copies.
	slow := nl || clean
	if nl {
		md = sc.maxDiffs(n)
		computeMaxDiffs(width, height, argb, usedSubGreen, md)
	}
	var bits int
	if lowEffort {
		bits = maxBits
		for i := range modes[:subSampleSize(width, bits)*subSampleSize(height, bits)] {
			modes[i] = argbBlack | predLowEffort<<8
		}
	} else {
		ps := predSearch{width: width, height: height, minBits: minBits, maxBits: maxBits,
			maxQuant: maxQuant, md: md, usedSubGreen: usedSubGreen, clean: clean, slow: slow, argb: argb, sc: sc}
		bits = ps.run(modes)
	}
	// Replace the image by residuals, row by row, keeping the original (or,
	// with near-lossless, reconstructed) previous row.
	tilesX := subSampleSize(width, bits)
	prev := sc.rowA(width + 1)
	curCopy := sc.rowB(width + 1)
	out := sc.residBuf(width)
	tile := 1 << uint(bits)
	for y := 0; y < height; y++ {
		row := argb[y*width : (y+1)*width]
		copy(curCopy, row)
		if y > 0 {
			prev[width] = curCopy[0]
		}
		mrow := modes[(y>>uint(bits))*tilesX:]
		for x := 0; x < width; x += tile {
			x1 := min(x+tile, width)
			mode := int(mrow[x>>uint(bits)]>>8) & 0xff
			if slow {
				var mdRow []uint8
				if nl {
					mdRow = md[y*width : (y+1)*width]
				}
				residualRowNL(mode, curCopy, prev, y, width, height, x, x1, mdRow, maxQuant, usedSubGreen, clean, out[x:x1])
			} else {
				residualRow(mode, curCopy, prev, y, x, x1, out[x:x1])
			}
		}
		copy(row, out[:width])
		prev, curCopy = curCopy, prev
	}
	return bits
}

// predSearch is libwebp's GetBestPredictorsAndSubSampling: the best mode of
// every tile is searched at every sampling from minBits to maxBits at once.
//
// Tiles of 1<<minBits pixels are visited in Z order within each max-tile
// (of 1<<maxBits pixels) so that super-tiles are completed as soon as
// possible:
//
//	1  2|  5  6
//	3  4|  7  8
//	-----------
//	9 10| 13 14
//	11 12| 15 16
//
// The residual histograms of a tile (for each mode) are added to those of
// its enclosing super-tiles; when a super-tile is complete its best mode is
// chosen like for a tile, with its own accumulated histogram.
type predSearch struct {
	width, height    int
	minBits, maxBits int
	maxQuant         int
	md               []uint8
	usedSubGreen     bool
	clean, slow      bool
	argb             []uint32
	sc               *scratch

	hist      []histo4 // [level][mode] residual histograms of the current (super-)tiles
	acc       []histo4 // [level] accumulated histograms of the chosen modes
	levelMode [][]uint32
	predHisto [(maxTransformBits - minTransformBits + 1) * numPredModes]int32
}

func (ps *predSearch) run(modes []uint32) int {
	width, height := ps.width, ps.height
	minBits := ps.minBits
	sc := ps.sc
	tilesPerRow := subSampleSize(width, minBits)
	tilesPerCol := subSampleSize(height, minBits)
	maxSub := ps.maxBits - minBits
	// Residuals are added directly to the histograms of the super-tiles up
	// to updateUpTo (cheaper for small tiles); larger super-tiles add the
	// histograms of the finished smaller ones.
	updateUpTo := max(min(4, ps.maxBits), minBits) - minBits
	levels := maxSub + 1

	total := 0
	for l := 0; l < levels; l++ {
		total += subSampleSize(width, minBits+l) * subSampleSize(height, minBits+l)
	}
	all := sc.predModeBuf(total)
	if cap(ps.levelMode) < levels {
		ps.levelMode = make([][]uint32, levels)
	}
	ps.levelMode = ps.levelMode[:levels]
	off := 0
	for l := 0; l < levels; l++ {
		sz := subSampleSize(width, minBits+l) * subSampleSize(height, minBits+l)
		ps.levelMode[l] = all[off : off+sz]
		off += sz
	}
	ps.hist = sc.predHistos(levels * numPredModes)
	ps.acc = sc.predAcc(levels)

	tile := 1 << uint(minBits)
	resid := sc.residBuf(tile * tile)
	maxTileSize := 1 << uint(maxSub) // in tiles
	var localX, localY, maxTileX, maxTileY, tileX, tileY int
	for tileY < tilesPerCol {
		// Residuals of the tile for every mode.
		sx, sy := tileX<<uint(minBits), tileY<<uint(minBits)
		maxX := min(tile, width-sx)
		maxY := min(tile, height-sy)
		for mode := 0; mode < numPredModes; mode++ {
			r := resid[:maxX*maxY]
			ps.tileResiduals(mode, sx, sy, maxX, maxY, r)
			for l := 0; l <= updateUpTo; l++ {
				h := &ps.hist[l*numPredModes+mode]
				for _, p := range r {
					h.add(p)
				}
			}
		}
		// Update all the super-tiles that are complete.
		sub := 0
		for {
			ps.bestForTile(sub, tileX>>uint(sub), tileY>>uint(sub), subSampleSize(width, minBits+sub))
			if sub == maxSub {
				break
			}
			sub++
			if sub > updateUpTo {
				dst := ps.hist[sub*numPredModes : (sub+1)*numPredModes]
				src := ps.hist[(sub-1)*numPredModes : sub*numPredModes]
				for m := range dst {
					addHisto4(&dst[m], &src[m])
				}
			}
			if !((tileX == tilesPerRow-1 || (localX+1)%(1<<uint(sub)) == 0) &&
				(tileY == tilesPerCol-1 || (localY+1)%(1<<uint(sub)) == 0)) {
				sub-- // index of the last finished super-tile
				break
			}
		}
		// Reset the histograms of the finished (super-)tiles.
		clear(ps.hist[:(sub+1)*numPredModes])
		if sub == maxSub {
			// Start a new max-tile.
			if tileX == tilesPerRow-1 {
				maxTileX = 0
				maxTileY++
			} else {
				maxTileX++
			}
			localX, localY = 0, 0
		} else {
			// Proceed with the Z traversal.
			cx := localX >> uint(sub)
			cy := localY >> uint(sub)
			switch {
			case tileX == tilesPerRow-1 && cx%2 == 0:
				cy++
			case cx%2 == 0:
				cx++
			default:
				cy++
				cx--
			}
			localX = cx << uint(sub)
			localY = cy << uint(sub)
		}
		tileX = maxTileX*maxTileSize + localX
		tileY = maxTileY*maxTileSize + localY
	}

	// Pick the sampling with the lowest cost: entropy of the mode image
	// plus entropy of the residuals.
	bestCost := math.Inf(1)
	best := 0
	for l := 0; l < levels; l++ {
		cost := shannonEntropy(ps.predHisto[l*numPredModes : (l+1)*numPredModes])
		for c := 0; c < 4; c++ {
			cost += shannonEntropy(ps.acc[l][c][:])
		}
		if cost < bestCost {
			bestCost = cost
			best = l
		}
	}
	bits := minBits + best
	copy(modes, ps.levelMode[best])
	return optimizeSampling(modes, width, height, bits, maxTransformBits)
}

// bestForTile chooses the mode of the (super-)tile (tx, ty) at level sub
// (libwebp GetBestPredictorForTile).
func (ps *predSearch) bestForTile(sub, tx, ty, perRow int) {
	modes := ps.levelMode[sub]
	acc := &ps.acc[sub]
	leftMode, aboveMode := -1, -1
	if tx > 0 {
		leftMode = int(modes[ty*perRow+tx-1]>>8) & 0xff
	}
	if ty > 0 {
		aboveMode = int(modes[(ty-1)*perRow+tx]>>8) & 0xff
	}
	var ae [4]accEntropy
	for i := range ae {
		ae[i].init(&acc[i])
	}
	hs := ps.hist[sub*numPredModes : (sub+1)*numPredModes]
	bestDiff := math.Inf(1)
	bestMode := 0
	for mode := range hs {
		d := predictionCostSpatialHistogram(acc, &hs[mode], &ae)
		if mode == leftMode {
			d -= spatialPredictorBias
		}
		if mode == aboveMode {
			d -= spatialPredictorBias
		}
		if d < bestDiff {
			bestDiff = d
			bestMode = mode
		}
	}
	addHisto4(acc, &hs[bestMode])
	modes[ty*perRow+tx] = argbBlack | uint32(bestMode)<<8
	ps.predHisto[sub*numPredModes+bestMode]++
}

func addHisto4(dst, src *histo4) {
	for i := range dst {
		d, s := &dst[i], &src[i]
		for j := range d {
			d[j] += s[j]
		}
	}
}

// shannonEntropy returns sum(X)*log2(sum(X)) - sum(x*log2(x)) (libwebp
// VP8LShannonEntropy).
func shannonEntropy(x []int32) float64 {
	var r float64
	var sum uint32
	for _, v := range x {
		if v != 0 {
			sum += uint32(v)
			r -= slog2(uint32(v))
		}
	}
	return r + slog2(sum)
}

// tileResiduals computes into out the residuals of one tile for mode.
func (ps *predSearch) tileResiduals(mode, sx, sy, maxX, maxY int, out []uint32) {
	width := ps.width
	argb := ps.argb
	if !ps.slow {
		for ry := 0; ry < maxY; ry++ {
			y := sy + ry
			row := argb[y*width : (y+1)*width]
			var up []uint32
			if y > 0 {
				up = argb[(y-1)*width : y*width+1]
			}
			residualRow(mode, row, up, y, sx, sx+maxX, out[ry*maxX:(ry+1)*maxX])
		}
		return
	}
	// Row copies, so that predictions use the reconstructed (near-lossless)
	// or cleaned (transparent) pixels.
	cur := ps.sc.rowA(width + 1)
	up := ps.sc.rowB(width + 1)
	lo := sx
	if lo > 0 {
		lo--
	}
	hi := sx + maxX + 1 // one pixel past the tile (top-right context)
	n := width * ps.height
	for ry := 0; ry < maxY; ry++ {
		y := sy + ry
		if ry == 0 {
			if y > 0 {
				e := min((y-1)*width+hi, n)
				copy(up[lo:], argb[(y-1)*width+lo:e])
			}
		} else {
			up, cur = cur, up
		}
		e := min(y*width+hi, n)
		copy(cur[lo:], argb[y*width+lo:e])
		var mdRow []uint8
		if ps.md != nil {
			mdRow = ps.md[y*width : (y+1)*width]
		}
		residualRowNL(mode, cur, up, y, width, ps.height, sx, sx+maxX, mdRow, ps.maxQuant, ps.usedSubGreen, ps.clean,
			out[ry*maxX:(ry+1)*maxX])
	}
}

// optimizeSampling checks whether image (a sub-image of a fullWidth x
// fullHeight image sampled with bits) is made of uniform squares of a larger
// power of 2 and, if so, subsamples it in place. It returns the new bits
// (at most maxBits). Port of libwebp's VP8LOptimizeSampling.
func optimizeSampling[T uint16 | uint32](image []T, fullWidth, fullHeight, bits, maxBits int) int {
	width := subSampleSize(fullWidth, bits)
	height := subSampleSize(fullHeight, bits)
	bestBits := bits
	// Check rows first.
	for bestBits < maxBits {
		newSquare := 1 << uint(bestBits+1-bits)
		square := 1 << uint(bestBits-bits)
		good := true
		for y := 0; y+square < height; y += newSquare {
			a := image[y*width : (y+1)*width]
			b := image[(y+square)*width : (y+square+1)*width]
			if !equalSlice(a, b) {
				good = false
				break
			}
		}
		if !good {
			break
		}
		bestBits++
	}
	if bestBits == bits {
		return bits
	}
	// Check columns.
	for bestBits > bits {
		square := 1 << uint(bestBits-bits)
		good := true
	rows:
		for y := 0; y < height; y++ {
			row := image[y*width : (y+1)*width]
			for x := 0; x < width; x += square {
				for i := x + 1; i < min(x+square, width); i++ {
					if row[i] != row[x] {
						good = false
						break rows
					}
				}
			}
		}
		if good {
			break
		}
		bestBits--
	}
	if bestBits == bits {
		return bits
	}
	// Subsample the image.
	oldWidth := width
	square := 1 << uint(bestBits-bits)
	width = subSampleSize(fullWidth, bestBits)
	height = subSampleSize(fullHeight, bestBits)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			image[y*width+x] = image[square*(y*oldWidth+x)]
		}
	}
	return bestBits
}

func equalSlice[T uint16 | uint32](a, b []T) bool {
	b = b[:len(a)]
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
