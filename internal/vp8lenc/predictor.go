package vp8lenc

// Spatial prediction transform.
// Algorithm adapted from libwebp (BSD-3-Clause).

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

// predictorImage selects the best mode for every tile and replaces argb by
// the prediction residuals. modes receives tile modes as ARGB pixels.
// maxQuant > 1 enables near-lossless quantization of the residuals.
//
// clean enables libwebp's exact=0 handling of fully transparent pixels
// (ignored with lowEffort, as in libwebp).
func predictorImage(width, height, bits int, lowEffort bool, maxQuant int, usedSubGreen, clean bool,
	argb []uint32, modes []uint32, sc *scratch) {
	tilesX := subSampleSize(width, bits)
	tilesY := subSampleSize(height, bits)
	n := width * height
	var md []uint8
	nl := maxQuant > 1 && !lowEffort && height >= 3 && width >= 3
	if !nl {
		maxQuant = 1
	}
	clean = clean && !lowEffort
	// slow selects the per-pixel GetResidual path working on row copies.
	slow := nl || clean
	var nlCur, nlUp []uint32
	if nl {
		md = sc.maxDiffs(n)
		computeMaxDiffs(width, height, argb, usedSubGreen, md)
	}
	if slow {
		nlCur = sc.rowA(width + 1)
		nlUp = sc.rowB(width + 1)
	}
	if lowEffort {
		for i := range modes[:tilesX*tilesY] {
			modes[i] = argbBlack | predLowEffort<<8
		}
	} else {
		var acc histo4
		var ae [4]accEntropy
		h1 := &sc.h4a
		h2 := &sc.h4b
		resid := sc.residBuf(1 << uint(bits))
		tile := 1 << uint(bits)
		for ty := 0; ty < tilesY; ty++ {
			for tx := 0; tx < tilesX; tx++ {
				sx := tx << uint(bits)
				sy := ty << uint(bits)
				maxX := min(tile, width-sx)
				maxY := min(tile, height-sy)
				leftMode, aboveMode := -1, -1
				if tx > 0 {
					leftMode = int(modes[ty*tilesX+tx-1]>>8) & 0xff
				}
				if ty > 0 {
					aboveMode = int(modes[(ty-1)*tilesX+tx]>>8) & 0xff
				}
				bestDiff := 1e30
				bestMode := 0
				cur, best := h1, h2
				for i := range ae {
					ae[i].init(&acc[i])
				}
				for mode := 0; mode < numPredModes; mode++ {
					*cur = histo4{}
					if slow {
						nlTileResiduals(mode, argb, width, height, sx, sy, maxX, maxY, md, maxQuant, usedSubGreen, clean, nlCur, nlUp, resid, cur)
					} else {
						for ry := 0; ry < maxY; ry++ {
							y := sy + ry
							row := argb[y*width : (y+1)*width]
							var up []uint32
							if y > 0 {
								up = argb[(y-1)*width : y*width+1]
							}
							r := resid[:maxX]
							residualRow(mode, row, up, y, sx, sx+maxX, r)
							for _, p := range r {
								cur.add(p)
							}
						}
					}
					d := predictionCostSpatialHistogram(&acc, cur, &ae)
					if mode == leftMode {
						d -= spatialPredictorBias
					}
					if mode == aboveMode {
						d -= spatialPredictorBias
					}
					if d < bestDiff {
						bestDiff = d
						bestMode = mode
						cur, best = best, cur
					}
				}
				for i := 0; i < 4; i++ {
					for j := 0; j < 256; j++ {
						acc[i][j] += best[i][j]
					}
				}
				modes[ty*tilesX+tx] = argbBlack | uint32(bestMode)<<8
			}
		}
	}
	// Replace the image by residuals, row by row, keeping the original (or,
	// with near-lossless, reconstructed) previous row.
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
}

// nlTileResiduals accumulates into h the near-lossless residuals of one tile
// for mode, using row copies so that predictions use reconstructed pixels.
func nlTileResiduals(mode int, argb []uint32, width, height, sx, sy, maxX, maxY int, md []uint8,
	maxQuant int, usedSubGreen, clean bool, cur, up []uint32, resid []uint32, h *histo4) {
	lo := sx
	if lo > 0 {
		lo--
	}
	hi := sx + maxX + 1 // one pixel past the tile (top-right context)
	n := width * height
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
		r := resid[:maxX]
		var mdRow []uint8
		if md != nil {
			mdRow = md[y*width : (y+1)*width]
		}
		residualRowNL(mode, cur, up, y, width, height, sx, sx+maxX, mdRow, maxQuant, usedSubGreen, clean, r)
		for _, p := range r {
			h.add(p)
		}
	}
}
