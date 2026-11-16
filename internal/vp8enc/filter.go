package vp8enc

// Loop-filter strength selection (autofilter) based on SSIM of the filtered
// reconstruction.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	maxLFLevels   = 64
	ssimKernel    = 3
	ssimKernelLen = 2*ssimKernel + 1
)

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func sclip1(v int) int { return clipInt(v, -128, 127) } // clips [-1020, 1020] to [-128, 127]
func sclip2(v int) int { return clipInt(v, -16, 15) }   // clips [-112, 112] to [-16, 15]
func uclip(v int) byte { return byte(clipInt(v, 0, 255)) }

func doFilter2(p []byte, i, step int) {
	p1, p0, q0, q1 := int(p[i-2*step]), int(p[i-step]), int(p[i]), int(p[i+step])
	a := 3*(q0-p0) + sclip1(p1-q1)
	a1 := sclip2((a + 4) >> 3)
	a2 := sclip2((a + 3) >> 3)
	p[i-step] = uclip(p0 + a2)
	p[i] = uclip(q0 - a1)
}

func doFilter4(p []byte, i, step int) {
	p1, p0, q0, q1 := int(p[i-2*step]), int(p[i-step]), int(p[i]), int(p[i+step])
	a := 3 * (q0 - p0)
	a1 := sclip2((a + 4) >> 3)
	a2 := sclip2((a + 3) >> 3)
	a3 := (a1 + 1) >> 1
	p[i-2*step] = uclip(p1 + a3)
	p[i-step] = uclip(p0 + a2)
	p[i] = uclip(q0 - a1)
	p[i+step] = uclip(q1 - a3)
}

func hev(p []byte, i, step, thresh int) bool {
	p1, p0, q0, q1 := int(p[i-2*step]), int(p[i-step]), int(p[i]), int(p[i+step])
	return absInt(p1-p0) > thresh || absInt(q1-q0) > thresh
}

func needsFilter(p []byte, i, step, t int) bool {
	p1, p0, q0, q1 := int(p[i-2*step]), int(p[i-step]), int(p[i]), int(p[i+step])
	return 4*absInt(p0-q0)+absInt(p1-q1) <= t
}

func needsFilter2(p []byte, i, step, t, it int) bool {
	p3, p2, p1, p0 := int(p[i-4*step]), int(p[i-3*step]), int(p[i-2*step]), int(p[i-step])
	q0, q1, q2, q3 := int(p[i]), int(p[i+step]), int(p[i+2*step]), int(p[i+3*step])
	if 4*absInt(p0-q0)+absInt(p1-q1) > t {
		return false
	}
	return absInt(p3-p2) <= it && absInt(p2-p1) <= it && absInt(p1-p0) <= it &&
		absInt(q3-q2) <= it && absInt(q2-q1) <= it && absInt(q1-q0) <= it
}

// simpleFilter16i filters the 3 inner vertical (hstride=1) or horizontal
// edges of a 16x16 block starting at p[0].
func simpleFilter16i(p []byte, hstride, vstride, thresh int) {
	thresh2 := 2*thresh + 1
	for k := 1; k <= 3; k++ {
		base := 4 * k * hstride
		for i := 0; i < 16; i++ {
			o := base + i*vstride
			if needsFilter(p, o, hstride, thresh2) {
				doFilter2(p, o, hstride)
			}
		}
	}
}

func filterLoop24(p []byte, base, hstride, vstride, size, thresh, ithresh, hevThresh int) {
	thresh2 := 2*thresh + 1
	for i := 0; i < size; i++ {
		o := base + i*vstride
		if needsFilter2(p, o, hstride, thresh2, ithresh) {
			if hev(p, o, hstride, hevThresh) {
				doFilter2(p, o, hstride)
			} else {
				doFilter4(p, o, hstride)
			}
		}
	}
}

// doFilter applies the inner-edge loop filter of the given level to a copy
// of the reconstruction (into yuvOut2).
func (it *iterator) doFilter(level int) {
	e := it.enc
	ilevel := getILevel(e.cfg.FilterSharpness, level)
	limit := 2*level + ilevel
	dst := it.yuvOut2
	copy(dst[:yuvSize], it.yuvOut[:yuvSize])
	y := dst[yOff:]
	if e.filterSimple {
		simpleFilter16i(y, 1, bps, limit) // horizontal filtering of vertical edges
		simpleFilter16i(y, bps, 1, limit) // vertical filtering of horizontal edges
		return
	}
	hevThresh := 0
	if level >= 40 {
		hevThresh = 2
	} else if level >= 15 {
		hevThresh = 1
	}
	for k := 1; k <= 3; k++ { // HFilter16i
		filterLoop24(y, 4*k, 1, bps, 16, limit, ilevel, hevThresh)
	}
	filterLoop24(dst, uOff+4, 1, bps, 8, limit, ilevel, hevThresh) // HFilter8i
	filterLoop24(dst, vOff+4, 1, bps, 8, limit, ilevel, hevThresh)
	for k := 1; k <= 3; k++ { // VFilter16i
		filterLoop24(y, 4*k*bps, bps, 1, 16, limit, ilevel, hevThresh)
	}
	filterLoop24(dst, uOff+4*bps, bps, 1, 8, limit, ilevel, hevThresh) // VFilter8i
	filterLoop24(dst, vOff+4*bps, bps, 1, 8, limit, ilevel, hevThresh)
}

func getILevel(sharpness, level int) int {
	if sharpness > 0 {
		if sharpness > 4 {
			level >>= 2
		} else {
			level >>= 1
		}
		if level > 9-sharpness {
			level = 9 - sharpness
		}
	}
	if level < 1 {
		level = 1
	}
	return level
}

var kSSIMWeight = [ssimKernelLen]uint32{1, 2, 3, 4, 3, 2, 1}

func ssimGetClipped(src1, src2 []byte, xo, yo, W, H int) float64 {
	var w, xm, ym, xxm, xym, yym uint32
	ymin := max(yo-ssimKernel, 0)
	ymax := min(yo+ssimKernel, H-1)
	xmin := max(xo-ssimKernel, 0)
	xmax := min(xo+ssimKernel, W-1)
	for y := ymin; y <= ymax; y++ {
		r1 := src1[y*bps:]
		r2 := src2[y*bps:]
		wy := kSSIMWeight[ssimKernel+y-yo]
		for x := xmin; x <= xmax; x++ {
			ww := kSSIMWeight[ssimKernel+x-xo] * wy
			s1 := uint32(r1[x])
			s2 := uint32(r2[x])
			w += ww
			xm += ww * s1
			ym += ww * s2
			xxm += ww * s1 * s1
			xym += ww * s1 * s2
			yym += ww * s2 * s2
		}
	}
	return ssimCalculation(w, xm, ym, xxm, xym, yym)
}

func ssimCalculation(N, xm, ym, xxm, xym, yym uint32) float64 {
	w2 := uint64(N) * uint64(N)
	C1 := 20 * w2
	C2 := 60 * w2
	C3 := 8 * 8 * w2 // 'dark' limit ~= 6
	xmxm := uint64(xm) * uint64(xm)
	ymym := uint64(ym) * uint64(ym)
	if xmxm+ymym >= C3 {
		xmym := int64(xm) * int64(ym)
		sxy := int64(xym)*int64(N) - xmym // can be negative
		sxx := uint64(xxm)*uint64(N) - xmxm
		syy := uint64(yym)*uint64(N) - ymym
		if sxy < 0 {
			sxy = 0
		}
		numS := (2*uint64(sxy) + C2) >> 8
		denS := (sxx + syy + C2) >> 8
		fnum := (2*uint64(xmym) + C1) * numS
		fden := (xmxm + ymym + C1) * denS
		return float64(fnum) / float64(fden)
	}
	return 1
}

func getMBSSIM(yuv1, yuv2 []byte) float64 {
	sum := 0.
	for y := ssimKernel; y < 16-ssimKernel; y++ {
		for x := ssimKernel; x < 16-ssimKernel; x++ {
			sum += ssimGetClipped(yuv1[yOff:], yuv2[yOff:], x, y, 16, 16)
		}
	}
	for x := 1; x < 7; x++ {
		for y := 1; y < 7; y++ {
			sum += ssimGetClipped(yuv1[uOff:], yuv2[uOff:], x, y, 8, 8)
			sum += ssimGetClipped(yuv1[vOff:], yuv2[vOff:], x, y, 8, 8)
		}
	}
	return sum
}

func (e *encoder) initFilter() {
	if e.lfStats != nil {
		*e.lfStats = [numMBSegments][64]float64{}
	}
}

func (e *encoder) storeFilterStats(it *iterator) {
	if e.lfStats == nil {
		return
	}
	mb := it.mb()
	if mb.typ == 1 && mb.skip {
		return
	}
	s := int(mb.segment & 3)
	level0 := e.dqm[s].fstrength
	deltaMin := -e.dqm[s].quant
	deltaMax := e.dqm[s].quant
	stepSize := 1
	if deltaMax-deltaMin >= 4 {
		stepSize = 4
	}
	e.lfStats[s][0] += getMBSSIM(it.yuvIn, it.yuvOut)
	for d := deltaMin; d <= deltaMax; d += stepSize {
		level := level0 + d
		if level <= 0 || level >= maxLFLevels {
			continue
		}
		it.doFilter(level)
		e.lfStats[s][level] += getMBSSIM(it.yuvIn, it.yuvOut2)
	}
}

func (e *encoder) adjustFilterStrength() {
	if e.lfStats != nil {
		for s := 0; s < numMBSegments; s++ {
			bestLevel := 0
			bestV := 1.00001 * e.lfStats[s][0]
			for i := 1; i < maxLFLevels; i++ {
				v := e.lfStats[s][i]
				if v > bestV {
					bestV = v
					bestLevel = i
				}
			}
			e.dqm[s].fstrength = bestLevel
		}
		// With a single segment the level is only carried by the filter
		// header (libwebp leaves it unchanged, making autofilter a no-op).
		if e.numSegments == 1 {
			e.filterLevel = e.dqm[0].fstrength
		}
		return
	}
	if e.cfg.FilterStrength > 0 {
		maxLevel := 0
		for s := 0; s < numMBSegments; s++ {
			dqm := &e.dqm[s]
			delta := (dqm.maxEdge * int(dqm.y2.q[1])) >> 3
			level := filterStrengthFromDelta(e.filterSharpness, delta)
			if level > dqm.fstrength {
				dqm.fstrength = level
			}
			if maxLevel < dqm.fstrength {
				maxLevel = dqm.fstrength
			}
		}
		e.filterLevel = maxLevel
	}
}
