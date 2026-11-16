package vp8enc

import "sync"

// Analysis pass: collects per-macroblock susceptibility ("alpha") and assigns
// segments with a simplified k-means.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	maxItersKMeans = 6
	maxAlphaValue  = 255               // 8b of precision for susceptibilities.
	alphaScale     = 2 * maxAlphaValue // scaling factor for alpha.
	defaultAlpha   = -1
)

func (e *encoder) smoothSegmentMap() {
	w, h := e.mbW, e.mbH
	const majorityCnt3x3 = 5
	tmp := make([]uint8, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			var cnt [numMBSegments]int
			i := x + w*y
			mb := e.mbInfo
			majority := mb[i].segment
			cnt[mb[i-w-1].segment&3]++
			cnt[mb[i-w].segment&3]++
			cnt[mb[i-w+1].segment&3]++
			cnt[mb[i-1].segment&3]++
			cnt[mb[i+1].segment&3]++
			cnt[mb[i+w-1].segment&3]++
			cnt[mb[i+w].segment&3]++
			cnt[mb[i+w+1].segment&3]++
			for n := 0; n < numMBSegments; n++ {
				if cnt[n] >= majorityCnt3x3 {
					majority = uint8(n)
					break
				}
			}
			tmp[i] = majority
		}
	}
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			e.mbInfo[x+w*y].segment = tmp[x+y*w]
		}
	}
}

func (e *encoder) setSegmentAlphas(centers *[numMBSegments]int, mid int) {
	nb := e.numSegments
	mn, mx := centers[0], centers[0]
	if nb > 1 {
		for n := 0; n < nb; n++ {
			mn = min(mn, centers[n])
			mx = max(mx, centers[n])
		}
	}
	if mx == mn {
		mx = mn + 1
	}
	for n := 0; n < nb; n++ {
		alpha := 255 * (centers[n] - mid) / (mx - mn)
		beta := 255 * (centers[n] - mn) / (mx - mn)
		e.dqm[n].alpha = clipInt(alpha, -127, 127)
		e.dqm[n].beta = clipInt(beta, 0, 255)
	}
}

func (e *encoder) assignSegments(alphas *[maxAlphaValue + 1]int) {
	nb := min(e.numSegments, numMBSegments)
	var centers [numMBSegments]int
	weightedAverage := 0
	var segMap [maxAlphaValue + 1]int
	var accum, distAccum [numMBSegments]int

	n := 0
	for n = 0; n <= maxAlphaValue && alphas[n] == 0; n++ {
	}
	minA := n
	for n = maxAlphaValue; n > minA && alphas[n] == 0; n-- {
	}
	maxA := n
	rangeA := maxA - minA

	for k, n := 0, 1; k < nb; k, n = k+1, n+2 {
		centers[k] = minA + (n*rangeA)/(2*nb)
	}

	for k := 0; k < maxItersKMeans; k++ {
		for n := 0; n < nb; n++ {
			accum[n] = 0
			distAccum[n] = 0
		}
		n := 0
		for a := minA; a <= maxA; a++ {
			if alphas[a] != 0 {
				for n+1 < nb && absInt(a-centers[n+1]) < absInt(a-centers[n]) {
					n++
				}
				segMap[a] = n
				distAccum[n] += a * alphas[a]
				accum[n] += alphas[a]
			}
		}
		displaced := 0
		weightedAverage = 0
		totalWeight := 0
		for n := 0; n < nb; n++ {
			if accum[n] != 0 {
				newCenter := (distAccum[n] + accum[n]/2) / accum[n]
				displaced += absInt(centers[n] - newCenter)
				centers[n] = newCenter
				weightedAverage += newCenter * accum[n]
				totalWeight += accum[n]
			}
		}
		weightedAverage = (weightedAverage + totalWeight/2) / totalWeight
		if displaced < 5 {
			break
		}
	}

	for i := range e.mbInfo {
		mb := &e.mbInfo[i]
		a := int(mb.alpha)
		mb.segment = uint8(segMap[a])
		mb.alpha = uint8(centers[segMap[a]])
	}
	if nb > 1 && e.cfg.Preprocessing&1 != 0 {
		e.smoothSegmentMap()
	}
	e.setSegmentAlphas(&centers, weightedAverage)
}

func (it *iterator) mbAnalyzeBestIntra16Mode() int {
	bestAlpha := defaultAlpha
	bestMode := 0
	it.makeLuma16Preds()
	for mode := 0; mode < 2; mode++ {
		maxValue, lastNonZero := collectHistogram(it.yuvIn[yOff:], it.yuvP[vp8I16ModeOffsets[mode]:], vp8Scan[:])
		alpha := 0
		if maxValue > 1 {
			alpha = alphaScale * lastNonZero / maxValue
		}
		if alpha > bestAlpha {
			bestAlpha = alpha
			bestMode = mode
		}
	}
	it.setIntra16Mode(bestMode)
	return bestAlpha
}

func (it *iterator) fastMBAnalyze() int {
	q := int(it.enc.cfg.Quality)
	threshold := uint32(8 + (17-8)*q/100)
	var m, m2 uint32
	for k := 0; k < 16; k += 4 {
		ref := it.yuvIn[yOff+k*bps:]
		for b := 0; b < 4; b++ {
			var avg uint32
			for y := 0; y < 4; y++ {
				r := ref[y*bps+b*4 : y*bps+b*4+4]
				avg += uint32(r[0]) + uint32(r[1]) + uint32(r[2]) + uint32(r[3])
			}
			m += avg
			m2 += avg * avg
		}
	}
	if threshold*m2 < m*m {
		it.setIntra16Mode(0) // DC16
	} else {
		var modes [16]uint8 // DC4
		it.setIntra4Mode(&modes)
	}
	return 0
}

func (it *iterator) mbAnalyzeBestUVMode() int {
	bestAlpha := defaultAlpha
	smallestAlpha := 0
	bestMode := 0
	it.makeChroma8Preds()
	for mode := 0; mode < 2; mode++ {
		maxValue, lastNonZero := collectHistogram(it.yuvIn[uOff:], it.yuvP[vp8UVModeOffsets[mode]:], vp8ScanUV[:])
		alpha := 0
		if maxValue > 1 {
			alpha = alphaScale * lastNonZero / maxValue
		}
		if alpha > bestAlpha {
			bestAlpha = alpha
		}
		if mode == 0 || alpha < smallestAlpha {
			smallestAlpha = alpha
			bestMode = mode
		}
	}
	it.mb().uvMode = uint8(bestMode)
	return bestAlpha
}

type segmentJob struct {
	alphas          [maxAlphaValue + 1]int
	alpha, uvAlpha  int
	it              *iterator
	startRow, nRows int
}

func (j *segmentJob) run() {
	it := j.it
	e := it.enc
	it.setRow(j.startRow)
	it.setCountDown(j.nRows * e.mbW)
	if it.countDown <= 0 {
		return
	}
	for {
		it.importMB()
		mb := it.mb()
		it.setIntra16Mode(0)
		mb.skip = false
		mb.segment = 0
		var bestAlpha int
		if e.method <= 1 {
			bestAlpha = it.fastMBAnalyze()
		} else {
			bestAlpha = it.mbAnalyzeBestIntra16Mode()
		}
		bestUVAlpha := it.mbAnalyzeBestUVMode()
		bestAlpha = (3*bestAlpha + bestUVAlpha + 2) >> 2
		bestAlpha = clipInt(maxAlphaValue-bestAlpha, 0, maxAlphaValue)
		j.alphas[bestAlpha]++
		mb.alpha = uint8(bestAlpha)
		j.alpha += bestAlpha
		j.uvAlpha += bestUVAlpha
		if !it.next() {
			break
		}
	}
}

var sideIteratorPool = sync.Pool{New: func() any { return new(iterator) }}

// analyze decides the segmentation and complexity, and assigns a first
// guess of the prediction modes.
func (e *encoder) analyze() {
	doSegments := e.cfg.EmulateJPEGSize || e.numSegments > 1 || e.method <= 1
	if !doSegments {
		for i := range e.mbInfo {
			e.mbInfo[i] = mbInfo{typ: 1}
		}
		e.dqm[0].alpha = 0
		e.dqm[0].beta = 0
		e.alpha = 0
		e.uvAlpha = 0
		return
	}
	lastRow := e.mbH
	totalMB := lastRow * e.mbW
	splitRow := (9*lastRow + 15) >> 4
	doMT := e.cfg.ThreadLevel && splitRow >= 2 && splitRow < lastRow

	it := &e.it
	it.init(e)
	it.analysis = true
	var main segmentJob
	main.it = it
	if doMT {
		side := new(segmentJob)
		sideIt := sideIteratorPool.Get().(*iterator)
		sideIt.init(e)
		sideIt.analysis = true
		side.it = sideIt
		side.startRow, side.nRows = splitRow, lastRow-splitRow
		main.startRow, main.nRows = 0, splitRow
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			side.run()
		}()
		main.run()
		wg.Wait()
		sideIt.enc = nil
		sideIteratorPool.Put(sideIt)
		for i := range main.alphas {
			main.alphas[i] += side.alphas[i]
		}
		main.alpha += side.alpha
		main.uvAlpha += side.uvAlpha
	} else {
		main.startRow, main.nRows = 0, lastRow
		main.run()
	}
	it.analysis = false
	e.alpha = main.alpha / totalMB
	e.uvAlpha = main.uvAlpha / totalMB
	e.assignSegments(&main.alphas)
}
