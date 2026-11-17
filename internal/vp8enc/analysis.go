package vp8enc

import (
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

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
	tmp := bytePool.Get(w * h) // only the inner macroblocks are written and read
	defer bytePool.Put(tmp)
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
	q := 0 // a NaN quality converts to 0, as libwebp's (int) cast on arm64
	if qf := it.enc.cfg.Quality; qf == qf {
		q = int(qf)
	}
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

// segmentJob holds the analysis results of a band of macroblock rows.
type segmentJob struct {
	alphas          [maxAlphaValue + 1]int
	alpha, uvAlpha  int
	startRow, nRows int
}

// run analyzes the job's rows with it (an iterator in analysis mode). Every
// macroblock is analyzed from the source samples only, so bands are
// independent of each other and of the iterator used.
func (j *segmentJob) run(it *iterator) {
	e := it.enc
	j.alphas = [maxAlphaValue + 1]int{}
	j.alpha, j.uvAlpha = 0, 0
	it.setRow(j.startRow)
	it.setCountDown(j.nRows * e.mbW)
	if it.countDown <= 0 {
		return
	}
	for {
		if it.x == 0 && e.cfg.Interrupt != nil && e.pollInterrupt() {
			return
		}
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

const (
	// maxAnalysisBands bounds the number of row bands the analysis pass is
	// split into with ThreadLevel.
	maxAnalysisBands = 8
	// minBandMBs is the smallest band worth handing to a helper goroutine,
	// about 0.5 ms of analysis: methods 0-1 analyze a macroblock in about
	// 1 us, the others in about 2.7 us (Apple M5, source.jpg).
	minBandMBs     = 192
	minBandMBsFast = 512
)

var sideIteratorPool = sync.Pool{New: func() any { return new(iterator) }}

// testHookAnalysisBand, if set (tests only), is called before a band is
// analyzed, on the goroutine that analyzes it.
var testHookAnalysisBand func(band int)

// analysisBands returns the number of row bands of the analysis pass.
func (e *encoder) analysisBands() int {
	if !e.cfg.ThreadLevel {
		return 1
	}
	minMBs := minBandMBs
	if e.method <= 1 {
		minMBs = minBandMBsFast
	}
	return max(1, min(maxAnalysisBands, e.mbH, e.mbW*e.mbH/minMBs))
}

// pollInterrupt polls Config.Interrupt (not nil) from an analysis band,
// about once per macroblock row. Once it has fired, every band stops
// without polling it again.
func (e *encoder) pollInterrupt() bool {
	if e.stopped.Load() {
		return true
	}
	if e.cfg.Interrupt() {
		e.stopped.Store(true)
		return true
	}
	return false
}

// analyze decides the segmentation and complexity, and assigns a first
// guess of the prediction modes. It reports whether Config.Interrupt fired
// (the analysis is then incomplete and the encode must stop).
func (e *encoder) analyze() bool {
	doSegments := e.cfg.EmulateJPEGSize || e.numSegments > 1 || e.method <= 1
	if !doSegments {
		for i := range e.mbInfo {
			e.mbInfo[i] = mbInfo{typ: 1}
		}
		e.dqm[0].alpha = 0
		e.dqm[0].beta = 0
		e.alpha = 0
		e.uvAlpha = 0
		return false
	}
	e.stopped.Store(false)
	e.analyzeInBands(e.analysisBands())
	return e.stopped.Load()
}

// analyzeInBands runs the analysis pass in nb (1..min(maxAnalysisBands,
// mbH)) row bands and merges their results.
func (e *encoder) analyzeInBands(nb int) {
	totalMB := e.mbH * e.mbW
	it := &e.it
	it.init(e)
	it.analysis = true
	if nb == 1 {
		job := &e.bands[0]
		job.startRow, job.nRows = 0, e.mbH
		if testHookAnalysisBand != nil {
			testHookAnalysisBand(0)
		}
		job.run(it)
	} else {
		e.analyzeBands(nb)
	}
	it.analysis = false
	if e.stopped.Load() {
		return // interrupted: the bands are incomplete (and may be empty)
	}

	// Integer sums: the merge order does not matter.
	var alphas [maxAlphaValue + 1]int
	alpha, uvAlpha := 0, 0
	for b := range e.bands[:nb] {
		job := &e.bands[b]
		for i, n := range job.alphas {
			alphas[i] += n
		}
		alpha += job.alpha
		uvAlpha += job.uvAlpha
	}
	e.alpha = alpha / totalMB
	e.uvAlpha = uvAlpha / totalMB
	e.assignSegments(&alphas)
}

// analyzeBands analyzes nb row bands on the calling goroutine and on up to
// nb-1 helper goroutines taken from the process-wide budget
// (internal/workers); with no helper available, the caller analyzes all
// bands itself. Workers claim bands in order through e.nextBand. Every band
// writes only its own macroblocks and its own segmentJob, so the result
// does not depend on how many helpers ran or which band each one took.
func (e *encoder) analyzeBands(nb int) {
	for b := range e.bands[:nb] {
		job := &e.bands[b]
		job.startRow = e.mbH * b / nb
		job.nRows = e.mbH*(b+1)/nb - job.startRow
	}
	e.nextBand.Store(0)
	work := func(it *iterator) {
		for {
			b := int(e.nextBand.Add(1)) - 1
			if b >= nb || e.stopped.Load() {
				return
			}
			if testHookAnalysisBand != nil {
				testHookAnalysisBand(b)
			}
			e.bands[b].run(it)
		}
	}

	var (
		its   [maxAnalysisBands - 1]*iterator
		tasks [maxAnalysisBands - 1]*workers.Task
		n     int
	)
	defer func() {
		// If the caller panicked, let the helpers stop after their
		// current band. Wait for all of them before recycling their
		// iterators, then re-raise a helper panic, if any.
		e.nextBand.Store(int32(nb))
		for _, t := range tasks[:n] {
			<-t.Done()
		}
		for _, sit := range its[:n] {
			sit.enc = nil
			sideIteratorPool.Put(sit)
		}
		for _, t := range tasks[:n] {
			t.Wait()
		}
	}()
	for n < nb-1 {
		sit := sideIteratorPool.Get().(*iterator)
		sit.reset(e)
		sit.analysis = true
		t := workers.Spawn(func() error {
			work(sit)
			return nil
		})
		if t == nil { // budget exhausted
			sit.enc = nil
			sideIteratorPool.Put(sit)
			break
		}
		its[n], tasks[n] = sit, t
		n++
	}
	work(&e.it)
}
