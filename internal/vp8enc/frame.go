package vp8enc

import (
	"math"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Main coding loop, probabilities finalization and bitstream assembly.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	maxPartition0Size   = 1 << 19
	maxPartitionSize    = 1 << 24
	headerSizeEstimate  = 12 + 8 + 10 // RIFF + chunk header + VP8 frame header
	dqLimit             = 0.4
	partition0SizeLimit = uint64(maxPartition0Size-2048) << 11
	minCount            = 96
	skipProbaThreshold  = 250
)

type passStats struct {
	isFirst      bool
	dq           float32
	q, lastQ     float32
	qmin, qmax   float32
	value        float64
	lastValue    float64
	target       float64
	doSizeSearch bool
}

func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func (e *encoder) initPassStats(s *passStats) {
	s.isFirst = true
	s.dq = 10
	s.qmin = 0
	s.qmax = 100
	s.q = clampF(e.cfg.Quality, s.qmin, s.qmax)
	s.lastQ = s.q
	s.doSizeSearch = e.cfg.TargetSize != 0
	switch {
	case s.doSizeSearch:
		s.target = float64(e.cfg.TargetSize)
	case e.cfg.TargetPSNR > 0:
		s.target = float64(e.cfg.TargetPSNR)
	default:
		s.target = 40
	}
	s.value, s.lastValue = 0, 0
}

func (s *passStats) computeNextQ() float32 {
	var dq float32
	if s.isFirst {
		if s.value > s.target {
			dq = -s.dq
		} else {
			dq = s.dq
		}
		s.isFirst = false
	} else if s.value != s.lastValue {
		slope := (s.target - s.value) / (s.lastValue - s.value)
		dq = float32(slope * float64(s.lastQ-s.q))
	} else {
		dq = 0
	}
	s.dq = clampF(dq, -30, 30)
	s.lastQ = s.q
	s.lastValue = s.value
	s.q = clampF(s.q+s.dq, s.qmin, s.qmax)
	return s.q
}

func getProba(a, b int) uint8 {
	total := a + b
	if total == 0 {
		return 255
	}
	return uint8((255*a + total/2) / total)
}

func (e *encoder) setSegmentProbas() {
	var p [numMBSegments]int
	for i := range e.mbInfo {
		p[e.mbInfo[i].segment&3]++
	}
	if e.numSegments > 1 {
		probas := &e.proba.segments
		probas[0] = getProba(p[0]+p[1], p[2]+p[3])
		probas[1] = getProba(p[0], p[1])
		probas[2] = getProba(p[2], p[3])
		e.updateMap = probas[0] != 255 || probas[1] != 255 || probas[2] != 255
		if !e.updateMap {
			for i := range e.mbInfo {
				e.mbInfo[i].segment = 0
			}
		}
		e.segHdrSize = p[0]*(bitCost(0, probas[0])+bitCost(0, probas[1])) +
			p[1]*(bitCost(0, probas[0])+bitCost(1, probas[1])) +
			p[2]*(bitCost(1, probas[0])+bitCost(0, probas[2])) +
			p[3]*(bitCost(1, probas[0])+bitCost(1, probas[2]))
	} else {
		e.updateMap = false
		e.segHdrSize = 0
	}
}

func (e *encoder) setLoopParams(q float32) {
	q = clampF(q, 0, 100)
	e.setSegmentParams(float64(q))
	e.setSegmentProbas()
	e.proba.calculateLevelCosts()
	e.proba.nbSkip = 0
}

func getPSNR(mse, size uint64) float64 {
	if mse > 0 && size > 0 {
		return 10 * math.Log10(255*255*float64(size)/float64(mse))
	}
	return 99
}

// recordTokens records the coefficient tokens of the current macroblock.
func (it *iterator) recordTokens(rd *modeScore) {
	e := it.enc
	tb := &e.tokens
	stats := &e.proba.stats
	var res residual
	it.nzToBytes()
	if it.mb().typ == 1 { // i16x16
		ctx := it.topNz[8] + it.leftNz[8]
		res.first, res.ctype = 0, typeI16DC
		res.setCoeffs(&rd.yDCLevels)
		v := tb.recordCoeffTokens(ctx, &res, stats)
		it.topNz[8] = v
		it.leftNz[8] = v
		res.first, res.ctype = 1, typeI16AC
	} else {
		res.first, res.ctype = 0, typeI4AC
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			ctx := it.topNz[x] + it.leftNz[y]
			res.setCoeffs(&rd.yACLevels[x+y*4])
			v := tb.recordCoeffTokens(ctx, &res, stats)
			it.topNz[x] = v
			it.leftNz[y] = v
		}
	}
	res.first, res.ctype = 0, typeChromaA
	for ch := 0; ch <= 2; ch += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				ctx := it.topNz[4+ch+x] + it.leftNz[4+ch+y]
				res.setCoeffs(&rd.uvLevels[ch*2+x+y*2])
				v := tb.recordCoeffTokens(ctx, &res, stats)
				it.topNz[4+ch+x] = v
				it.leftNz[4+ch+y] = v
			}
		}
	}
	it.bytesToNz()
}

// tokenLoop runs the coding passes, recording all tokens.
func (e *encoder) tokenLoop() error {
	nbMBs := e.mbW * e.mbH
	maxCount := max(nbMBs>>3, minCount)
	numPassLeft := e.cfg.Pass
	rdOpt := e.rdOptLevel
	pixelCount := uint64(nbMBs) * 384
	it := &e.it
	var stats passStats
	e.initPassStats(&stats)

	for numPassLeft > 0 {
		numPassLeft--
		isLastPass := math.Abs(float64(stats.dq)) <= dqLimit || numPassLeft == 0 || e.maxI4HeaderBits == 0
		var sizeP0, distortion uint64
		cnt := maxCount
		it.init(e)
		e.setLoopParams(stats.q)
		if isLastPass {
			e.proba.stats = [numTypes]statsArray{}
			e.initFilter()
		}
		e.tokens.reset()
		rd := &it.rd
		for {
			if it.x == 0 && e.cfg.interrupted() {
				return ErrInterrupted
			}
			it.importMB()
			if rdOpt > rdOptNone {
				cnt--
				if cnt < 0 {
					e.proba.finalizeTokenProbas()
					e.proba.calculateLevelCosts()
					cnt = maxCount
				}
			}
			it.decimate(rd, rdOpt)
			e.mbTokStart[it.mbIdx] = int32(e.tokens.len())
			it.recordTokens(rd)
			sizeP0 += uint64(rd.H)
			distortion += uint64(rd.D)
			if isLastPass {
				e.storeFilterStats(it)
			}
			it.saveBoundary()
			if !it.next() {
				break
			}
		}
		e.mbTokStart[nbMBs] = int32(e.tokens.len())
		sizeP0 += uint64(e.segHdrSize)
		if stats.doSizeSearch {
			size := uint64(e.proba.finalizeTokenProbas())
			flattenProbas(&e.flat, &e.proba.coeffs)
			size += e.tokens.estimateSize(&e.flat)
			size = (size + sizeP0 + 1024) >> 11
			size += headerSizeEstimate
			stats.value = float64(size)
		} else {
			stats.value = getPSNR(distortion, pixelCount)
		}
		if e.maxI4HeaderBits > 0 && sizeP0 > partition0SizeLimit {
			numPassLeft++
			e.maxI4HeaderBits >>= 1 // strengthen header bit limitation...
			continue                // ...and start over
		}
		if isLastPass {
			break
		}
		if e.doSearch {
			stats.computeNextQ()
		}
	}

	// Skip probability: only use it when it saves bits.
	nbSkip := 0
	for i := range e.mbInfo {
		if e.mbInfo[i].skip {
			nbSkip++
		}
	}
	e.proba.nbSkip = nbSkip
	skipProba := 255
	if nbMBs > 0 {
		skipProba = (nbMBs - nbSkip) * 255 / nbMBs
	}
	e.proba.skipProba = uint8(skipProba)
	e.proba.useSkipProba = skipProba < skipProbaThreshold
	if e.proba.useSkipProba {
		// recompute the statistics without the skipped macroblocks
		e.proba.stats = [numTypes]statsArray{}
		for i := 0; i < nbMBs; i++ {
			if !e.mbInfo[i].skip {
				e.tokens.recordStatsRange(int(e.mbTokStart[i]), int(e.mbTokStart[i+1]), &e.proba.stats)
			}
		}
	}
	e.proba.finalizeTokenProbas()
	e.adjustFilterStrength()
	return nil
}

//------------------------------------------------------------------------------
// Bitstream writing

func putI4Mode(bw *bitWriter, mode int, prob *[numBModes - 1]uint8) int {
	if bw.putBit(mode != bDCPred, int(prob[0])) {
		if bw.putBit(mode != bTMPred, int(prob[1])) {
			if bw.putBit(mode != bVEPred, int(prob[2])) {
				if !bw.putBit(mode >= bLDPred, int(prob[3])) {
					if bw.putBit(mode != bHEPred, int(prob[4])) {
						bw.putBit(mode != bRDPred, int(prob[5]))
					}
				} else {
					if bw.putBit(mode != bLDPred, int(prob[6])) {
						if bw.putBit(mode != bVLPred, int(prob[7])) {
							bw.putBit(mode != bHDPred, int(prob[8]))
						}
					}
				}
			}
		}
	}
	return mode
}

// Intra4 modes (libwebp numbering).
const (
	bDCPred = iota
	bTMPred
	bVEPred
	bHEPred
	bRDPred
	bVRPred
	bLDPred
	bVLPred
	bHDPred
	bHUPred
)

func putI16Mode(bw *bitWriter, mode int) {
	if bw.putBit(mode == predTM || mode == predHE, 156) {
		bw.putBit(mode == predTM, 128) // TM or HE
	} else {
		bw.putBit(mode == predVE, 163) // VE or DC
	}
}

func putUVMode(bw *bitWriter, uvMode int) {
	if bw.putBit(uvMode != predDC, 142) {
		if bw.putBit(uvMode != predVE, 114) {
			bw.putBit(uvMode != predHE, 183) // else: TM_PRED
		}
	}
}

func putSegment(bw *bitWriter, s int, p *[3]uint8) {
	if bw.putBit(s >= 2, int(p[0])) {
		bw.putBit(s&1 != 0, int(p[2]))
	} else {
		bw.putBit(s&1 != 0, int(p[1]))
	}
}

func (e *encoder) codeIntraModes(bw *bitWriter) {
	predsW := e.predsW
	for my := 0; my < e.mbH; my++ {
		for mx := 0; mx < e.mbW; mx++ {
			mb := &e.mbInfo[my*e.mbW+mx]
			p := predsW + 1 + my*4*predsW + mx*4
			if e.updateMap {
				putSegment(bw, int(mb.segment), &e.proba.segments)
			}
			if e.proba.useSkipProba {
				bw.putBit(mb.skip, int(e.proba.skipProba))
			}
			if bw.putBit(mb.typ != 0, 145) { // i16x16
				putI16Mode(bw, int(e.preds[p]))
			} else {
				topPred := p - predsW
				for y := 0; y < 4; y++ {
					left := int(e.preds[p-1])
					for x := 0; x < 4; x++ {
						probas := &bModesProba[e.preds[topPred+x]%numBModes][left%numBModes]
						left = putI4Mode(bw, int(e.preds[p+x]%numBModes), probas)
					}
					topPred = p
					p += predsW
				}
			}
			putUVMode(bw, int(mb.uvMode))
		}
	}
}

func (e *encoder) writeProbas(bw *bitWriter) {
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					p0 := e.proba.coeffs[t][b][c][p]
					update := p0 != coeffsProba0[t][b][c][p]
					if bw.putBit(update, int(coeffsUpdateProba[t][b][c][p])) {
						bw.putBits(uint32(p0), 8)
					}
				}
			}
		}
	}
	if bw.putBitUniform(e.proba.useSkipProba) {
		bw.putBits(uint32(e.proba.skipProba), 8)
	}
}

func (e *encoder) putSegmentHeader(bw *bitWriter) {
	if bw.putBitUniform(e.numSegments > 1) {
		bw.putBitUniform(e.updateMap)
		if bw.putBitUniform(true) { // update_data
			bw.putBitUniform(true) // absolute values
			for s := 0; s < numMBSegments; s++ {
				bw.putSignedBits(e.dqm[s].quant, 7)
			}
			for s := 0; s < numMBSegments; s++ {
				bw.putSignedBits(e.dqm[s].fstrength, 6)
			}
		}
		if e.updateMap {
			for s := 0; s < 3; s++ {
				if bw.putBitUniform(e.proba.segments[s] != 255) {
					bw.putBits(uint32(e.proba.segments[s]), 8)
				}
			}
		}
	}
}

func (e *encoder) putFilterHeader(bw *bitWriter) {
	bw.putBitUniform(e.filterSimple)
	bw.putBits(uint32(e.filterLevel), 6)
	bw.putBits(uint32(e.filterSharpness), 3)
	bw.putBitUniform(false) // no lf-delta
}

func (e *encoder) putQuant(bw *bitWriter) {
	bw.putBits(uint32(e.baseQuant), 7)
	bw.putSignedBits(e.dqY1DC, 4)
	bw.putSignedBits(e.dqY2DC, 4)
	bw.putSignedBits(e.dqY2AC, 4)
	bw.putSignedBits(e.dqUVDC, 4)
	bw.putSignedBits(e.dqUVAC, 4)
}

func (e *encoder) generatePartition0() []byte {
	bw := &e.bw0
	bw.init(e.mbW*e.mbH*7/8 + 1024)
	bw.putBitUniform(false) // colorspace
	bw.putBitUniform(false) // clamp type
	e.putSegmentHeader(bw)
	e.putFilterHeader(bw)
	var np uint32
	switch e.numParts {
	case 8:
		np = 3
	case 4:
		np = 2
	case 2:
		np = 1
	}
	bw.putBits(np, 2)
	e.putQuant(bw)
	bw.putBitUniform(false) // no proba update
	e.writeProbas(bw)
	e.codeIntraModes(bw)
	return bw.finish()
}

// write assembles the frame after headroom bytes left for the caller.
func (e *encoder) write(headroom int) ([]byte, error) {
	// token partitions
	flattenProbas(&e.flat, &e.proba.coeffs)
	nbMBs := e.mbW * e.mbH
	estimate := e.tokens.len()/(6*e.numParts) + 1024
	for p := 0; p < e.numParts; p++ {
		e.parts[p].init(estimate)
	}
	mask := e.numParts - 1
	for i := 0; i < nbMBs; i++ {
		if e.proba.useSkipProba && e.mbInfo[i].skip {
			continue
		}
		y := i / e.mbW
		e.tokens.emitTokenRange(&e.parts[y&mask], int(e.mbTokStart[i]), int(e.mbTokStart[i+1]), &e.flat)
	}
	var parts [8][]byte
	total := 0
	for p := 0; p < e.numParts; p++ {
		parts[p] = e.parts[p].finish()
		if p < e.numParts-1 && len(parts[p]) >= maxPartitionSize {
			return nil, status.InvalidParam("vp8enc: token partition overflow")
		}
		total += len(parts[p])
	}
	part0 := e.generatePartition0()
	size0 := len(part0)
	if size0 >= maxPartition0Size {
		return nil, status.InvalidParam("vp8enc: partition #0 overflow (try a higher PartitionLimit)")
	}
	total += 10 + size0 + 3*(e.numParts-1)
	// One spare byte for the RIFF padding byte that a writer framing the
	// payload in place (out[:headroom]) appends to an odd-sized payload.
	out := make([]byte, headroom, headroom+total+1)
	bits := uint32(0) | uint32(e.profile)<<1 | 1<<4 | uint32(size0)<<5
	out = append(out, byte(bits), byte(bits>>8), byte(bits>>16), 0x9d, 0x01, 0x2a,
		byte(e.pic.Width), byte(e.pic.Width>>8), byte(e.pic.Height), byte(e.pic.Height>>8))
	out = append(out, part0...)
	for p := 0; p < e.numParts-1; p++ {
		s := len(parts[p])
		out = append(out, byte(s), byte(s>>8), byte(s>>16))
	}
	for p := 0; p < e.numParts; p++ {
		out = append(out, parts[p]...)
	}
	return out, nil
}
