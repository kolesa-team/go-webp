package vp8enc

import "math"

// Quantization setup, mode decision and reconstruction.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	midAlpha = 64  // neutral value for susceptibility
	minAlpha = 30  // lowest usable value for susceptibility
	maxAlpha = 100 // higher meaningful value for susceptibility

	snsToDQ = 0.9 // scaling constant between the sns value and the QP power-law modulation

	flatnessLimitI16 = 0
	flatnessLimitI4  = 3
	flatnessLimitUV  = 2
	flatnessPenalty  = 140

	rdDistoMult = 256

	maxDQUV = 6
	minDQUV = -4

	fstrengthCutoff = 2

	maxCost = int64(0x7fffffffffffff)
)

// Rate-distortion optimization levels.
const (
	rdOptNone       = 0
	rdOptBasic      = 1
	rdOptTrellis    = 2
	rdOptTrellisAll = 3
)

var kDcTable = [128]uint8{
	4, 5, 6, 7, 8, 9, 10, 10,
	11, 12, 13, 14, 15, 16, 17, 17,
	18, 19, 20, 20, 21, 21, 22, 22,
	23, 23, 24, 25, 25, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36,
	37, 37, 38, 39, 40, 41, 42, 43,
	44, 45, 46, 46, 47, 48, 49, 50,
	51, 52, 53, 54, 55, 56, 57, 58,
	59, 60, 61, 62, 63, 64, 65, 66,
	67, 68, 69, 70, 71, 72, 73, 74,
	75, 76, 76, 77, 78, 79, 80, 81,
	82, 83, 84, 85, 86, 87, 88, 89,
	91, 93, 95, 96, 98, 100, 101, 102,
	104, 106, 108, 110, 112, 114, 116, 118,
	122, 124, 126, 128, 130, 132, 134, 136,
	138, 140, 143, 145, 148, 151, 154, 157,
}

var kAcTable = [128]uint16{
	4, 5, 6, 7, 8, 9, 10, 11,
	12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 24, 25, 26, 27,
	28, 29, 30, 31, 32, 33, 34, 35,
	36, 37, 38, 39, 40, 41, 42, 43,
	44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 60,
	62, 64, 66, 68, 70, 72, 74, 76,
	78, 80, 82, 84, 86, 88, 90, 92,
	94, 96, 98, 100, 102, 104, 106, 108,
	110, 112, 114, 116, 119, 122, 125, 128,
	131, 134, 137, 140, 143, 146, 149, 152,
	155, 158, 161, 164, 167, 170, 173, 177,
	181, 185, 189, 193, 197, 201, 205, 209,
	213, 217, 221, 225, 229, 234, 239, 245,
	249, 254, 259, 264, 269, 274, 279, 284,
}

var kAcTable2 = [128]uint16{
	8, 8, 9, 10, 12, 13, 15, 17,
	18, 20, 21, 23, 24, 26, 27, 29,
	31, 32, 34, 35, 37, 38, 40, 41,
	43, 44, 46, 48, 49, 51, 52, 54,
	55, 57, 58, 60, 62, 63, 65, 66,
	68, 69, 71, 72, 74, 75, 77, 79,
	80, 82, 83, 85, 86, 88, 89, 93,
	96, 99, 102, 105, 108, 111, 114, 117,
	120, 124, 127, 130, 133, 136, 139, 142,
	145, 148, 151, 155, 158, 161, 164, 167,
	170, 173, 176, 179, 184, 189, 193, 198,
	203, 207, 212, 217, 221, 226, 230, 235,
	240, 244, 249, 254, 258, 263, 268, 274,
	280, 286, 292, 299, 305, 311, 317, 323,
	330, 336, 342, 348, 354, 362, 370, 379,
	385, 393, 401, 409, 416, 424, 432, 440,
}

// [luma-ac,luma-dc,chroma][dc,ac]
var kBiasMatrices = [3][2]uint32{{96, 110}, {96, 108}, {110, 115}}

const sharpenBits = 11

var kFreqSharpening = [16]uint32{
	0, 30, 60, 90,
	30, 60, 90, 90,
	60, 90, 90, 90,
	90, 90, 90, 90,
}

var kWeightY = [16]uint16{38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2}

var kWeightTrellis = [16]int32{
	30, 27, 19, 11,
	27, 24, 17, 10,
	19, 17, 12, 8,
	11, 10, 8, 6,
}

func clipInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// segmentInfo holds the per-segment quantization parameters.
type segmentInfo struct {
	y1, y2, uv matrix
	alpha      int // quant-susceptibility, range [-127,127]
	beta       int // filter-susceptibility, range [0,255]
	quant      int // final segment quantizer
	fstrength  int // final in-loop filtering strength
	maxEdge    int // max edge delta (for filtering strength)
	minDisto   int // minimum distortion required to trigger filtering record

	lambdaI16, lambdaI4, lambdaUV                      int
	lambdaMode, tlambda                                int
	lambdaTrellisI16, lambdaTrellisI4, lambdaTrellisUV int

	i4Penalty int64 // penalty for using Intra4
}

func expandMatrix(m *matrix, typ int) int {
	for i := 0; i < 2; i++ {
		isAC := 0
		if i > 0 {
			isAC = 1
		}
		bias := kBiasMatrices[typ][isAC]
		m.iq[i] = (1 << qfix) / m.q[i]
		m.bias[i] = biasOf(bias)
		m.zthresh[i] = ((1 << qfix) - 1 - m.bias[i]) / m.iq[i]
	}
	for i := 2; i < 16; i++ {
		m.q[i] = m.q[1]
		m.iq[i] = m.iq[1]
		m.bias[i] = m.bias[1]
		m.zthresh[i] = m.zthresh[1]
	}
	sum := uint32(0)
	for i := 0; i < 16; i++ {
		if typ == 0 {
			m.sharpen[i] = (kFreqSharpening[i] * m.q[i]) >> sharpenBits
		} else {
			m.sharpen[i] = 0
		}
		sum += m.q[i]
	}
	return int((sum + 8) >> 4)
}

func atLeast1(v int) int {
	if v < 1 {
		return 1
	}
	return v
}

func (e *encoder) setupMatrices() {
	tlambdaScale := 0
	if e.method >= 4 {
		tlambdaScale = e.cfg.SNSStrength
	}
	for i := 0; i < e.numSegments; i++ {
		m := &e.dqm[i]
		q := m.quant
		m.y1.q[0] = uint32(kDcTable[clipInt(q+e.dqY1DC, 0, 127)])
		m.y1.q[1] = uint32(kAcTable[clipInt(q, 0, 127)])

		m.y2.q[0] = uint32(kDcTable[clipInt(q+e.dqY2DC, 0, 127)]) * 2
		m.y2.q[1] = uint32(kAcTable2[clipInt(q+e.dqY2AC, 0, 127)])

		m.uv.q[0] = uint32(kDcTable[clipInt(q+e.dqUVDC, 0, 117)])
		m.uv.q[1] = uint32(kAcTable[clipInt(q+e.dqUVAC, 0, 127)])

		qI4 := expandMatrix(&m.y1, 0)
		qI16 := expandMatrix(&m.y2, 1)
		qUV := expandMatrix(&m.uv, 2)

		m.lambdaI4 = atLeast1((3 * qI4 * qI4) >> 7)
		m.lambdaI16 = atLeast1(3 * qI16 * qI16)
		m.lambdaUV = atLeast1((3 * qUV * qUV) >> 6)
		m.lambdaMode = atLeast1((1 * qI4 * qI4) >> 7)
		m.lambdaTrellisI4 = atLeast1((7 * qI4 * qI4) >> 3)
		m.lambdaTrellisI16 = atLeast1((qI16 * qI16) >> 2)
		m.lambdaTrellisUV = atLeast1((qUV * qUV) << 1)
		m.tlambda = atLeast1((tlambdaScale * qI4) >> 5)

		m.minDisto = 20 * int(m.y1.q[0])
		m.maxEdge = 0
		m.i4Penalty = 1000 * int64(qI4) * int64(qI4)
	}
}

func filterStrengthFromDelta(sharpness, delta int) int {
	pos := delta
	if pos >= 64 {
		pos = 63
	}
	if pos < 0 {
		pos = 0
	}
	return int(levelsFromDelta[sharpness&7][pos])
}

func (e *encoder) setupFilterStrength() {
	level0 := 5 * e.cfg.FilterStrength
	for i := 0; i < numMBSegments; i++ {
		m := &e.dqm[i]
		qstep := int(kAcTable[clipInt(m.quant, 0, 127)]) >> 2
		baseStrength := filterStrengthFromDelta(e.filterSharpness, qstep)
		f := baseStrength * level0 / (256 + m.beta)
		if f < fstrengthCutoff {
			m.fstrength = 0
		} else if f > 63 {
			m.fstrength = 63
		} else {
			m.fstrength = f
		}
	}
	e.filterLevel = e.dqm[0].fstrength
	e.filterSimple = e.cfg.FilterType == 0
	e.filterSharpness = e.cfg.FilterSharpness
}

func qualityToCompression(c float64) float64 {
	linearC := 2*c - 1
	if c < 0.75 {
		linearC = c * (2. / 3.)
	}
	return math.Pow(linearC, 1/3.)
}

func qualityToJPEGCompression(c, alpha float64) float64 {
	const (
		amin   = 0.30
		amax   = 0.85
		expMin = 0.4
		expMax = 0.9
		slope  = (expMin - expMax) / (amax - amin)
	)
	expn := expMax + slope*(alpha-amin)
	if alpha > amax {
		expn = expMin
	} else if alpha < amin {
		expn = expMax
	}
	return math.Pow(c, expn)
}

func (e *encoder) simplifySegments() {
	segMap := [numMBSegments]uint8{0, 1, 2, 3}
	numSegments := e.numSegments
	numFinal := 1
	for s1 := 1; s1 < numSegments; s1++ {
		S1 := &e.dqm[s1]
		found := false
		s2 := 0
		for ; s2 < numFinal; s2++ {
			S2 := &e.dqm[s2]
			if S1.quant == S2.quant && S1.fstrength == S2.fstrength {
				found = true
				break
			}
		}
		segMap[s1] = uint8(s2)
		if !found {
			if numFinal != s1 {
				e.dqm[numFinal] = e.dqm[s1]
			}
			numFinal++
		}
	}
	if numFinal < numSegments {
		for i := range e.mbInfo {
			e.mbInfo[i].segment = segMap[e.mbInfo[i].segment&3]
		}
		e.numSegments = numFinal
		for i := numFinal; i < numSegments; i++ {
			e.dqm[i] = e.dqm[numFinal-1]
		}
	}
}

// setSegmentParams sets up the segments' quantization values, base_quant and
// filter strengths for the given quality.
func (e *encoder) setSegmentParams(quality float64) {
	numSegments := e.numSegments
	amp := snsToDQ * float64(e.cfg.SNSStrength) / 100. / 128.
	Q := quality / 100.
	var cBase float64
	if e.cfg.EmulateJPEGSize {
		cBase = qualityToJPEGCompression(Q, float64(e.alpha)/255.)
	} else {
		cBase = qualityToCompression(Q)
	}
	for i := 0; i < numSegments; i++ {
		expn := 1. - amp*float64(e.dqm[i].alpha)
		c := math.Pow(cBase, expn)
		// A NaN quality yields NaN here; libwebp's (int) cast of it ends up
		// as quantizer 0 after clipping (0 on arm64, INT_MIN on x86), so
		// make that explicit instead of relying on Go's platform-dependent
		// NaN conversion.
		q := 0
		if f := 127. * (1. - c); f == f {
			q = int(f)
		}
		e.dqm[i].quant = clipInt(q, 0, 127)
	}
	e.baseQuant = e.dqm[0].quant
	for i := numSegments; i < numMBSegments; i++ {
		e.dqm[i].quant = e.baseQuant
	}

	dqUVAC := (e.uvAlpha - midAlpha) * (maxDQUV - minDQUV) / (maxAlpha - minAlpha)
	dqUVAC = dqUVAC * e.cfg.SNSStrength / 100
	dqUVAC = clipInt(dqUVAC, minDQUV, maxDQUV)
	dqUVDC := -4 * e.cfg.SNSStrength / 100
	dqUVDC = clipInt(dqUVDC, -15, 15)

	e.dqY1DC = 0
	e.dqY2DC = 0
	e.dqY2AC = 0
	e.dqUVDC = dqUVDC
	e.dqUVAC = dqUVAC

	e.setupFilterStrength()
	if numSegments > 1 {
		e.simplifySegments()
	}
	e.setupMatrices()
}

//------------------------------------------------------------------------------
// Mode scores

type modeScore struct {
	D, SD, H, R, score int64
	nz                 uint32
	modeI16            int
	modeUV             int
	modesI4            [16]uint8
	derr               [2][3]int8
	yDCLevels          [16]int16
	yACLevels          [16][16]int16
	uvLevels           [8][16]int16
}

func (rd *modeScore) init() {
	rd.D, rd.SD, rd.R, rd.H, rd.nz = 0, 0, 0, 0, 0
	rd.score = maxCost
}

func (rd *modeScore) setRDScore(lambda int) {
	rd.score = (rd.R+rd.H)*int64(lambda) + rdDistoMult*(rd.D+rd.SD)
}

// scoreT is the score-only part of a modeScore.
type scoreT struct {
	D, SD, H, R, score int64
	nz                 uint32
}

func (s *scoreT) setRDScore(lambda int) {
	s.score = (s.R+s.H)*int64(lambda) + rdDistoMult*(s.D+s.SD)
}

func mult8b(a, b int) int64 { return int64((a*b + 128) >> 8) }

//------------------------------------------------------------------------------
// Trellis-optimized quantization

type trellisNode struct {
	prev  int8
	sign  bool
	level int16
}

type scoreState struct {
	score int64
	costs *levelCostTable
}

func rdScoreTrellis(lambda int, rate, distortion int64) int64 {
	return rate*int64(lambda) + rdDistoMult*distortion
}

func (it *iterator) trellisQuantizeBlock(in *[16]int16, out *[16]int16, ctx0 int, coeffType int, mtx *matrix, lambda int) bool {
	const numNodes = 2 // MIN_DELTA = 0, MAX_DELTA = 1
	e := it.enc
	probas := &e.proba.coeffs[coeffType]
	costs := &e.proba.levelCost[coeffType]
	first := 0
	if coeffType == typeI16AC {
		first = 1
	}
	var nodes [16][numNodes]trellisNode
	var states [2][numNodes]scoreState
	ssCur := &states[0]
	ssPrev := &states[1]
	bestPath := [3]int{-1, -1, -1}
	var bestScore int64
	last := first - 1
	{
		thresh := int32(mtx.q[1] * mtx.q[1] / 4)
		lastProba := probas[encBands[first]][ctx0][0]
		for n := 15; n >= first; n-- {
			j := kZigzag[n] & 15
			err := int32(in[j]) * int32(in[j])
			if err > thresh {
				last = n
				break
			}
		}
		if last < 15 {
			last++
		}
		cost := int64(bitCost(0, lastProba))
		bestScore = rdScoreTrellis(lambda, cost, 0)
		rate := int64(0)
		if ctx0 == 0 {
			rate = int64(bitCost(1, lastProba))
		}
		for m := 0; m < numNodes; m++ {
			ssCur[m].score = rdScoreTrellis(lambda, rate, 0)
			ssCur[m].costs = &costs[encBands[first]][ctx0]
		}
	}

	for n := first; n <= last; n++ {
		j := kZigzag[n] & 15
		Q := mtx.q[j]
		iQ := mtx.iq[j]
		B := biasOf(0x00)
		v := int32(in[j])
		sign := v < 0
		if sign {
			v = -v
		}
		coeff0 := uint32(v) + mtx.sharpen[j]
		level0 := int(quantDiv(coeff0, iQ, B))
		threshLevel := int(quantDiv(coeff0, iQ, biasOf(0x80)))
		if threshLevel > maxLevel {
			threshLevel = maxLevel
		}
		if level0 > maxLevel {
			level0 = maxLevel
		}
		ssCur, ssPrev = ssPrev, ssCur

		band := encBands[n+1]
		for m := 0; m < numNodes; m++ {
			cur := &nodes[n][m]
			level := level0 + m
			ctx := level
			if ctx > 2 {
				ctx = 2
			}
			ssCur[m].costs = &costs[band][ctx]
			if level < 0 || level > threshLevel {
				ssCur[m].score = maxCost
				continue
			}
			var baseScore int64
			{
				newError := int64(coeff0) - int64(level)*int64(Q)
				deltaError := int64(kWeightTrellis[j]) * (newError*newError - int64(coeff0)*int64(coeff0))
				baseScore = rdScoreTrellis(lambda, 0, deltaError)
			}
			cost := int64(levelCost(ssPrev[0].costs, level))
			bestCurScore := ssPrev[0].score + rdScoreTrellis(lambda, cost, 0)
			bestPrev := 0
			{
				cost = int64(levelCost(ssPrev[1].costs, level))
				score := ssPrev[1].score + rdScoreTrellis(lambda, cost, 0)
				if score < bestCurScore {
					bestCurScore = score
					bestPrev = 1
				}
			}
			bestCurScore += baseScore
			cur.sign = sign
			cur.level = int16(level)
			cur.prev = int8(bestPrev)
			ssCur[m].score = bestCurScore

			if level != 0 && bestCurScore < bestScore {
				lastPosCost := int64(0)
				if n < 15 {
					lastPosCost = int64(bitCost(0, probas[band][ctx][0]))
				}
				score := bestCurScore + rdScoreTrellis(lambda, lastPosCost, 0)
				if score < bestScore {
					bestScore = score
					bestPath[0] = n
					bestPath[1] = m
					bestPath[2] = bestPrev
				}
			}
		}
	}

	if coeffType == typeI16AC {
		for i := 1; i < 16; i++ {
			in[i] = 0
			out[i] = 0
		}
	} else {
		*in = [16]int16{}
		*out = [16]int16{}
	}
	if bestPath[0] == -1 {
		return false
	}
	nz := 0
	bestNode := bestPath[1]
	n := bestPath[0]
	nodes[n][bestNode].prev = int8(bestPath[2])
	for ; n >= first; n-- {
		node := &nodes[n][bestNode&1]
		j := kZigzag[n] & 15
		l := node.level
		if node.sign {
			l = -l
		}
		out[n] = l
		nz |= int(node.level)
		in[j] = int16(int32(l) * int32(mtx.q[j]))
		bestNode = int(node.prev)
	}
	return nz != 0
}

//------------------------------------------------------------------------------
// Reconstruction

func (it *iterator) reconstructIntra16(dcLevels *[16]int16, acLevels *[16][16]int16, yuvOut []byte, mode int) uint32 {
	e := it.enc
	ref := it.yuvP[vp8I16ModeOffsets[mode&3]:]
	src := it.yuvIn[yOff:]
	dqm := &e.dqm[it.mb().segment&3]
	nz := uint32(0)
	tmp := &it.tmp16
	var dcTmp [16]int16

	for n := 0; n < 16; n++ {
		off := vp8Scan[n]
		fTransform(src[off:], ref[off:], &tmp[n])
	}
	fTransformWHT(tmp, &dcTmp)
	if quantizeBlock(&dcTmp, dcLevels, &dqm.y2) {
		nz |= 1 << 24
	}

	if it.doTrellis {
		it.nzToBytes()
		for y, n := 0, 0; y < 4; y++ {
			for x := 0; x < 4; x, n = x+1, n+1 {
				ctx := it.topNz[x] + it.leftNz[y]
				nonZero := it.trellisQuantizeBlock(&tmp[n], &acLevels[n], ctx, typeI16AC, &dqm.y1, dqm.lambdaTrellisI16)
				v := b2i(nonZero)
				it.topNz[x] = v
				it.leftNz[y] = v
				acLevels[n][0] = 0
				nz |= uint32(v) << uint(n)
			}
		}
	} else {
		for n := 0; n < 16; n++ {
			tmp[n][0] = 0
			if quantizeBlock(&tmp[n], &acLevels[n], &dqm.y1) {
				nz |= 1 << uint(n)
			}
		}
	}

	iTransformWHT(&dcTmp, tmp)
	for n := 0; n < 16; n++ {
		off := vp8Scan[n]
		iTransform(ref[off:], &tmp[n], yuvOut[off:])
	}
	return nz
}

func (it *iterator) reconstructIntra4(levels *[16]int16, src, yuvOut []byte, mode int) bool {
	ref := it.yuvP[vp8I4ModeOffsets[mode]:]
	dqm := &it.enc.dqm[it.mb().segment&3]
	var tmp [16]int16
	fTransform(src, ref, &tmp)
	var nz bool
	if it.doTrellis {
		x, y := it.i4&3, it.i4>>2
		ctx := it.topNz[x] + it.leftNz[y]
		nz = it.trellisQuantizeBlock(&tmp, levels, ctx, typeI4AC, &dqm.y1, dqm.lambdaTrellisI4)
	} else {
		nz = quantizeBlock(&tmp, levels, &dqm.y1)
	}
	iTransform(ref, &tmp, yuvOut)
	return nz
}

// DC-error diffusion constants.
const (
	derrC1     = 7 // fraction of error sent to the 4x4 block below
	derrC2     = 8 // fraction of error sent to the 4x4 block on the right
	derrDShift = 4
	derrDScale = 1 // storage descaling, needed to make the error fit int8
)

func quantizeSingle(v *int16, mtx *matrix) int {
	V := int(*v)
	sign := V < 0
	if sign {
		V = -V
	}
	if V > int(mtx.zthresh[0]) {
		qV := int(quantDiv(uint32(V), mtx.iq[0], mtx.bias[0])) * int(mtx.q[0])
		err := V - qV
		if sign {
			*v = int16(-qV)
			return (-err) >> derrDScale
		}
		*v = int16(qV)
		return err >> derrDScale
	}
	*v = 0
	if sign {
		return (-V) >> derrDScale
	}
	return V >> derrDScale
}

func (it *iterator) correctDCValues(mtx *matrix, tmp *[8][16]int16, derr *[2][3]int8) {
	for ch := 0; ch <= 1; ch++ {
		top := &it.enc.topDerr[it.x][ch]
		left := &it.leftDerr[ch]
		c := tmp[ch*4 : ch*4+4 : ch*4+4]
		c[0][0] += int16((derrC1*int(top[0]) + derrC2*int(left[0])) >> (derrDShift - derrDScale))
		err0 := quantizeSingle(&c[0][0], mtx)
		c[1][0] += int16((derrC1*int(top[1]) + derrC2*err0) >> (derrDShift - derrDScale))
		err1 := quantizeSingle(&c[1][0], mtx)
		c[2][0] += int16((derrC1*err0 + derrC2*int(left[1])) >> (derrDShift - derrDScale))
		err2 := quantizeSingle(&c[2][0], mtx)
		c[3][0] += int16((derrC1*err1 + derrC2*err2) >> (derrDShift - derrDScale))
		err3 := quantizeSingle(&c[3][0], mtx)
		derr[ch][0] = int8(err1)
		derr[ch][1] = int8(err2)
		derr[ch][2] = int8(err3)
	}
}

func (it *iterator) storeDiffusionErrors(derr *[2][3]int8) {
	for ch := 0; ch <= 1; ch++ {
		top := &it.enc.topDerr[it.x][ch]
		left := &it.leftDerr[ch]
		left[0] = derr[ch][0]
		left[1] = int8((3 * int(derr[ch][2])) >> 2)
		top[0] = derr[ch][1]
		top[1] = derr[ch][2] - left[1]
	}
}

func (it *iterator) reconstructUV(levels *[8][16]int16, derr *[2][3]int8, yuvOut []byte, mode int) uint32 {
	e := it.enc
	ref := it.yuvP[vp8UVModeOffsets[mode&3]:]
	src := it.yuvIn[uOff:]
	dqm := &e.dqm[it.mb().segment&3]
	nz := uint32(0)
	tmp := &it.tmpUV
	for n := 0; n < 8; n++ {
		off := vp8ScanUV[n]
		fTransform(src[off:], ref[off:], &tmp[n])
	}
	if e.topDerr != nil {
		it.correctDCValues(&dqm.uv, tmp, derr)
	}
	for n := 0; n < 8; n++ {
		if quantizeBlock(&tmp[n], &levels[n], &dqm.uv) {
			nz |= 1 << uint(n)
		}
	}
	for n := 0; n < 8; n++ {
		off := vp8ScanUV[n]
		iTransform(ref[off:], &tmp[n], yuvOut[off:])
	}
	return nz << 16
}

//------------------------------------------------------------------------------
// RD-opt decision

func (it *iterator) swapOut() {
	it.yuvOut, it.yuvOut2 = it.yuvOut2, it.yuvOut
}

func storeMaxDelta(dqm *segmentInfo, dcs *[16]int16) {
	v0 := abs16(dcs[1])
	v1 := abs16(dcs[2])
	v2 := abs16(dcs[4])
	maxV := v0
	if v1 > maxV {
		maxV = v1
	}
	if v2 > maxV {
		maxV = v2
	}
	if maxV > dqm.maxEdge {
		dqm.maxEdge = maxV
	}
}

func (it *iterator) pickBestIntra16(rd *modeScore) {
	e := it.enc
	dqm := &e.dqm[it.mb().segment&3]
	lambda := dqm.lambdaI16
	tlambda := dqm.tlambda
	src := it.yuvIn[yOff:]
	rdCur := &it.rdTmp
	rdBest := rd
	isFlatSrc := isFlatSource16(src)

	if tlambda != 0 { // the source part of the texture distortion is constant
		for n := 0; n < 16; n++ {
			it.srcTT[n] = tTransform(src[vp8Scan[n]:], &kWeightY)
		}
	}
	rd.modeI16 = -1
	for mode := 0; mode < numPredModes; mode++ {
		tmpDst := it.yuvOut2[yOff:]
		rdCur.modeI16 = mode
		rdCur.nz = it.reconstructIntra16(&rdCur.yDCLevels, &rdCur.yACLevels, tmpDst, mode)

		rdCur.D = int64(sse16x16(src, tmpDst))
		if tlambda != 0 {
			d := 0
			for n := 0; n < 16; n++ {
				d += int(iabs(tTransform(tmpDst[vp8Scan[n]:], &kWeightY)-it.srcTT[n]) >> 5)
			}
			rdCur.SD = mult8b(tlambda, d)
		} else {
			rdCur.SD = 0
		}
		rdCur.H = int64(fixedCostsI16[mode])
		rdCur.R = int64(it.getCostLuma16(rdCur))
		if isFlatSrc {
			isFlatSrc = isFlat(rdCur.yACLevels[:], flatnessLimitI16)
			if isFlatSrc {
				rdCur.D *= 2
				rdCur.SD *= 2
			}
		}
		rdCur.setRDScore(lambda)
		if mode == 0 || rdCur.score < rdBest.score {
			rdCur, rdBest = rdBest, rdCur
			it.swapOut()
		}
	}
	if rdBest != rd {
		*rd = *rdBest
	}
	rd.setRDScore(dqm.lambdaMode)
	it.setIntra16Mode(rd.modeI16)

	if rd.nz&0x100ffff == 0x1000000 && rd.D > int64(dqm.minDisto) {
		storeMaxDelta(dqm, &rd.yDCLevels)
	}
}

func (it *iterator) getCostModeI4(modes *[16]uint8) *[numBModes]uint16 {
	predsW := it.enc.predsW
	x, y := it.i4&3, it.i4>>2
	var left, top uint8
	if x == 0 {
		left = it.enc.preds[it.predsIdx+y*predsW-1]
	} else {
		left = modes[(it.i4-1)&15]
	}
	if y == 0 {
		top = it.enc.preds[it.predsIdx-predsW+x]
	} else {
		top = modes[(it.i4-4)&15]
	}
	return &fixedCostsI4[top%numBModes][left%numBModes]
}

func (it *iterator) pickBestIntra4(rd *modeScore) bool {
	e := it.enc
	dqm := &e.dqm[it.mb().segment&3]
	lambda := dqm.lambdaI4
	tlambda := dqm.tlambda
	src0 := it.yuvIn[yOff:]
	bestBlocks := it.yuvOut2[yOff:]
	totalHeaderBits := 0
	var rdBest scoreT
	bestLevels := &it.i4Levels

	if e.maxI4HeaderBits == 0 {
		return false
	}
	rdBest.score = maxCost
	rdBest.H = 211 // '211' is the value of VP8BitCost(0, 145)
	rdBest.setRDScore(dqm.lambdaMode)
	it.startI4()
	for {
		var rdI4 scoreT
		rdI4.score = maxCost
		bestMode := -1
		src := src0[vp8Scan[it.i4]:]
		modeCosts := it.getCostModeI4(&rd.modesI4)
		bestBlock := bestBlocks[vp8Scan[it.i4]:]
		tmpDst := it.yuvP[i4TMP:]
		bestInPlace := true // whether bestBlock points into bestBlocks
		var srcTT int32
		if tlambda != 0 {
			srcTT = tTransform(src, &kWeightY)
		}

		it.makeIntra4Preds()
		for mode := 0; mode < numBModes; mode++ {
			var rdTmp scoreT
			tmpLevels := &it.tmpLevels
			nz := it.reconstructIntra4(&tmpLevels[0], src, tmpDst, mode)
			if nz {
				rdTmp.nz = 1 << uint(it.i4)
			}
			rdTmp.D = int64(sse4x4(src, tmpDst))
			if tlambda != 0 {
				rdTmp.SD = mult8b(tlambda, int(iabs(tTransform(tmpDst, &kWeightY)-srcTT)>>5))
			}
			rdTmp.H = int64(modeCosts[mode])
			if mode > 0 && (!nz || isFlat(tmpLevels[:], flatnessLimitI4)) {
				rdTmp.R = flatnessPenalty
			} else {
				rdTmp.R = 0
			}
			// early-out check
			rdTmp.setRDScore(lambda)
			if bestMode >= 0 && rdTmp.score >= rdI4.score {
				continue
			}
			rdTmp.R += int64(it.getCostLuma4(&tmpLevels[0], nz))
			rdTmp.setRDScore(lambda)
			if bestMode < 0 || rdTmp.score < rdI4.score {
				rdI4 = rdTmp
				bestMode = mode
				tmpDst, bestBlock = bestBlock, tmpDst
				bestInPlace = !bestInPlace
				bestLevels[it.i4] = tmpLevels[0]
			}
		}
		rdI4.setRDScore(dqm.lambdaMode)
		rdBest.D += rdI4.D
		rdBest.SD += rdI4.SD
		rdBest.R += rdI4.R
		rdBest.H += rdI4.H
		rdBest.nz |= rdI4.nz
		rdBest.score += rdI4.score
		if rdBest.score >= rd.score {
			return false
		}
		totalHeaderBits += int(rdI4.H)
		if totalHeaderBits > e.maxI4HeaderBits {
			return false
		}
		if !bestInPlace {
			copy4x4(bestBlock, bestBlocks[vp8Scan[it.i4]:])
		}
		rd.modesI4[it.i4] = uint8(bestMode)
		v := 0
		if rdI4.nz != 0 {
			v = 1
		}
		it.topNz[it.i4&3] = v
		it.leftNz[it.i4>>2] = v
		if !it.rotateI4(bestBlocks) {
			break
		}
	}

	rd.D, rd.SD, rd.R, rd.H, rd.nz, rd.score = rdBest.D, rdBest.SD, rdBest.R, rdBest.H, rdBest.nz, rdBest.score
	it.setIntra4Mode(&rd.modesI4)
	it.swapOut()
	rd.yACLevels = *bestLevels
	return true
}

func (it *iterator) pickBestUV(rd *modeScore) {
	const kNumBlocks = 8
	e := it.enc
	dqm := &e.dqm[it.mb().segment&3]
	lambda := dqm.lambdaUV
	src := it.yuvIn[uOff:]
	tmpDst := it.yuvOut2[uOff:]
	dst0 := it.yuvOut[uOff:]
	dst := dst0
	dstIsDst0 := true
	var rdBest scoreT
	rdUV := &it.rdUV

	rd.modeUV = -1
	rdBest.score = maxCost
	for mode := 0; mode < numPredModes; mode++ {
		var s scoreT
		s.nz = it.reconstructUV(&rdUV.uvLevels, &rdUV.derr, tmpDst, mode)
		s.D = int64(sse16x8(src, tmpDst))
		s.SD = 0
		s.H = int64(fixedCostsUV[mode])
		s.R = int64(it.getCostUV(&rdUV.uvLevels))
		if mode > 0 && isFlat(rdUV.uvLevels[:], flatnessLimitUV) {
			s.R += flatnessPenalty * kNumBlocks
		}
		s.setRDScore(lambda)
		if mode == 0 || s.score < rdBest.score {
			rdBest = s
			rd.modeUV = mode
			rd.uvLevels = rdUV.uvLevels
			if e.topDerr != nil {
				rd.derr = rdUV.derr
			}
			dst, tmpDst = tmpDst, dst
			dstIsDst0 = !dstIsDst0
		}
	}
	it.mb().uvMode = uint8(rd.modeUV)
	rd.D += rdBest.D
	rd.SD += rdBest.SD
	rd.R += rdBest.R
	rd.H += rdBest.H
	rd.nz |= rdBest.nz
	rd.score += rdBest.score
	if !dstIsDst0 {
		copy16x8(dst, dst0)
	}
	if e.topDerr != nil {
		it.storeDiffusionErrors(&rd.derr)
	}
}

//------------------------------------------------------------------------------
// Final reconstruction and quantization.

// simpleQuantize redoes the final quantization of the chosen modes (used
// with trellis). The chroma part, which is never trellis-quantized, is kept
// from pickBestUV when uvDone is true.
func (it *iterator) simpleQuantize(rd *modeScore, uvDone bool) {
	e := it.enc
	mb := it.mb()
	nz := uint32(0)
	if mb.typ == 1 {
		nz = it.reconstructIntra16(&rd.yDCLevels, &rd.yACLevels, it.yuvOut[yOff:], int(e.preds[it.predsIdx]))
	} else {
		it.startI4()
		for {
			mode := int(e.preds[it.predsIdx+(it.i4&3)+(it.i4>>2)*e.predsW])
			src := it.yuvIn[yOff+vp8Scan[it.i4]:]
			dst := it.yuvOut[yOff+vp8Scan[it.i4]:]
			it.makeIntra4Preds()
			if it.reconstructIntra4(&rd.yACLevels[it.i4], src, dst, mode%numBModes) {
				nz |= 1 << uint(it.i4)
			}
			if !it.rotateI4(it.yuvOut[yOff:]) {
				break
			}
		}
	}
	if uvDone {
		nz |= rd.nz & 0xff0000
	} else {
		nz |= it.reconstructUV(&rd.uvLevels, &rd.derr, it.yuvOut[uOff:], int(mb.uvMode))
	}
	rd.nz = nz
}

const (
	lambdaDI16 = 106
	lambdaDI4  = 11
	lambdaDUV  = 120
)

// pickI16BySSE selects the intra16 mode with the lowest distortion-based
// score. It returns the score and whether i16 must be kept (flat border
// macroblock).
func (it *iterator) pickI16BySSE(bitLimit int64) (int64, bool) {
	bestScore := maxCost
	bestMode := -1
	stick := false
	src := it.yuvIn[yOff:]
	for mode := 0; mode < numPredModes; mode++ {
		ref := it.yuvP[vp8I16ModeOffsets[mode]:]
		score := int64(sse16x16(src, ref))*rdDistoMult + int64(fixedCostsI16[mode])*lambdaDI16
		if mode > 0 && int64(fixedCostsI16[mode]) > bitLimit {
			continue
		}
		if score < bestScore {
			bestMode = mode
			bestScore = score
		}
	}
	if it.x == 0 || it.y == 0 {
		// avoid starting a checkerboard resonance from the border.
		if isFlatSource16(src) {
			if it.x == 0 {
				bestMode = 0
			} else {
				bestMode = 2
			}
			stick = true
		}
	}
	it.setIntra16Mode(bestMode)
	return bestScore, stick
}

// refineUsingDistortion refines intra16/intra4 sub-modes based on
// distortion only (not rate).
func (it *iterator) refineUsingDistortion(tryBothModes, refineUVMode bool, rd *modeScore) {
	e := it.enc
	bestScore := maxCost
	nz := uint32(0)
	isI16 := tryBothModes || it.mb().typ == 1
	evaluatedI16 := false
	dqm := &e.dqm[it.mb().segment&3]
	scoreI4 := dqm.i4Penalty
	i4BitSum := int64(0)
	bitLimit := maxCost
	if tryBothModes {
		bitLimit = e.mbHeaderLimit
	}
	// Also honour the intra4 header budget (partition #0 size control).
	i4Limit := min(bitLimit, int64(e.maxI4HeaderBits))

	if isI16 {
		var stick bool
		bestScore, stick = it.pickI16BySSE(bitLimit)
		evaluatedI16 = true
		if stick {
			tryBothModes = false
		}
	}

	if tryBothModes || !isI16 {
		isI16 = false
		it.startI4()
		for {
			bestI4Mode := -1
			bestI4Score := maxCost
			src := it.yuvIn[yOff+vp8Scan[it.i4]:]
			modeCosts := it.getCostModeI4(&rd.modesI4)
			it.makeIntra4Preds()
			for mode := 0; mode < numBModes; mode++ {
				ref := it.yuvP[vp8I4ModeOffsets[mode]:]
				score := int64(sse4x4(src, ref))*rdDistoMult + int64(modeCosts[mode])*lambdaDI4
				if score < bestI4Score {
					bestI4Mode = mode
					bestI4Score = score
				}
			}
			i4BitSum += int64(modeCosts[bestI4Mode])
			rd.modesI4[it.i4] = uint8(bestI4Mode)
			scoreI4 += bestI4Score
			if scoreI4 >= bestScore || i4BitSum > i4Limit {
				isI16 = true
				break
			}
			tmpDst := it.yuvOut2[yOff+vp8Scan[it.i4]:]
			if it.reconstructIntra4(&rd.yACLevels[it.i4], src, tmpDst, bestI4Mode) {
				nz |= 1 << uint(it.i4)
			}
			if !it.rotateI4(it.yuvOut2[yOff:]) {
				break
			}
		}
		if isI16 && !evaluatedI16 {
			it.pickI16BySSE(bitLimit)
		}
	}

	// Header bits and distortion are not used for the decision, but feed
	// the partition #0 size control and the target size/PSNR search.
	if !isI16 {
		it.setIntra4Mode(&rd.modesI4)
		it.swapOut()
		bestScore = scoreI4
		rd.H = 211 + i4BitSum // 211 is VP8BitCost(0, 145)
	} else {
		mode := int(e.preds[it.predsIdx])
		nz = it.reconstructIntra16(&rd.yDCLevels, &rd.yACLevels, it.yuvOut[yOff:], mode)
		rd.H = int64(fixedCostsI16[mode&3])
	}

	if refineUVMode {
		bestMode := -1
		bestUVScore := maxCost
		src := it.yuvIn[uOff:]
		for mode := 0; mode < numPredModes; mode++ {
			ref := it.yuvP[vp8UVModeOffsets[mode]:]
			score := int64(sse16x8(src, ref))*rdDistoMult + int64(fixedCostsUV[mode])*lambdaDUV
			if score < bestUVScore {
				bestMode = mode
				bestUVScore = score
			}
		}
		it.mb().uvMode = uint8(bestMode)
	}
	uvMode := int(it.mb().uvMode)
	nz |= it.reconstructUV(&rd.uvLevels, &rd.derr, it.yuvOut[uOff:], uvMode)
	if e.topDerr != nil {
		it.storeDiffusionErrors(&rd.derr)
	}
	rd.H += int64(fixedCostsUV[uvMode&3])
	rd.D = int64(sse16x16(it.yuvIn[yOff:], it.yuvOut[yOff:]) + sse16x8(it.yuvIn[uOff:], it.yuvOut[uOff:]))
	rd.nz = nz
	rd.score = bestScore
}

// decimate picks the best modes and fills the levels. Returns true if the
// macroblock has no non-zero coefficient.
func (it *iterator) decimate(rd *modeScore, rdOpt int) bool {
	method := it.enc.method
	rd.init()
	it.makeLuma16Preds()
	it.makeChroma8Preds()
	if rdOpt > rdOptNone {
		it.doTrellis = rdOpt >= rdOptTrellisAll
		it.pickBestIntra16(rd)
		if method >= 2 {
			it.pickBestIntra4(rd)
		}
		it.pickBestUV(rd)
		if rdOpt == rdOptTrellis {
			it.doTrellis = true
			it.simpleQuantize(rd, true)
		}
	} else {
		it.refineUsingDistortion(method >= 2, method >= 1, rd)
	}
	skipped := rd.nz == 0
	it.mb().skip = skipped
	return skipped
}

//------------------------------------------------------------------------------
// Rate calculation

func (it *iterator) getCostLuma16(rd *modeScore) int {
	p := &it.enc.proba
	var res residual
	r := 0
	it.nzToBytes()
	// DC
	res.first, res.ctype = 0, typeI16DC
	res.setCoeffs(&rd.yDCLevels)
	r += p.residualCost(it.topNz[8]+it.leftNz[8], &res)
	// AC
	res.first, res.ctype = 1, typeI16AC
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			ctx := it.topNz[x] + it.leftNz[y]
			res.setCoeffs(&rd.yACLevels[x+y*4])
			r += p.residualCost(ctx, &res)
			v := b2i(res.last >= 0)
			it.topNz[x] = v
			it.leftNz[y] = v
		}
	}
	return r
}

func (it *iterator) getCostLuma4(levels *[16]int16, nz bool) int {
	x, y := it.i4&3, it.i4>>2
	ctx := it.topNz[x] + it.leftNz[y]
	if !nz { // empty block: only the EOB is coded
		return bitCost(0, it.enc.proba.coeffs[typeI4AC][0][ctx%numCtx][0])
	}
	var res residual
	res.first, res.ctype = 0, typeI4AC
	res.setCoeffs(levels)
	return it.enc.proba.residualCost(it.topNz[x]+it.leftNz[y], &res)
}

func (it *iterator) getCostUV(levels *[8][16]int16) int {
	p := &it.enc.proba
	var res residual
	r := 0
	it.nzToBytes()
	res.first, res.ctype = 0, typeChromaA
	for ch := 0; ch <= 2; ch += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				ctx := it.topNz[4+ch+x] + it.leftNz[4+ch+y]
				res.setCoeffs(&levels[ch*2+x+y*2])
				r += p.residualCost(ctx, &res)
				v := b2i(res.last >= 0)
				it.topNz[4+ch+x] = v
				it.leftNz[4+ch+y] = v
			}
		}
	}
	return r
}
