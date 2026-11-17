package vp8enc

import (
	"math/rand"
	"testing"
)

// trellisQuantizeBlockRef is the straightforward transcription of libwebp's
// TrellisQuantizeBlock that trellisQuantizeBlock must reproduce exactly.
func (it *iterator) trellisQuantizeBlockRef(in *[16]int16, out *[16]int16, ctx0 int, coeffType int, mtx *matrix, lambda int) bool {
	const numNodes = 2 // MIN_DELTA = 0, MAX_DELTA = 1
	type scoreState struct {
		score int64
		costs *levelCostTable
	}
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
		v := int32(in[j])
		sign := v < 0
		if sign {
			v = -v
		}
		coeff0 := uint32(v) + mtx.sharpen[j]
		level0 := int(quantDiv(coeff0, iQ, biasOf(0x00)))
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
			ctx := min(level, 2)
			ssCur[m].costs = &costs[band][ctx]
			if level < 0 || level > threshLevel {
				ssCur[m].score = maxCost
				continue
			}
			newError := int64(coeff0) - int64(level)*int64(Q)
			deltaError := int64(kWeightTrellis[j]) * (newError*newError - int64(coeff0)*int64(coeff0))
			baseScore := rdScoreTrellis(lambda, 0, deltaError)
			cost := int64(levelCost(ssPrev[0].costs, level))
			bestCurScore := ssPrev[0].score + rdScoreTrellis(lambda, cost, 0)
			bestPrev := 0
			cost = int64(levelCost(ssPrev[1].costs, level))
			if score := ssPrev[1].score + rdScoreTrellis(lambda, cost, 0); score < bestCurScore {
				bestCurScore = score
				bestPrev = 1
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

// TestTrellisExact compares trellisQuantizeBlock with the reference on
// random blocks, quantizers, probabilities and lambdas, including
// coefficients far outside the range the transforms produce.
func TestTrellisExact(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	e := new(encoder)
	var it iterator
	it.enc = e
	n := 200000
	if testing.Short() {
		n = 20000
	}
	for iter := 0; iter < n; iter++ {
		if iter%500 == 0 {
			e.proba.reset()
			for ty := range e.proba.coeffs {
				for b := range e.proba.coeffs[ty] {
					for c := range e.proba.coeffs[ty][b] {
						for p := range e.proba.coeffs[ty][b][c] {
							switch rng.Intn(5) {
							case 0:
								e.proba.coeffs[ty][b][c][p] = uint8(rng.Intn(3) * 127) // 0, 127, 254
							case 1:
								e.proba.coeffs[ty][b][c][p] = 255
							default:
								e.proba.coeffs[ty][b][c][p] = uint8(1 + rng.Intn(255))
							}
						}
					}
				}
			}
			e.proba.calculateLevelCosts()
		}
		var mtx matrix
		q := rng.Intn(128)
		mtx.q[0] = uint32(kDcTable[q])
		mtx.q[1] = uint32(kAcTable[q])
		expandMatrix(&mtx, 0)
		coeffType := typeI4AC
		if rng.Intn(2) == 0 {
			coeffType = typeI16AC
		}
		var in [16]int16
		scale := []int{4, 64, 1024, 4096, 32768}[rng.Intn(5)]
		for i := range in {
			if rng.Intn(3) > 0 {
				in[i] = int16(rng.Intn(2*scale) - scale)
			}
		}
		if rng.Intn(50) == 0 {
			in[rng.Intn(16)] = -32768
		}
		lambda := 1 + rng.Intn([]int{10, 1000, 100000, 1 << 22}[rng.Intn(4)])
		ctx0 := rng.Intn(3)
		inRef, inGot := in, in
		var outRef, outGot [16]int16
		for i := range outRef {
			outRef[i] = int16(rng.Intn(100)) // garbage the DC of i16 keeps
		}
		outGot = outRef
		nzRef := it.trellisQuantizeBlockRef(&inRef, &outRef, ctx0, coeffType, &mtx, lambda)
		info := it.trellisQuantizeBlock(&inGot, &outGot, ctx0, coeffType, &mtx, lambda)
		nzGot := info != 0
		outDC := outGot
		if coeffType == typeI16AC {
			outDC[0] = 0 // the caller clears the DC position
		}
		if want := levelInfoOf(&outDC); info != want {
			t.Fatalf("iter %d: level summary %#x, want %#x for %v", iter, info, want, outDC)
		}
		if nzRef != nzGot || inRef != inGot || outRef != outGot {
			t.Fatalf("iter %d (q %d type %d ctx %d lambda %d, in %v): got %v %v %v, want %v %v %v",
				iter, q, coeffType, ctx0, lambda, in, nzGot, inGot, outGot, nzRef, inRef, outRef)
		}
	}
}

func BenchmarkTrellis(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	e := new(encoder)
	e.proba.reset()
	e.proba.calculateLevelCosts()
	var it iterator
	it.enc = e
	var mtx matrix
	mtx.q[0], mtx.q[1] = uint32(kDcTable[40]), uint32(kAcTable[40])
	expandMatrix(&mtx, 0)
	blocks := make([][16]int16, 1024)
	for i := range blocks {
		for k := range blocks[i] {
			if rng.Intn(2) == 0 {
				blocks[i][k] = int16(rng.Intn(400) - 200)
			}
		}
	}
	for b.Loop() {
		for i := range blocks {
			in := blocks[i]
			var out [16]int16
			it.trellisQuantizeBlock(&in, &out, i%3, typeI4AC, &mtx, 2000)
		}
	}
}
