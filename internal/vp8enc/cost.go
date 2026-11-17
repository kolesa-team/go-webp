package vp8enc

// Token probabilities, statistics and bit-cost estimation.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	numTypes         = 4
	numBands         = 8
	numCtx           = 3
	numProbas        = 11
	numMBSegments    = 4
	maxVariableLevel = 67
)

// Coefficient types.
const (
	typeI16AC   = 0
	typeI16DC   = 1
	typeChromaA = 2
	typeI4AC    = 3
)

var encBands = [16 + 1]uint8{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0}

var (
	vp8Cat3 = []uint8{173, 148, 140}
	vp8Cat4 = []uint8{176, 155, 140, 135}
	vp8Cat5 = []uint8{180, 157, 141, 134, 130}
	vp8Cat6 = []uint8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129}
)

// Mode costs: fixed probabilities in the coding trees turned into bit-cost.
var (
	fixedCostsUV  = [4]uint16{302, 984, 439, 642}
	fixedCostsI16 = [4]uint16{663, 919, 872, 919} // includes VP8BitCost(1, 145)
)

// bitCost is the cost (in 1/256 bit) of coding 'bit' with probability 'p'.
func bitCost(bit int, p uint8) int {
	if bit == 0 {
		return int(entropyCost[p])
	}
	return int(entropyCost[255-p])
}

type levelCostTable [maxVariableLevel + 1]uint16

type encProba struct {
	segments     [3]uint8
	skipProba    uint8
	useSkipProba bool
	dirty        bool
	nbSkip       int
	coeffs       [numTypes][numBands][numCtx][numProbas]uint8
	stats        [numTypes][numBands][numCtx][numProbas]uint32
	levelCost    [numTypes][numBands][numCtx]levelCostTable
}

func (p *encProba) reset() {
	p.useSkipProba = false
	p.skipProba = 255
	p.segments = [3]uint8{255, 255, 255}
	p.coeffs = coeffsProba0
	p.stats = [numTypes]statsArray{}
	p.dirty = true
	p.nbSkip = 0
}

func variableLevelCost(level int, probas *[numProbas]uint8) int {
	pattern := int(levelCodes[level-1][0])
	bits := int(levelCodes[level-1][1])
	cost := 0
	for i := 2; pattern != 0; i++ {
		if pattern&1 != 0 {
			cost += bitCost(bits&1, probas[i])
		}
		bits >>= 1
		pattern >>= 1
	}
	return cost
}

// calculateLevelCosts refreshes the level cost tables from the current
// probabilities.
func (p *encProba) calculateLevelCosts() {
	if !p.dirty {
		return
	}
	for ctype := 0; ctype < numTypes; ctype++ {
		for band := 0; band < numBands; band++ {
			for ctx := 0; ctx < numCtx; ctx++ {
				pr := &p.coeffs[ctype][band][ctx]
				table := &p.levelCost[ctype][band][ctx]
				cost0 := 0
				if ctx > 0 {
					cost0 = bitCost(1, pr[0])
				}
				costBase := bitCost(1, pr[1]) + cost0
				table[0] = uint16(bitCost(0, pr[1]) + cost0)
				for v := 1; v <= maxVariableLevel; v++ {
					table[v] = uint16(costBase + variableLevelCost(v, pr))
				}
			}
		}
	}
	p.dirty = false
}

func levelCost(table *levelCostTable, level int) int {
	if level > maxVariableLevel {
		return int(levelFixedCosts[level&2047]) + int(table[maxVariableLevel])
	}
	return int(levelFixedCosts[level]) + int(table[level])
}

// residual describes one block of quantized levels (zigzag order).
type residual struct {
	first  int
	last   int
	ctype  int
	coeffs *[16]int16
}

func (r *residual) setCoeffs(c *[16]int16) {
	r.coeffs = c
	r.last = lastNonZero(c)
}

// setCoeffsInfo is setCoeffs with the levelInfoOf summary of c known.
func (r *residual) setCoeffsInfo(c *[16]int16, info uint16) {
	r.coeffs = c
	r.last = infoLast(info)
}

// lastNonZero returns the index of the last non-zero coefficient, or -1.
func lastNonZero(c *[16]int16) int {
	n := 15
	for ; n >= 3; n -= 4 { // skip empty groups of 4 quickly
		if c[n]|c[n-1]|c[n-2]|c[n-3] != 0 {
			break
		}
	}
	for ; n >= 0; n-- {
		if c[n&15] != 0 {
			return n
		}
	}
	return -1
}

func abs16(v int16) int {
	if v < 0 {
		return -int(v)
	}
	return int(v)
}

// residualCost returns the bit-cost of coding the residual with context ctx0.
func (p *encProba) residualCost(ctx0 int, res *residual) int {
	n := res.first
	prob := &p.coeffs[res.ctype]
	costs := &p.levelCost[res.ctype]
	// should be prob[encBands[n]], but it's equivalent for n=0 or 1
	p0 := prob[n][ctx0][0]
	if res.last < 0 {
		return bitCost(0, p0)
	}
	cost := 0
	if ctx0 == 0 {
		cost = bitCost(1, p0)
	}
	t := &costs[n][ctx0]
	coeffs := res.coeffs
	last := res.last
	// The signs, magnitudes and contexts of the levels are data dependent:
	// select with min() and masks rather than branch (levelCost, with the
	// variable part clamped at maxVariableLevel, and ctx = min(level, 2)).
	for ; n < last; n++ {
		v := absLevel(coeffs[n&15])
		cost += int(levelFixedCosts[v&maxLevel]) + int(t[min(v, maxVariableLevel)])
		t = &costs[encBands[(n+1)&15]&7][min(v, 2)]
	}
	// Last coefficient is always non-zero
	v := absLevel(coeffs[n&15])
	cost += int(levelFixedCosts[v&maxLevel]) + int(t[min(v, maxVariableLevel)])
	if n < 15 {
		ctx := 2
		if v == 1 {
			ctx = 1
		}
		cost += bitCost(0, prob[encBands[n+1]&7][ctx][0])
	}
	return cost
}

// absLevel returns |v| (at most 32768) without a data-dependent branch.
func absLevel(v int16) uint {
	a := int32(v)
	m := a >> 31
	return uint((a ^ m) - m)
}

// recordStats records a bit event in the 16b+16b stats counter.
func recordStats(bit int, s *uint32) int {
	p := *s
	if p >= 0xfffe0000 {
		p = ((p + 1) >> 1) & 0x7fff7fff // divide the stats by 2.
	}
	p += 0x00010000 + uint32(bit)
	*s = p
	return bit
}

func calcTokenProba(nb, total int) int {
	if nb == 0 {
		return 255
	}
	return 255 - nb*255/total
}

func branchCost(nb, total, proba int) int {
	return nb*bitCost(1, uint8(proba)) + (total-nb)*bitCost(0, uint8(proba))
}

// finalizeTokenProbas derives new probabilities from the statistics and
// returns the bit-cost of the probability updates.
func (p *encProba) finalizeTokenProbas() int {
	hasChanged := false
	size := 0
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for q := 0; q < numProbas; q++ {
					stats := p.stats[t][b][c][q]
					nb := int(stats & 0xffff)
					total := int(stats >> 16)
					updateProba := coeffsUpdateProba[t][b][c][q]
					oldP := int(coeffsProba0[t][b][c][q])
					newP := calcTokenProba(nb, total)
					oldCost := branchCost(nb, total, oldP) + bitCost(0, updateProba)
					newCost := branchCost(nb, total, newP) + bitCost(1, updateProba) + 8*256
					finalP := oldP
					if oldCost > newCost {
						size += bitCost(1, updateProba) + 8*256
						finalP = newP
					} else {
						size += bitCost(0, updateProba)
					}
					if int(p.coeffs[t][b][c][q]) != finalP {
						hasChanged = true
						p.coeffs[t][b][c][q] = uint8(finalP)
					}
				}
			}
		}
	}
	if hasChanged {
		p.dirty = true
	}
	return size
}
