package vp8lenc

import "math"

// Histograms and histogram clustering for meta-Huffman codes.
// Algorithm adapted from libwebp (BSD-3-Clause).

// histo is a view on a flat population array laid out as
// literal[litLen] | red[256] | blue[256] | alpha[256] | dist[40].
type histo struct {
	d        []uint32
	bitCost  float64
	litCost  float64
	redCost  float64
	blueCost float64
	used     bool
}

// histoSet owns the storage of several histograms of the same layout.
type histoSet struct {
	litLen int
	stride int
	data   []uint32
	h      []histo
}

func literalLen(cacheBits int) int {
	n := numLiteralCodes + numLengthCodes
	if cacheBits > 0 {
		n += 1 << uint(cacheBits)
	}
	return n
}

// init prepares the set for n histograms with the given cache bits.
func (s *histoSet) init(n, cacheBits int) {
	s.litLen = literalLen(cacheBits)
	s.stride = s.litLen + 3*256 + numDistanceCodes
	total := n * s.stride
	if cap(s.data) < total {
		s.data = make([]uint32, total)
	} else {
		s.data = s.data[:total]
		for i := range s.data {
			s.data[i] = 0
		}
	}
	if cap(s.h) < n {
		s.h = make([]histo, n)
	}
	s.h = s.h[:n]
	for i := range s.h {
		s.h[i] = histo{d: s.data[i*s.stride : (i+1)*s.stride : (i+1)*s.stride], used: true}
	}
}

func (s *histoSet) clear(h *histo) {
	for i := range h.d {
		h.d[i] = 0
	}
}

// addRef adds one reference to h.
func (s *histoSet) addRef(d []uint32, r pixOrCopy, xsize int) {
	L := s.litLen
	switch r.mode() {
	case modeLiteral:
		p := r.val
		d[(p>>8)&0xff]++
		d[L+int((p>>16)&0xff)]++
		d[L+256+int(p&0xff)]++
		d[L+512+int(p>>24)]++
	case modeCache:
		d[numLiteralCodes+numLengthCodes+int(r.val)]++
	default:
		d[numLiteralCodes+prefixCode(r.length())]++
		d[L+768+prefixCode(distanceToPlaneCode(xsize, int(r.val)))]++
	}
}

func (s *histoSet) addRefs(h *histo, refs []pixOrCopy, xsize int) {
	d := h.d
	for _, r := range refs {
		s.addRef(d, r, xsize)
	}
}

// estimate computes and stores the bit cost of h.
func (s *histoSet) estimate(h *histo) float64 {
	L := s.litLen
	d := h.d
	h.litCost = populationCost(d[:L])
	h.redCost = populationCost(d[L : L+256])
	h.blueCost = populationCost(d[L+256 : L+512])
	h.bitCost = h.litCost + h.redCost + h.blueCost +
		populationCost(d[L+512:L+768]) +
		populationCost(d[L+768:L+808]) +
		extraCost(d[numLiteralCodes:numLiteralCodes+numLengthCodes]) +
		extraCost(d[L+768:L+808])
	return h.bitCost
}

// combinedCost returns the cost of a+b, or +Inf when it exceeds limit.
func (s *histoSet) combinedCost(a, b *histo, limit float64) float64 {
	L := s.litLen
	x, y := a.d, b.d
	c := extraCostCombined(x[numLiteralCodes:numLiteralCodes+numLengthCodes], y[numLiteralCodes:numLiteralCodes+numLengthCodes]) +
		extraCostCombined(x[L+768:L+808], y[L+768:L+808])
	c += combinedPopulationCost(x[:L], y[:L])
	if c > limit {
		return math.Inf(1)
	}
	c += combinedPopulationCost(x[L:L+256], y[L:L+256])
	if c > limit {
		return math.Inf(1)
	}
	c += combinedPopulationCost(x[L+256:L+512], y[L+256:L+512])
	if c > limit {
		return math.Inf(1)
	}
	c += combinedPopulationCost(x[L+512:L+768], y[L+512:L+768])
	if c > limit {
		return math.Inf(1)
	}
	c += combinedPopulationCost(x[L+768:L+808], y[L+768:L+808])
	if c > limit {
		return math.Inf(1)
	}
	return c
}

// addInto adds b into a.
func addInto(a, b *histo) {
	x := a.d
	y := b.d[:len(x)]
	for i := range x {
		x[i] += y[i]
	}
}

// ---- clustering ----

type histoPair struct {
	i1, i2   int
	costDiff float64
	costCmb  float64
}

// pairQueue keeps the best (lowest costDiff) pair at index 0.
type pairQueue struct {
	q   []histoPair
	max int
}

func (pq *pairQueue) updateHead(k int) {
	if pq.q[k].costDiff < pq.q[0].costDiff {
		pq.q[k], pq.q[0] = pq.q[0], pq.q[k]
	}
}

func (pq *pairQueue) pop(k int) {
	last := len(pq.q) - 1
	pq.q[k] = pq.q[last]
	pq.q = pq.q[:last]
}

// push evaluates the pair (i1,i2) and inserts it if merging gains more than
// threshold. Returns the cost difference (0 when not pushed).
func (pq *pairQueue) push(s *histoSet, hs []histo, i1, i2 int, threshold float64) float64 {
	if len(pq.q) >= pq.max {
		return 0
	}
	if i1 > i2 {
		i1, i2 = i2, i1
	}
	a, b := &hs[i1], &hs[i2]
	sumCost := a.bitCost + b.bitCost
	cmb := s.combinedCost(a, b, sumCost+threshold)
	diff := cmb - sumCost
	if !(diff < threshold) {
		return 0
	}
	pq.q = append(pq.q, histoPair{i1, i2, diff, cmb})
	pq.updateHead(len(pq.q) - 1)
	return diff
}

const numPartitions = 4
const binSize = numPartitions * numPartitions * numPartitions

func binID(min, max, v float64) int {
	r := max - min
	if r > 0 {
		return int((numPartitions - 1e-6) * (v - min) / r)
	}
	return 0
}

func combineCostFactor(histoSize, quality int) float64 {
	f := 0.16
	if quality < 90 {
		if histoSize > 256 {
			f /= 2
		}
		if histoSize > 512 {
			f /= 2
		}
		if histoSize > 1024 {
			f /= 2
		}
		if quality <= 50 {
			f /= 2
		}
	}
	return f
}

type clusterer struct {
	orig     histoSet // per-tile histograms
	symbols  []uint16
	mapping  []int
	queue    pairQueue
	binFirst [binSize]int
	binFail  [binSize]int
}

// buildHistoImage computes the clustered histograms for refs. It returns the
// final histograms (in out) and fills c.symbols with the tile->histogram map.
func (c *clusterer) buildHistoImage(xsize, ysize int, refs []pixOrCopy, quality int,
	lowEffort bool, histoBits, cacheBits int, out *histoSet) int {
	hx := subSampleSize(xsize, histoBits)
	hy := subSampleSize(ysize, histoBits)
	nt := hx * hy
	c.orig.init(nt, cacheBits)
	// Build per-tile histograms.
	x, y := 0, 0
	for _, r := range refs {
		ix := (y>>uint(histoBits))*hx + (x >> uint(histoBits))
		c.orig.addRef(c.orig.h[ix].d, r, xsize)
		x += r.length()
		for x >= xsize {
			x -= xsize
			y++
		}
	}
	if cap(c.symbols) < nt {
		c.symbols = make([]uint16, nt)
	}
	c.symbols = c.symbols[:nt]

	// Copy non-empty histograms into the working set.
	numUsed := 0
	for i := range c.orig.h {
		h := &c.orig.h[i]
		empty := true
		for _, v := range h.d {
			if v != 0 {
				empty = false
				break
			}
		}
		h.used = !empty
		if !empty {
			numUsed++
		}
	}
	out.init(numUsed, cacheBits)
	if cap(c.mapping) < nt {
		c.mapping = make([]int, nt)
	}
	k := 0
	for i := range c.orig.h {
		if !c.orig.h[i].used {
			continue
		}
		copy(out.h[k].d, c.orig.h[i].d)
		out.estimate(&out.h[k])
		k++
	}
	if numUsed == 0 {
		// Empty image (cannot happen for valid input); single histogram.
		out.init(1, cacheBits)
		for i := range c.symbols {
			c.symbols[i] = 0
		}
		return 1
	}
	hs := out.h

	entropyBins := binSize
	if lowEffort {
		entropyBins = numPartitions
	}
	if numUsed > 2*entropyBins && quality < 100 {
		c.entropyBinCombine(out, lowEffort, combineCostFactor(nt, quality))
	}
	if !lowEffort || !(numUsed > 2*entropyBins && quality < 100) {
		thresholdSize := 1 + (quality*quality*quality*(100-1)+(100*100*100)/2)/(100*100*100)
		doGreedy := c.stochasticCombine(out, thresholdSize)
		if doGreedy {
			c.greedyCombine(out)
		}
	}
	// Compact the used histograms.
	n := 0
	for i := range hs {
		if hs[i].used {
			if n != i {
				copy(hs[n].d, hs[i].d)
				hs[n].bitCost = hs[i].bitCost
			}
			hs[n].used = true
			n++
		}
	}
	out.h = hs[:n]
	c.remap(out)
	return len(out.h)
}

func (c *clusterer) entropyBinCombine(s *histoSet, lowEffort bool, factor float64) {
	hs := s.h
	litMin, litMax := math.MaxFloat64, -math.MaxFloat64
	redMin, redMax := litMin, litMax
	blueMin, blueMax := litMin, litMax
	for i := range hs {
		h := &hs[i]
		litMin = math.Min(litMin, h.litCost)
		litMax = math.Max(litMax, h.litCost)
		redMin = math.Min(redMin, h.redCost)
		redMax = math.Max(redMax, h.redCost)
		blueMin = math.Min(blueMin, h.blueCost)
		blueMax = math.Max(blueMax, h.blueCost)
	}
	for i := range c.binFirst {
		c.binFirst[i] = -1
		c.binFail[i] = 0
	}
	for i := range hs {
		h := &hs[i]
		b := binID(litMin, litMax, h.litCost)
		if !lowEffort {
			b = b*numPartitions + binID(redMin, redMax, h.redCost)
			b = b*numPartitions + binID(blueMin, blueMax, h.blueCost)
		}
		f := c.binFirst[b]
		if f == -1 {
			c.binFirst[b] = i
			continue
		}
		first := &hs[f]
		if lowEffort {
			addInto(first, h)
			s.estimate(first)
			h.used = false
			continue
		}
		thresh := -h.bitCost * factor
		sumCost := first.bitCost + h.bitCost
		cmb := s.combinedCost(first, h, sumCost+thresh)
		if cmb-sumCost < thresh {
			addInto(first, h)
			first.bitCost = cmb
			// Keep per-channel costs roughly up to date for later bins.
			h.used = false
		}
	}
}

// myRand is libwebp's MyRand LCG.
func myRand(seed *uint32) uint32 {
	*seed = (*seed * 16807) & 0xffffffff
	if *seed == 0 {
		*seed = 1
	}
	return *seed
}

// stochasticCombine merges random pairs until the number of histograms is
// below minClusterSize. Returns whether a greedy pass should follow.
func (c *clusterer) stochasticCombine(s *histoSet, minClusterSize int) bool {
	hs := s.h
	idx := c.mapping[:0]
	for i := range hs {
		if hs[i].used {
			idx = append(idx, i)
		}
	}
	numUsed := len(idx)
	if numUsed < minClusterSize {
		return true
	}
	const queueSize = 9
	c.queue.q = c.queue.q[:0]
	c.queue.max = queueSize
	seed := uint32(1)
	outerIters := numUsed
	numTriesNoSuccess := outerIters / 2
	triesNoSuccess := 0
	for iter := 0; iter < outerIters && numUsed >= minClusterSize; iter++ {
		triesNoSuccess++
		if triesNoSuccess >= numTriesNoSuccess {
			break
		}
		bestCost := 0.0
		if len(c.queue.q) > 0 {
			bestCost = c.queue.q[0].costDiff
		}
		randRange := uint32((numUsed - 1) * numUsed)
		numTries := numUsed / 2
		for j := 0; numUsed >= 2 && j < numTries; j++ {
			t := myRand(&seed) % randRange
			i1 := t / uint32(numUsed-1)
			i2 := t % uint32(numUsed-1)
			if i2 >= i1 {
				i2++
			}
			cur := c.queue.push(s, hs, idx[i1], idx[i2], bestCost)
			if cur < 0 {
				bestCost = cur
				if len(c.queue.q) == c.queue.max {
					break
				}
			}
		}
		if len(c.queue.q) == 0 {
			continue
		}
		b1, b2 := c.queue.q[0].i1, c.queue.q[0].i2
		// Remove b2 from idx.
		for k := range idx {
			if idx[k] == b2 {
				copy(idx[k:], idx[k+1:])
				idx = idx[:len(idx)-1]
				break
			}
		}
		addInto(&hs[b1], &hs[b2])
		hs[b1].bitCost = c.queue.q[0].costCmb
		hs[b2].used = false
		numUsed--
		for j := 0; j < len(c.queue.q); {
			p := &c.queue.q[j]
			is1 := p.i1 == b1 || p.i1 == b2
			is2 := p.i2 == b1 || p.i2 == b2
			if is1 && is2 {
				c.queue.pop(j)
				continue
			}
			eval := false
			if is1 {
				p.i1 = b1
				eval = true
			} else if is2 {
				p.i2 = b1
				eval = true
			}
			if p.i1 > p.i2 {
				p.i1, p.i2 = p.i2, p.i1
			}
			if eval {
				a, b := &hs[p.i1], &hs[p.i2]
				sumCost := a.bitCost + b.bitCost
				cmb := s.combinedCost(a, b, sumCost)
				p.costCmb = cmb
				p.costDiff = cmb - sumCost
				if !(p.costDiff < 0) {
					c.queue.pop(j)
					continue
				}
			}
			c.queue.updateHead(j)
			j++
		}
		triesNoSuccess = 0
	}
	return numUsed <= minClusterSize
}

// greedyCombine repeatedly merges the best pair while it reduces the cost.
func (c *clusterer) greedyCombine(s *histoSet) {
	hs := s.h
	n := len(hs)
	c.queue.q = c.queue.q[:0]
	c.queue.max = n * n
	if cap(c.queue.q) < 64 {
		c.queue.q = make([]histoPair, 0, 64)
	}
	for i := 0; i < n; i++ {
		if !hs[i].used {
			continue
		}
		for j := i + 1; j < n; j++ {
			if !hs[j].used {
				continue
			}
			c.queue.push(s, hs, i, j, 0)
		}
	}
	for len(c.queue.q) > 0 {
		i1, i2 := c.queue.q[0].i1, c.queue.q[0].i2
		addInto(&hs[i1], &hs[i2])
		hs[i1].bitCost = c.queue.q[0].costCmb
		hs[i2].used = false
		for k := 0; k < len(c.queue.q); {
			p := &c.queue.q[k]
			if p.i1 == i1 || p.i2 == i1 || p.i1 == i2 || p.i2 == i2 {
				c.queue.pop(k)
			} else {
				c.queue.updateHead(k)
				k++
			}
		}
		for k := 0; k < n; k++ {
			if k == i1 || !hs[k].used {
				continue
			}
			c.queue.push(s, hs, i1, k, 0)
		}
	}
}

// remap assigns each tile histogram to the best final cluster, renumbers
// clusters by first use and recomputes the final histograms.
func (c *clusterer) remap(out *histoSet) {
	hs := out.h
	syms := c.symbols
	orig := c.orig.h
	if len(hs) > 1 {
		for i := range orig {
			if !orig[i].used {
				if i > 0 {
					syms[i] = syms[i-1]
				} else {
					syms[i] = 0
				}
				continue
			}
			best := 0
			bestBits := math.MaxFloat64
			for k := range hs {
				limit := bestBits + hs[k].bitCost
				cmb := out.combinedCost(&hs[k], &orig[i], limit)
				cur := cmb - hs[k].bitCost
				if k == 0 || cur < bestBits {
					bestBits = cur
					best = k
				}
			}
			syms[i] = uint16(best)
		}
	} else {
		for i := range syms {
			syms[i] = 0
		}
	}
	// Renumber by first appearance; drop clusters nobody maps to.
	order := c.mapping[:len(hs)]
	for i := range order {
		order[i] = -1
	}
	n := 0
	for i, sym := range syms {
		if order[sym] == -1 {
			order[sym] = n
			n++
		}
		syms[i] = uint16(order[sym])
	}
	out.h = out.h[:n]
	for i := range out.data[:n*out.stride] {
		out.data[i] = 0
	}
	for i := range orig {
		if orig[i].used {
			addInto(&out.h[syms[i]], &orig[i])
		}
	}
}
