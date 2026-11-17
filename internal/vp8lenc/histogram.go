package vp8lenc

import "math"

// Histograms and histogram clustering for meta-Huffman codes.
// Algorithm adapted from libwebp (BSD-3-Clause).

// histo is a view on a flat population array laid out as
// literal[litLen] | red[256] | blue[256] | alpha[256] | dist[40].
type histo struct {
	d []uint32
	// Clustering state (libwebp VP8LHistogram): bitCost is the sum of the
	// population costs of the 5 codes (without the extra bits), costs the
	// cost of each code, trivial the single symbol of a code (or
	// nonTrivialSym) and isUsed whether a code has any symbol. As in
	// libwebp, trivial and isUsed are propagated (not recomputed) when
	// histograms are merged.
	bitCost float64
	costs   [5]float64
	trivial [5]int32
	isUsed  [5]bool
	binID   int
	used    bool // the histogram is not empty
	// shared marks a tile histogram of the clusterer (clusterer.orig) used
	// as is in the working set: its populations and costs must not change
	// (the working set copies it on write, see clusterer.mergeInto).
	shared bool
}

// nonTrivialSym marks a code with zero or several symbols.
const nonTrivialSym = -1

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

// addRefsRawDist is addRefs with the distances coded as linear distances
// instead of plane codes. libwebp compares candidate parses with such
// histograms (VP8LHistogramCreate in GetBackwardReferences, before
// BackwardReferences2DLocality), which favours short linear distances.
func (s *histoSet) addRefsRawDist(h *histo, refs []pixOrCopy) {
	d := h.d
	L := s.litLen
	for _, r := range refs {
		if r.mode() == modeCopy {
			d[numLiteralCodes+prefixCode(r.length())]++
			d[L+768+prefixCode(int(r.val))]++
			continue
		}
		s.addRef(d, r, 0)
	}
}

func (s *histoSet) addRefs(h *histo, refs []pixOrCopy, xsize int) {
	d := h.d
	for _, r := range refs {
		s.addRef(d, r, xsize)
	}
}

// code returns the population of code k (0 literal, 1 red, 2 blue, 3
// alpha, 4 distance) of h.
func (s *histoSet) code(h *histo, k int) []uint32 {
	L := s.litLen
	switch k {
	case 0:
		return h.d[:L]
	case 4:
		return h.d[L+768 : L+808]
	}
	return h.d[L+256*(k-1) : L+256*k]
}

// estimate returns the estimated number of bits to code h, including the
// extra bits of lengths and distances (libwebp VP8LHistogramEstimateBits).
func (s *histoSet) estimate(h *histo) float64 {
	L := s.litLen
	d := h.d
	return populationCost(d[:L]) + populationCost(d[L:L+256]) + populationCost(d[L+256:L+512]) +
		populationCost(d[L+512:L+768]) +
		populationCost(d[L+768:L+808]) +
		extraCost(d[numLiteralCodes:numLiteralCodes+numLengthCodes]) +
		extraCost(d[L+768:L+808])
}

// computeCost sets the clustering state of h from its populations (libwebp
// ComputeHistogramCost).
func (s *histoSet) computeCost(h *histo) {
	h.bitCost = 0
	for k := 0; k < 5; k++ {
		pop := s.code(h, k)
		h.costs[k] = populationCost(pop)
		nz, sym := 0, 0
		for i, v := range pop {
			if v != 0 {
				nz++
				sym = i
				if nz > 1 {
					break
				}
			}
		}
		h.isUsed[k] = nz > 0
		h.trivial[k] = nonTrivialSym
		if nz == 1 {
			h.trivial[k] = int32(sym)
		}
		h.bitCost += h.costs[k]
	}
}

// combinedCost computes the cost of a+b (without extra bits) into costs and
// returns it, or ok = false if it is not below threshold (libwebp
// GetCombinedHistogramEntropy).
func (s *histoSet) combinedCost(a, b *histo, threshold float64, costs *[5]float64) (cost float64, ok bool) {
	if threshold <= 0 {
		return 0, false
	}
	for k := 0; k < 5; k++ {
		var c float64
		switch {
		case a.trivial[k] != nonTrivialSym && a.trivial[k] == b.trivial[k], !a.isUsed[k] || !b.isUsed[k]:
			if a.isUsed[k] {
				c = a.costs[k]
			} else {
				c = b.costs[k]
			}
		default:
			c = combinedPopulationCost(s.code(a, k), s.code(b, k))
		}
		costs[k] = c
		cost += c
		if cost >= threshold {
			return cost, false
		}
	}
	return cost, true
}

// addHisto sets out = a + b, propagating the clustering state like libwebp's
// HistogramAdd (out may be b).
func addHisto(a, b, out *histo) {
	x, y, o := a.d, b.d[:len(a.d)], out.d[:len(a.d)]
	for i := range o {
		o[i] = x[i] + y[i]
	}
	for k := 0; k < 5; k++ {
		if a.trivial[k] != b.trivial[k] {
			out.trivial[k] = nonTrivialSym
		} else {
			out.trivial[k] = a.trivial[k]
		}
		out.isUsed[k] = a.isUsed[k] || b.isUsed[k]
	}
}

// addInto adds b into a (data only).
func addInto(a, b *histo) {
	x := a.d
	y := b.d[:len(x)]
	for i := range x {
		x[i] += y[i]
	}
}

// ---- clustering ----
//
// The clustering follows libwebp 1.6 (histogram_enc.c) closely, including the
// order in which histograms are removed from the working set (the last one
// takes the place of the removed one), since it drives the random pair
// selection and the final cluster order.

type histoPair struct {
	i1, i2   int
	costDiff float64
	costCmb  float64
	costs    [5]float64
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

// fixPair replaces bad by good in p.
func fixPair(bad, good int, p *histoPair) {
	if p.i1 == bad {
		p.i1 = good
	}
	if p.i2 == bad {
		p.i2 = good
	}
	if p.i1 > p.i2 {
		p.i1, p.i2 = p.i2, p.i1
	}
}

// updatePair evaluates the merge of h1 and h2 into p; it fails when the
// cost difference is not below threshold.
func updatePair(s *histoSet, h1, h2 *histo, threshold float64, p *histoPair) bool {
	sum := h1.bitCost + h2.bitCost
	cmb, ok := s.combinedCost(h1, h2, sum+threshold, &p.costs)
	if !ok {
		return false
	}
	p.costCmb = cmb
	p.costDiff = cmb - sum
	return true
}

// push evaluates the pair (i1,i2) and inserts it if merging gains more than
// threshold. Returns the cost difference (0 when not pushed).
func (pq *pairQueue) push(s *histoSet, set []*histo, i1, i2 int, threshold float64) float64 {
	if len(pq.q) >= pq.max {
		return 0
	}
	if i1 > i2 {
		i1, i2 = i2, i1
	}
	p := histoPair{i1: i1, i2: i2}
	if !updatePair(s, set[i1], set[i2], threshold, &p) {
		return 0
	}
	pq.q = append(pq.q, p)
	pq.updateHead(len(pq.q) - 1)
	return p.costDiff
}

const numPartitions = 4
const binSize = numPartitions * numPartitions * numPartitions
const maxHistoGreedy = 100

func binID(min, max, v float64) int {
	r := max - min
	if r > 0 {
		return int((numPartitions - 1e-6) * (v - min) / r)
	}
	return 0
}

// combineCostFactor is libwebp's GetCombineCostFactor, divided by 100.
func combineCostFactor(histoSize, quality int) float64 {
	f := 16
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
	return float64(f) / 100
}

type clusterer struct {
	orig    histoSet // per-tile histograms
	symbols []uint16
	mapping []int
	queue   pairQueue
	// set[:size] is the working set: the live histograms. It starts with
	// the (shared) non-empty tile histograms themselves; a merge writes its
	// result into a histogram of owned (copy on write), so that only the
	// merged histograms take memory besides the tile histograms.
	set       []*histo
	size      int
	combo     *histo // scratch histogram of the entropy-bin combination
	cacheBits int
	binFirst  [binSize]int
	binFail   [binSize]int

	owned     []*histo // working histograms (kept for later calls)
	numOwned  int      // owned[:numOwned] are in use or in free
	freeOwned []*histo

	// interrupt is Config.Interrupt during buildHistoImage: when it fires,
	// the clustering loops stop early (the clustering stays valid).
	interrupt func() bool
	// crew runs the parallel remap (the crew of the scratch, set during
	// buildHistoImage).
	crew *crew
}

// newHisto returns an unused working histogram (contents unspecified).
func (c *clusterer) newHisto() *histo {
	var h *histo
	if k := len(c.freeOwned); k > 0 {
		h = c.freeOwned[k-1]
		c.freeOwned = c.freeOwned[:k-1]
		return h
	}
	if c.numOwned == len(c.owned) {
		c.owned = append(c.owned, new(histo))
	}
	h = c.owned[c.numOwned]
	c.numOwned++
	stride := c.orig.stride
	if cap(h.d) < stride {
		h.d = make([]uint32, stride)
	}
	h.d = h.d[:stride:stride]
	return h
}

// release hands h (no longer in the working set) back to newHisto if it is
// a working histogram.
func (c *clusterer) release(h *histo) {
	if !h.shared {
		c.freeOwned = append(c.freeOwned, h)
	}
}

// mergeInto sets set[i] to set[j] + set[i] (libwebp HistogramAdd(set[j],
// set[i], set[i])), in place unless set[i] is a shared tile histogram.
func (c *clusterer) mergeInto(i, j int) *histo {
	dst := c.set[i]
	if dst.shared {
		dst = c.newHisto()
	}
	addHisto(c.set[j], c.set[i], dst)
	c.set[i] = dst
	return dst
}

func (c *clusterer) interrupted() bool { return c.interrupt != nil && c.interrupt() }

// remove removes set[i], replacing it by the last histogram (libwebp
// HistogramSetRemoveHistogram).
func (c *clusterer) remove(i int) {
	c.release(c.set[i])
	c.set[i] = c.set[c.size-1]
	c.size--
}

// minParallelRemap is the smallest remap (tiles times clusters) spread over
// helper goroutines (a variable for the tests).
var minParallelRemap = 1 << 12

// buildHistoImage computes the clustered histograms for refs. It returns the
// final histograms (in out) and fills c.symbols with the tile->histogram map.
// helpers > 0 allows a parallel remap.
func (c *clusterer) buildHistoImage(xsize, ysize int, refs []pixOrCopy, quality int,
	lowEffort bool, histoBits, cacheBits int, out *histoSet, helpers int) int {
	hx := subSampleSize(xsize, histoBits)
	hy := subSampleSize(ysize, histoBits)
	nt := hx * hy
	c.cacheBits = cacheBits
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

	// Analyze the tile histograms; the non-empty ones form the working set.
	numUsed := 0
	for i := range c.orig.h {
		h := &c.orig.h[i]
		c.orig.computeCost(h)
		h.used = h.isUsed[0] || h.isUsed[1] || h.isUsed[2] || h.isUsed[3] || h.isUsed[4]
		h.shared = true
		if h.used {
			numUsed++
		}
	}
	if numUsed == 0 {
		// Empty image (cannot happen for valid input); single histogram.
		out.init(1, cacheBits)
		clear(c.symbols)
		return 1
	}
	if cap(c.set) < numUsed {
		c.set = make([]*histo, numUsed)
	}
	c.set = c.set[:numUsed]
	k := 0
	for i := range c.orig.h {
		if o := &c.orig.h[i]; o.used {
			c.set[k] = o
			k++
		}
	}
	c.size = numUsed
	c.numOwned = 0
	c.freeOwned = c.freeOwned[:0]
	c.combo = c.newHisto()
	// The working histograms have the layout of the tile histograms: the
	// combination steps use c.orig for it.
	s := &c.orig

	entropyBins := binSize
	if lowEffort {
		entropyBins = numPartitions
	}
	entropyCombine := numUsed > 2*entropyBins && quality < 100
	if entropyCombine {
		c.entropyBinCombine(s, entropyBins, lowEffort, combineCostFactor(nt, quality))
	}
	if !lowEffort || !entropyCombine {
		thresholdSize := 1 + (quality*quality*quality*(maxHistoGreedy-1)+(100*100*100)/2)/(100*100*100)
		if c.stochasticCombine(s, thresholdSize) {
			c.greedyCombine(s)
		}
	}
	c.remap(out, helpers)
	return len(out.h)
}

// entropyBinCombine merges histograms with similar dominant costs (libwebp
// HistogramAnalyzeEntropyBin and HistogramCombineEntropyBin).
func (c *clusterer) entropyBinCombine(s *histoSet, numBins int, lowEffort bool, factor float64) {
	set := c.set
	litMin, litMax := math.MaxFloat64, -math.MaxFloat64
	redMin, redMax := litMin, litMax
	blueMin, blueMax := litMin, litMax
	for _, h := range set[:c.size] {
		litMin = math.Min(litMin, h.costs[0])
		litMax = math.Max(litMax, h.costs[0])
		redMin = math.Min(redMin, h.costs[1])
		redMax = math.Max(redMax, h.costs[1])
		blueMin = math.Min(blueMin, h.costs[2])
		blueMax = math.Max(blueMax, h.costs[2])
	}
	// (binID is scratch state: it is also set on shared histograms.)
	for _, h := range set[:c.size] {
		b := binID(litMin, litMax, h.costs[0])
		if !lowEffort {
			b = b*numPartitions + binID(redMin, redMax, h.costs[1])
			b = b*numPartitions + binID(blueMin, blueMax, h.costs[2])
		}
		h.binID = b
	}
	for i := 0; i < numBins; i++ {
		c.binFirst[i] = -1
		c.binFail[i] = 0
	}
	const maxCombineFailures = 32
	for idx := 0; idx < c.size; {
		h := set[idx]
		b := h.binID
		first := c.binFirst[b]
		if first == -1 {
			c.binFirst[b] = idx
			idx++
			continue
		}
		f := set[first]
		if lowEffort {
			c.mergeInto(first, idx)
			c.remove(idx)
			continue
		}
		thresh := -h.bitCost * factor
		sum := f.bitCost + h.bitCost
		combo := c.combo
		cmb, ok := s.combinedCost(f, h, sum+thresh, &combo.costs)
		if !ok {
			idx++
			continue
		}
		addHisto(f, h, combo)
		combo.bitCost = cmb
		// Merge only if the combination is trivial or both histograms are
		// already non-trivial (unless merges keep failing for this bin).
		try := combo.trivial[1] != nonTrivialSym && combo.trivial[2] != nonTrivialSym &&
			combo.trivial[3] != nonTrivialSym
		if !try {
			try = (h.trivial[1] == nonTrivialSym || h.trivial[2] == nonTrivialSym || h.trivial[3] == nonTrivialSym) &&
				(f.trivial[1] == nonTrivialSym || f.trivial[2] == nonTrivialSym || f.trivial[3] == nonTrivialSym)
		}
		if try || c.binFail[b] >= maxCombineFailures {
			// combo replaces f, which becomes the scratch histogram (unless
			// it is shared).
			set[first], c.combo = combo, f
			if f.shared {
				c.combo = c.newHisto()
			}
			c.remove(idx)
		} else {
			c.binFail[b]++
			idx++
		}
	}
	if lowEffort {
		// (The costs of the unmerged, shared histograms are up to date.)
		for _, h := range set[:c.size] {
			if !h.shared {
				s.computeCost(h)
			}
		}
	}
}

// myRand is libwebp's MyRand Lehmer generator.
func myRand(seed *uint32) uint32 {
	*seed = uint32(uint64(*seed) * 48271 % 2147483647)
	return *seed
}

// stochasticCombine merges random pairs until the number of histograms is
// below minClusterSize. Returns whether a greedy pass should follow.
func (c *clusterer) stochasticCombine(s *histoSet, minClusterSize int) bool {
	if c.size < minClusterSize {
		return true
	}
	set := c.set
	const queueSize = 9
	pq := &c.queue
	pq.q = pq.q[:0]
	pq.max = queueSize
	seed := uint32(1)
	outerIters := c.size
	numTriesNoSuccess := outerIters / 2
	triesNoSuccess := 0
	for iter := 0; iter < outerIters && c.size >= minClusterSize && !c.interrupted(); iter++ {
		triesNoSuccess++
		if triesNoSuccess >= numTriesNoSuccess {
			break
		}
		bestCost := 0.0
		if len(pq.q) > 0 {
			bestCost = pq.q[0].costDiff
		}
		randRange := uint32((c.size - 1) * c.size)
		numTries := c.size / 2
		for j := 0; c.size >= 2 && j < numTries; j++ {
			t := myRand(&seed) % randRange
			i1 := t / uint32(c.size-1)
			i2 := t % uint32(c.size-1)
			if i2 >= i1 {
				i2++
			}
			cur := pq.push(s, set, int(i1), int(i2), bestCost)
			if cur < 0 {
				bestCost = cur
				if len(pq.q) == pq.max {
					break
				}
			}
		}
		if len(pq.q) == 0 {
			continue
		}
		b1, b2 := pq.q[0].i1, pq.q[0].i2
		m := c.mergeInto(b1, b2)
		m.bitCost = pq.q[0].costCmb
		m.costs = pq.q[0].costs
		c.remove(b2)
		for j := 0; j < len(pq.q); {
			p := &pq.q[j]
			is1 := p.i1 == b1 || p.i1 == b2
			is2 := p.i2 == b1 || p.i2 == b2
			if is1 && is2 {
				pq.pop(j)
				continue
			}
			if is1 || is2 {
				fixPair(b2, b1, p)
				if !updatePair(s, set[p.i1], set[p.i2], 0, p) {
					pq.pop(j)
					continue
				}
			}
			fixPair(c.size, b2, p)
			pq.updateHead(j)
			j++
		}
		triesNoSuccess = 0
	}
	return c.size <= minClusterSize
}

// greedyCombine repeatedly merges the best pair while it reduces the cost.
func (c *clusterer) greedyCombine(s *histoSet) {
	set := c.set
	n := c.size
	pq := &c.queue
	pq.q = pq.q[:0]
	pq.max = n * n
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			pq.push(s, set, i, j, 0)
		}
	}
	for len(pq.q) > 0 && !c.interrupted() {
		i1, i2 := pq.q[0].i1, pq.q[0].i2
		m := c.mergeInto(i1, i2)
		m.bitCost = pq.q[0].costCmb
		m.costs = pq.q[0].costs
		c.remove(i2)
		for k := 0; k < len(pq.q); {
			p := &pq.q[k]
			if p.i1 == i1 || p.i2 == i1 || p.i1 == i2 || p.i2 == i2 {
				pq.pop(k)
			} else {
				fixPair(c.size, i2, p)
				pq.updateHead(k)
				k++
			}
		}
		for k := 0; k < c.size; k++ {
			if k == i1 {
				continue
			}
			pq.push(s, set, i1, k, 0)
		}
	}
}

// remapChunk is the number of tiles per work item of a parallel remap.
const remapChunk = 32

// remap assigns each tile histogram to the best final cluster (libwebp
// HistogramRemap), renumbers the clusters by first use (dropping the
// unused ones) and recomputes the final histograms into out. With helpers
// > 0 and enough work, the assignments (independent per tile) are spread
// over helper goroutines.
func (c *clusterer) remap(out *histoSet, helpers int) {
	set := c.set[:c.size]
	syms := c.symbols
	orig := c.orig.h
	if len(set) > 1 {
		assign := func(lo, hi int) {
			var costs [5]float64
			for i := lo; i < hi; i++ {
				if !orig[i].used {
					continue
				}
				best := 0
				bestBits := math.MaxFloat64
				for k, h := range set {
					cmb, ok := c.orig.combinedCost(h, &orig[i], h.bitCost+bestBits, &costs)
					if ok {
						bestBits = cmb - h.bitCost
						best = k
					}
				}
				syms[i] = uint16(best)
			}
		}
		// Config.Interrupt is polled once per chunk of remapChunk tiles,
		// also on the helpers; an interrupted remap leaves every tile in
		// cluster 0 (a valid, if useless, clustering).
		chunks := (len(orig) + remapChunk - 1) / remapChunk
		if helpers > 0 && chunks > 1 && len(orig)*len(set) >= minParallelRemap {
			c.crew.run(chunks, helpers, func(_ int, claim func() (int, bool)) {
				for {
					k, ok := claim()
					if !ok || c.interrupted() {
						return
					}
					assign(k*remapChunk, min((k+1)*remapChunk, len(orig)))
				}
			})
		} else {
			for k := 0; k < chunks && !c.interrupted(); k++ {
				assign(k*remapChunk, min((k+1)*remapChunk, len(orig)))
			}
		}
		if c.interrupted() {
			clear(syms)
		} else {
			// Unused tiles take the cluster of the previous tile.
			for i := range orig {
				if !orig[i].used {
					if i > 0 {
						syms[i] = syms[i-1]
					} else {
						syms[i] = 0
					}
				}
			}
		}
	} else {
		clear(syms)
	}
	// Renumber by first appearance; drop clusters nobody maps to.
	order := c.mapping[:0]
	for range set {
		order = append(order, -1)
	}
	n := 0
	for i, sym := range syms {
		if order[sym] == -1 {
			order[sym] = n
			n++
		}
		syms[i] = uint16(order[sym])
	}
	c.mapping = order
	out.init(n, c.cacheBits)
	for i := range orig {
		if orig[i].used {
			addInto(&out.h[syms[i]], &orig[i])
		}
	}
}
