package vp8lenc

import "math"

// Backward reference (LZ77) computation.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	modeLiteral = 0
	modeCache   = 1
	modeCopy    = 2
)

// pixOrCopy is a literal pixel, a color cache index or a (distance, length)
// copy. Distances are linear pixel distances (converted to plane codes when
// the stream is written).
type pixOrCopy struct {
	val uint32 // argb, cache index or distance
	lm  uint32 // len<<2 | mode
}

func (p pixOrCopy) mode() uint32 { return p.lm & 3 }
func (p pixOrCopy) length() int  { return int(p.lm >> 2) }

func mkLiteral(argb uint32) pixOrCopy     { return pixOrCopy{argb, 1<<2 | modeLiteral} }
func mkCache(idx uint32) pixOrCopy        { return pixOrCopy{idx, 1<<2 | modeCache} }
func mkCopy(dist uint32, l int) pixOrCopy { return pixOrCopy{dist, uint32(l)<<2 | modeCopy} }

// initRefsCap returns the initial capacity of a reference buffer for the
// w x h image argb. A parse has at most one entry per pixel, but graphics
// and alpha planes typically need 100-1000 times fewer, so the buffer is
// sized from a cheap estimate: the number of pixels that differ from both
// their left and top neighbors (each such pixel starts a literal or a
// copy), with 50% headroom. Photos thus get one allocation of about n
// entries, flat images a small one; growRefs covers underestimates. The
// pixels are counted in every rleSampleStep-th row only (all rows below
// rleSampleMinRows): a full pass took 3% of a level 0 encode.
func initRefsCap(argb []uint32, w, h int) int {
	n := w * h
	if n <= 4096 {
		return n
	}
	step := 1
	if h >= rleSampleMinRows {
		step = rleSampleStep
	}
	// (The first row, which has no row above, is counted apart.)
	est, rows := 0, 0
	for y := step; y < h; y += step {
		est += newPixels(argb[y*w:(y+1)*w], argb[(y-1)*w:y*w], argb[y*w-1])
		rows++
	}
	if step > 1 {
		est = int(int64(est) * int64(h-1) / int64(rows))
	}
	est += 1 + newPixels(argb[1:w], nil, argb[0])
	return refsCapFromEstimate(n, est)
}

// refsCapFromEstimate is initRefsCap for an image of n pixels of which est
// differ from their left and top neighbors.
func refsCapFromEstimate(n, est int) int {
	if n <= 4096 {
		return n
	}
	return min(n, est+est/2+4096)
}

// newPixels returns the number of pixels of row that differ from their left
// neighbor (left for the first one) and, if up is not nil, from their top
// neighbor in up.
func newPixels(row, up []uint32, left uint32) int {
	n := 0
	if up == nil {
		for _, p := range row {
			if p != left {
				n++
			}
			left = p
		}
		return n
	}
	up = up[:len(row)]
	for x, p := range row {
		if p != left && p != up[x] {
			n++
		}
		left = p
	}
	return n
}

// growRefs returns refs with room for at least one more entry, doubling its
// capacity but never beyond limit entries (the most a parse of limit pixels
// can produce) unless needed.
func growRefs(refs []pixOrCopy, limit int) []pixOrCopy {
	c := max(min(2*cap(refs), limit), cap(refs)+1, 64)
	r := make([]pixOrCopy, len(refs), c)
	copy(r, refs)
	return r
}

// backwardRefsLz77 computes greedy LZ77 references with one-step lookahead.
func backwardRefsLz77(n int, argb []uint32, hc *hashChain, refs []pixOrCopy) []pixOrCopy {
	refs = refs[:0]
	ol := hc.ol[:n]
	for i := 0; i < n; {
		v := ol[i]
		l := int(v & (1<<maxLengthBits - 1))
		off := v >> maxLengthBits
		if l >= minLength {
			jMax := i + l
			if jMax >= n {
				jMax = n - 1
			}
			maxReach := 0
			for j := i + 1; j <= jMax; j++ {
				lj := int(ol[j] & (1<<maxLengthBits - 1))
				reach := j + 1
				if lj >= minLength {
					reach = j + lj
				}
				if reach > maxReach {
					l = j - i
					maxReach = reach
					if maxReach >= n {
						break
					}
				}
			}
		} else {
			l = 1
		}
		if len(refs) == cap(refs) {
			refs = growRefs(refs, n)
		}
		if l == 1 {
			refs = append(refs, mkLiteral(argb[i]))
		} else {
			refs = append(refs, mkCopy(off, l))
		}
		i += l
	}
	return refs
}

// rleSampleStep is the row sampling of the estimates of the parse sizes
// (initRefsCap, rleRefsEstimate) for images of at least rleSampleMinRows
// rows.
const (
	rleSampleStep    = 16
	rleSampleMinRows = 64
)

// rleRefsEstimate estimates the number of entries of the RLE parse of the
// w x h image argb (backwardRefsRle) from its entries starting in every
// rleSampleStep-th row (it counts them all in images of fewer than
// rleSampleMinRows rows). initRefsCap undercounts them on photos (pixels
// equal to a neighbor are often literals of the RLE parse, in runs too short
// for a copy), whose RLE parse would then outgrow its buffer (a copy into
// a buffer of one entry per pixel, both alive at once).
func rleRefsEstimate(argb []uint32, w, h int) int {
	n := w * h
	argb = argb[:n]
	if h < rleSampleMinRows {
		return rleEntries(argb, w, 0, n)
	}
	// (The first row, which has no row above, is counted apart.)
	count, rows := 0, 0
	for y := rleSampleStep; y < h; y += rleSampleStep {
		count += rleEntries(argb, w, y*w, (y+1)*w)
		rows++
	}
	return rleEntries(argb, w, 0, w) + int(int64(count)*int64(h-1)/int64(rows))
}

// rleEntries returns the number of entries of backwardRefsRle that start
// in argb[lo:hi] (w pixels wide), as if one started at lo.
func rleEntries(argb []uint32, w, lo, hi int) int {
	n := len(argb)
	i, k := lo, 0
	if i == 0 {
		k, i = 1, 1 // the first pixel is a literal
	}
	for ; i < hi; k++ {
		maxLen := min(n-i, maxLength)
		cur := argb[i : i+maxLen]
		rle := vectorMismatch(cur, argb[i-1:], maxLen)
		prow := 0
		if i >= w {
			prow = vectorMismatch(cur, argb[i-w:], maxLen)
		}
		switch {
		case rle >= prow && rle >= minLength:
			i += rle
		case prow >= minLength:
			i += prow
		default:
			i++
		}
	}
	return k
}

// backwardRefsRle computes references using only distance 1 and xsize.
func backwardRefsRle(xsize, n int, argb []uint32, refs []pixOrCopy) []pixOrCopy {
	refs = refs[:0]
	argb = argb[:n]
	if cap(refs) == 0 {
		refs = growRefs(refs, n)
	}
	refs = append(refs, mkLiteral(argb[0]))
	for i := 1; i < n; {
		if len(refs) == cap(refs) {
			refs = growRefs(refs, n)
		}
		maxLen := n - i
		if maxLen > maxLength {
			maxLen = maxLength
		}
		cur := argb[i : i+maxLen]
		rle := vectorMismatch(cur, argb[i-1:], maxLen)
		prow := 0
		if i >= xsize {
			prow = vectorMismatch(cur, argb[i-xsize:], maxLen)
		}
		if rle >= prow && rle >= minLength {
			refs = append(refs, mkCopy(1, rle))
			i += rle
		} else if prow >= minLength {
			refs = append(refs, mkCopy(uint32(xsize), prow))
			i += prow
		} else {
			refs = append(refs, mkLiteral(argb[i]))
			i++
		}
	}
	return refs
}

// applyColorCache converts literals into cache references where possible.
func applyColorCache(argb []uint32, cacheBits int, refs []pixOrCopy, cache []uint32) {
	if cacheBits == 0 {
		return
	}
	cache = cache[:1<<uint(cacheBits)]
	for i := range cache {
		cache[i] = 0
	}
	shift := uint(32 - cacheBits)
	pos := 0
	// The cache starts zero-filled but an index is only used on an exact
	// hit, and a hit on an unset slot would require argb == 0 which is
	// valid since the decoder's cache is zero-initialised as well.
	for k, r := range refs {
		if r.mode() == modeLiteral {
			a := r.val
			key := colorCacheHash(a, shift)
			if cache[key] == a {
				refs[k] = mkCache(key)
			} else {
				cache[key] = a
			}
			pos++
		} else {
			l := r.length()
			// Runs of the same color insert it once.
			prev := ^argb[pos]
			for _, a := range argb[pos : pos+l] {
				if a != prev {
					cache[colorCacheHash(a, shift)] = a
					prev = a
				}
			}
			pos += l
		}
	}
}

// bestCacheBits returns the color cache size minimizing the estimated
// entropy of refs (which must contain no cache references).
func bestCacheBits(argb []uint32, quality int, refs []pixOrCopy, maxBits int, sc *scratch) int {
	if quality <= 25 || maxBits == 0 {
		return 0
	}
	type cacheHisto struct {
		lit              []uint32
		red, blue, alpha [256]uint32
	}
	// Literal histograms: [0..280) + cache indices.
	var hs [maxColorCacheBits + 1]cacheHisto
	total := 0
	for i := 0; i <= maxBits; i++ {
		n := numLiteralCodes + numLengthCodes
		if i > 0 {
			n += 1 << uint(i)
		}
		total += n
	}
	buf := sc.u32a(total + (2 << uint(maxBits)))
	for i := range buf {
		buf[i] = 0
	}
	off := 0
	var caches [maxColorCacheBits + 1][]uint32
	for i := 0; i <= maxBits; i++ {
		n := numLiteralCodes + numLengthCodes
		if i > 0 {
			n += 1 << uint(i)
		}
		hs[i].lit = buf[off : off+n]
		off += n
	}
	for i := 1; i <= maxBits; i++ {
		caches[i] = buf[off : off+(1<<uint(i))]
		off += 1 << uint(i)
	}
	shift := uint(32 - maxBits)
	pos := 0
	for _, r := range refs {
		if r.mode() == modeLiteral {
			pix := argb[pos]
			pos++
			a, rr, g, b := pix>>24, (pix>>16)&0xff, (pix>>8)&0xff, pix&0xff
			key := colorCacheHash(pix, shift)
			hs[0].alpha[a]++
			hs[0].red[rr]++
			hs[0].lit[g]++
			hs[0].blue[b]++
			for i := maxBits; i >= 1; i-- {
				c := caches[i]
				if c[key] == pix {
					hs[i].lit[numLiteralCodes+numLengthCodes+int(key)]++
				} else {
					c[key] = pix
					hs[i].alpha[a]++
					hs[i].red[rr]++
					hs[i].lit[g]++
					hs[i].blue[b]++
				}
				key >>= 1
			}
		} else {
			l := r.length()
			code := prefixCode(l)
			for i := 0; i <= maxBits; i++ {
				hs[i].lit[numLiteralCodes+code]++
			}
			prev := argb[pos] ^ 0xffffffff
			for _, p := range argb[pos : pos+l] {
				if p != prev {
					key := colorCacheHash(p, shift)
					for i := maxBits; i >= 1; i-- {
						caches[i][key] = p
						key >>= 1
					}
					prev = p
				}
			}
			pos += l
		}
	}
	best := 0
	bestCost := math.MaxFloat64
	for i := 0; i <= maxBits; i++ {
		c := populationCost(hs[i].lit) + populationCost(hs[i].red[:]) +
			populationCost(hs[i].blue[:]) + populationCost(hs[i].alpha[:]) +
			extraCost(hs[i].lit[numLiteralCodes:numLiteralCodes+numLengthCodes])
		if i == 0 || c < bestCost {
			bestCost = c
			best = i
		}
	}
	return best
}

// ---- cost-based optimal parsing ----
//
// Port of libwebp 1.6 src/enc/backward_references_cost_enc.c (BSD-3-Clause),
// with its fixed-point costs.

// costModel holds per-symbol bit cost estimates (fixed point).
type costModel struct {
	alpha, red, blue [256]uint32
	lit              []uint32 // green + length prefixes + cache
	dist             [numDistanceCodes]uint32
}

// popToBits is libwebp's ConvertPopulationCountTableToBitEstimates.
func popToBits(pop []uint32, out []uint32) {
	var sum uint32
	nz := 0
	for _, v := range pop {
		sum += v
		if v > 0 {
			nz++
		}
	}
	if nz <= 1 {
		clear(out)
		return
	}
	ls := fastLog2(sum)
	for i, v := range pop {
		out[i] = ls - fastLog2(v)
	}
}

func (m *costModel) build(xsize int, refs []pixOrCopy, h *histo, hs *histoSet) {
	hs.clear(h)
	hs.addRefs(h, refs, xsize)
	L := hs.litLen
	d := h.d
	if cap(m.lit) < L {
		m.lit = make([]uint32, L)
	}
	m.lit = m.lit[:L]
	popToBits(d[:L], m.lit)
	popToBits(d[L:L+256], m.red[:])
	popToBits(d[L+256:L+512], m.blue[:])
	popToBits(d[L+512:L+768], m.alpha[:])
	popToBits(d[L+768:L+808], m.dist[:])
}

// prefixEncodeBits is libwebp's VP8LPrefixEncodeBits (0 maps to code 0).
func prefixEncodeBits(v int) (code, nbits int) {
	if v == 0 {
		return 0, 0
	}
	code, nbits, _ = prefixEncode(v)
	return
}

func (m *costModel) lengthCost(length int) int64 {
	code, nb := prefixEncodeBits(length)
	return int64(m.lit[numLiteralCodes+code]) + int64(nb)<<log2PrecisionBits
}

func (m *costModel) distanceCost(planeCode int) int64 {
	code, nb := prefixEncodeBits(planeCode)
	return int64(m.dist[code]) + int64(nb)<<log2PrecisionBits
}

// costInterval is a range [start, end) of pixels whose cost may be lowered
// to cost by a copy starting at index.
type costInterval struct {
	cost            int64
	start, end, idx int
	prev, next      *costInterval
}

type costCacheInterval struct {
	cost       int64
	start, end int
}

const (
	costCacheIntervalSizeMax = 500

	// A copy found at pixel j <= i+maxLength (see the reach logic in
	// traceBackwards) covers pixels up to j+maxLength-1, so costs are
	// written at most costAhead pixels ahead of the current one.
	costAhead = 2*maxLength + 2
	costChunk = 4096
	costRing  = 1 << 15 // > costAhead + costChunk + 1
	costMask  = costRing - 1
)

// costManager keeps the minimal copy costs of the pixels ahead as
// non-overlapping intervals sorted by start (libwebp CostManager). At most
// costCacheIntervalSizeMax intervals exist at a time; they live in slab.
type costManager struct {
	slab      [costCacheIntervalSizeMax]costInterval
	used      int // slab[:used] have been handed out
	free      *costInterval
	head      *costInterval
	count     int
	cacheIv   []costCacheInterval
	costCache []int64
	// costs holds the cost of pixels [i-1, i+costAhead+costChunk) (i being
	// the current pixel) in a ring indexed by pixel&costMask: older costs
	// are never read again.
	costs    *[costRing]int64
	n        int
	initUpTo int // the costs of pixels below initUpTo are initialized
	dist     []uint16
}

func (cm *costManager) init(dist []uint16, costs *[costRing]int64, n int, m *costModel) {
	cm.used = 0
	cm.free = nil
	cm.head = nil
	cm.count = 0
	cm.dist = dist
	sz := min(n, maxLength)
	if cap(cm.costCache) < sz {
		cm.costCache = make([]int64, sz)
	}
	cm.costCache = cm.costCache[:sz]
	for i := range cm.costCache {
		cm.costCache[i] = m.lengthCost(i)
	}
	cm.cacheIv = cm.cacheIv[:0]
	cur := costCacheInterval{cost: cm.costCache[0], start: 0, end: 1}
	for i := 1; i < sz; i++ {
		if c := cm.costCache[i]; c != cur.cost {
			cm.cacheIv = append(cm.cacheIv, cur)
			cur = costCacheInterval{cost: c, start: i}
		}
		cur.end = i + 1
	}
	cm.cacheIv = append(cm.cacheIv, cur)
	cm.costs = costs
	cm.n = n
	cm.initUpTo = 0
}

// prepare initializes (to MaxInt64) the costs of the pixels that may be
// written while processing pixel i.
func (cm *costManager) prepare(i int) {
	need := min(i+costAhead, cm.n)
	if need <= cm.initUpTo {
		return
	}
	end := min(need+costChunk, cm.n)
	for k := cm.initUpTo; k < end; {
		// Fill up to the end of the ring or to end.
		lo := k & costMask
		c := cm.costs[lo:min(costRing, lo+end-k)]
		for j := range c {
			c[j] = math.MaxInt64
		}
		k += len(c)
	}
	cm.initUpTo = end
}

func (cm *costManager) updateCost(i, position int, cost int64) {
	if c := &cm.costs[i&costMask]; *c > cost {
		*c = cost
		cm.dist[i] = uint16(i - position + 1)
	}
}

func (cm *costManager) updateCostPerInterval(start, end, position int, cost int64) {
	for i := start; i < end; i++ {
		cm.updateCost(i, position, cost)
	}
}

func (cm *costManager) connect(prev, next *costInterval) {
	if prev != nil {
		prev.next = next
	} else {
		cm.head = next
	}
	if next != nil {
		next.prev = prev
	}
}

func (cm *costManager) pop(iv *costInterval) {
	if iv == nil {
		return
	}
	cm.connect(iv.prev, iv.next)
	iv.next = cm.free
	cm.free = iv
	cm.count--
}

// updateCostAtIndex lowers the cost of pixel i with the intervals covering
// it; with clean, it drops the intervals ending before i.
func (cm *costManager) updateCostAtIndex(i int, clean bool) {
	cur := cm.head
	for cur != nil && cur.start <= i {
		next := cur.next
		if cur.end <= i {
			if clean {
				cm.pop(cur)
			}
		} else if c := &cm.costs[i&costMask]; *c > cur.cost {
			*c = cur.cost
			cm.dist[i] = uint16(i - cur.idx + 1)
		}
		cur = next
	}
}

// positionOrphan links the unlinked interval cur into the list, using prev
// as a hint.
func (cm *costManager) positionOrphan(cur, prev *costInterval) {
	if prev == nil {
		prev = cm.head
	}
	start := cur.start
	for prev != nil && start < prev.start {
		prev = prev.prev
	}
	for prev != nil && prev.next != nil && prev.next.start < start {
		prev = prev.next
	}
	if prev != nil {
		cm.connect(cur, prev.next)
	} else {
		cm.connect(cur, cm.head)
	}
	cm.connect(prev, cur)
}

func (cm *costManager) insert(hint *costInterval, cost int64, position, start, end int) {
	if start >= end {
		return
	}
	if cm.count >= costCacheIntervalSizeMax {
		cm.updateCostPerInterval(start, end, position, cost)
		return
	}
	var iv *costInterval
	if cm.free != nil {
		iv = cm.free
		cm.free = iv.next
	} else {
		iv = &cm.slab[cm.used]
		cm.used++
	}
	*iv = costInterval{cost: cost, idx: position, start: start, end: end}
	cm.positionOrphan(iv, hint)
	cm.count++
}

// push adds the contributions of a copy of up to len pixels at position
// whose distance cost (plus the cost before position) is distCost.
func (cm *costManager) push(distCost int64, position, length int) {
	const skipDistance = 10
	if length < skipDistance {
		cc := cm.costCache[:length]
		for k, c := range cc {
			c += distCost
			if p := &cm.costs[(position+k)&costMask]; *p > c {
				*p = c
				cm.dist[position+k] = uint16(k + 1)
			}
		}
		return
	}
	iv := cm.head
	for _, ci := range cm.cacheIv {
		if ci.start >= length {
			break
		}
		start := position + ci.start
		end := position + min(ci.end, length)
		cost := distCost + ci.cost
		for iv != nil && iv.start < end {
			next := iv.next
			if start >= iv.end {
				iv = next
				continue
			}
			if cost >= iv.cost {
				startNew := iv.end
				cm.insert(iv, cost, position, start, iv.start)
				start = startNew
				if start >= end {
					break
				}
				iv = next
				continue
			}
			if start <= iv.start {
				if iv.end <= end {
					cm.pop(iv)
				} else {
					iv.start = end
					break
				}
			} else {
				if end < iv.end {
					endOrig := iv.end
					iv.end = start
					cm.insert(iv, iv.cost, iv.idx, end, endOrig)
					iv = iv.next
					break
				}
				iv.end = start
			}
			iv = next
		}
		cm.insert(iv, cost, position, start, end)
	}
}

// traceBackwards computes cost-optimal references using the matches found by
// the hash chain and the statistics of an initial parse (refsIn), like
// libwebp's VP8LBackwardReferencesTraceBackwards. The returned references
// use no color cache (applyColorCache converts them).
func traceBackwards(xsize, n int, argb []uint32, cacheBits int, hc *hashChain,
	refsIn []pixOrCopy, refsOut []pixOrCopy, sc *scratch) []pixOrCopy {
	m := &sc.cm
	h := sc.tmpHisto(cacheBits)
	m.build(xsize, refsIn, h, &sc.tmpSet)

	dist := sc.u16(n)
	cm := &sc.costMgr
	if sc.costRing == nil {
		sc.costRing = new([costRing]int64)
	}
	cm.init(dist, sc.costRing, n, m)
	costs := cm.costs
	var cache []uint32
	shift := uint(32 - cacheBits)
	if cacheBits > 0 {
		cache = sc.cacheBuf(1 << uint(cacheBits))
	}
	argb = argb[:n]
	ol := hc.ol[:n]
	dist = dist[:n]
	// Literal costs are scaled by 82/100 and cache costs by 68/100 (with
	// rounding, both positive), as in libwebp.
	lit := m.lit
	cacheLit := lit[numLiteralCodes+numLengthCodes:]
	alpha, red, blue := &m.alpha, &m.red, &m.blue
	green := (*[256]uint32)(lit[:256])
	dist[0] = 0
	offsetPrev, lenPrev := -1, -1
	var offsetCost int64 = -1
	firstOffsetIsConstant := -1
	reach := 0
	var cacheCost []int64
	if cacheBits > 0 {
		cacheCost = sc.cacheCosts(1 << uint(cacheBits))
		for k := range cacheCost {
			cacheCost[k] = (int64(cacheLit[k])*68 + 50) / 100
		}
	}
	cm.prepare(0)
	initUpTo := cm.initUpTo
	// The cost before pixel 0 is 0; its ring slot (that of the pixel
	// costRing-1) is reinitialized before that pixel is reached.
	costs[costMask] = 0
	for i := 0; i < n; i++ {
		if i+costAhead > initUpTo {
			cm.prepare(i)
			initUpTo = cm.initUpTo
		}
		prevCost := costs[(i-1)&costMask]
		// The pixel as a literal (or cache hit).
		c := argb[i]
		v := prevCost
		if cacheBits > 0 {
			key := colorCacheHash(c, shift)
			if cache[key] == c {
				v += cacheCost[key]
			} else {
				cache[key] = c
				lc := uint64(alpha[c>>24]) + uint64(red[(c>>16)&0xff]) + uint64(green[(c>>8)&0xff]) + uint64(blue[c&0xff])
				v += int64((lc*82 + 50) / 100)
			}
		} else {
			lc := uint64(alpha[c>>24]) + uint64(red[(c>>16)&0xff]) + uint64(green[(c>>8)&0xff]) + uint64(blue[c&0xff])
			v += int64((lc*82 + 50) / 100)
		}
		if c := &costs[i&costMask]; *c > v {
			*c = v
			dist[i] = 1
		}
		// (Pixel 0 has no match: offset and length are 0.)
		offset := int(ol[i] >> maxLengthBits)
		length := int(ol[i] & (1<<maxLengthBits - 1))
		if length >= 2 {
			if offset != offsetPrev {
				offsetCost = m.distanceCost(distanceToPlaneCode(xsize, offset))
				firstOffsetIsConstant = 1
				cm.push(prevCost+offsetCost, i, length)
			} else {
				if firstOffsetIsConstant == 1 {
					reach = i - 1 + lenPrev - 1
					firstOffsetIsConstant = 0
				}
				if i+length-1 > reach {
					var lenJ int
					j := i
					for ; j <= reach; j++ {
						if int(ol[j+1]>>maxLengthBits) != offset {
							lenJ = int(ol[j] & (1<<maxLengthBits - 1))
							break
						}
						lenJ = int(ol[j+1] & (1<<maxLengthBits - 1))
					}
					cm.updateCostAtIndex(j-1, false)
					cm.updateCostAtIndex(j, false)
					cm.push(costs[(j-1)&costMask]+offsetCost, j, lenJ)
					reach = j + lenJ - 1
				}
			}
		}
		if h := cm.head; h != nil && h.start <= i {
			if h.end > i && (h.next == nil || h.next.start > i) {
				// Fast path: only the head interval covers i.
				if c := &costs[i&costMask]; *c > h.cost {
					*c = h.cost
					dist[i] = uint16(i - h.idx + 1)
				}
			} else {
				cm.updateCostAtIndex(i, true)
			}
		}
		offsetPrev = offset
		lenPrev = length
	}

	// Trace back the chosen path, packing it at the end of dist.
	p := n
	for cur := n - 1; cur >= 0; {
		k := int(dist[cur])
		p--
		dist[p] = uint16(k)
		cur -= k
	}
	path := dist[p:]
	if cap(refsOut) < len(path) {
		refsOut = make([]pixOrCopy, 0, len(path))
	}
	refsOut = refsOut[:0]
	pos := 0
	for _, s := range path {
		if s != 1 {
			refsOut = append(refsOut, mkCopy(ol[pos]>>maxLengthBits, int(s)))
		} else {
			refsOut = append(refsOut, mkLiteral(argb[pos]))
		}
		pos += int(s)
	}
	return refsOut
}
