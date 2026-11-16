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
// entries, flat images a small one; growRefs covers underestimates.
func initRefsCap(argb []uint32, w, h int) int {
	n := w * h
	if n <= 4096 {
		return n
	}
	argb = argb[:n]
	est := 1
	for i := 1; i < w; i++ {
		if argb[i] != argb[i-1] {
			est++
		}
	}
	for i := w; i < n; i++ {
		p := argb[i]
		if p != argb[i-1] && p != argb[i-w] {
			est++
		}
	}
	return min(n, est+est/2+4096)
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
			for _, a := range argb[pos : pos+l] {
				cache[colorCacheHash(a, shift)] = a
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

// costModel holds per-symbol bit cost estimates.
type costModel struct {
	alpha, red, blue [256]float32
	lit              []float32 // green + length prefixes + cache
	dist             [numDistanceCodes]float32
	lenCost          []float32 // cost of a copy length (prefix + extra bits)
}

func popToBits(pop []uint32, out []float32) {
	var sum uint32
	nz := 0
	for _, v := range pop {
		sum += v
		if v > 0 {
			nz++
		}
	}
	if nz <= 1 {
		for i := range out {
			out[i] = 0
		}
		return
	}
	ls := flog2(sum)
	for i, v := range pop {
		out[i] = float32(ls - flog2(v))
	}
}

func (m *costModel) build(xsize, cacheBits int, refs []pixOrCopy, h *histo, hs *histoSet) {
	hs.clear(h)
	hs.addRefs(h, refs, xsize)
	L := hs.litLen
	d := h.d
	if cap(m.lit) < L {
		m.lit = make([]float32, L)
	}
	m.lit = m.lit[:L]
	popToBits(d[:L], m.lit)
	popToBits(d[L:L+256], m.red[:])
	popToBits(d[L+256:L+512], m.blue[:])
	popToBits(d[L+512:L+768], m.alpha[:])
	popToBits(d[L+768:L+808], m.dist[:])
	if cap(m.lenCost) < maxLength+1 {
		m.lenCost = make([]float32, maxLength+1)
	}
	m.lenCost = m.lenCost[:maxLength+1]
	for l := 1; l <= maxLength; l++ {
		code, nb, _ := prefixEncode(l)
		m.lenCost[l] = m.lit[numLiteralCodes+code] + float32(nb)
	}
}

func (m *costModel) literalCost(p uint32) float32 {
	return m.alpha[p>>24] + m.red[(p>>16)&0xff] + m.lit[(p>>8)&0xff] + m.blue[p&0xff]
}

func (m *costModel) distCost(xsize int, dist int) float32 {
	code, nb, _ := prefixEncode(distanceToPlaneCode(xsize, dist))
	return m.dist[code] + float32(nb)
}

// traceBackwards computes cost-optimal references using the matches found by
// the hash chain and the statistics of an initial parse (refsIn).
func traceBackwards(xsize, n int, argb []uint32, cacheBits int, hc *hashChain,
	refsIn []pixOrCopy, refsOut []pixOrCopy, quality int, sc *scratch) []pixOrCopy {
	m := &sc.cm
	h := sc.tmpHisto(cacheBits)
	m.build(xsize, cacheBits, refsIn, h, &sc.tmpSet)

	costs := sc.f32(n + 1)
	steps := sc.u16(n + 1)
	inf := float32(math.MaxFloat32)
	for i := range costs {
		costs[i] = inf
	}
	costs[0] = 0
	var cache []uint32
	shift := uint(32 - cacheBits)
	if cacheBits > 0 {
		cache = sc.cacheBuf(1 << uint(cacheBits))
	}
	window := 16 + quality/4 // number of short lengths evaluated exhaustively
	const skipLen = 128      // matches at least this long are taken greedily
	ol := hc.ol[:n]
	argb = argb[:n]
	lenCost := m.lenCost
	for i := 0; i < n; {
		c := costs[i]
		p := argb[i]
		var lc float32
		if cacheBits > 0 {
			key := colorCacheHash(p, shift)
			if cache[key] == p {
				lc = m.lit[numLiteralCodes+numLengthCodes+int(key)]
			} else {
				lc = m.literalCost(p)
			}
		} else {
			lc = m.literalCost(p)
		}
		if nc := c + lc; nc < costs[i+1] {
			costs[i+1] = nc
			steps[i+1] = 1
		}
		v := ol[i]
		l := int(v & (1<<maxLengthBits - 1))
		if l >= 2 {
			base := c + m.distCost(xsize, int(v>>maxLengthBits))
			kmax := l
			if kmax > window {
				kmax = window
			}
			for k := 2; k <= kmax; k++ {
				if nc := base + lenCost[k]; nc < costs[i+k] {
					costs[i+k] = nc
					steps[i+k] = uint16(k)
				}
			}
			if l > kmax {
				if nc := base + lenCost[l]; nc < costs[i+l] {
					costs[i+l] = nc
					steps[i+l] = uint16(l)
				}
			}
			if l >= skipLen {
				if cacheBits > 0 {
					for _, q := range argb[i : i+l] {
						cache[colorCacheHash(q, shift)] = q
					}
				}
				// Force the long match: make sure i+l is reached from i.
				if costs[i+l] > base+lenCost[l] {
					costs[i+l] = base + lenCost[l]
					steps[i+l] = uint16(l)
				}
				i += l
				continue
			}
		}
		if cacheBits > 0 {
			cache[colorCacheHash(p, shift)] = p
		}
		i++
	}
	// Trace back the chosen steps, storing them in place at the path's end.
	cnt := 0
	for pos := n; pos > 0; {
		s := int(steps[pos])
		pos -= s
		cnt++
	}
	path := sc.u16b(cnt)
	k := cnt
	for pos := n; pos > 0; {
		s := steps[pos]
		k--
		path[k] = s
		pos -= int(s)
	}
	if cap(refsOut) < cnt {
		refsOut = make([]pixOrCopy, 0, cnt)
	}
	refsOut = refsOut[:0]
	pos := 0
	for _, s := range path {
		if s == 1 {
			refsOut = append(refsOut, mkLiteral(argb[pos]))
		} else {
			refsOut = append(refsOut, mkCopy(ol[pos]>>maxLengthBits, int(s)))
		}
		pos += int(s)
	}
	return refsOut
}
