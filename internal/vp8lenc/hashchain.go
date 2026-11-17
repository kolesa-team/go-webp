package vp8lenc

import "sync"

// Hash chain based longest-match search.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	hashBits = 18
	hashSize = 1 << hashBits
)

var hashTablePool = sync.Pool{New: func() any { t := make([]int32, hashSize); return &t }}

// hashChain stores, for every pixel, the best match as dist<<12 | length.
type hashChain struct {
	ol []uint32
}

func maxItersForQuality(quality int) int {
	return 8 + quality*quality/128
}

func windowSizeForHashChain(quality, xsize int) int {
	var w int
	switch {
	case quality > 75:
		w = windowSize
	case quality > 50:
		w = xsize << 8
	case quality > 25:
		w = xsize << 6
	default:
		w = xsize << 4
	}
	if w > windowSize {
		w = windowSize
	}
	return w
}

func pixPairHash(a0, a1 uint32) uint32 {
	key := a1 * 0xc6a4a793
	key += a0 * 0x5bd1e996
	return key >> (32 - hashBits)
}

// vectorMismatch returns the length of the common prefix of a and b, at most max.
func vectorMismatch(a, b []uint32, max int) int {
	a = a[:max]
	b = b[:max]
	i := 0
	for i+2 <= max {
		if a[i] != b[i] {
			return i
		}
		if a[i+1] != b[i+1] {
			return i + 1
		}
		i += 2
	}
	if i < max && a[i] == b[i] {
		i++
	}
	return i
}

// findMatchLength is vectorMismatch with a quick reject on the bestLen pixel.
func findMatchLength(a, b []uint32, bestLen, max int) int {
	if a[bestLen] != b[bestLen] {
		return 0
	}
	return vectorMismatch(a, b, max)
}

// fill computes the best match (distance, length) for every pixel.
func (hc *hashChain) fill(argb []uint32, xsize, ysize, quality int, lowEffort bool, windowCap int) {
	hc.fillPar(argb, xsize, ysize, quality, lowEffort, windowCap, 0, nil)
}

// fillPar is fill, with the match search spread over the caller and up to
// helpers helper goroutines when helpers > 0 (see fillStripes); sc then
// provides the extra buffers.
func (hc *hashChain) fillPar(argb []uint32, xsize, ysize, quality int, lowEffort bool, windowCap int, helpers int, sc *scratch) {
	size := xsize * ysize
	if cap(hc.ol) < size {
		hc.ol = make([]uint32, size)
	}
	ol := hc.ol[:size]
	hc.ol = ol
	if size <= 2 {
		ol[0] = 0
		ol[size-1] = 0
		return
	}
	argb = argb[:size]
	window := windowSizeForHashChain(quality, xsize)
	if windowCap > 0 && window > windowCap {
		window = windowCap
	}
	s := hcSearcher{argb: argb, ol: ol, xsize: xsize, size: size,
		iterMax: maxItersForQuality(quality), window: window, lowEffort: lowEffort}
	if helpers > 0 {
		// The chain must stay intact while the stripes write ol.
		s.chain = sc.hcChainBuf(size)
		buildChain(argb, s.chain)
		ol[0] = 0
		ol[size-1] = 0
		s.fillStripes(helpers, sc)
		return
	}
	s.chain = ol // reused as int32 chain storage
	buildChain(argb, s.chain)
	ol[0] = 0
	ol[size-1] = 0
	// The searches go from the end to the start; each one writes the
	// results of the positions it covers (at and below it), so the chain
	// entries it reads (below it) are still intact.
	s.run(size-2, 0, 1, nil)
}

// buildChain links every position of argb (but the last) to the previous
// position with the same hash of its pixel pair (or of the color and run
// length in runs), as int32 (-1: none) in chain.
func buildChain(argb []uint32, chain []uint32) {
	size := len(argb)
	tp := hashTablePool.Get().(*[]int32)
	first := *tp
	for i := range first {
		first[i] = -1
	}
	argbComp := argb[0] == argb[1]
	pos := 0
	for pos < size-2 {
		argbCompNext := argb[pos+1] == argb[pos+2]
		if argbComp && argbCompNext {
			// Run of identical pixels: hash the color with the run length.
			length := 1
			c := argb[pos]
			for pos+length+2 < size && argb[pos+length+2] == c {
				length++
			}
			if length > maxLength {
				for k := 0; k < length-maxLength; k++ {
					chain[pos+k] = 0xffffffff
				}
				pos += length - maxLength
				length = maxLength
			}
			for length > 0 {
				h := pixPairHash(c, uint32(length))
				length--
				chain[pos] = uint32(first[h])
				first[h] = int32(pos)
				pos++
			}
			argbComp = false
		} else {
			h := pixPairHash(argb[pos], argb[pos+1])
			chain[pos] = uint32(first[h])
			first[h] = int32(pos)
			pos++
			argbComp = argbCompNext
		}
	}
	chain[pos] = uint32(first[pixPairHash(argb[pos], argb[pos+1])])
	hashTablePool.Put(tp)
}

// hcSearcher holds the state of the match search of hashChain.fill.
type hcSearcher struct {
	argb            []uint32
	chain           []uint32 // int32 links (buildChain)
	ol              []uint32
	xsize, size     int
	iterMax, window int
	lowEffort       bool
}

// run is the search loop of fill: it searches the best match at hi, writes
// it, extended to the left as far as possible, into ol, and repeats from the
// next position while it is at least stop. Positions below lo are never
// written: an extension stops there. When marks is not nil, every searched
// position is marked in it. run returns the last position searched and the
// next position (below stop).
//
// The position following a search only depends on the searched position,
// so the searches form a chain from size-2 down to 0 (see fillStripes).
// (One loop for all uses: a call per search costs the sequential path
// 7-19%.)
func (s *hcSearcher) run(hi, lo, stop int, marks []uint64) (last, next int) {
	argb, chain, ol := s.argb, s.chain, s.ol
	xsize, size, window, iterMax, lowEffort := s.xsize, s.size, s.window, s.iterMax, s.lowEffort
	base := hi
	for base >= stop {
		var bestLen, bestDist int
		{
			maxLen := size - 1 - base
			if maxLen > maxLength {
				maxLen = maxLength
			}
			start := argb[base:]
			iter := iterMax
			minPos := 0
			if base > window {
				minPos = base - window
			}
			lengthMax := maxLen
			if lengthMax > 256 {
				lengthMax = 256
			}
			p := int(int32(chain[base]))
			if !lowEffort {
				if base >= xsize {
					l := findMatchLength(argb[base-xsize:], start, bestLen, maxLen)
					if l > bestLen {
						bestLen = l
						bestDist = xsize
					}
					iter--
				}
				l := findMatchLength(argb[base-1:], start, bestLen, maxLen)
				if l > bestLen {
					bestLen = l
					bestDist = 1
				}
				iter--
				if bestLen == maxLength {
					p = minPos - 1
				}
			}
			bestArgb := start[bestLen]
			for ; p >= minPos; p = int(int32(chain[p])) {
				iter--
				if iter <= 0 {
					break
				}
				if argb[p+bestLen] != bestArgb {
					continue
				}
				l := vectorMismatch(argb[p:], start, maxLen)
				if bestLen < l {
					bestLen = l
					bestDist = base - p
					bestArgb = start[bestLen]
					if bestLen >= lengthMax {
						break
					}
				}
			}
			if marks != nil {
				marks[base>>6] |= 1 << uint(base&63)
			}
		}
		last = base
		// Extend the match to the left as far as possible.
		maxBase := base
		for {
			ol[base] = uint32(bestDist)<<maxLengthBits | uint32(bestLen)
			base--
			if bestDist == 0 || base == 0 || base < lo {
				break
			}
			if base < bestDist || argb[base-bestDist] != argb[base] {
				break
			}
			if bestLen == maxLength && bestDist != 1 && base+maxLength < maxBase {
				break
			}
			if bestLen < maxLength {
				bestLen++
				maxBase = base
			}
		}
	}
	return last, base
}

// minHCStripe is the smallest stripe of fillStripes, in positions (a
// multiple of 64; a variable for the tests).
var minHCStripe = 1 << 16

// fillStripes is the search loop of fill, run speculatively in stripes.
//
// The sequential search visits a chain of positions, from size-2 down: each
// search at base writes the positions from base down to the next search
// position, which only depends on base. Each stripe of positions starts
// such a chain at its top, writing only within the stripe and recording
// the positions it searched. The chain of a stripe does not start where
// the true chain enters it, but as soon as the true chain hits a position
// the stripe searched, both coincide below it (and the positions at or
// below a position are only written by searches at or below it). So a
// sequential fix-up follows the true chain from the top: it searches
// positions itself until it hits a position its stripe searched, then
// jumps to that stripe's exit search, whose extension (into the stripes
// below) it replays. In practice the chains meet within a few searches, so
// the fix-up is short, and the result is identical to fill's.
func (s *hcSearcher) fillStripes(helpers int, sc *scratch) {
	size := s.size
	workers := helpers + 1
	// Stripes of a multiple of 64 positions (the searched-position bitmap
	// words are not shared), at least minHCStripe, a few per worker for
	// balance.
	per := (size + 4*workers - 1) / (4 * workers)
	per = max((per+63)&^63, minHCStripe)
	stripes := (size-2)/per + 1 // stripe k covers [k*per, (k+1)*per) within [1, size-2]
	marks := sc.hcMarkBuf((size + 63) / 64)
	exits := sc.hcExitBuf(stripes)
	parallel(stripes, helpers, func(worker int, claim func() (int, bool)) {
		for {
			k, ok := claim()
			if !ok || sc.interrupted() {
				return
			}
			k = stripes - 1 - k // top stripes first: the fix-up starts there
			lo := max(k*per, 1)
			exits[k], _ = s.run(min((k+1)*per-1, size-2), lo, lo, marks)
		}
	})
	if sc.interrupted() {
		// Some stripes may not have run: the matches are garbage, and the
		// caller (encodeImageInternal) stops right after the fill.
		return
	}
	// Fix-up, from the top: every position above t is final.
	for t := size - 2; t > 0; {
		if marks[t>>6]&(1<<uint(t&63)) != 0 {
			// The stripe's chain from t is the true one: its results are
			// final down to its exit, whose search is redone, with its
			// extension in full.
			e := exits[t/per]
			_, t = s.run(e, 0, e, nil)
			continue
		}
		_, t = s.run(t, 0, t, nil)
	}
}

func (hc *hashChain) findLength(pos int) int { return int(hc.ol[pos] & (1<<maxLengthBits - 1)) }
func (hc *hashChain) findOffset(pos int) int { return int(hc.ol[pos] >> maxLengthBits) }

// windowOffsetsMax bounds the 2D distances considered by the box parse.
const windowOffsetsMax = 32

// fillBox computes, for every pixel, the longest match among the short 2D
// distances only (the 32 first plane codes), using best (the regular hash
// chain) to skip pixels that already have a maximal match at such a
// distance. Port of libwebp's BackwardReferencesLz77Box.
func (hc *hashChain) fillBox(argb []uint32, xsize, n int, best *hashChain, sc *scratch) {
	if cap(hc.ol) < n {
		hc.ol = make([]uint32, n)
	}
	ol := hc.ol[:n]
	hc.ol = ol
	argb = argb[:n]
	if cap(sc.counts) < n {
		sc.counts = make([]uint16, n)
	}
	counts := sc.counts[:n]
	// counts[i] is how many times argb[i] repeats from i (at most maxLength).
	counts[n-1] = 1
	for i := n - 2; i >= 0; i-- {
		if argb[i] == argb[i+1] {
			c := counts[i+1]
			if c != maxLength {
				c++
			}
			counts[i] = c
		} else {
			counts[i] = 1
		}
	}
	// The window offsets around a pixel, in plane code order, and those
	// that reach pixels unreachable from the previous pixel.
	var offsets, offsetsNew [windowOffsetsMax]int
	for y := 0; y <= 6; y++ {
		for x := -6; x <= 6; x++ {
			off := y*xsize + x
			if off <= 0 {
				continue
			}
			code := distanceToPlaneCode(xsize, off) - 1
			if code >= windowOffsetsMax {
				continue
			}
			offsets[code] = off
		}
	}
	nOff := 0
	for _, o := range offsets {
		if o != 0 {
			offsets[nOff] = o
			nOff++
		}
	}
	nNew := 0
	for i := 0; i < nOff; i++ {
		reachable := false
		for j := 0; j < nOff && !reachable; j++ {
			reachable = offsets[i] == offsets[j]+1
		}
		if !reachable {
			offsetsNew[nNew] = offsets[i]
			nNew++
		}
	}

	ol[0] = 0
	bestOffsetPrev, bestLengthPrev := -1, -1
	for i := 1; i < n; i++ {
		bestLength := best.findLength(i)
		var bestOffset int
		compute := true
		if bestLength >= maxLength {
			// Keep a maximal match already in the window.
			bestOffset = best.findOffset(i)
			for _, o := range offsets[:nOff] {
				if bestOffset == o {
					compute = false
					break
				}
			}
		}
		if compute {
			// Start from the previous pixel's match (shortened by one), then
			// only look at the offsets it could not reach.
			usePrev := bestLengthPrev > 1 && bestLengthPrev < maxLength
			offs := offsets[:nOff]
			bestLength, bestOffset = 0, 0
			if usePrev {
				offs = offsetsNew[:nNew]
				bestLength, bestOffset = bestLengthPrev-1, bestOffsetPrev
			}
			for _, o := range offs {
				jOff := i - o
				if jOff < 0 || argb[jOff] != argb[i] {
					continue
				}
				// The match length is the sum of the repetition counts.
				cur := 0
				j := i
				for {
					cOff, cj := int(counts[jOff]), int(counts[j])
					if cOff != cj {
						cur += min(cOff, cj)
						break
					}
					cur += cOff
					jOff += cOff
					j += cOff
					if !(cur <= maxLength && j < n && argb[jOff] == argb[j]) {
						break
					}
				}
				if bestLength < cur {
					bestOffset = o
					if cur >= maxLength {
						bestLength = maxLength
						break
					}
					bestLength = cur
				}
			}
		}
		if bestLength <= minLength {
			ol[i] = 0
			bestOffsetPrev, bestLengthPrev = 0, 0
		} else {
			ol[i] = uint32(bestOffset)<<maxLengthBits | uint32(bestLength)
			bestOffsetPrev, bestLengthPrev = bestOffset, bestLength
		}
	}
	ol[0] = 0
}
