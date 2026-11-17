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
	iterMax := maxItersForQuality(quality)
	window := windowSizeForHashChain(quality, xsize)
	if windowCap > 0 && window > windowCap {
		window = windowCap
	}

	tp := hashTablePool.Get().(*[]int32)
	first := *tp
	for i := range first {
		first[i] = -1
	}
	chain := ol // reused as int32 chain storage
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

	ol[0] = 0
	ol[size-1] = 0
	for base := size - 2; base > 0; {
		maxLen := size - 1 - base
		if maxLen > maxLength {
			maxLen = maxLength
		}
		start := argb[base:]
		iter := iterMax
		bestLen := 0
		bestDist := 0
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
		// Extend the match to the left as far as possible.
		maxBase := base
		for {
			ol[base] = uint32(bestDist)<<maxLengthBits | uint32(bestLen)
			base--
			if bestDist == 0 || base == 0 {
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
