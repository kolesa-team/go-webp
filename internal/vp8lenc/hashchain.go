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
