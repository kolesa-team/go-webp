package vp8lenc

import (
	"math"
	"math/bits"
)

// Entropy estimation helpers. Algorithm adapted from libwebp (BSD-3-Clause).

const logTabSize = 4096

var (
	slog2Tab [logTabSize]float64 // v*log2(v)
	log2Tab  [logTabSize]float64 // log2(v), log2(0) = 0
)

func init() {
	for i := 1; i < logTabSize; i++ {
		f := float64(i)
		log2Tab[i] = math.Log2(f)
		slog2Tab[i] = f * log2Tab[i]
	}
}

// slog2 returns v*log2(v).
func slog2(v uint32) float64 {
	if v < logTabSize {
		return slog2Tab[v]
	}
	return slog2Slow(v)
}

// log2Slow approximates log2(v) for v >= logTabSize from the table using the
// 12 most significant bits plus a linear correction of the remainder.
func log2Slow(v uint32) float64 {
	shift := uint(bits.Len32(v) - 12)
	m := v >> shift
	rem := v & (1<<shift - 1)
	// log2(m + r) ~= log2(m) + r/(m*ln2)
	return log2Tab[m] + float64(shift) + float64(rem)/(float64(m)*math.Ln2*float64(uint32(1)<<shift))
}

func slog2Slow(v uint32) float64 {
	return float64(v) * log2Slow(v)
}

type bitEntropy struct {
	entropy  float64
	sum      uint32
	nonzeros int
	maxVal   uint32
}

type streaks struct {
	counts  [2]int    // [zero/nonzero] number of streaks longer than 3
	streaks [2][2]int // [zero/nonzero][short/long] total streak length
}

// refine returns the refined bit entropy estimate (libwebp BitsEntropyRefine).
func (e *bitEntropy) refine() float64 {
	var mix float64
	if e.nonzeros < 5 {
		if e.nonzeros <= 1 {
			return 0
		}
		if e.nonzeros == 2 {
			return 0.99*float64(e.sum) + 0.01*e.entropy
		}
		if e.nonzeros == 3 {
			mix = 0.95
		} else {
			mix = 0.7
		}
	} else {
		mix = 0.627
	}
	minLimit := 2*float64(e.sum) - float64(e.maxVal)
	minLimit = mix*minLimit + (1-mix)*e.entropy
	if e.entropy < minLimit {
		return minLimit
	}
	return e.entropy
}

// finalHuffmanCost estimates the cost of storing a Huffman code from its
// streak statistics.
func (s *streaks) finalHuffmanCost() float64 {
	r := float64(codeLengthCodes*3) - 9.1
	r += float64(s.counts[0])*1.5625 + 0.234375*float64(s.streaks[0][1])
	r += float64(s.counts[1])*2.578125 + 0.703125*float64(s.streaks[1][1])
	r += 1.796875 * float64(s.streaks[0][0])
	r += 3.28125 * float64(s.streaks[1][0])
	return r
}

// bitsEntropy returns the refined Shannon entropy of a population.
func bitsEntropy(pop []uint32) float64 {
	var e bitEntropy
	for _, v := range pop {
		if v != 0 {
			e.sum += v
			e.nonzeros++
			e.entropy -= slog2(v)
			if v > e.maxVal {
				e.maxVal = v
			}
		}
	}
	e.entropy += slog2(e.sum)
	return e.refine()
}

// populationCost estimates the number of bits to code pop with a Huffman
// code, including the cost of storing the code itself.
func populationCost(pop []uint32) float64 {
	var e bitEntropy
	var s streaks
	n := len(pop)
	if n == 0 {
		return 0
	}
	prev := pop[0]
	iPrev := 0
	for i := 1; i <= n; i++ {
		var v uint32
		if i < n {
			v = pop[i]
			if v == prev {
				continue
			}
		}
		streak := i - iPrev
		nz := 0
		if prev != 0 {
			nz = 1
			e.sum += prev * uint32(streak)
			e.nonzeros += streak
			e.entropy -= slog2(prev) * float64(streak)
			if e.maxVal < prev {
				e.maxVal = prev
			}
		}
		long := 0
		if streak > 3 {
			long = 1
		}
		s.counts[nz] += long
		s.streaks[nz][long] += streak
		prev = v
		iPrev = i
	}
	e.entropy += slog2(e.sum)
	return e.refine() + s.finalHuffmanCost()
}

// combinedPopulationCostGo is populationCost(a+b) without materializing
// a+b. It is the Go reference of combinedPopulationCost (dsp_amd64.go,
// dsp_generic.go).
func combinedPopulationCostGo(a, b []uint32) float64 {
	var e bitEntropy
	var s streaks
	n := len(a)
	if n == 0 {
		return 0
	}
	b = b[:n]
	prev := a[0] + b[0]
	iPrev := 0
	for i := 1; i <= n; i++ {
		var v uint32
		if i < n {
			v = a[i] + b[i]
			if v == prev {
				continue
			}
		}
		streak := i - iPrev
		nz := 0
		if prev != 0 {
			nz = 1
			e.sum += prev * uint32(streak)
			e.nonzeros += streak
			e.entropy -= slog2(prev) * float64(streak)
			if e.maxVal < prev {
				e.maxVal = prev
			}
		}
		long := 0
		if streak > 3 {
			long = 1
		}
		s.counts[nz] += long
		s.streaks[nz][long] += streak
		prev = v
		iPrev = i
	}
	e.entropy += slog2(e.sum)
	return e.refine() + s.finalHuffmanCost()
}

// extraCost returns the number of extra bits implied by a prefix-code
// population of the given length.
func extraCost(pop []uint32) float64 {
	var cost uint64
	for i := 4; i < len(pop); i++ {
		cost += uint64((i-2)>>1) * uint64(pop[i])
	}
	return float64(cost)
}

// accEntropy caches per-bin x*log2(x) of an accumulated histogram, so that
// combinedShannonEntropy only needs to visit the non-zero bins of the other
// operand.
type accEntropy struct {
	s     [256]float64
	total float64
	sum   uint32
}

func (a *accEntropy) init(h *[256]int32) {
	a.total = 0
	a.sum = 0
	for i, v := range h {
		f := slog2(uint32(v))
		a.s[i] = f
		a.total += f
		a.sum += uint32(v)
	}
}

// combinedShannonEntropyGo returns the entropy of X plus the entropy of X+Y
// (libwebp VP8LCombinedShannonEntropy), where Y is described by acc. It is
// the Go reference of combinedShannonEntropy (dsp_amd64.go,
// dsp_generic.go).
func combinedShannonEntropyGo(x, y *[256]int32, acc *accEntropy) float64 {
	r := -acc.total
	var sumX uint32
	for i := 0; i < 256; i++ {
		xv := uint32(x[i])
		if xv != 0 {
			sumX += xv
			r += acc.s[i] - slog2(xv) - slog2(xv+uint32(y[i]))
		}
	}
	return r + slog2(sumX) + slog2(sumX+acc.sum)
}
