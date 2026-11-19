//go:build amd64 && !purego

package vp8lenc

import (
	"math"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// SSE4.1 and AVX2 implementations of the kernel entry points, written in
// dsp_amd64.s. The wrappers below check every bound before calling the
// assembly, which then reads and writes exactly the elements the Go
// reference touches (no read past the end of a slice: the tails are
// handled by an overlapping last block or by the Go code). All kernels are
// bit-exact with the Go reference, so the encoded bytes do not depend on
// the machine.

// x86Level selects the kernels: the highest level of this machine, lowered
// by the tests to run every level against the Go reference.
var x86Level = cpuinfo.Max

// maxKernelLevel is the highest level with kernels of its own: the levels
// above it run its kernels.
const maxKernelLevel = cpuinfo.AVX2

//go:noescape
func predictRowSSE41(mode int, cur, up, out *uint32, n int)

//go:noescape
func predictRowAVX2(mode int, cur, up, out *uint32, n int)

// predictRow is predictRowGo: 4 pixels per step with SSE4.1, 8 with AVX2
// (rows of at least that many pixels).
func predictRow(mode int, cur, up []uint32, x0, x1 int, out []uint32) {
	n := x1 - x0
	if x86Level >= cpuinfo.SSE41 && n >= 4 && x0 >= 1 && uint(mode) < numPredModes {
		// The kernel reads cur[x0-1:x1] and up[x0-1:x1+1], and writes
		// out[:n].
		_ = up[x1]
		_ = cur[x1-1]
		out = out[:n]
		if x86Level >= cpuinfo.AVX2 && n >= 8 {
			predictRowAVX2(mode, &cur[x0], &up[x0], &out[0], n)
		} else {
			predictRowSSE41(mode, &cur[x0], &up[x0], &out[0], n)
		}
		return
	}
	predictRowGo(mode, cur, up, x0, x1, out)
}

//go:noescape
func combinedShannonEntropySSE41(x, y *[256]int32, acc *accEntropy) float64

// combinedShannonEntropy is combinedShannonEntropyGo: the kernel visits
// only the non-zero bins of x (a bit mask built 16 bins at a time) instead
// of branching on every bin, and adds the same terms in the same order.
func combinedShannonEntropy(x, y *[256]int32, acc *accEntropy) float64 {
	if x86Level >= cpuinfo.SSE41 {
		return combinedShannonEntropySSE41(x, y, acc)
	}
	return combinedShannonEntropyGo(x, y, acc)
}

//go:noescape
func hcRunSSE41(s *hcSearcher, hi, lo, stop int, marks *uint64, pf int) (last, next int)

// hcPrefetch is how many searches ahead hcRunSSE41 prefetches the first
// node of the hash chain (a variable for the benchmarks; 0: none). (4: -2%
// at level 6 on the photo, 8 and 16 less; the chains are short, 1.8 links
// per search at level 0 and 4.9 at level 6 on the photo.)
var hcPrefetch = 4

// run is runGo: the kernel runs the same searches and extensions, with the
// match lengths compared 4 pixels at a time, and prefetches the pixels and
// the next link of the first chain node of the search hcPrefetch positions
// below (the searches go down by one position most of the time).
//
// The kernel trusts the chain (buildChain): every link from a position is
// -1 or a lower position, as are the links of the positions it searches.
func (s *hcSearcher) run(hi, lo, stop int, marks []uint64) (last, next int) {
	if x86Level >= cpuinfo.SSE41 && stop >= 1 && hi <= s.size-2 && s.size >= 2 {
		// Every access is below size: positions and links are below hi.
		_ = s.argb[s.size-1]
		_ = s.chain[s.size-1]
		_ = s.ol[s.size-1]
		var m *uint64
		if marks != nil && hi >= stop {
			_ = marks[hi>>6]
			m = &marks[0]
		}
		return hcRunSSE41(s, hi, lo, stop, m, hcPrefetch)
	}
	return s.runGo(hi, lo, stop, marks)
}

// cacheHistos are the histograms of bestCacheBits for each cache size i
// (0: no cache): the literal histogram (green, length prefixes and cache
// indices) and the cache are in a scratch buffer.
type cacheHistos struct {
	lit, cache       [maxColorCacheBits + 1]*uint32
	red, blue, alpha [maxColorCacheBits + 1][256]uint32
}

//go:noescape
func cacheCountsSSE41(argb *uint32, npix int, refs *pixOrCopy, nrefs, maxBits int, h *cacheHistos) int

// bestCacheBits is bestCacheBitsGo, with the histograms of every cache
// size counted by cacheCountsSSE41 (the counting loop of bestCacheBitsGo
// without bound checks; it returns -1, and the Go code then runs, if refs
// does not fit argb). The costs are computed as in the Go code.
func bestCacheBits(argb []uint32, quality int, refs []pixOrCopy, maxBits int, sc *scratch) int {
	if x86Level < cpuinfo.SSE41 || quality <= 25 || maxBits <= 0 || maxBits > maxColorCacheBits || len(refs) == 0 || len(argb) == 0 {
		return bestCacheBitsGo(argb, quality, refs, maxBits, sc)
	}
	var h cacheHistos
	var lits [maxColorCacheBits + 1][]uint32
	total := 0
	for i := 0; i <= maxBits; i++ {
		n := numLiteralCodes + numLengthCodes
		if i > 0 {
			n += 1 << uint(i)
		}
		total += n
	}
	buf := sc.u32a(total + (2 << uint(maxBits)))
	clear(buf)
	off := 0
	for i := 0; i <= maxBits; i++ {
		n := numLiteralCodes + numLengthCodes
		if i > 0 {
			n += 1 << uint(i)
		}
		lits[i] = buf[off : off+n]
		h.lit[i] = &buf[off]
		off += n
	}
	for i := 1; i <= maxBits; i++ {
		h.cache[i] = &buf[off]
		off += 1 << uint(i)
	}
	_ = buf[off-1]
	if cacheCountsSSE41(&argb[0], len(argb), &refs[0], len(refs), maxBits, &h) < 0 {
		return bestCacheBitsGo(argb, quality, refs, maxBits, sc)
	}
	best := 0
	bestCost := math.MaxFloat64
	for i := 0; i <= maxBits; i++ {
		c := populationCost(lits[i]) + populationCost(h.red[i][:]) +
			populationCost(h.blue[i][:]) + populationCost(h.alpha[i][:]) +
			extraCost(lits[i][numLiteralCodes:numLiteralCodes+numLengthCodes])
		if i == 0 || c < bestCost {
			bestCost = c
			best = i
		}
	}
	return best
}

//go:noescape
func rleSSE41(argb *uint32, n, xsize, i int, refs *pixOrCopy, room int) (count, next int)

// backwardRefsRle is backwardRefsRleGo: the kernel parses from position i
// until the end or until refs is full (the buffer then grows as in the Go
// code), with a pixel equal to neither its left nor its top neighbor taken
// as a literal at once and the match lengths compared 4 pixels at a time.
func backwardRefsRle(xsize, n int, argb []uint32, refs []pixOrCopy) []pixOrCopy {
	if x86Level < cpuinfo.SSE41 || n < 1 || xsize < 1 {
		return backwardRefsRleGo(xsize, n, argb, refs)
	}
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
		room := refs[len(refs):cap(refs)]
		k, next := rleSSE41(&argb[0], n, xsize, i, &room[0], len(room))
		refs = refs[:len(refs)+k]
		i = next
	}
	return refs
}

// popStats are the streak statistics of combinedPopulationCostGo.
type popStats struct {
	e bitEntropy
	s streaks
}

//go:noescape
func popStatsSSE41(a, b *uint32, n int, st *popStats)

// combinedPopulationCost is combinedPopulationCostGo: the kernel finds the
// ends of the streaks of a+b 16 bins at a time (a bit mask) and updates the
// statistics of each streak without branches, with the floating-point
// operations of the Go code in its order; the cost is then computed from
// the statistics as in the Go code.
func combinedPopulationCost(a, b []uint32) float64 {
	n := len(a)
	if x86Level >= cpuinfo.SSE41 && n > 16 && n < 1<<16 {
		b = b[:n]
		var st popStats
		popStatsSSE41(&a[0], &b[0], n, &st)
		st.e.entropy += slog2(st.e.sum)
		return st.e.refine() + st.s.finalHuffmanCost()
	}
	return combinedPopulationCostGo(a, b)
}

// zeroPop is the second operand of popStatsSSE41 for populationCost: zeros
// as long as the longest population (the literal histogram with the
// largest color cache). Never written.
var zeroPop [numLiteralCodes + numLengthCodes + 1<<maxColorCacheBits]uint32

// populationCost is populationCostGo, which is combinedPopulationCostGo
// with a zero second operand: the same kernel with zeroPop as that operand
// (pop[i]+0 == pop[i]), so the statistics and the cost are those of the Go
// code.
func populationCost(pop []uint32) float64 {
	n := len(pop)
	if x86Level >= cpuinfo.SSE41 && n > 16 && n <= len(zeroPop) {
		var st popStats
		popStatsSSE41(&pop[0], &zeroPop[0], n, &st)
		st.e.entropy += slog2(st.e.sum)
		return st.e.refine() + st.s.finalHuffmanCost()
	}
	return populationCostGo(pop)
}

//go:noescape
func accEntropyInitSSE41(h *[256]int32, a *accEntropy)

// init is initGo: the kernel adds up the bins with vector adds, stores
// zeros for the zero bins and visits only the non-zero ones (a bit mask
// built 16 bins at a time), whose terms it adds to the total in the order
// of the Go code. The skipped terms of the Go code are slog2(0) = +0, and
// x + (+0) == x for every total it can reach (a sum of non-negative terms
// from +0, never -0), so the total is bit-identical.
func (a *accEntropy) init(h *[256]int32) {
	if x86Level >= cpuinfo.SSE41 {
		accEntropyInitSSE41(h, a)
		return
	}
	a.initGo(h)
}

//go:noescape
func ccHistoSSE41(histo *[256]int32, d, s1, s2 *int8, c *int32, n int, m1, m2 int)

// ccHistoRed is ccHistoRedGo: the kernel computes the transformed values 8
// at a time in 16-bit lanes (the products of two int8 fit), then adds the
// counts to their bins one by one. The red transform is the blue one with
// a zero second multiplier.
func ccHistoRed(histo *[256]int32, r, g []int8, c []int32, m int8) {
	n := len(r)
	if x86Level >= cpuinfo.SSE41 && n > 0 {
		g = g[:n]
		c = c[:n]
		ccHistoSSE41(histo, &r[0], &g[0], &g[0], &c[0], n, int(m), 0)
		return
	}
	ccHistoRedGo(histo, r, g, c, m)
}

// ccHistoBlue is ccHistoBlueGo (see ccHistoRed).
func ccHistoBlue(histo *[256]int32, b, g, r []int8, c []int32, mg, mr int8) {
	n := len(b)
	if x86Level >= cpuinfo.SSE41 && n > 0 {
		g = g[:n]
		r = r[:n]
		c = c[:n]
		ccHistoSSE41(histo, &b[0], &g[0], &r[0], &c[0], n, int(mg), int(mr))
		return
	}
	ccHistoBlueGo(histo, b, g, r, c, mg, mr)
}
