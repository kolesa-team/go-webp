//go:build amd64 && !purego

package vp8lenc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

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
