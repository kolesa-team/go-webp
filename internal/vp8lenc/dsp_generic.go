//go:build !amd64 || purego

package vp8lenc

// Portable bindings of the kernel entry points: the Go reference
// implementations. (The NEON versions of these kernels gave no gain on
// arm64, so only amd64 has assembly; see dsp_amd64.go.)

func predictRow(mode int, cur, up []uint32, x0, x1 int, out []uint32) {
	predictRowGo(mode, cur, up, x0, x1, out)
}

func combinedShannonEntropy(x, y *[256]int32, acc *accEntropy) float64 {
	return combinedShannonEntropyGo(x, y, acc)
}

func combinedPopulationCost(a, b []uint32) float64 { return combinedPopulationCostGo(a, b) }

func (s *hcSearcher) run(hi, lo, stop int, marks []uint64) (last, next int) {
	return s.runGo(hi, lo, stop, marks)
}

func bestCacheBits(argb []uint32, quality int, refs []pixOrCopy, maxBits int, sc *scratch) int {
	return bestCacheBitsGo(argb, quality, refs, maxBits, sc)
}

func backwardRefsRle(xsize, n int, argb []uint32, refs []pixOrCopy) []pixOrCopy {
	return backwardRefsRleGo(xsize, n, argb, refs)
}
