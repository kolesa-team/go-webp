// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

func populationCost(pop []uint32) float64 { return populationCostGo(pop) }

func (a *accEntropy) init(h *[256]int32) { a.initGo(h) }

func subtractGreen(p []uint32) { subtractGreenGo(p) }

func (s *hcSearcher) run(hi, lo, stop int, marks []uint64) (last, next int) {
	return s.runGo(hi, lo, stop, marks)
}

func bestCacheBits(argb []uint32, quality int, refs []pixOrCopy, maxBits int, sc *scratch) int {
	return bestCacheBitsGo(argb, quality, refs, maxBits, sc)
}

func backwardRefsRle(xsize, n int, argb []uint32, refs []pixOrCopy) []pixOrCopy {
	return backwardRefsRleGo(xsize, n, argb, refs)
}

func nrgbaRow(dst []uint32, src []byte, exact bool) (hasAlpha, transparent bool) {
	return nrgbaRowGo(dst, src, exact)
}

func writeImageData(bw *bitWriter, w int, refs []pixOrCopy, histoBits int, symbols []uint16, codes []huffCode) {
	writeImageDataGo(bw, w, refs, histoBits, symbols, codes)
}

func (s *histoSet) addRefs(h *histo, refs []pixOrCopy, xsize int) { s.addRefsGo(h, refs, xsize) }

func (s *histoSet) addRefsRawDist(h *histo, refs []pixOrCopy) { s.addRefsRawDistGo(h, refs) }

func (s *histoSet) addTileRefs(refs []pixOrCopy, xsize, histoBits, hx int) {
	s.addTileRefsGo(refs, xsize, histoBits, hx)
}

func ccHistoRed(histo *[256]int32, r, g []int8, c []int32, m int8) { ccHistoRedGo(histo, r, g, c, m) }

func ccHistoBlue(histo *[256]int32, b, g, r []int8, c []int32, mg, mr int8) {
	ccHistoBlueGo(histo, b, g, r, c, mg, mr)
}

func lz77Reach(ol []uint32, i, jMax, n, l int) int { return lz77ReachGo(ol, i, jMax, n, l) }
