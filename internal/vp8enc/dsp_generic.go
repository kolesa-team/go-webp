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

//go:build (!arm64 && !amd64) || purego

package vp8enc

// Portable bindings of the batched DSP entry points: the Go reference
// implementations of dsp_batch.go and dsp.go.

// vecKernels is false: where a batch of single-block Go calls is slower
// than the mode-by-mode loop (the intra4 search), the mode search keeps
// the loop.
const vecKernels = false

func fTransformPairs(src, ref []byte, t *pairTable, out [][16]int16) {
	fTransformPairsRef(src, ref, t, out)
}

func iTransformPairs(ref []byte, in [][16]int16, dst []byte, t *pairTable) {
	iTransformPairsRef(ref, in, dst, t)
}

func quantizeBlocks(in, out [][16]int16, m *matrix, info []uint16) uint32 {
	return quantizeBlocksRef(in, out, m, info)
}

func tTransformPairs(a []byte, t *pairTable, out []int32) { tTransformPairsRef(a, t, out) }

func sse4x4Pairs(a, b []byte, t *pairTable, out []int32) { sse4x4PairsRef(a, b, t, out) }

func sse16x16(a, b []byte) int { return sse16x16Ref(a, b) }

func sse16x8(a, b []byte) int { return sse16x8Ref(a, b) }

func predLuma16(p []byte, left, top []byte) { predLuma16Ref(p, left, top) }

func predChroma8(p []byte, uLeft, vLeft, uvTop []byte) { predChroma8Ref(p, uLeft, vLeft, uvTop) }

func predLuma4(p []byte, b []byte, t int) { predLuma4Ref(p, b, t) }

func zeroBinStats(coeffs [][16]int16) (int, int) { return zeroBinStatsRef(coeffs) }
