//go:build !arm64 || purego

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
