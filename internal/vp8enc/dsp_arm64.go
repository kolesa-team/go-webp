//go:build arm64 && !purego

package vp8enc

// NEON implementations of the batched DSP entry points of dsp_batch.go,
// written in dsp_arm64.s. The wrappers below check every bound before
// calling the assembly, which then reads and writes exactly the bytes the
// reference implementation touches. All kernels are bit-exact with the Go
// reference code, so the encoder output does not depend on the build.

// vecKernels tells the mode search to call the batched entry points (one
// call per macroblock or per intra4 sub-block) rather than loop over the
// single-block Go functions.
const vecKernels = true

// neonEnabled selects the assembly kernels. Tests clear it to run the Go
// reference implementation in the same binary and compare whole encodes.
var neonEnabled = true

//go:noescape
func fTransformPairsNEON(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)

//go:noescape
func iTransformPairsNEON(ref *byte, in *[16]int16, dst *byte, off *[8][2]uint16, n int)

//go:noescape
func quantizeBlocksNEON(in, out *[16]int16, n int, m *matrix, info *uint16) uint32

//go:noescape
func tTransformPairsNEON(a *byte, off *[8][2]uint16, n int, out *int32)

//go:noescape
func sse4x4PairsNEON(a, b *byte, off *[8][2]uint16, n int, out *int32)

//go:noescape
func sse16xNNEON(a, b *byte, rows int) int

func fTransformPairs(src, ref []byte, t *pairTable, out [][16]int16) {
	if len(src) < t.need[0] || len(ref) < t.need[1] || len(out) < 2*t.n {
		panic("vp8enc: fTransformPairs: buffer too small")
	}
	if !neonEnabled || t.n == 0 {
		fTransformPairsRef(src, ref, t, out)
		return
	}
	fTransformPairsNEON(&src[0], &ref[0], &t.off, t.n, &out[0])
}

func iTransformPairs(ref []byte, in [][16]int16, dst []byte, t *pairTable) {
	if len(ref) < t.need[0] || len(dst) < t.need[1] || len(in) < 2*t.n {
		panic("vp8enc: iTransformPairs: buffer too small")
	}
	if !neonEnabled || t.n == 0 {
		iTransformPairsRef(ref, in, dst, t)
		return
	}
	iTransformPairsNEON(&ref[0], &in[0], &dst[0], &t.off, t.n)
}

func quantizeBlocks(in, out [][16]int16, m *matrix, info []uint16) uint32 {
	n := len(in)
	if len(out) < n || n > 32 || (info != nil && len(info) < n) {
		panic("vp8enc: quantizeBlocks: bad block count")
	}
	if !neonEnabled || n == 0 || !m.vecOK {
		return quantizeBlocksRef(in, out, m, info)
	}
	var pinfo *uint16
	if info != nil {
		pinfo = &info[0]
	}
	return quantizeBlocksNEON(&in[0], &out[0], n, m, pinfo)
}

func tTransformPairs(a []byte, t *pairTable, out []int32) {
	if len(a) < t.need[0] || len(out) < 2*t.n {
		panic("vp8enc: tTransformPairs: buffer too small")
	}
	if !neonEnabled || t.n == 0 {
		tTransformPairsRef(a, t, out)
		return
	}
	tTransformPairsNEON(&a[0], &t.off, t.n, &out[0])
}

func sse4x4Pairs(a, b []byte, t *pairTable, out []int32) {
	if len(a) < t.need[0] || len(b) < t.need[1] || len(out) < 2*t.n {
		panic("vp8enc: sse4x4Pairs: buffer too small")
	}
	if !neonEnabled || t.n == 0 {
		sse4x4PairsRef(a, b, t, out)
		return
	}
	sse4x4PairsNEON(&a[0], &b[0], &t.off, t.n, &out[0])
}

func sse16x16(a, b []byte) int {
	if len(a) < 15*bps+16 || len(b) < 15*bps+16 {
		panic("vp8enc: sse16x16: buffer too small")
	}
	if !neonEnabled {
		return sse16x16Ref(a, b)
	}
	return sse16xNNEON(&a[0], &b[0], 16)
}

func sse16x8(a, b []byte) int {
	if len(a) < 7*bps+16 || len(b) < 7*bps+16 {
		panic("vp8enc: sse16x8: buffer too small")
	}
	if !neonEnabled {
		return sse16x8Ref(a, b)
	}
	return sse16xNNEON(&a[0], &b[0], 8)
}
