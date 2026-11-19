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

//go:noescape
func predLuma16NEON(p, left, top *byte)

//go:noescape
func predChroma8NEON(p, uLeft, vLeft, uvTop *byte)

//go:noescape
func predLuma4NEON(p, s *byte)

//go:noescape
func zeroBinStatsNEON(c *[16]int16, n int) (count0, maxBin int)

func zeroBinStats(coeffs [][16]int16) (int, int) {
	if len(coeffs) > 16 {
		panic("vp8enc: zeroBinStats: too many blocks")
	}
	if !neonEnabled || len(coeffs) == 0 {
		return zeroBinStatsRef(coeffs)
	}
	return zeroBinStatsNEON(&coeffs[0], len(coeffs))
}

// predLuma4 computes the 10 intra4 predictions into p from the boundary
// samples b[t-5..t+7] (see predLuma4Ref). The kernel loads 16 bytes from
// b[t-5]; the iterator's boundary array always has them.
func predLuma4(p []byte, b []byte, t int) {
	if !neonEnabled || t < 5 || t+11 > len(b) {
		predLuma4Ref(p, b, t)
		return
	}
	if len(p) < i4DC4+7*bps+8 {
		panic("vp8enc: predLuma4: buffer too small")
	}
	predLuma4NEON(&p[i4DC4], &b[t-5])
}

// predLuma16 computes the 4 intra16 predictions into p; left (top-left
// then 16 samples) and top (16 samples) are nil when not available, which
// only happens on the picture edges and takes the Go code.
func predLuma16(p []byte, left, top []byte) {
	if !neonEnabled || left == nil || top == nil {
		predLuma16Ref(p, left, top)
		return
	}
	if len(p) < i16HE16+15*bps+16 || len(left) < 17 || len(top) < 16 {
		panic("vp8enc: predLuma16: buffer too small")
	}
	predLuma16NEON(&p[0], &left[0], &top[0])
}

// predChroma8 computes the 4 chroma predictions into p (U and V side by
// side); uLeft, vLeft (top-left then 8 samples) and uvTop (8 U then 8 V
// samples) are nil when not available (picture edges: Go code).
func predChroma8(p []byte, uLeft, vLeft, uvTop []byte) {
	if !neonEnabled || uLeft == nil || vLeft == nil || uvTop == nil {
		predChroma8Ref(p, uLeft, vLeft, uvTop)
		return
	}
	if len(p) < c8HE8+7*bps+16 || len(uLeft) < 9 || len(vLeft) < 9 || len(uvTop) < 16 {
		panic("vp8enc: predChroma8: buffer too small")
	}
	predChroma8NEON(&p[c8DC8], &uLeft[0], &vLeft[0], &uvTop[0])
}

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
