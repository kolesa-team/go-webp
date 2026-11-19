//go:build amd64 && !purego

package vp8enc

import (
	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
	"golang.org/x/sys/cpu"
)

// x86 implementations of the batched DSP entry points of dsp_batch.go:
// SSE4.1 kernels (128-bit, legacy encoding) in dsp_amd64.s and, where the
// 256-bit width pays off, AVX2 ones in dsp_avx2_amd64.s (the residual cost
// kernel is in cost_amd64.s). The wrappers below check every bound before
// calling the assembly, which then reads and writes exactly the bytes the
// reference implementation touches. All kernels are bit-exact with the Go
// reference code, so the encoder output does not depend on the build or
// on the CPU.

// x86MaxLevel is the highest kernel level this machine runs: cpuinfo.Max,
// except that the kernels also use POPCNT (present on every CPU with
// SSE4.2, but not part of the SSE41 level).
var x86MaxLevel = func() cpuinfo.Level {
	if !cpu.X86.HasPOPCNT {
		return cpuinfo.Go
	}
	return cpuinfo.Max
}()

// x86Level selects the kernels: below cpuinfo.SSE41 the Go reference, then
// the SSE4.1 kernels, and from cpuinfo.AVX2 the AVX2 kernels where there is
// one. Tests lower it to run every level against the Go reference.
//
// There are no AVX-512 kernels: AVX-512 machines run the AVX2 ones. The
// best candidate, quantizeBlocks with VPERMW for the zigzag order and a
// mask register for the non-zero summary, was 12% faster than the AVX2
// kernel but only 0.4% faster on a whole method 4 encode (AMD EPYC 8024P),
// and the other kernels are bound by their loads, not by the vector width.
var x86Level = x86MaxLevel

// vecKernels tells the mode search to call the batched entry points (one
// call per macroblock or per intra4 sub-block) rather than loop over the
// single-block Go functions; with the Go reference (x86Level lowered by a
// test) both give the same output.
var vecKernels = x86MaxLevel >= cpuinfo.SSE41

//go:noescape
func fTransformPairsSSE41(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)

//go:noescape
func iTransformPairsSSE41(ref *byte, in *[16]int16, dst *byte, off *[2]uint16, n int) int

//go:noescape
func quantizeBlocksSSE41(in, out *[16]int16, n int, m *matrix, info *uint16) uint32

//go:noescape
func tTransformPairsSSE41(a *byte, off *[8][2]uint16, n int, out *int32)

//go:noescape
func sse4x4PairsSSE41(a, b *byte, off *[8][2]uint16, n int, out *int32)

//go:noescape
func sse16xNSSE41(a, b *byte, rows int) int

//go:noescape
func predLuma16SSE41(p, left, top *byte)

//go:noescape
func predChroma8SSE41(p, uLeft, vLeft, uvTop *byte)

//go:noescape
func predLuma4SSE41(p, s *byte)

//go:noescape
func zeroBinStatsSSE41(c *[16]int16, n int) (count0, maxBin int)

//go:noescape
func fTransformPairsAVX2(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)

//go:noescape
func quantizeBlocksAVX2(in, out *[16]int16, n int, m *matrix, info *uint16) uint32

//go:noescape
func tTransformPairsAVX2(a *byte, off *[8][2]uint16, n int, out *int32)

//go:noescape
func iTransformPairsAVX2(ref *byte, in *[16]int16, dst *byte, off *[2]uint16, n int) int

//go:noescape
func sse16xNAVX2(a, b *byte, rows int) int

func zeroBinStats(coeffs [][16]int16) (int, int) {
	if len(coeffs) > 16 {
		panic("vp8enc: zeroBinStats: too many blocks")
	}
	if x86Level < cpuinfo.SSE41 || len(coeffs) == 0 {
		return zeroBinStatsRef(coeffs)
	}
	return zeroBinStatsSSE41(&coeffs[0], len(coeffs))
}

// predLuma4 computes the 10 intra4 predictions into p from the boundary
// samples b[t-5..t+7] (see predLuma4Ref). The kernel loads 16 bytes from
// b[t-5]; the iterator's boundary array always has them.
func predLuma4(p []byte, b []byte, t int) {
	if x86Level < cpuinfo.SSE41 || t < 5 || t+11 > len(b) {
		predLuma4Ref(p, b, t)
		return
	}
	if len(p) < i4DC4+7*bps+8 {
		panic("vp8enc: predLuma4: buffer too small")
	}
	predLuma4SSE41(&p[i4DC4], &b[t-5])
}

// predLuma16 computes the 4 intra16 predictions into p; left (top-left
// then 16 samples) and top (16 samples) are nil when not available, which
// only happens on the picture edges and takes the Go code.
func predLuma16(p []byte, left, top []byte) {
	if x86Level < cpuinfo.SSE41 || left == nil || top == nil {
		predLuma16Ref(p, left, top)
		return
	}
	if len(p) < i16HE16+15*bps+16 || len(left) < 17 || len(top) < 16 {
		panic("vp8enc: predLuma16: buffer too small")
	}
	predLuma16SSE41(&p[0], &left[0], &top[0])
}

// predChroma8 computes the 4 chroma predictions into p (U and V side by
// side); uLeft, vLeft (top-left then 8 samples) and uvTop (8 U then 8 V
// samples) are nil when not available (picture edges: Go code).
func predChroma8(p []byte, uLeft, vLeft, uvTop []byte) {
	if x86Level < cpuinfo.SSE41 || uLeft == nil || vLeft == nil || uvTop == nil {
		predChroma8Ref(p, uLeft, vLeft, uvTop)
		return
	}
	if len(p) < c8HE8+7*bps+16 || len(uLeft) < 9 || len(vLeft) < 9 || len(uvTop) < 16 {
		panic("vp8enc: predChroma8: buffer too small")
	}
	predChroma8SSE41(&p[c8DC8], &uLeft[0], &vLeft[0], &uvTop[0])
}

func fTransformPairs(src, ref []byte, t *pairTable, out [][16]int16) {
	if len(src) < t.need[0] || len(ref) < t.need[1] || len(out) < 2*t.n {
		panic("vp8enc: fTransformPairs: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 || t.n == 0 {
		fTransformPairsRef(src, ref, t, out)
		return
	}
	if x86Level >= cpuinfo.AVX2 {
		fTransformPairsAVX2(&src[0], &ref[0], &t.off, t.n, &out[0])
		return
	}
	fTransformPairsSSE41(&src[0], &ref[0], &t.off, t.n, &out[0])
}

// iTransformPairs runs the kernel, which stops before a pair holding a
// coefficient beyond its 16-bit range (|in| > 2047, see dsp_amd64.s);
// such a pair, rare in practice, takes the Go code, then the kernel
// resumes, so the pairs are still written in order. As in the reference,
// ref and dst must not overlap (the kernels read ahead of their stores).
func iTransformPairs(ref []byte, in [][16]int16, dst []byte, t *pairTable) {
	if len(ref) < t.need[0] || len(dst) < t.need[1] || len(in) < 2*t.n {
		panic("vp8enc: iTransformPairs: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 || t.n == 0 {
		iTransformPairsRef(ref, in, dst, t)
		return
	}
	avx2 := x86Level >= cpuinfo.AVX2
	for k := 0; k < t.n; k++ {
		if avx2 {
			k += iTransformPairsAVX2(&ref[0], &in[2*k], &dst[0], &t.off[k], t.n-k)
		} else {
			k += iTransformPairsSSE41(&ref[0], &in[2*k], &dst[0], &t.off[k], t.n-k)
		}
		if k < t.n {
			r, d := int(t.off[k][0]), int(t.off[k][1])
			iTransform(ref[r:], &in[2*k], dst[d:])
			iTransform(ref[r+4:], &in[2*k+1], dst[d+4:])
		}
	}
}

func quantizeBlocks(in, out [][16]int16, m *matrix, info []uint16) uint32 {
	n := len(in)
	if len(out) < n || n > 32 || (info != nil && len(info) < n) {
		panic("vp8enc: quantizeBlocks: bad block count")
	}
	if x86Level < cpuinfo.SSE41 || n == 0 || !m.vecOK {
		return quantizeBlocksRef(in, out, m, info)
	}
	var pinfo *uint16
	if info != nil {
		pinfo = &info[0]
	}
	if x86Level >= cpuinfo.AVX2 {
		return quantizeBlocksAVX2(&in[0], &out[0], n, m, pinfo)
	}
	return quantizeBlocksSSE41(&in[0], &out[0], n, m, pinfo)
}

func tTransformPairs(a []byte, t *pairTable, out []int32) {
	if len(a) < t.need[0] || len(out) < 2*t.n {
		panic("vp8enc: tTransformPairs: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 || t.n == 0 {
		tTransformPairsRef(a, t, out)
		return
	}
	if x86Level >= cpuinfo.AVX2 {
		tTransformPairsAVX2(&a[0], &t.off, t.n, &out[0])
		return
	}
	tTransformPairsSSE41(&a[0], &t.off, t.n, &out[0])
}

// sse4x4Pairs has no AVX2 kernel: two pairs per 256-bit register measured
// slower than the SSE4.1 kernel (the row loads dominate).
func sse4x4Pairs(a, b []byte, t *pairTable, out []int32) {
	if len(a) < t.need[0] || len(b) < t.need[1] || len(out) < 2*t.n {
		panic("vp8enc: sse4x4Pairs: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 || t.n == 0 {
		sse4x4PairsRef(a, b, t, out)
		return
	}
	sse4x4PairsSSE41(&a[0], &b[0], &t.off, t.n, &out[0])
}

func sse16x16(a, b []byte) int {
	if len(a) < 15*bps+16 || len(b) < 15*bps+16 {
		panic("vp8enc: sse16x16: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 {
		return sse16x16Ref(a, b)
	}
	if x86Level >= cpuinfo.AVX2 {
		return sse16xNAVX2(&a[0], &b[0], 16)
	}
	return sse16xNSSE41(&a[0], &b[0], 16)
}

func sse16x8(a, b []byte) int {
	if len(a) < 7*bps+16 || len(b) < 7*bps+16 {
		panic("vp8enc: sse16x8: buffer too small")
	}
	if x86Level < cpuinfo.SSE41 {
		return sse16x8Ref(a, b)
	}
	if x86Level >= cpuinfo.AVX2 {
		return sse16xNAVX2(&a[0], &b[0], 8)
	}
	return sse16xNSSE41(&a[0], &b[0], 8)
}
