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

//go:build amd64 && !purego

package vp8

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// x86 kernels for the macroblock reconstruction (recon_amd64.s), chosen at
// run time by reconLevel: AVX2 (x86-64-v3) and SSE4.1 versions of the three
// per-macroblock entry points, an SSE4.1 inverse WHT and SSE2 edge setup and
// row cache copy. Levels without a kernel use the next lower one, down to
// the Go reference.
//
// The inverse DCT runs in 16-bit lanes as in libwebp, which is exact only
// while the coefficients it uses stay within [-4095, 4095] (a valid encoder
// stays far below: the transform of 8-bit residuals is bounded by 2040, plus
// the quantization error). The kernels check that bound and return false
// before they use a coefficient outside it; the wrappers then rerun the
// macroblock with the Go reference, which recomputes every pixel the kernel
// may have written from context the kernels never write. The DC-only path
// saturates instead and needs no check.

// reconLevel is the SIMD level the wrappers use; tests set it to every level
// of cpuinfo.Levels.
var reconLevel = cpuinfo.Max

//go:noescape
func reconI4AVX2(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) bool

//go:noescape
func reconI16AVX2(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) bool

//go:noescape
func reconUVAVX2(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int) bool

//go:noescape
func reconI4SSE41(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) bool

//go:noescape
func reconI16SSE41(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) bool

//go:noescape
func reconUVSSE41(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int) bool

//go:noescape
func transformWHTSSE41(in *[16]int16, out *[384]int16)

//go:noescape
func loadEdgesSSE2(b *[yuvSize]byte, top *topSamples, tr, flags uint32)

//go:noescape
func storeMBRowsSSE2(b *[yuvSize]byte, top *topSamples, cy, cu, cv *byte, ys, uvs int)

func reconI4(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) {
	switch {
	case reconLevel >= cpuinfo.AVX2:
		if reconI4AVX2(b, coeffs, bits, modes) {
			return
		}
	case reconLevel >= cpuinfo.SSE41:
		if reconI4SSE41(b, coeffs, bits, modes) {
			return
		}
	}
	reconI4Ref(b, coeffs, bits, modes)
}

func reconI16(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	switch {
	case reconLevel >= cpuinfo.AVX2:
		if reconI16AVX2(b, coeffs, bits, mode) {
			return
		}
	case reconLevel >= cpuinfo.SSE41:
		if reconI16SSE41(b, coeffs, bits, mode) {
			return
		}
	}
	reconI16Ref(b, coeffs, bits, mode)
}

func reconUV(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	switch {
	case reconLevel >= cpuinfo.AVX2:
		if reconUVAVX2(b, coeffs, uvPlaneCodes(bits)|uvPlaneCodes(bits>>8)<<8, mode) {
			return
		}
	case reconLevel >= cpuinfo.SSE41:
		if reconUVSSE41(b, coeffs, uvPlaneCodes(bits)|uvPlaneCodes(bits>>8)<<8, mode) {
			return
		}
	}
	reconUVRef(b, coeffs, bits, mode)
}

// uvPlaneCodes maps the non-zero bits of one chroma plane (the low byte of
// bits) to the transform code the kernels apply to all four of its blocks,
// following doUVTransform: none (0) without any bit, the full transform (3)
// if any block has AC coefficients, otherwise the DC-only transform (1),
// which leaves a block with a zero DC coefficient unchanged just as the
// reference's skipping it does. The result repeats the code for the four
// blocks (0x00, 0x55 or 0xff).
func uvPlaneCodes(bits uint32) uint32 {
	switch {
	case bits&0xff == 0:
		return 0
	case bits&0xaa != 0:
		return 0xff
	}
	return 0x55
}

func transformWHT(in *[16]int16, out *[384]int16) {
	if reconLevel >= cpuinfo.SSE41 {
		transformWHTSSE41(in, out)
		return
	}
	transformWHTRef(in, out)
}

func loadEdges(b *[yuvSize]byte, top *topSamples, tr, flags uint32) {
	if reconLevel >= cpuinfo.SSE41 {
		loadEdgesSSE2(b, top, tr, flags)
		return
	}
	loadEdgesRef(b, top, tr, flags)
}

// storeMBRows checks the cache geometry once and lets the kernel copy the
// rows; anything unusual goes to the reference (which panics on a cache
// that is too small).
func storeMBRows(b *[yuvSize]byte, top *topSamples, cy, cu, cv []byte, ys, uvs int) {
	if reconLevel < cpuinfo.SSE41 || ys < 0 || uvs < 0 || ys > len(cy) || uvs > len(cu) ||
		len(cy) < 15*ys+16 || len(cu) < 7*uvs+8 || len(cv) < 7*uvs+8 {
		storeMBRowsRef(b, top, cy, cu, cv, ys, uvs)
		return
	}
	storeMBRowsSSE2(b, top, &cy[0], &cu[0], &cv[0], ys, uvs)
}
