//go:build amd64 && !purego

package pipeline

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// x86Level is the SIMD level of the kernels; tests lower it to run every
// level against the Go reference.
var x86Level = cpuinfo.Max

// yuvToRGBARowSSE41 converts pixels [0, n) of full resolution Y, U, V rows
// (and alpha, or opaque when a is nil) to RGBA, n >= 16. It reads y, u, v
// and a [:n] and writes dst[:4*n].
//
//go:noescape
func yuvToRGBARowSSE41(y, u, v, a, dst *byte, n int)

// yuvToRGBARowAVX2 is yuvToRGBARowSSE41 for n >= 32.
//
//go:noescape
func yuvToRGBARowAVX2(y, u, v, a, dst *byte, n int)

// yuvToRGBARow converts full resolution Y'CbCr rows to RGBA with SSE4.1 or
// AVX2; it matches yuvToRGBARowGo bit for bit.
func yuvToRGBARow(yr, ur, vr, ar, out []byte) {
	w := len(yr)
	if w < 16 || x86Level < cpuinfo.SSE41 {
		yuvToRGBARowGo(yr, ur, vr, ar, out)
		return
	}
	ur, vr = ur[:w], vr[:w]
	out = out[:4*w]
	var pa *byte
	if ar != nil {
		ar = ar[:w]
		pa = &ar[0]
	}
	if w >= 32 && x86Level >= cpuinfo.AVX2 {
		yuvToRGBARowAVX2(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
		return
	}
	yuvToRGBARowSSE41(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
}
