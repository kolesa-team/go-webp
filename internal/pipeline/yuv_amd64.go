//go:build amd64 && !purego

package pipeline

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// yuvToRGBARowSSE41 converts pixels [0, n) of full resolution Y, U, V rows
// (and alpha, or opaque when a is nil) to RGBA, n >= 16;
// yuvToRGBARowAVX2 needs n >= 32. They read y, u, v and a [:n] and write
// dst[:4*n].
//
//go:noescape
func yuvToRGBARowSSE41(y, u, v, a, dst *byte, n int)

//go:noescape
func yuvToRGBARowAVX2(y, u, v, a, dst *byte, n int)

// yuvToRGBARow converts full resolution Y'CbCr rows to RGBA with SSE4.1 or
// AVX2 (selected by yuvLevel); it matches yuvToRGBARowGo bit for bit.
func yuvToRGBARow(yr, ur, vr, ar, out []byte) {
	w := len(yr)
	if w < 16 || yuvLevel < cpuinfo.SSE41 {
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
	if w >= 32 && yuvLevel >= cpuinfo.AVX2 {
		yuvToRGBARowAVX2(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
	} else {
		yuvToRGBARowSSE41(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
	}
}
