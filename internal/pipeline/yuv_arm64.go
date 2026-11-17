//go:build arm64 && !purego

package pipeline

// yuvToRGBARowNEON converts pixels [0, n) of full resolution Y, U, V rows
// (and alpha, or opaque when a is nil) to RGBA, n >= 16. It reads y, u, v
// and a [:n] and writes dst[:4*n].
//
//go:noescape
func yuvToRGBARowNEON(y, u, v, a, dst *byte, n int)

// yuvToRGBARow converts full resolution Y'CbCr rows to RGBA with NEON; it
// matches yuvToRGBARowGo bit for bit.
func yuvToRGBARow(yr, ur, vr, ar, out []byte) {
	w := len(yr)
	if w < 16 {
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
	yuvToRGBARowNEON(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
}
