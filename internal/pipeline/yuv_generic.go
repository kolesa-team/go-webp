//go:build (!arm64 && !amd64) || purego

package pipeline

// yuvToRGBARow converts full resolution Y'CbCr rows to RGBA; see
// yuvToRGBARowGo.
func yuvToRGBARow(yr, ur, vr, ar, out []byte) {
	yuvToRGBARowGo(yr, ur, vr, ar, out)
}
