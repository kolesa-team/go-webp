//go:build (!arm64 && !amd64) || purego

package vp8

// Without an assembly kernel the macroblock reconstruction entry points are
// the Go reference implementations of dsp.go.

func reconI4(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) {
	reconI4Ref(b, coeffs, bits, modes)
}

func reconI16(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	reconI16Ref(b, coeffs, bits, mode)
}

func reconUV(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	reconUVRef(b, coeffs, bits, mode)
}

func loadEdges(b *[yuvSize]byte, top *topSamples, tr, flags uint32) {
	loadEdgesRef(b, top, tr, flags)
}

func storeMBRows(b *[yuvSize]byte, top *topSamples, cy, cu, cv []byte, ys, uvs int) {
	storeMBRowsRef(b, top, cy, cu, cv, ys, uvs)
}

func transformWHT(in *[16]int16, out *[384]int16) {
	transformWHTRef(in, out)
}
