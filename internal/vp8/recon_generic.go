//go:build !arm64 || purego

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
