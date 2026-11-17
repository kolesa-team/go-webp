//go:build arm64 && !purego

package vp8

// NEON kernels for the macroblock reconstruction (recon_arm64.s). Each call
// handles a whole macroblock (16 luma sub-blocks, or both chroma planes), so
// the cost of the assembly call is paid once per macroblock rather than once
// per 4x4 block.
//
// All buffers are passed as pointers to fixed-size arrays and the kernels
// only touch the offsets the Go reference touches (see recon_arm64.s), so no
// run-time bounds checks are needed. Mode values outside the valid range
// select the same predictor as the default case of the Go reference.

//go:noescape
func reconI4NEON(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8)

//go:noescape
func reconI16NEON(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int)

//go:noescape
func reconUVNEON(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int)

func reconI4(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) {
	reconI4NEON(b, coeffs, bits, modes)
}

func reconI16(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	reconI16NEON(b, coeffs, bits, mode)
}

func reconUV(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	reconUVNEON(b, coeffs, uvCodes(bits&0xff, coeffs[16*16:])|uvCodes(bits>>8&0xff, coeffs[20*16:])<<8, mode)
}

// uvCodes maps the non-zero bits of one chroma plane to the per-block
// transform codes the kernel applies (0 none, 1 DC only, 3 full), following
// doUVTransform exactly: any AC coefficient in the plane selects the full
// transform for all four blocks, otherwise blocks with a non-zero DC get
// the DC-only transform.
func uvCodes(bits uint32, in []int16) uint32 {
	if bits == 0 {
		return 0
	}
	if bits&0xaa != 0 {
		return 0xff
	}
	in = in[:64]
	var c uint32
	if in[0] != 0 {
		c |= 1 << 6
	}
	if in[16] != 0 {
		c |= 1 << 4
	}
	if in[32] != 0 {
		c |= 1 << 2
	}
	if in[48] != 0 {
		c |= 1
	}
	return c
}
