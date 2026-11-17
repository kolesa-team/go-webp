package vp8

import "encoding/binary"

// Algorithm adapted from libwebp (BSD-3-Clause): dsp/dec.c (inverse
// transforms, intra predictors and in-loop filters).

// bps is the stride of the reconstruction work buffer.
const bps = 32

// Offsets of the Y, U and V blocks inside the work buffer. One row and a
// few columns of context are kept above and to the left of each block.
const (
	yOff    = bps*1 + 8
	uOff    = yOff + bps*16 + bps
	vOff    = uOff + 16
	yuvSize = bps*17 + bps*9
)

// Intra prediction modes, in libwebp order.
const (
	bDCPred = iota
	bTMPred
	bVEPred
	bHEPred
	bRDPred
	bVRPred
	bLDPred
	bVLPred
	bHDPred
	bHUPred

	// Special DC modes used at frame edges (16x16 and chroma only).
	bDCPredNoTop     = 4
	bDCPredNoLeft    = 5
	bDCPredNoTopLeft = 6
)

func clip8(v int) uint8 {
	if uint(v) <= 255 {
		return uint8(v)
	}
	if v < 0 {
		return 0
	}
	return 255
}

// --- inverse transforms ---------------------------------------------------

func mul1(a int) int { return ((a * 20091) >> 16) + a }
func mul2(a int) int { return (a * 35468) >> 16 }

// transformOne adds the inverse DCT of in to the 4x4 block at b[o:].
func transformOne(in *[16]int16, b []byte, o int) {
	var tmp [16]int
	for i := 0; i < 4; i++ { // vertical pass
		i0, i4, i8, i12 := int(in[i]), int(in[4+i]), int(in[8+i]), int(in[12+i])
		a := i0 + i8
		bb := i0 - i8
		c := mul2(i4) - mul1(i12)
		d := mul1(i4) + mul2(i12)
		tmp[4*i+0] = a + d
		tmp[4*i+1] = bb + c
		tmp[4*i+2] = bb - c
		tmp[4*i+3] = a - d
	}
	dst := (*[3*bps + 4]byte)(b[o:])
	for i := 0; i < 4; i++ { // horizontal pass
		dc := tmp[i] + 4
		t4, t8, t12 := tmp[4+i], tmp[8+i], tmp[12+i]
		a := dc + t8
		bb := dc - t8
		c := mul2(t4) - mul1(t12)
		d := mul1(t4) + mul2(t12)
		r := i * bps
		dst[r+0] = clip8(int(dst[r+0]) + ((a + d) >> 3))
		dst[r+1] = clip8(int(dst[r+1]) + ((bb + c) >> 3))
		dst[r+2] = clip8(int(dst[r+2]) + ((bb - c) >> 3))
		dst[r+3] = clip8(int(dst[r+3]) + ((a - d) >> 3))
	}
}

// transformAC3 handles blocks whose only non-zero coefficients are in[0],
// in[1] and in[4].
func transformAC3(in *[16]int16, b []byte, o int) {
	a := int(in[0]) + 4
	c4 := mul2(int(in[4]))
	d4 := mul1(int(in[4]))
	c1 := mul2(int(in[1]))
	d1 := mul1(int(in[1]))
	dst := (*[3*bps + 4]byte)(b[o:])
	store2 := func(r, dc int) {
		dst[r+0] = clip8(int(dst[r+0]) + ((dc + d1) >> 3))
		dst[r+1] = clip8(int(dst[r+1]) + ((dc + c1) >> 3))
		dst[r+2] = clip8(int(dst[r+2]) + ((dc - c1) >> 3))
		dst[r+3] = clip8(int(dst[r+3]) + ((dc - d1) >> 3))
	}
	store2(0, a+d4)
	store2(bps, a+c4)
	store2(2*bps, a-c4)
	store2(3*bps, a-d4)
}

// transformDC handles blocks with only a DC coefficient.
func transformDC(in *[16]int16, b []byte, o int) {
	dc := (int(in[0]) + 4) >> 3
	dst := (*[3*bps + 4]byte)(b[o:])
	for r := 0; r < 4*bps; r += bps {
		dst[r+0] = clip8(int(dst[r+0]) + dc)
		dst[r+1] = clip8(int(dst[r+1]) + dc)
		dst[r+2] = clip8(int(dst[r+2]) + dc)
		dst[r+3] = clip8(int(dst[r+3]) + dc)
	}
}

// transformWHT computes the inverse Walsh-Hadamard transform of the luma DC
// coefficients and scatters them into the 16 luma blocks of out.
func transformWHT(in *[16]int16, out *[384]int16) {
	var tmp [16]int
	for i := 0; i < 4; i++ {
		a0 := int(in[0+i]) + int(in[12+i])
		a1 := int(in[4+i]) + int(in[8+i])
		a2 := int(in[4+i]) - int(in[8+i])
		a3 := int(in[0+i]) - int(in[12+i])
		tmp[0+i] = a0 + a1
		tmp[8+i] = a0 - a1
		tmp[4+i] = a3 + a2
		tmp[12+i] = a3 - a2
	}
	for i := 0; i < 4; i++ {
		dc := tmp[0+i*4] + 3 // w/ rounder
		a0 := dc + tmp[3+i*4]
		a1 := tmp[1+i*4] + tmp[2+i*4]
		a2 := tmp[1+i*4] - tmp[2+i*4]
		a3 := dc - tmp[3+i*4]
		o := i * 64
		out[o+0] = int16((a0 + a1) >> 3)
		out[o+16] = int16((a3 + a2) >> 3)
		out[o+32] = int16((a0 - a1) >> 3)
		out[o+48] = int16((a3 - a2) >> 3)
	}
}

// doTransform dispatches on the 2-bit non-zero code of a block.
func doTransform(code uint32, in *[16]int16, b []byte, o int) {
	switch code {
	case 3:
		transformOne(in, b, o)
	case 2:
		transformAC3(in, b, o)
	case 1:
		transformDC(in, b, o)
	}
}

// doUVTransform reconstructs the four 4x4 blocks of one chroma plane.
func doUVTransform(bits uint32, in *[384]int16, first int, b []byte, o int) {
	if bits&0xff == 0 {
		return
	}
	b0 := (*[16]int16)(in[first:])
	b1 := (*[16]int16)(in[first+16:])
	b2 := (*[16]int16)(in[first+32:])
	b3 := (*[16]int16)(in[first+48:])
	if bits&0xaa != 0 { // any non-zero AC coefficient
		transformOne(b0, b, o)
		transformOne(b1, b, o+4)
		transformOne(b2, b, o+4*bps)
		transformOne(b3, b, o+4*bps+4)
		return
	}
	if b0[0] != 0 {
		transformDC(b0, b, o)
	}
	if b1[0] != 0 {
		transformDC(b1, b, o+4)
	}
	if b2[0] != 0 {
		transformDC(b2, b, o+4*bps)
	}
	if b3[0] != 0 {
		transformDC(b3, b, o+4*bps+4)
	}
}

// --- per-macroblock reconstruction -------------------------------------------
//
// These are the reference implementations of the macroblock-level entry
// points reconI4, reconI16 and reconUV. Architectures with an assembly
// kernel (recon_arm64.s) bind the entry points to it; the others, and the
// purego build, bind them to these functions (recon_generic.go). The
// assembly must stay bit-exact with them for every input, which the
// differential tests in recon_test.go check.

// reconI4Ref predicts and reconstructs the 16 luma sub-blocks of an
// intra-4x4 macroblock in raster order. The top-right samples of the
// macroblock must already be in place (and replicated below, see
// reconstructMB). bits holds the 2-bit transform code of each sub-block,
// sub-block 0 in the top bits.
func reconI4Ref(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) {
	for n := 0; n < 16; n++ {
		o := yOff + kScan[n]
		predLuma4(modes[n], b[:], o)
		doTransform(bits>>30, (*[16]int16)(coeffs[n*16:]), b[:], o)
		bits <<= 2
	}
}

// reconI16Ref runs the 16x16 luma predictor mode (after checkMode) and adds
// the residuals of the 16 luma blocks.
func reconI16Ref(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	predLuma16(mode, b[:])
	if bits != 0 {
		for n := 0; n < 16; n++ {
			doTransform(bits>>30, (*[16]int16)(coeffs[n*16:]), b[:], yOff+kScan[n])
			bits <<= 2
		}
	}
}

// reconUVRef runs the 8x8 chroma predictor mode (after checkMode) on both
// chroma planes and adds their residuals; bits is mbBlock.nonZeroUV.
func reconUVRef(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) {
	predChroma8(mode, b[:], uOff)
	predChroma8(mode, b[:], vOff)
	doUVTransform(bits>>0, coeffs, 16*16, b[:], uOff)
	doUVTransform(bits>>8, coeffs, 20*16, b[:], vOff)
}

// loadEdges flags.
const (
	edgeRotate   = 1 << iota // copy the right samples of the previous macroblock to the left
	edgeTop                  // load the top samples
	edgeTopRight             // store tr as the top-right samples (intra 4x4)
	edgeI4                   // replicate the top-right samples below (intra 4x4)
)

// loadEdgesRef sets up the context samples of the next macroblock in the
// work buffer (the first half of libwebp's ReconstructRow loop body): the
// left column from the previous macroblock, the top row from top and the
// four top-right samples tr (little endian), replicated on the rows above
// sub-block rows 1-3 for intra 4x4.
func loadEdgesRef(b *[yuvSize]byte, top *topSamples, tr, flags uint32) {
	if flags&edgeRotate != 0 {
		for j := -1; j < 16; j++ {
			o := yOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+12:])
		}
		for j := -1; j < 8; j++ {
			o := uOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+4:])
			o = vOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+4:])
		}
	}
	if flags&edgeTop != 0 {
		*(*[16]byte)(b[yOff-bps:]) = top.y
		*(*[8]byte)(b[uOff-bps:]) = top.u
		*(*[8]byte)(b[vOff-bps:]) = top.v
	}
	if flags&edgeI4 != 0 {
		const tro = yOff - bps + 16
		if flags&edgeTopRight != 0 {
			binary.LittleEndian.PutUint32(b[tro:], tr)
		}
		for k := 1; k <= 3; k++ {
			*(*[4]byte)(b[tro+4*k*bps:]) = *(*[4]byte)(b[tro:])
		}
	}
}

// storeMBRowsRef saves the bottom samples of the reconstructed macroblock
// in top (unless it is nil) and copies the macroblock into the row cache:
// luma row j to cy[j*ys:], chroma row j to cu[j*uvs:] and cv[j*uvs:].
func storeMBRowsRef(b *[yuvSize]byte, top *topSamples, cy, cu, cv []byte, ys, uvs int) {
	if top != nil {
		top.y = *(*[16]byte)(b[yOff+15*bps:])
		top.u = *(*[8]byte)(b[uOff+7*bps:])
		top.v = *(*[8]byte)(b[vOff+7*bps:])
	}
	for j := 0; j < 16; j++ {
		*(*[16]byte)(cy[j*ys:]) = *(*[16]byte)(b[yOff+j*bps:])
	}
	for j := 0; j < 8; j++ {
		*(*[8]byte)(cu[j*uvs:]) = *(*[8]byte)(b[uOff+j*bps:])
		*(*[8]byte)(cv[j*uvs:]) = *(*[8]byte)(b[vOff+j*bps:])
	}
}

// --- intra prediction -------------------------------------------------------

func avg3(a, b, c int) uint8 { return uint8((a + 2*b + c + 2) >> 2) }
func avg2(a, b int) uint8    { return uint8((a + b + 1) >> 1) }

// trueMotion implements TM prediction for a size x size block at b[o:].
func trueMotion(b []byte, o, size int) {
	tl := int(b[o-bps-1])
	top := b[o-bps : o-bps+size]
	for y := 0; y < size; y++ {
		row := b[o+y*bps : o+y*bps+size]
		l := int(b[o+y*bps-1]) - tl
		for x := range row {
			row[x] = clip8(int(top[x]) + l)
		}
	}
}

// put8 stores an 8-byte word at b[o:].
func put8(b []byte, o int, v uint64) {
	binary.LittleEndian.PutUint64((*[8]byte)(b[o : o+8])[:], v)
}

// fill sets a size x size block (size is 8 or 16) to v.
func fill(b []byte, o, size int, v uint8) {
	v64 := uint64(v) * 0x0101010101010101
	for y := 0; y < size; y++ {
		put8(b, o+y*bps, v64)
		if size == 16 {
			put8(b, o+y*bps+8, v64)
		}
	}
}

func verticalPred(b []byte, o, size int) {
	t0 := binary.LittleEndian.Uint64(b[o-bps:])
	t1 := uint64(0)
	if size == 16 {
		t1 = binary.LittleEndian.Uint64(b[o-bps+8:])
	}
	for y := 0; y < size; y++ {
		put8(b, o+y*bps, t0)
		if size == 16 {
			put8(b, o+y*bps+8, t1)
		}
	}
}

func horizontalPred(b []byte, o, size int) {
	for y := 0; y < size; y++ {
		v64 := uint64(b[o+y*bps-1]) * 0x0101010101010101
		put8(b, o+y*bps, v64)
		if size == 16 {
			put8(b, o+y*bps+8, v64)
		}
	}
}

func sumTop(b []byte, o, size int) int {
	s := 0
	for _, v := range b[o-bps : o-bps+size] {
		s += int(v)
	}
	return s
}

func sumLeft(b []byte, o, size int) int {
	s := 0
	for y := 0; y < size; y++ {
		s += int(b[o+y*bps-1])
	}
	return s
}

// predLuma16 runs the 16x16 predictor for mode (after checkMode).
func predLuma16(mode int, b []byte) {
	const o = yOff
	switch mode {
	case bDCPred:
		fill(b, o, 16, uint8((sumTop(b, o, 16)+sumLeft(b, o, 16)+16)>>5))
	case bTMPred:
		trueMotion(b, o, 16)
	case bVEPred:
		verticalPred(b, o, 16)
	case bHEPred:
		horizontalPred(b, o, 16)
	case bDCPredNoTop:
		fill(b, o, 16, uint8((sumLeft(b, o, 16)+8)>>4))
	case bDCPredNoLeft:
		fill(b, o, 16, uint8((sumTop(b, o, 16)+8)>>4))
	default: // bDCPredNoTopLeft
		fill(b, o, 16, 0x80)
	}
}

// predChroma8 runs the 8x8 chroma predictor for mode (after checkMode).
func predChroma8(mode int, b []byte, o int) {
	switch mode {
	case bDCPred:
		fill(b, o, 8, uint8((sumTop(b, o, 8)+sumLeft(b, o, 8)+8)>>4))
	case bTMPred:
		trueMotion(b, o, 8)
	case bVEPred:
		verticalPred(b, o, 8)
	case bHEPred:
		horizontalPred(b, o, 8)
	case bDCPredNoTop:
		fill(b, o, 8, uint8((sumLeft(b, o, 8)+4)>>3))
	case bDCPredNoLeft:
		fill(b, o, 8, uint8((sumTop(b, o, 8)+4)>>3))
	default:
		fill(b, o, 8, 0x80)
	}
}

// pred4 window layout: s[0] is the top-left pixel, s[1+i] the top row
// (8 pixels incl. top-right), s[bps*(j+1)] the left column, and
// s[bps*(y+1)+1+x] the destination pixel (x, y).
type pred4Win = [4*bps + 8]byte

const (
	p4X = 0 // top-left
	p4A = 1 // top[0]
	p4I = bps
	p4J = 2 * bps
	p4K = 3 * bps
	p4L = 4 * bps
)

func p4(x, y int) int { return bps*(y+1) + 1 + x }

func putRow4(s *pred4Win, y int, v uint8) {
	o := bps*(y+1) + 1
	s[o], s[o+1], s[o+2], s[o+3] = v, v, v, v
}

// predLuma4 runs the 4x4 predictor for mode on the sub-block at b[o:].
func predLuma4(mode uint8, b []byte, o int) {
	s := (*pred4Win)(b[o-bps-1:])
	switch mode {
	case bDCPred:
		dc := 4
		for i := 0; i < 4; i++ {
			dc += int(s[p4A+i]) + int(s[bps*(i+1)])
		}
		v := uint8(dc >> 3)
		for y := 0; y < 4; y++ {
			putRow4(s, y, v)
		}
	case bTMPred:
		tl := int(s[p4X])
		for y := 0; y < 4; y++ {
			l := int(s[bps*(y+1)]) - tl
			r := bps*(y+1) + 1
			s[r+0] = clip8(int(s[1]) + l)
			s[r+1] = clip8(int(s[2]) + l)
			s[r+2] = clip8(int(s[3]) + l)
			s[r+3] = clip8(int(s[4]) + l)
		}
	case bVEPred:
		v0 := avg3(int(s[0]), int(s[1]), int(s[2]))
		v1 := avg3(int(s[1]), int(s[2]), int(s[3]))
		v2 := avg3(int(s[2]), int(s[3]), int(s[4]))
		v3 := avg3(int(s[3]), int(s[4]), int(s[5]))
		for y := 0; y < 4; y++ {
			r := bps*(y+1) + 1
			s[r], s[r+1], s[r+2], s[r+3] = v0, v1, v2, v3
		}
	case bHEPred:
		A, B, C, D, E := int(s[p4X]), int(s[p4I]), int(s[p4J]), int(s[p4K]), int(s[p4L])
		putRow4(s, 0, avg3(A, B, C))
		putRow4(s, 1, avg3(B, C, D))
		putRow4(s, 2, avg3(C, D, E))
		putRow4(s, 3, avg3(D, E, E))
	case bRDPred:
		I, J, K, L, X := int(s[p4I]), int(s[p4J]), int(s[p4K]), int(s[p4L]), int(s[p4X])
		A, B, C, D := int(s[1]), int(s[2]), int(s[3]), int(s[4])
		s[p4(0, 3)] = avg3(J, K, L)
		v := avg3(I, J, K)
		s[p4(1, 3)], s[p4(0, 2)] = v, v
		v = avg3(X, I, J)
		s[p4(2, 3)], s[p4(1, 2)], s[p4(0, 1)] = v, v, v
		v = avg3(A, X, I)
		s[p4(3, 3)], s[p4(2, 2)], s[p4(1, 1)], s[p4(0, 0)] = v, v, v, v
		v = avg3(B, A, X)
		s[p4(3, 2)], s[p4(2, 1)], s[p4(1, 0)] = v, v, v
		v = avg3(C, B, A)
		s[p4(3, 1)], s[p4(2, 0)] = v, v
		s[p4(3, 0)] = avg3(D, C, B)
	case bVRPred:
		I, J, K, X := int(s[p4I]), int(s[p4J]), int(s[p4K]), int(s[p4X])
		A, B, C, D := int(s[1]), int(s[2]), int(s[3]), int(s[4])
		v := avg2(X, A)
		s[p4(0, 0)], s[p4(1, 2)] = v, v
		v = avg2(A, B)
		s[p4(1, 0)], s[p4(2, 2)] = v, v
		v = avg2(B, C)
		s[p4(2, 0)], s[p4(3, 2)] = v, v
		s[p4(3, 0)] = avg2(C, D)
		s[p4(0, 3)] = avg3(K, J, I)
		s[p4(0, 2)] = avg3(J, I, X)
		v = avg3(I, X, A)
		s[p4(0, 1)], s[p4(1, 3)] = v, v
		v = avg3(X, A, B)
		s[p4(1, 1)], s[p4(2, 3)] = v, v
		v = avg3(A, B, C)
		s[p4(2, 1)], s[p4(3, 3)] = v, v
		s[p4(3, 1)] = avg3(B, C, D)
	case bLDPred:
		A, B, C, D := int(s[1]), int(s[2]), int(s[3]), int(s[4])
		E, F, G, H := int(s[5]), int(s[6]), int(s[7]), int(s[8])
		s[p4(0, 0)] = avg3(A, B, C)
		v := avg3(B, C, D)
		s[p4(1, 0)], s[p4(0, 1)] = v, v
		v = avg3(C, D, E)
		s[p4(2, 0)], s[p4(1, 1)], s[p4(0, 2)] = v, v, v
		v = avg3(D, E, F)
		s[p4(3, 0)], s[p4(2, 1)], s[p4(1, 2)], s[p4(0, 3)] = v, v, v, v
		v = avg3(E, F, G)
		s[p4(3, 1)], s[p4(2, 2)], s[p4(1, 3)] = v, v, v
		v = avg3(F, G, H)
		s[p4(3, 2)], s[p4(2, 3)] = v, v
		s[p4(3, 3)] = avg3(G, H, H)
	case bVLPred:
		A, B, C, D := int(s[1]), int(s[2]), int(s[3]), int(s[4])
		E, F, G, H := int(s[5]), int(s[6]), int(s[7]), int(s[8])
		s[p4(0, 0)] = avg2(A, B)
		v := avg2(B, C)
		s[p4(1, 0)], s[p4(0, 2)] = v, v
		v = avg2(C, D)
		s[p4(2, 0)], s[p4(1, 2)] = v, v
		v = avg2(D, E)
		s[p4(3, 0)], s[p4(2, 2)] = v, v
		s[p4(0, 1)] = avg3(A, B, C)
		v = avg3(B, C, D)
		s[p4(1, 1)], s[p4(0, 3)] = v, v
		v = avg3(C, D, E)
		s[p4(2, 1)], s[p4(1, 3)] = v, v
		v = avg3(D, E, F)
		s[p4(3, 1)], s[p4(2, 3)] = v, v
		s[p4(3, 2)] = avg3(E, F, G)
		s[p4(3, 3)] = avg3(F, G, H)
	case bHDPred:
		I, J, K, L, X := int(s[p4I]), int(s[p4J]), int(s[p4K]), int(s[p4L]), int(s[p4X])
		A, B, C := int(s[1]), int(s[2]), int(s[3])
		v := avg2(I, X)
		s[p4(0, 0)], s[p4(2, 1)] = v, v
		v = avg2(J, I)
		s[p4(0, 1)], s[p4(2, 2)] = v, v
		v = avg2(K, J)
		s[p4(0, 2)], s[p4(2, 3)] = v, v
		s[p4(0, 3)] = avg2(L, K)
		s[p4(3, 0)] = avg3(A, B, C)
		s[p4(2, 0)] = avg3(X, A, B)
		v = avg3(I, X, A)
		s[p4(1, 0)], s[p4(3, 1)] = v, v
		v = avg3(J, I, X)
		s[p4(1, 1)], s[p4(3, 2)] = v, v
		v = avg3(K, J, I)
		s[p4(1, 2)], s[p4(3, 3)] = v, v
		s[p4(1, 3)] = avg3(L, K, J)
	default: // bHUPred
		I, J, K, L := int(s[p4I]), int(s[p4J]), int(s[p4K]), int(s[p4L])
		s[p4(0, 0)] = avg2(I, J)
		v := avg2(J, K)
		s[p4(2, 0)], s[p4(0, 1)] = v, v
		v = avg2(K, L)
		s[p4(2, 1)], s[p4(0, 2)] = v, v
		s[p4(1, 0)] = avg3(I, J, K)
		v = avg3(J, K, L)
		s[p4(3, 0)], s[p4(1, 1)] = v, v
		v = avg3(K, L, L)
		s[p4(3, 1)], s[p4(1, 2)] = v, v
		l := uint8(L)
		s[p4(3, 2)], s[p4(2, 2)] = l, l
		s[p4(0, 3)], s[p4(1, 3)], s[p4(2, 3)], s[p4(3, 3)] = l, l, l, l
	}
}
