//go:build arm64 && !purego

// NEON kernels of the VP8 encoder (see dsp_arm64.go for the Go wrappers,
// which do all the bounds checking, and dsp_batch.go for the reference
// semantics). Every kernel is bit-exact with the Go code of dsp.go.
// Algorithms adapted from libwebp's src/dsp/enc_neon.c (BSD-3-Clause).
//
// All 4x4 blocks use a row stride of bps = 32 bytes. Pair tables
// (pairTable.off) hold two uint16 offsets per pair of horizontally
// adjacent blocks.
//
// Go assembler operand order reminder: "VADD Vm, Vn, Vd" is Vd = Vn + Vm,
// "VSUB Vm, Vn, Vd" is Vd = Vn - Vm, and the permutes (VTRN1, VZIP1, VUZP2)
// likewise take Vm first: "VTRN1 Vm, Vn, Vd" is Vd = trn1(Vn, Vm).

#include "go_asm.h"
#include "textflag.h"

// Instructions the Go 1.27 arm64 assembler has no mnemonic for, as raw
// encodings (each one checked against clang's encoding of the instruction
// named in the comment). d, n, m are vector register numbers.

// USUBL Vd.8H, Vn.8B, Vm.8B (unsigned widening subtract)
#define USUBL8H(d, n, m) WORD $(0x2E202000 | ((m)<<16) | ((n)<<5) | (d))
// UABD Vd.8B, Vn.8B, Vm.8B (unsigned absolute difference)
#define UABD8B(d, n, m) WORD $(0x2E207400 | ((m)<<16) | ((n)<<5) | (d))
// UABD Vd.16B, Vn.16B, Vm.16B
#define UABD16B(d, n, m) WORD $(0x6E207400 | ((m)<<16) | ((n)<<5) | (d))
// UADALP Vd.4S, Vn.8H (unsigned add and accumulate long pairwise)
#define UADALP4S(d, n) WORD $(0x6E606800 | ((n)<<5) | (d))

// TRANSPOSE4x4H transposes the two 4x4 int16 matrices held side by side
// in r0..r3 (r_i = [A row i | B row i]) in place, using t0..t3 as scratch:
// afterwards r_j = [A column j | B column j].
#define TRANSPOSE4x4H(r0, r1, r2, r3, t0, t1, t2, t3) \
	VTRN1	r1.H8, r0.H8, t0.H8 \
	VTRN2	r1.H8, r0.H8, t1.H8 \
	VTRN1	r3.H8, r2.H8, t2.H8 \
	VTRN2	r3.H8, r2.H8, t3.H8 \
	VTRN1	t2.S4, t0.S4, r0.S4 \
	VTRN2	t2.S4, t0.S4, r2.S4 \
	VTRN1	t3.S4, t1.S4, r1.S4 \
	VTRN2	t3.S4, t1.S4, r3.S4

// TRANSPOSE4x4S transposes the 4x4 int32 matrix in r0..r3 (one row per
// register) into d0..d3 (one column per register), using t0..t3.
#define TRANSPOSE4x4S(r0, r1, r2, r3, d0, d1, d2, d3, t0, t1, t2, t3) \
	VTRN1	r1.S4, r0.S4, t0.S4 \
	VTRN2	r1.S4, r0.S4, t1.S4 \
	VTRN1	r3.S4, r2.S4, t2.S4 \
	VTRN2	r3.S4, r2.S4, t3.S4 \
	VTRN1	t2.D2, t0.D2, d0.D2 \
	VTRN2	t2.D2, t0.D2, d2.D2 \
	VTRN1	t3.D2, t1.D2, d1.D2 \
	VTRN2	t3.D2, t1.D2, d3.D2

// Byte shuffle putting 16 int16 levels in zigzag order:
// out[n] = level[kZigzag[n]] (two 16-byte TBL indexes over a 32-byte table).
DATA ·zigzagTBL+0(SB)/8, $0x1110090803020100
DATA ·zigzagTBL+8(SB)/8, $0x0d0c070605040b0a
DATA ·zigzagTBL+16(SB)/8, $0x15141b1a19181312
DATA ·zigzagTBL+24(SB)/8, $0x1f1e1d1c17160f0e
GLOBL ·zigzagTBL(SB), RODATA|NOPTR, $32

// Bytes 1..16 (index of a zigzag position plus one) and the mask of the
// AC positions 1..15, for the level summaries of quantizeBlocksNEON.
DATA ·levelIdx+0(SB)/8, $0x0807060504030201
DATA ·levelIdx+8(SB)/8, $0x100f0e0d0c0b0a09
DATA ·levelIdx+16(SB)/8, $0x0101010101010100
DATA ·levelIdx+24(SB)/8, $0x0101010101010101
GLOBL ·levelIdx(SB), RODATA|NOPTR, $32

// ---------------------------------------------------------------------------
// func quantizeBlocksNEON(in, out *[16]int16, n int, m *matrix, info *uint16) uint32
//
// quantizeBlock (dsp.go) on n consecutive blocks; mirrors libwebp's
// QuantizeBlock_NEON. Per block, two vectors of 8 coefficients:
//   sign  = x >> 15                       (SSHR)
//   coeff = |x| + sharpen                 (ABS, ADD; uint16, no overflow)
//   level = min((coeff*iq + bias) >> 17, 2047)
//           (UMLAL into 32-bit accumulators preset to bias, UZP2 takes the
//           high halves = >> 16, USHR 1, UMIN)
//   level = (level ^ sign) - sign
//   in    = level * q                     (MUL, low 16 bits as in Go)
//   out   = zigzag(level)                 (TBL)
// then the summary levelInfoOf(out) (dsp_batch.go): CMTST gives the
// non-zero lanes, UZP1 narrows them to bytes, UMAXV of (mask & 1..16) is
// the last index plus one, ADDV of (mask & AC positions) the AC count.
// It is stored to info[b] unless info is nil. Bit b of the result is set
// if block b has a non-zero level.
//
// Registers: R0 in, R1 out, R2 blocks left, R3 matrix, R6 nz mask, R7
// current bit, R10 info. V16-V17 iq, V18-V19 sharpen, V20-V21 q, V22-V25
// bias, V26-V27 zigzag index, V28 2047, V29 1..16, V30 AC mask.
TEXT ·quantizeBlocksNEON(SB), NOSPLIT, $0-44
	MOVD	in+0(FP), R0
	MOVD	out+8(FP), R1
	MOVD	n+16(FP), R2
	MOVD	m+24(FP), R3
	MOVD	info+32(FP), R10
	MOVD	$·levelIdx(SB), R4
	VLD1	(R4), [V29.B16, V30.B16]
	ADD	$matrix_iq16, R3, R4
	VLD1	(R4), [V16.H8, V17.H8]
	ADD	$matrix_sharpen16, R3, R4
	VLD1	(R4), [V18.H8, V19.H8]
	ADD	$matrix_q16, R3, R4
	VLD1	(R4), [V20.H8, V21.H8]
	ADD	$matrix_bias, R3, R4
	VLD1	(R4), [V22.S4, V23.S4, V24.S4, V25.S4]
	MOVD	$·zigzagTBL(SB), R4
	VLD1	(R4), [V26.B16, V27.B16]
	MOVD	$2047, R5
	VDUP	R5, V28.H8
	MOVD	ZR, R6
	MOVD	$1, R7
	CBZ	R2, qdone

qloop:
	VLD1	(R0), [V0.H8, V1.H8]
	VSSHR	$15, V0.H8, V2.H8
	VSSHR	$15, V1.H8, V3.H8
	VABS	V0.H8, V0.H8
	VABS	V1.H8, V1.H8
	VADD	V18.H8, V0.H8, V0.H8
	VADD	V19.H8, V1.H8, V1.H8
	VMOV	V22.B16, V4.B16
	VMOV	V23.B16, V5.B16
	VMOV	V24.B16, V6.B16
	VMOV	V25.B16, V7.B16
	VUMLAL	V16.H4, V0.H4, V4.S4
	VUMLAL2	V16.H8, V0.H8, V5.S4
	VUMLAL	V17.H4, V1.H4, V6.S4
	VUMLAL2	V17.H8, V1.H8, V7.S4
	VUZP2	V5.H8, V4.H8, V8.H8
	VUZP2	V7.H8, V6.H8, V9.H8
	VUSHR	$1, V8.H8, V8.H8
	VUSHR	$1, V9.H8, V9.H8
	VUMIN	V28.H8, V8.H8, V8.H8
	VUMIN	V28.H8, V9.H8, V9.H8
	VEOR	V2.B16, V8.B16, V8.B16
	VEOR	V3.B16, V9.B16, V9.B16
	VSUB	V2.H8, V8.H8, V8.H8
	VSUB	V3.H8, V9.H8, V9.H8
	VMUL	V20.H8, V8.H8, V10.H8
	VMUL	V21.H8, V9.H8, V11.H8
	VST1.P	[V10.H8, V11.H8], 32(R0)
	VTBL	V26.B16, [V8.B16, V9.B16], V12.B16
	VTBL	V27.B16, [V8.B16, V9.B16], V13.B16
	VST1.P	[V12.H8, V13.H8], 32(R1)
	VCMTST	V12.H8, V12.H8, V14.H8
	VCMTST	V13.H8, V13.H8, V15.H8
	VUZP1	V15.B16, V14.B16, V14.B16 // 0xff where the level is non-zero
	VAND	V29.B16, V14.B16, V15.B16
	VUMAXV	V15.B16, V15
	VMOV	V15.B[0], R8             // last + 1
	CMP	$0, R8
	CSEL	NE, R7, ZR, R9
	ORR	R9, R6, R6
	CBZ	R10, qnext
	VAND	V30.B16, V14.B16, V15.B16
	VADDV	V15.B16, V15
	VMOV	V15.B[0], R9             // non-zero AC levels
	ORR	R9<<8, R8, R8
	MOVH	R8, (R10)
	ADD	$2, R10

qnext:
	LSL	$1, R7, R7
	SUBS	$1, R2, R2
	BNE	qloop

qdone:
	MOVW	R6, ret+40(FP)
	RET

// ---------------------------------------------------------------------------
// func fTransformPairsNEON(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)
//
// fTransform (dsp.go) of the n block pairs of a pair table; mirrors
// libwebp's FTransform_NEON, two blocks at a time. Lanes 0-3 hold block A
// and lanes 4-7 block B of the pair.
//   load:   4 rows of 8 bytes from src and ref, USUBL -> int16 rows
//   transpose -> D0..D3 (column k of each row)
//   pass 1 (lanes = rows): a0..a3 butterflies,
//           t0 = (a0+a1)<<3, t2 = (a0-a1)<<3,
//           t1 = (a2*2217 + a3*5352 + 1812) >> 9  (SMLAL, SHRN),
//           t3 = (a3*2217 - a2*5352 + 937) >> 9
//   transpose -> lanes = columns
//   pass 2: out0 = (a0+a1+7)>>4, out2 = (a0-a1+7)>>4,
//           out1 = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3 != 0)
//                  (accumulator preset to 12000+65536, plus the CMEQ mask),
//           out3 = (a3*2217 - a2*5352 + 51000) >> 16
// All intermediate values fit their lanes for any byte input (|t| <= 8160,
// |a0+a1| <= 32640), so the narrowing shifts are exact.
//
// Registers: R0 src, R1 ref, R2 table, R3 pairs left, R4 out, R5/R6 row
// pointers, R7 = bps. V24 2217, V25 5352, V26 1812, V27 937, V28 77536,
// V29 51000, V30 7.
TEXT ·fTransformPairsNEON(SB), NOSPLIT, $0-40
	MOVD	src+0(FP), R0
	MOVD	ref+8(FP), R1
	MOVD	off+16(FP), R2
	MOVD	n+24(FP), R3
	MOVD	out+32(FP), R4
	MOVD	$32, R7
	MOVD	$2217, R5
	VDUP	R5, V24.H8
	MOVD	$5352, R5
	VDUP	R5, V25.H8
	MOVD	$1812, R5
	VDUP	R5, V26.S4
	MOVD	$937, R5
	VDUP	R5, V27.S4
	MOVD	$(12000+65536), R5
	VDUP	R5, V28.S4
	MOVD	$51000, R5
	VDUP	R5, V29.S4
	MOVD	$7, R5
	VDUP	R5, V30.H8
	CBZ	R3, fdone

floop:
	MOVHU	(R2), R5
	MOVHU	2(R2), R6
	ADD	$4, R2
	ADD	R0, R5, R5
	ADD	R1, R6, R6
	VLD1.P	(R5)(R7), [V0.B8]
	VLD1.P	(R5)(R7), [V1.B8]
	VLD1.P	(R5)(R7), [V2.B8]
	VLD1	(R5), [V3.B8]
	VLD1.P	(R6)(R7), [V4.B8]
	VLD1.P	(R6)(R7), [V5.B8]
	VLD1.P	(R6)(R7), [V6.B8]
	VLD1	(R6), [V7.B8]
	USUBL8H(0, 0, 4)
	USUBL8H(1, 1, 5)
	USUBL8H(2, 2, 6)
	USUBL8H(3, 3, 7)
	TRANSPOSE4x4H(V0, V1, V2, V3, V4, V5, V6, V7)

	// pass 1
	VADD	V3.H8, V0.H8, V4.H8 // a0 = d0 + d3
	VADD	V2.H8, V1.H8, V5.H8 // a1 = d1 + d2
	VSUB	V2.H8, V1.H8, V6.H8 // a2 = d1 - d2
	VSUB	V3.H8, V0.H8, V7.H8 // a3 = d0 - d3
	VADD	V5.H8, V4.H8, V0.H8
	VSUB	V5.H8, V4.H8, V2.H8
	VSHL	$3, V0.H8, V0.H8    // t0
	VSHL	$3, V2.H8, V2.H8    // t2
	VMOV	V26.B16, V8.B16
	VMOV	V26.B16, V9.B16
	VSMLAL	V24.H4, V6.H4, V8.S4
	VSMLAL2	V24.H8, V6.H8, V9.S4
	VSMLAL	V25.H4, V7.H4, V8.S4
	VSMLAL2	V25.H8, V7.H8, V9.S4
	VSHRN	$9, V8.S4, V1.H4
	VSHRN2	$9, V9.S4, V1.H8    // t1
	VMOV	V27.B16, V10.B16
	VMOV	V27.B16, V11.B16
	VSMLAL	V24.H4, V7.H4, V10.S4
	VSMLAL2	V24.H8, V7.H8, V11.S4
	VSMLSL	V25.H4, V6.H4, V10.S4
	VSMLSL2	V25.H8, V6.H8, V11.S4
	VSHRN	$9, V10.S4, V3.H4
	VSHRN2	$9, V11.S4, V3.H8   // t3
	TRANSPOSE4x4H(V0, V1, V2, V3, V4, V5, V6, V7)

	// pass 2
	VADD	V3.H8, V0.H8, V4.H8 // a0 = t0 + t3
	VADD	V2.H8, V1.H8, V5.H8 // a1 = t1 + t2
	VSUB	V2.H8, V1.H8, V6.H8 // a2 = t1 - t2
	VSUB	V3.H8, V0.H8, V7.H8 // a3 = t0 - t3
	VADD	V30.H8, V4.H8, V4.H8
	VADD	V5.H8, V4.H8, V0.H8
	VSUB	V5.H8, V4.H8, V2.H8
	VSSHR	$4, V0.H8, V0.H8    // out[0..3]
	VSSHR	$4, V2.H8, V2.H8    // out[8..11]
	VMOV	V28.B16, V8.B16
	VMOV	V28.B16, V9.B16
	VSMLAL	V24.H4, V6.H4, V8.S4
	VSMLAL2	V24.H8, V6.H8, V9.S4
	VSMLAL	V25.H4, V7.H4, V8.S4
	VSMLAL2	V25.H8, V7.H8, V9.S4
	VSHRN	$16, V8.S4, V1.H4
	VSHRN2	$16, V9.S4, V1.H8
	VCMEQ	$0, V7.H8, V12.H8
	VADD	V12.H8, V1.H8, V1.H8 // out[4..7]
	VMOV	V29.B16, V10.B16
	VMOV	V29.B16, V11.B16
	VSMLAL	V24.H4, V7.H4, V10.S4
	VSMLAL2	V24.H8, V7.H8, V11.S4
	VSMLSL	V25.H4, V6.H4, V10.S4
	VSMLSL2	V25.H8, V6.H8, V11.S4
	VSHRN	$16, V10.S4, V3.H4
	VSHRN2	$16, V11.S4, V3.H8  // out[12..15]

	// block A = low halves, block B = high halves
	VZIP1	V1.D2, V0.D2, V4.D2
	VZIP1	V3.D2, V2.D2, V5.D2
	VZIP2	V1.D2, V0.D2, V6.D2
	VZIP2	V3.D2, V2.D2, V7.D2
	VST1.P	[V4.H8, V5.H8, V6.H8, V7.H8], 64(R4)
	SUBS	$1, R3, R3
	BNE	floop

fdone:
	RET

// SQDMULH Vd.8H, Vn.8H, Vm.8H (signed saturating doubling multiply high)
#define SQDMULH8H(d, n, m) WORD $(0x4E60B400 | ((m)<<16) | ((n)<<5) | (d))
// SSRA Vd.8H, Vn.8H, #1 (signed shift right by 1 and accumulate)
#define SSRA8H1(d, n) WORD $(0x4F1F1400 | ((n)<<5) | (d))

// ---------------------------------------------------------------------------
// func iTransformPairsNEON(ref *byte, in *[16]int16, dst *byte, off *[8][2]uint16, n int)
//
// iTransform (dsp.go) of the 2n blocks of the n pairs of a pair table:
// dst = clip8(ref + IDCT(in)), with
//   pass 1 (lanes = columns i): a = in[i]+in[8+i], b = in[i]-in[8+i],
//           c = mul2(in[4+i]) - mul1(in[12+i]), d = mul1(in[4+i]) + mul2(in[12+i])
//           with mul1(x) = ((x*20091)>>16)+x, mul2(x) = (x*35468)>>16,
//           outputs a+d, b+c, b-c, a-d
//   transpose -> lanes = rows
//   pass 2: same butterfly with dc = c0+4, then >> 3
//   clip8(ref + v)
//
// The Go code computes in int32. When every coefficient of the pair has
// |in| <= 2047 (all the blocks the encoder produces in practice), no
// intermediate value exceeds 14.8*2047+4 < 2^15, so 16-bit lanes give the
// same results: the pair is transformed at once as in libwebp's
// ITransform_NEON, with
//   mul2(x) = (x*35468)>>16 = SQDMULH(x, 17734)
//   mul1(x) = ((x*20091)>>16) + x = x + (SQDMULH(x, 20091) >> 1)  (SSRA)
// (exact: floor(floor(2*x*20091/2^16)/2) = floor(x*20091/2^16)), and
// SQADD/SQXTUN add the prediction and clip. Otherwise each block takes the int32 path, which
// reproduces the Go arithmetic (including its int32 wrap-around) for any
// input: MUL/SSHR for mul1/mul2, SQXTN to int16 (saturating), SQADD with
// the widened reference, SQXTUN to bytes; saturating at each step gives the
// same result as Go's clip8 of the int32 sum.
//
// Registers: R0 ref, R1 in, R2 dst, R3 table, R4 pairs left, R5/R6 pair
// pointers, R7 = bps, R8 blocks left in the pair (int32 path), R9 max |in|,
// R10/R11 row pointers. 16-bit path: V25 20091, V26 17734, V27 4.
// int32 path: V28 20091, V29 35468, V30 4.
TEXT ·iTransformPairsNEON(SB), NOSPLIT, $0-40
	MOVD	ref+0(FP), R0
	MOVD	in+8(FP), R1
	MOVD	dst+16(FP), R2
	MOVD	off+24(FP), R3
	MOVD	n+32(FP), R4
	MOVD	$32, R7
	MOVD	$20091, R5
	VDUP	R5, V28.S4
	VDUP	R5, V25.H8
	MOVD	$35468, R5
	VDUP	R5, V29.S4
	MOVD	$17734, R5
	VDUP	R5, V26.H8
	MOVD	$4, R5
	VDUP	R5, V30.S4
	VDUP	R5, V27.H8
	CBZ	R4, idone

iloop:
	MOVHU	(R3), R5
	MOVHU	2(R3), R6
	ADD	$4, R3
	ADD	R0, R5, R5
	ADD	R2, R6, R6

	// max |in| over the pair (ABS(-32768) reads as 32768 unsigned)
	VLD1	(R1), [V0.H8, V1.H8, V2.H8, V3.H8]
	VABS	V0.H8, V4.H8
	VABS	V1.H8, V5.H8
	VABS	V2.H8, V6.H8
	VABS	V3.H8, V7.H8
	VUMAX	V5.H8, V4.H8, V4.H8
	VUMAX	V7.H8, V6.H8, V6.H8
	VUMAX	V6.H8, V4.H8, V4.H8
	VUMAXV	V4.H8, V4
	VMOV	V4.H[0], R9
	CMP	$2047, R9
	BHI	islow

	// 16-bit path: rows [A row k | B row k]
	ADD	$64, R1
	VZIP1	V2.D2, V0.D2, V4.D2
	VZIP2	V2.D2, V0.D2, V5.D2
	VZIP1	V3.D2, V1.D2, V6.D2
	VZIP2	V3.D2, V1.D2, V7.D2
	// pass 1: V4..V7 -> V0..V3 (V5, V7 clobbered)
	VADD	V6.H8, V4.H8, V8.H8   // a = in0 + in8
	VSUB	V6.H8, V4.H8, V9.H8   // b = in0 - in8
	SQDMULH8H(10, 5, 26)        // mul2(in4)
	SQDMULH8H(12, 5, 25)
	SSRA8H1(5, 12)              // mul1(in4)
	SQDMULH8H(11, 7, 26)        // mul2(in12)
	SQDMULH8H(13, 7, 25)
	SSRA8H1(7, 13)              // mul1(in12)
	VSUB	V7.H8, V10.H8, V12.H8 // c = mul2(in4) - mul1(in12)
	VADD	V11.H8, V5.H8, V13.H8 // d = mul1(in4) + mul2(in12)
	VADD	V13.H8, V8.H8, V0.H8  // a + d
	VADD	V12.H8, V9.H8, V1.H8  // b + c
	VSUB	V12.H8, V9.H8, V2.H8  // b - c
	VSUB	V13.H8, V8.H8, V3.H8  // a - d
	TRANSPOSE4x4H(V0, V1, V2, V3, V4, V5, V6, V7)

	// pass 2: V0..V3 -> V4..V7 (V1, V3 clobbered)
	VADD	V27.H8, V0.H8, V0.H8  // dc = c0 + 4
	VADD	V2.H8, V0.H8, V8.H8
	VSUB	V2.H8, V0.H8, V9.H8
	SQDMULH8H(10, 1, 26)
	SQDMULH8H(12, 1, 25)
	SSRA8H1(1, 12)
	SQDMULH8H(11, 3, 26)
	SQDMULH8H(13, 3, 25)
	SSRA8H1(3, 13)
	VSUB	V3.H8, V10.H8, V12.H8
	VADD	V11.H8, V1.H8, V13.H8
	VADD	V13.H8, V8.H8, V4.H8
	VADD	V12.H8, V9.H8, V5.H8
	VSUB	V12.H8, V9.H8, V6.H8
	VSUB	V13.H8, V8.H8, V7.H8
	VSSHR	$3, V4.H8, V4.H8
	VSSHR	$3, V5.H8, V5.H8
	VSSHR	$3, V6.H8, V6.H8
	VSSHR	$3, V7.H8, V7.H8
	TRANSPOSE4x4H(V4, V5, V6, V7, V0, V1, V2, V3)
	VLD1.P	(R5)(R7), [V0.B8]
	VLD1.P	(R5)(R7), [V1.B8]
	VLD1.P	(R5)(R7), [V2.B8]
	VLD1	(R5), [V3.B8]
	VUXTL	V0.B8, V0.H8
	VUXTL	V1.B8, V1.H8
	VUXTL	V2.B8, V2.H8
	VUXTL	V3.B8, V3.H8
	VSQADD	V0.H8, V4.H8, V4.H8
	VSQADD	V1.H8, V5.H8, V5.H8
	VSQADD	V2.H8, V6.H8, V6.H8
	VSQADD	V3.H8, V7.H8, V7.H8
	VSQXTUN	V4.H8, V0.B8
	VSQXTUN	V5.H8, V1.B8
	VSQXTUN	V6.H8, V2.B8
	VSQXTUN	V7.H8, V3.B8
	VST1.P	[V0.B8], (R6)(R7)
	VST1.P	[V1.B8], (R6)(R7)
	VST1.P	[V2.B8], (R6)(R7)
	VST1	[V3.B8], (R6)
	SUBS	$1, R4, R4
	BNE	iloop
	RET

islow:
	MOVD	$2, R8

iblock:
	VLD1.P	32(R1), [V0.H8, V1.H8]
	VSXTL	V0.H4, V2.S4        // in[0..3]
	VSXTL2	V0.H8, V3.S4        // in[4..7]
	VSXTL	V1.H4, V4.S4        // in[8..11]
	VSXTL2	V1.H8, V5.S4        // in[12..15]

	// pass 1 (vertical)
	VADD	V4.S4, V2.S4, V6.S4 // a
	VSUB	V4.S4, V2.S4, V7.S4 // b
	VMUL	V28.S4, V3.S4, V8.S4
	VSSHR	$16, V8.S4, V8.S4
	VADD	V3.S4, V8.S4, V8.S4 // mul1(in4)
	VMUL	V29.S4, V3.S4, V9.S4
	VSSHR	$16, V9.S4, V9.S4   // mul2(in4)
	VMUL	V28.S4, V5.S4, V10.S4
	VSSHR	$16, V10.S4, V10.S4
	VADD	V5.S4, V10.S4, V10.S4 // mul1(in12)
	VMUL	V29.S4, V5.S4, V11.S4
	VSSHR	$16, V11.S4, V11.S4 // mul2(in12)
	VSUB	V10.S4, V9.S4, V12.S4 // c
	VADD	V11.S4, V8.S4, V13.S4 // d
	VADD	V13.S4, V6.S4, V14.S4 // C0 = a + d
	VADD	V12.S4, V7.S4, V15.S4 // C1 = b + c
	VSUB	V12.S4, V7.S4, V16.S4 // C2 = b - c
	VSUB	V13.S4, V6.S4, V17.S4 // C3 = a - d
	TRANSPOSE4x4S(V14, V15, V16, V17, V2, V3, V4, V5, V6, V7, V8, V9)

	// pass 2 (horizontal), lanes = rows
	VADD	V30.S4, V2.S4, V2.S4  // dc
	VADD	V4.S4, V2.S4, V6.S4   // a
	VSUB	V4.S4, V2.S4, V7.S4   // b
	VMUL	V28.S4, V3.S4, V8.S4
	VSSHR	$16, V8.S4, V8.S4
	VADD	V3.S4, V8.S4, V8.S4   // mul1(c1)
	VMUL	V29.S4, V3.S4, V9.S4
	VSSHR	$16, V9.S4, V9.S4     // mul2(c1)
	VMUL	V28.S4, V5.S4, V10.S4
	VSSHR	$16, V10.S4, V10.S4
	VADD	V5.S4, V10.S4, V10.S4 // mul1(c3)
	VMUL	V29.S4, V5.S4, V11.S4
	VSSHR	$16, V11.S4, V11.S4   // mul2(c3)
	VSUB	V10.S4, V9.S4, V12.S4 // c
	VADD	V11.S4, V8.S4, V13.S4 // d
	VADD	V13.S4, V6.S4, V14.S4
	VADD	V12.S4, V7.S4, V15.S4
	VSUB	V12.S4, V7.S4, V16.S4
	VSUB	V13.S4, V6.S4, V17.S4
	VSSHR	$3, V14.S4, V14.S4    // column 0, rows 0-3
	VSSHR	$3, V15.S4, V15.S4    // column 1
	VSSHR	$3, V16.S4, V16.S4    // column 2
	VSSHR	$3, V17.S4, V17.S4    // column 3
	VSQXTN	V14.S4, V6.H4
	VSQXTN2	V15.S4, V6.H8         // [col0 | col1]
	VSQXTN	V16.S4, V7.H4
	VSQXTN2	V17.S4, V7.H8         // [col2 | col3]
	VZIP1	V7.H8, V6.H8, V8.H8
	VZIP2	V7.H8, V6.H8, V9.H8
	VZIP1	V9.H8, V8.H8, V2.H8   // [row0 | row1]
	VZIP2	V9.H8, V8.H8, V3.H8   // [row2 | row3]

	// add the prediction and store
	MOVD	R5, R10
	VLD1.P	(R10)(R7), V10.S[0]
	VLD1.P	(R10)(R7), V10.S[1]
	VLD1.P	(R10)(R7), V10.S[2]
	VLD1	(R10), V10.S[3]
	VUXTL	V10.B8, V11.H8
	VUXTL2	V10.B16, V12.H8
	VSQADD	V11.H8, V2.H8, V2.H8
	VSQADD	V12.H8, V3.H8, V3.H8
	VSQXTUN	V2.H8, V13.B8
	VSQXTUN2	V3.H8, V13.B16
	MOVD	R6, R11
	VST1.P	V13.S[0], (R11)(R7)
	VST1.P	V13.S[1], (R11)(R7)
	VST1.P	V13.S[2], (R11)(R7)
	VST1	V13.S[3], (R11)

	ADD	$4, R5
	ADD	$4, R6
	SUBS	$1, R8, R8
	BNE	iblock
	SUBS	$1, R4, R4
	BNE	iloop

idone:
	RET

// Weights of tTransform (kWeightY) for the horizontal coefficient h of
// each vertical index v, i.e. row h of the (symmetric) weight matrix,
// duplicated for the two blocks of a pair.
DATA ·ttWeights+0(SB)/8, $0x0009001400200026
DATA ·ttWeights+8(SB)/8, $0x0009001400200026
DATA ·ttWeights+16(SB)/8, $0x00070011001c0020
DATA ·ttWeights+24(SB)/8, $0x00070011001c0020
DATA ·ttWeights+32(SB)/8, $0x0004000a00110014
DATA ·ttWeights+40(SB)/8, $0x0004000a00110014
DATA ·ttWeights+48(SB)/8, $0x0002000400070009
DATA ·ttWeights+56(SB)/8, $0x0002000400070009
GLOBL ·ttWeights(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// func tTransformPairsNEON(a *byte, off *[8][2]uint16, n int, out *int32)
//
// tTransform(a, &kWeightY) (dsp.go) of the 2n blocks of the n pairs of a
// pair table (first offsets only); the computation of libwebp's
// TTransform_NEON / Disto4x4_NEON for one input. The Hadamard transform has
// no rounding, so the vertical pass can run first (lanes = columns): after a
// transpose the horizontal pass works lane-wise, and coefficient
// (vertical v, horizontal h) sits in lane v of register h, with weight
// kWeightY[4v+h] = kWeightY[4h+v]. The weighted sum of |coeff| (<= 4080)
// accumulates exactly in 32 bits (UMULL/UMLAL), block A in the low lanes,
// block B in the high lanes.
//
// Registers: R0 a, R1 table, R2 pairs left, R3 out, R5 row pointer,
// R7 = bps, V20-V23 weights.
TEXT ·tTransformPairsNEON(SB), NOSPLIT, $0-32
	MOVD	a+0(FP), R0
	MOVD	off+8(FP), R1
	MOVD	n+16(FP), R2
	MOVD	out+24(FP), R3
	MOVD	$32, R7
	MOVD	$·ttWeights(SB), R4
	VLD1	(R4), [V20.H8, V21.H8, V22.H8, V23.H8]
	CBZ	R2, tdone

tloop:
	MOVHU	(R1), R5
	ADD	$4, R1
	ADD	R0, R5, R5
	VLD1.P	(R5)(R7), [V0.B8]
	VLD1.P	(R5)(R7), [V1.B8]
	VLD1.P	(R5)(R7), [V2.B8]
	VLD1	(R5), [V3.B8]
	VUXTL	V0.B8, V0.H8
	VUXTL	V1.B8, V1.H8
	VUXTL	V2.B8, V2.H8
	VUXTL	V3.B8, V3.H8

	// vertical pass, lanes = columns
	VADD	V2.H8, V0.H8, V4.H8 // a0 = r0 + r2
	VADD	V3.H8, V1.H8, V5.H8 // a1 = r1 + r3
	VSUB	V3.H8, V1.H8, V6.H8 // a2 = r1 - r3
	VSUB	V2.H8, V0.H8, V7.H8 // a3 = r0 - r2
	VADD	V5.H8, V4.H8, V0.H8 // v0 = a0 + a1
	VADD	V6.H8, V7.H8, V1.H8 // v1 = a3 + a2
	VSUB	V6.H8, V7.H8, V2.H8 // v2 = a3 - a2
	VSUB	V5.H8, V4.H8, V3.H8 // v3 = a0 - a1
	TRANSPOSE4x4H(V0, V1, V2, V3, V4, V5, V6, V7)

	// horizontal pass, lane-wise across the column registers
	VADD	V2.H8, V0.H8, V4.H8
	VADD	V3.H8, V1.H8, V5.H8
	VSUB	V3.H8, V1.H8, V6.H8
	VSUB	V2.H8, V0.H8, V7.H8
	VADD	V5.H8, V4.H8, V0.H8 // h0
	VADD	V6.H8, V7.H8, V1.H8 // h1
	VSUB	V6.H8, V7.H8, V2.H8 // h2
	VSUB	V5.H8, V4.H8, V3.H8 // h3
	VABS	V0.H8, V0.H8
	VABS	V1.H8, V1.H8
	VABS	V2.H8, V2.H8
	VABS	V3.H8, V3.H8
	VUMULL	V20.H4, V0.H4, V8.S4
	VUMULL2	V20.H8, V0.H8, V9.S4
	VUMLAL	V21.H4, V1.H4, V8.S4
	VUMLAL2	V21.H8, V1.H8, V9.S4
	VUMLAL	V22.H4, V2.H4, V8.S4
	VUMLAL2	V22.H8, V2.H8, V9.S4
	VUMLAL	V23.H4, V3.H4, V8.S4
	VUMLAL2	V23.H8, V3.H8, V9.S4
	VADDP	V9.S4, V8.S4, V10.S4
	VADDP	V10.S4, V10.S4, V10.S4 // [A, B, A, B]
	VST1.P	V10.D[0], 8(R3)
	SUBS	$1, R2, R2
	BNE	tloop

tdone:
	RET

// ---------------------------------------------------------------------------
// func sse4x4PairsNEON(a, b *byte, off *[8][2]uint16, n int, out *int32)
//
// sse4x4 (dsp.go) of the 2n block pairs of a pair table: UABD, UMULL
// (squares <= 65025 fit 16 bits), UADALP into 32-bit lanes (block A in
// lanes 0-1, block B in lanes 2-3), ADDP.
//
// Registers: R0 a, R1 b, R2 table, R3 pairs left, R4 out, R5/R6 row
// pointers, R7 = bps.
TEXT ·sse4x4PairsNEON(SB), NOSPLIT, $0-40
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	off+16(FP), R2
	MOVD	n+24(FP), R3
	MOVD	out+32(FP), R4
	MOVD	$32, R7
	CBZ	R3, sdone

sloop:
	MOVHU	(R2), R5
	MOVHU	2(R2), R6
	ADD	$4, R2
	ADD	R0, R5, R5
	ADD	R1, R6, R6
	VLD1.P	(R5)(R7), [V0.B8]
	VLD1.P	(R5)(R7), [V1.B8]
	VLD1.P	(R5)(R7), [V2.B8]
	VLD1	(R5), [V3.B8]
	VLD1.P	(R6)(R7), [V4.B8]
	VLD1.P	(R6)(R7), [V5.B8]
	VLD1.P	(R6)(R7), [V6.B8]
	VLD1	(R6), [V7.B8]
	UABD8B(0, 0, 4)
	UABD8B(1, 1, 5)
	UABD8B(2, 2, 6)
	UABD8B(3, 3, 7)
	VUMULL	V0.B8, V0.B8, V0.H8
	VUMULL	V1.B8, V1.B8, V1.H8
	VUMULL	V2.B8, V2.B8, V2.H8
	VUMULL	V3.B8, V3.B8, V3.H8
	VEOR	V8.B16, V8.B16, V8.B16
	UADALP4S(8, 0)
	UADALP4S(8, 1)
	UADALP4S(8, 2)
	UADALP4S(8, 3)
	VADDP	V8.S4, V8.S4, V9.S4 // [A, B, A, B]
	VST1.P	V9.D[0], 8(R4)
	SUBS	$1, R3, R3
	BNE	sloop

sdone:
	RET

// ---------------------------------------------------------------------------
// func sse16xNNEON(a, b *byte, rows int) int
//
// Sum of squared differences of a 16-pixel wide block of the given number
// of rows (sse16x16, sse16x8 of dsp.go): UABD, UMULL/UMULL2, UADALP; the
// ARMv8.0 baseline variant of libwebp's SSE16x16_NEON.
//
// Registers: R0 a, R1 b, R2 rows left, R3 = bps, V16/V17 accumulators.
TEXT ·sse16xNNEON(SB), NOSPLIT, $0-32
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	rows+16(FP), R2
	MOVD	$32, R3
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	CBZ	R2, edone

eloop:
	VLD1.P	(R0)(R3), [V0.B16]
	VLD1.P	(R1)(R3), [V1.B16]
	UABD16B(2, 0, 1)
	VUMULL	V2.B8, V2.B8, V3.H8
	VUMULL2	V2.B16, V2.B16, V4.H8
	UADALP4S(16, 3)
	UADALP4S(17, 4)
	SUBS	$1, R2, R2
	BNE	eloop

edone:
	VADD	V17.S4, V16.S4, V16.S4
	VADDV	V16.S4, V16
	VMOV	V16.S[0], R0
	MOVD	R0, ret+24(FP)
	RET
