//go:build arm64 && !purego

#include "go_asm.h"
#include "textflag.h"

// NEON macroblock reconstruction: intra prediction plus inverse transforms.
//
// Algorithm adapted from libwebp (BSD-3-Clause): dsp/dec.c and
// dsp/dec_neon.c (TransformOne, TransformDC, TransformAC3, the 4x4, 8x8 and
// 16x16 intra predictors). The results are bit-exact with the Go reference
// in dsp.go for every input, which differs from libwebp's NEON code in one
// respect: libwebp computes the inverse DCT in saturating 16-bit lanes,
// which is only exact for coefficients in the range a valid encoder emits.
// Here the transform runs in 32-bit lanes (SQDMULH by 20091<<15 and
// 35468<<15 is exactly (x*20091)>>16 and (x*35468)>>16), so any int16 input
// gives the same pixels as the scalar C and Go code.
//
// The work buffer b (*[yuvSize]byte, stride const_bps) holds the luma block
// at const_yOff and the chroma blocks at const_uOff/const_vOff, each with
// its top row (and top-left/top-right samples) and left column of context.
// The kernels write only the samples the Go reference writes, and read only
// the samples it reads plus this over-read, which stays inside b:
//   - reconI4NEON loads columns -8..7 of the row above each 4x4 sub-block
//     (16 bytes); the lowest byte read is b[const_yOff - const_bps - 8] =
//     b[0], the highest b[const_yOff + 11*const_bps + 19].
// coeffs is *[384]int16 and modes *[16]uint8; only their elements are read,
// and nothing but b is written. TestReconGuardPages checks all of this with
// the buffers placed against inaccessible pages.

// Instructions the Go arm64 assembler lacks, as raw encodings (checked
// against clang: see the comment at each use).

// SQDMULH Vd.4S, Vn.4S, Vm.S[i]
#define SQDMULH_S4E(d, n, m, i) WORD $(0x4F80C000 | (((i)&1)<<21) | (((i)>>1)<<11) | ((m)<<16) | ((n)<<5) | (d))
// SQRSHRN Vd.4H, Vn.4S, #3
#define SQRSHRN_3(d, n) WORD $(0x0F1D9C00 | ((n)<<5) | (d))
// SQRSHRN2 Vd.8H, Vn.4S, #3
#define SQRSHRN2_3(d, n) WORD $(0x4F1D9C00 | ((n)<<5) | (d))

// reconConst: TBL indices that turn the column-major int16 output of the
// inverse DCT into rows (rows 2-3, rows 0-1), the multipliers of MUL1/MUL2
// as SQDMULH constants, the AC3 coefficient mask and the TBL indices that
// gather the 4x4 edge vector e (see reconI4NEON).
DATA reconConst<>+0(SB)/8, $0x1D1C15140D0C0504  // idx23: rows 2,3
DATA reconConst<>+8(SB)/8, $0x1F1E17160F0E0706
DATA reconConst<>+16(SB)/8, $0x1918111009080100 // idx01: rows 0,1
DATA reconConst<>+24(SB)/8, $0x1B1A13120B0A0302
DATA reconConst<>+32(SB)/8, $0x45460000273D8000 // S[0]=20091<<15, S[1]=35468<<15
DATA reconConst<>+40(SB)/8, $0
DATA reconConst<>+48(SB)/8, $0x00000000FFFFFFFF // AC3: keep in[0], in[1], in[4]
DATA reconConst<>+56(SB)/8, $0x000000000000FFFF
DATA reconConst<>+64(SB)/8, $0x19181703070B0F0F // e: 15 15 11 7 3 23 24 25
DATA reconConst<>+72(SB)/8, $0x1F1F1F1E1D1C1B1A //    26 27 28 29 30 31 31 31
GLOBL reconConst<>(SB), RODATA|NOPTR, $80

// reconI4Tab: for each 4x4 mode, the TBL indices (row-major 4x4) into the
// table {m3, m2, tm, dc} built by reconI4NEON (0-15: m3, 16-31: m2, 32-47:
// the TM prediction, 48: the DC value). VE and HE (modes 2 and 3) only use
// indices 1-9, so the same rows also gather A..D and I..L from e for TM.
DATA reconI4Tab<>+0(SB)/8, $0x3030303030303030   // DC
DATA reconI4Tab<>+8(SB)/8, $0x3030303030303030
DATA reconI4Tab<>+16(SB)/8, $0x2726252423222120  // TM
DATA reconI4Tab<>+24(SB)/8, $0x2F2E2D2C2B2A2928
DATA reconI4Tab<>+32(SB)/8, $0x0908070609080706  // VE
DATA reconI4Tab<>+40(SB)/8, $0x0908070609080706
DATA reconI4Tab<>+48(SB)/8, $0x0303030304040404  // HE
DATA reconI4Tab<>+56(SB)/8, $0x0101010102020202
DATA reconI4Tab<>+64(SB)/8, $0x0706050408070605  // RD
DATA reconI4Tab<>+72(SB)/8, $0x0504030206050403
DATA reconI4Tab<>+80(SB)/8, $0x0807060518171615  // VR
DATA reconI4Tab<>+88(SB)/8, $0x0706050317161504
DATA reconI4Tab<>+96(SB)/8, $0x0B0A09080A090807  // LD
DATA reconI4Tab<>+104(SB)/8, $0x0D0C0B0A0C0B0A09
DATA reconI4Tab<>+112(SB)/8, $0x0A09080719181716 // VL
DATA reconI4Tab<>+120(SB)/8, $0x0C0A09080B191817
DATA reconI4Tab<>+128(SB)/8, $0x0514041307060514 // HD
DATA reconI4Tab<>+136(SB)/8, $0x0312021104130312
DATA reconI4Tab<>+144(SB)/8, $0x0111021202120313 // HU
DATA reconI4Tab<>+152(SB)/8, $0x1010101010100111
GLOBL reconI4Tab<>(SB), RODATA|NOPTR, $160

// reconI4Gather: per-sub-block TBL indices into {V0 = left neighbour,
// V1 = row above} (see reconI4NEON): e[k+1], e[k-1], the left column
// I..L across the rows, and the 8 DC samples (I..L, A..D, then zeros).
DATA reconI4Gather<>+0(SB)/8, $0x1A19181703070B0F  // e[k+1]
DATA reconI4Gather<>+8(SB)/8, $0x1F1F1F1F1E1D1C1B
DATA reconI4Gather<>+16(SB)/8, $0x181703070B0F0F0F // e[k-1]
DATA reconI4Gather<>+24(SB)/8, $0x1F1F1E1D1C1B1A19
DATA reconI4Gather<>+32(SB)/8, $0x0707070703030303 // I x4, J x4, K x4, L x4
DATA reconI4Gather<>+40(SB)/8, $0x0F0F0F0F0B0B0B0B
DATA reconI4Gather<>+48(SB)/8, $0x1B1A19180F0B0703 // I J K L A B C D
DATA reconI4Gather<>+56(SB)/8, $-1
DATA reconI4Gather<>+64(SB)/8, $0x0B0A09080B0A0908 // A B C D on each row (in V1 alone)
DATA reconI4Gather<>+72(SB)/8, $0x0B0A09080B0A0908
GLOBL reconI4Gather<>(SB), RODATA|NOPTR, $80

// reconCodeMask: per transform code (0-3), the mask applied to in[0..15]
// before the full inverse DCT, which then equals the code's transform: none
// (all zero, giving a zero residual), TransformDC (in[0] only),
// TransformAC3 (in[0], in[1], in[4]) and TransformOne.
DATA reconCodeMask<>+0(SB)/8, $0
DATA reconCodeMask<>+8(SB)/8, $0
DATA reconCodeMask<>+16(SB)/8, $0
DATA reconCodeMask<>+24(SB)/8, $0
DATA reconCodeMask<>+32(SB)/8, $0x000000000000FFFF
DATA reconCodeMask<>+40(SB)/8, $0
DATA reconCodeMask<>+48(SB)/8, $0
DATA reconCodeMask<>+56(SB)/8, $0
DATA reconCodeMask<>+64(SB)/8, $0x00000000FFFFFFFF
DATA reconCodeMask<>+72(SB)/8, $0x000000000000FFFF
DATA reconCodeMask<>+80(SB)/8, $0
DATA reconCodeMask<>+88(SB)/8, $0
DATA reconCodeMask<>+96(SB)/8, $-1
DATA reconCodeMask<>+104(SB)/8, $-1
DATA reconCodeMask<>+112(SB)/8, $-1
DATA reconCodeMask<>+120(SB)/8, $-1
GLOBL reconCodeMask<>(SB), RODATA|NOPTR, $128

// --- inverse DCT --------------------------------------------------------------
//
// IDCT computes the residual of one 4x4 block (TransformOne):
//   in:  V16 = in[0..7], V17 = in[8..15] (int16)
//   out: V26 = rows 0,1 and V27 = rows 2,3 of ((x + 4) >> 3) saturated to
//        int16 (saturation cannot change the final clip to [0, 255])
//   constants: V29 = idx23, V30 = idx01, V31 = MUL1/MUL2 multipliers
//   clobbers V16-V25.
// Both passes run on four int32 lanes; for the vertical pass the vectors are
// the input rows (lane = column), then the 4x4 is transposed with TRN1/TRN2
// so that the horizontal pass works on columns of tmp (lane = output row).
// The result is column-major; SQRSHRN narrows it with the rounding and a
// two-register TBL reorders it into rows.

// IDCT_PASS: one 1-D pass on V18..V21 (x0, x4, x8, x12), result in V18..V21:
//   a = x0 + x8, b = x0 - x8
//   c = MUL2(x4) - MUL1(x12), d = MUL1(x4) + MUL2(x12)
//   out = a + d, b + c, b - c, a - d
// MUL1(x) = x + SQDMULH(x, 20091<<15), MUL2(x) = SQDMULH(x, 35468<<15).
#define IDCT_PASS \
	VADD	V20.S4, V18.S4, V22.S4 \
	VSUB	V20.S4, V18.S4, V23.S4 \
	SQDMULH_S4E(24, 19, 31, 1) /* sqdmulh v24.4s, v19.4s, v31.s[1] */ \
	SQDMULH_S4E(25, 19, 31, 0) /* sqdmulh v25.4s, v19.4s, v31.s[0] */ \
	SQDMULH_S4E(16, 21, 31, 1) /* sqdmulh v16.4s, v21.4s, v31.s[1] */ \
	SQDMULH_S4E(17, 21, 31, 0) /* sqdmulh v17.4s, v21.4s, v31.s[0] */ \
	VADD	V19.S4, V25.S4, V25.S4 \
	VADD	V21.S4, V17.S4, V17.S4 \
	VSUB	V17.S4, V24.S4, V24.S4 \
	VADD	V16.S4, V25.S4, V25.S4 \
	VADD	V25.S4, V22.S4, V18.S4 \
	VADD	V24.S4, V23.S4, V19.S4 \
	VSUB	V24.S4, V23.S4, V20.S4 \
	VSUB	V25.S4, V22.S4, V21.S4

// IDCT_TRANSPOSE: 4x4 int32 transpose of V18..V21 in place.
#define IDCT_TRANSPOSE \
	VTRN1	V19.S4, V18.S4, V22.S4 \
	VTRN2	V19.S4, V18.S4, V23.S4 \
	VTRN1	V21.S4, V20.S4, V24.S4 \
	VTRN2	V21.S4, V20.S4, V25.S4 \
	VTRN1	V24.D2, V22.D2, V18.D2 \
	VTRN1	V25.D2, V23.D2, V19.D2 \
	VTRN2	V24.D2, V22.D2, V20.D2 \
	VTRN2	V25.D2, V23.D2, V21.D2

#define IDCT \
	VSXTL	V16.H4, V18.S4 \
	VSXTL2	V16.H8, V19.S4 \
	VSXTL	V17.H4, V20.S4 \
	VSXTL2	V17.H8, V21.S4 \
	IDCT_PASS \
	IDCT_TRANSPOSE \
	IDCT_PASS \
	SQRSHRN_3(16, 18) /* sqrshrn v16.4h, v18.4s, #3 */ \
	SQRSHRN2_3(16, 19) /* sqrshrn2 v16.8h, v19.4s, #3 */ \
	SQRSHRN_3(17, 20) /* sqrshrn v17.4h, v20.4s, #3 */ \
	SQRSHRN2_3(17, 21) /* sqrshrn2 v17.8h, v21.4s, #3 */ \
	VTBL	V30.B16, [V16.B16, V17.B16], V26.B16 \
	VTBL	V29.B16, [V16.B16, V17.B16], V27.B16

// DCRES sets V26/V27 to the TransformDC residual (in[0] + 4) >> 3 of the
// block at R1. Clobbers R7.
#define DCRES \
	MOVH	(R1), R7 \
	ADD	$4, R7, R7 \
	ASR	$3, R7, R7 \
	VDUP	R7, V26.H8 \
	VMOV	V26.B16, V27.B16

// ADDBLOCK(base) adds the residual rows V26 (rows 0,1) and V27 (rows 2,3)
// to the 4x4 pixels at base (stride R10 = const_bps) with saturation:
// clip8(pixel + residual). Clobbers R8, V12-V14.
#define ADDBLOCK(base) \
	MOVD	base, R8 \
	VLD1.P	(R8)(R10), V12.S[0] \
	VLD1.P	(R8)(R10), V12.S[1] \
	VLD1.P	(R8)(R10), V12.S[2] \
	VLD1	(R8), V12.S[3] \
	VUXTL	V12.B8, V13.H8 \
	VUXTL2	V12.B16, V14.H8 \
	VSQADD	V26.H8, V13.H8, V13.H8 \
	VSQADD	V27.H8, V14.H8, V14.H8 \
	VSQXTUN	V13.H8, V12.B8 \
	VSQXTUN2	V14.H8, V12.B16 \
	MOVD	base, R8 \
	VST1.P	V12.S[0], (R8)(R10) \
	VST1.P	V12.S[1], (R8)(R10) \
	VST1.P	V12.S[2], (R8)(R10) \
	VST1	V12.S[3], (R8)

// LOADCONST loads V29-V31 (IDCT constants) and V2 (AC3 mask). Clobbers R8.
#define LOADCONST \
	MOVD	$reconConst<>(SB), R8 \
	VLD1.P	48(R8), [V29.B16, V30.B16, V31.B16] \
	VLD1	(R8), [V2.B16]

// --- intra 4x4 -----------------------------------------------------------------

// func reconI4NEON(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8)
//
// Mirrors reconI4Ref: for each sub-block n in raster order, predLuma4 with
// modes[n] then doTransform with the code bits>>30 (bits <<= 2).
//
// Each sub-block depends on the reconstruction of its left and upper
// neighbours, and that chain, not the arithmetic, bounds the speed. So the
// kernel has no data-dependent branch per sub-block and keeps the left
// neighbour in a register: from one sub-block to the next the chain is
// TBL, at most three vector ops, TBL4, UQADD, UQSUB.
//
// Prediction. V0 is the reconstructed left neighbour (I..L, its right
// column, are bytes 3, 7, 11, 15; at the start of a row V0 is a pseudo
// block holding the left context) and V1 the 16 bytes at columns -8..7 of
// the row above (X, the top-left sample, in byte 7, the top row A..H with
// the top-right samples in bytes 8..15). TBLs over {V0, V1} (indices in
// reconConst and reconI4Gather) gather
//   e = [L L K J I X A B C D E F G H H .]
// and e shifted by one either way, and then
//   m3[k] = avg3(e[k-1], e[k], e[k+1])  (V4, UHADD + URHADD as in libwebp)
//   m2[k] = avg2(e[k], e[k+1])          (V5, URHADD; m2[0] = avg2(L, L) = L)
//   tm    = clip8(left + top - X)       (V6, UQSUB(UQADD(left, max(top-X, 0)),
//                                        max(X-top, 0)), exact with
//                                        saturating byte ops)
//   dc    = (A+B+C+D+I+J+K+L + 4) >> 3  (V7 byte 0, UADDLV + SRSHR)
// are all computed, and the prediction of any of the 10 modes is one TBL
// over {m3, m2, tm, dc} with the mode's index row from reconI4Tab.
//
// Residual. Instead of branching on the transform code, the coefficients
// are masked with reconCodeMask[code] and the full inverse DCT is run: it
// equals TransformDC on a DC-only block, TransformAC3 on in[0], in[1],
// in[4], and gives 0 on a zero block. The int16 residual is split into its
// positive and negative parts clamped to [0, 255] (SQXTUN of r and of -r),
// and clip8(pred + r) = UQSUB(UQADD(pred, pos), neg). Once all remaining
// codes are zero (a branch that flips at most once per macroblock) the
// prediction is stored as is.
//
// Registers: R0 b, R1 coefficients of the current block, R2 bits, R3 modes,
// R4 reconCodeMask, R5 current sub-block, R6 mode, R7/R8 scratch, R9 = 9,
// R10 = const_bps, R11/R13 loop counters, R12 reconI4Tab, R14-R17 left
// context, R19 reconI4Gather. V0 left neighbour, V1 row above, V2 e,
// V3/V13 e shifted, V8-V10/V14 TM terms, V4-V7 TBL table, V11 e indices,
// V12 mode indices, V15 prediction, V16-V19 gather indices then V16-V27
// inverse DCT, V24/V25 residual parts, V28 top-row indices, V29-V31 IDCT
// constants.
TEXT ·reconI4NEON(SB), NOSPLIT, $0-32
	MOVD	b+0(FP), R0
	MOVD	coeffs+8(FP), R1
	MOVWU	bits+16(FP), R2
	MOVD	modes+24(FP), R3
	MOVD	$reconConst<>(SB), R8
	VLD1.P	48(R8), [V29.B16, V30.B16, V31.B16]
	ADD	$16, R8, R8
	VLD1	(R8), [V11.B16]
	MOVD	$reconI4Gather<>(SB), R19
	ADD	$64, R19, R8
	VLD1	(R8), [V28.B16]
	MOVD	$reconI4Tab<>(SB), R12
	MOVD	$reconCodeMask<>(SB), R4
	MOVD	$9, R9
	MOVD	$const_bps, R10
	ADD	$const_yOff, R0, R5
	MOVD	$4, R11

i4row:
	// Pseudo left neighbour: the left context I, J, K, L in bytes 3, 7,
	// 11, 15 of V0.
	MOVBU	-1(R5), R14
	MOVBU	(const_bps-1)(R5), R15
	MOVBU	(2*const_bps-1)(R5), R16
	MOVBU	(3*const_bps-1)(R5), R17
	LSL	$24, R14, R14
	ORR	R15<<56, R14, R14
	LSL	$24, R16, R16
	ORR	R17<<56, R16, R16
	VMOV	R14, V0.D[0]
	VMOV	R16, V0.D[1]
	MOVD	$4, R13

i4blk:
	MOVBU.P	1(R3), R6
	CMP	R9, R6
	CSEL	HI, R9, R6, R6 // modes > 9 predict like the default case (HU)
	ADD	R6<<4, R12, R7
	VLD1	(R7), [V12.B16]
	SUB	$(const_bps+8), R5, R8
	VLD1	(R8), [V1.B16]
	VLD1	(R19), [V16.B16, V17.B16, V18.B16, V19.B16] // reconI4Gather

	// Everything that depends on the left neighbour is gathered from
	// {V0, V1} directly, so that the chain from one sub-block to the next
	// is TBL, at most three vector ops, TBL4, UQADD, UQSUB.
	VTBL	V11.B16, [V0.B16, V1.B16], V2.B16  // e
	VTBL	V16.B16, [V0.B16, V1.B16], V3.B16  // e[k+1]
	VTBL	V17.B16, [V0.B16, V1.B16], V13.B16 // e[k-1]
	VTBL	V18.B16, [V0.B16, V1.B16], V9.B16  // I, J, K, L across the rows
	VTBL	V19.B16, [V0.B16, V1.B16], V7.B16  // DC samples
	VTBL	V28.B16, [V1.B16], V8.B16          // A B C D on each row
	VDUP	V1.B[7], V10.B16                   // X

	VUHADD	V3.B16, V13.B16, V13.B16
	VURHADD	V13.B16, V2.B16, V4.B16      // m3
	VURHADD	V3.B16, V2.B16, V5.B16       // m2

	VUQSUB	V10.B16, V8.B16, V14.B16     // max(top - X, 0)
	VUQSUB	V8.B16, V10.B16, V8.B16      // max(X - top, 0)
	VUQADD	V14.B16, V9.B16, V9.B16
	VUQSUB	V8.B16, V9.B16, V6.B16       // tm = clip8(left + (top - X))

	VUADDLV	V7.B8, V7
	VSRSHR	$3, V7.H4, V7.H4             // dc in byte 0

	VTBL	V12.B16, [V4.B16, V5.B16, V6.B16, V7.B16], V15.B16

	// Once the remaining codes are all zero (a branch that changes at most
	// once per macroblock) only the prediction is stored.
	CBZ	R2, i4nores
	LSRW	$30, R2, R7
	LSLW	$2, R2, R2
	ADD	R7<<5, R4, R7
	VLD1	(R7), [V26.B16, V27.B16]
	VLD1	(R1), [V16.H8, V17.H8]
	VAND	V26.B16, V16.B16, V16.B16
	VAND	V27.B16, V17.B16, V17.B16
	IDCT
	VSQXTUN	V26.H8, V24.B8
	VSQXTUN2	V27.H8, V24.B16 // max(r, 0)
	VSQNEG	V26.H8, V26.H8
	VSQNEG	V27.H8, V27.H8
	VSQXTUN	V26.H8, V25.B8
	VSQXTUN2	V27.H8, V25.B16 // max(-r, 0)
	VUQADD	V24.B16, V15.B16, V0.B16
	VUQSUB	V25.B16, V0.B16, V0.B16
	B	i4store

i4nores:
	VMOV	V15.B16, V0.B16

i4store:
	ADD	$32, R1, R1
	MOVD	R5, R8
	VST1.P	V0.S[0], (R8)(R10)
	VST1.P	V0.S[1], (R8)(R10)
	VST1.P	V0.S[2], (R8)(R10)
	VST1	V0.S[3], (R8)
	ADD	$4, R5, R5
	SUB	$1, R13, R13
	CBNZ	R13, i4blk
	ADD	$(4*const_bps-16), R5, R5
	SUB	$1, R11, R11
	CBNZ	R11, i4row
	RET

// --- residual loop shared by the 16x16 and chroma kernels ----------------------

// SUMLEFT16 sets R7 to the sum of the 16 left samples of the block at R5.
// Clobbers R15.
#define SUMLEFT16 \
	MOVBU	-1(R5), R7 \
	MOVBU	(1*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(2*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(3*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(4*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(5*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(6*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(7*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(8*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(9*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(10*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(11*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(12*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(13*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(14*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(15*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7

// SUMLEFT8 sets R7 to the sum of the 8 left samples of the block at R5.
// Clobbers R15.
#define SUMLEFT8 \
	MOVBU	-1(R5), R7 \
	MOVBU	(1*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(2*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(3*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(4*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(5*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(6*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7 \
	MOVBU	(7*const_bps-1)(R5), R15 \
	ADD	R15, R7, R7

// ST16x4 / ST8x4 store V0 to four rows at R9 (advanced by 4 rows).
#define ST16x4 \
	VST1.P	[V0.B16], (R9)(R10) \
	VST1.P	[V0.B16], (R9)(R10) \
	VST1.P	[V0.B16], (R9)(R10) \
	VST1.P	[V0.B16], (R9)(R10)

#define ST8x4 \
	VST1.P	[V0.B8], (R9)(R10) \
	VST1.P	[V0.B8], (R9)(R10) \
	VST1.P	[V0.B8], (R9)(R10) \
	VST1.P	[V0.B8], (R9)(R10)

// HE16ROW / HE8ROW replicate the left sample at R8 over the row at R9.
#define HE16ROW \
	VLD1R.P	(R8)(R10), [V0.B16] \
	VST1.P	[V0.B16], (R9)(R10)

#define HE8ROW \
	VLD1R.P	(R8)(R10), [V0.B8] \
	VST1.P	[V0.B8], (R9)(R10)

// TM16ROW / TM8ROW: row = clip8(top - X + left), top - X in V2 (and V3).
#define TM16ROW \
	MOVBU.P	const_bps(R8), R6 \
	VDUP	R6, V4.H8 \
	VADD	V4.H8, V2.H8, V5.H8 \
	VADD	V4.H8, V3.H8, V6.H8 \
	VSQXTUN	V5.H8, V7.B8 \
	VSQXTUN2	V6.H8, V7.B16 \
	VST1.P	[V7.B16], (R9)(R10)

#define TM8ROW \
	MOVBU.P	const_bps(R8), R6 \
	VDUP	R6, V4.H8 \
	VADD	V4.H8, V2.H8, V5.H8 \
	VSQXTUN	V5.H8, V7.B8 \
	VST1.P	[V7.B8], (R9)(R10)

// --- intra 16x16 -----------------------------------------------------------------

// func reconI16NEON(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int)
//
// Mirrors reconI16Ref: predLuma16(mode) written to b, then for each of the
// 16 sub-blocks with a non-zero code the DC-only, AC3 (masked full
// transform: identical by construction) or full residual is added in place.
//
// Registers: R0 b, R1 coefficients, R2 bits, R3 mode, R5 block, R6/R7/R9
// scratch, R8/R9 row pointers, R10 = const_bps, R11/R13 loop counters.
TEXT ·reconI16NEON(SB), NOSPLIT, $0-32
	MOVD	b+0(FP), R0
	MOVD	coeffs+8(FP), R1
	MOVWU	bits+16(FP), R2
	MOVD	mode+24(FP), R3
	MOVD	$const_bps, R10
	ADD	$const_yOff, R0, R5
	SUB	$const_bps, R5, R8 // top row
	MOVD	R5, R9
	CMP	$1, R3
	BEQ	i16tm
	CMP	$2, R3
	BEQ	i16ve
	CMP	$3, R3
	BEQ	i16he
	CMP	$5, R3
	BHI	i16dc80 // bDCPredNoTopLeft and anything out of range
	CMP	$4, R3
	BEQ	i16dcnotop
	VLD1	(R8), [V0.B16]
	VUADDLV	V0.B16, V0
	VMOV	V0.H[0], R6 // sum of the top row
	CMP	$5, R3
	BEQ	i16dcnoleft
	SUMLEFT16
	ADD	R7, R6, R6
	ADD	$16, R6, R6
	LSR	$5, R6, R6
	B	i16fill

i16dcnotop:
	SUMLEFT16
	ADD	$8, R7, R6
	LSR	$4, R6, R6
	B	i16fill

i16dcnoleft:
	ADD	$8, R6, R6
	LSR	$4, R6, R6
	B	i16fill

i16dc80:
	MOVD	$0x80, R6

i16fill:
	VDUP	R6, V0.B16
	ST16x4
	ST16x4
	ST16x4
	ST16x4
	B	i16res

i16ve:
	VLD1	(R8), [V0.B16]
	ST16x4
	ST16x4
	ST16x4
	ST16x4
	B	i16res

i16he:
	SUB	$1, R5, R8
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	HE16ROW
	B	i16res

i16tm:
	VLD1	(R8), [V0.B16]
	MOVBU	-1(R8), R6
	VDUP	R6, V1.H8
	VUXTL	V0.B8, V2.H8
	VUXTL2	V0.B16, V3.H8
	VSUB	V1.H8, V2.H8, V2.H8
	VSUB	V1.H8, V3.H8, V3.H8
	SUB	$1, R5, R8
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW
	TM16ROW

i16res:
	CBZ	R2, i16done
	LOADCONST
	MOVD	$4, R11

i16row:
	MOVD	$4, R13

i16blk:
	CBZ	R2, i16done
	LSRW	$30, R2, R7
	LSLW	$2, R2, R2
	CBZ	R7, i16next
	CMP	$1, R7
	BEQ	i16dcres
	VLD1	(R1), [V16.H8, V17.H8]
	CMP	$2, R7
	BNE	i16full
	VAND	V2.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16

i16full:
	IDCT
	B	i16add

i16dcres:
	DCRES

i16add:
	ADDBLOCK(R5)

i16next:
	ADD	$32, R1, R1
	ADD	$4, R5, R5
	SUB	$1, R13, R13
	CBNZ	R13, i16blk
	ADD	$(4*const_bps-16), R5, R5
	SUB	$1, R11, R11
	CBNZ	R11, i16row

i16done:
	RET

// --- chroma 8x8 ------------------------------------------------------------------

// func reconUVNEON(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int)
//
// Mirrors reconUVRef for both planes: predChroma8(mode) then the residuals
// of the four 4x4 blocks. codes holds one 2-bit code per block (U in bits
// 0-7, V in bits 8-15, block 0 in the top bits of each byte) as computed
// by uvCodes: 0 none, 1 DC only, 3 full transform.
//
// Registers: R0 b, R1 coefficients, R2 V codes, R3 mode, R4 codes of the
// current plane, R5 plane, R6/R7/R9 scratch, R8/R9 row pointers,
// R10 = const_bps, R11 plane counter, R12 block, R13 block counter,
// R14 coefficients of the current plane.
TEXT ·reconUVNEON(SB), NOSPLIT, $0-32
	MOVD	b+0(FP), R0
	MOVD	coeffs+8(FP), R14
	MOVWU	codes+16(FP), R2
	MOVD	mode+24(FP), R3
	MOVD	$const_bps, R10
	ADD	$(16*16*2), R14, R14 // block 16
	LSLW	$24, R2, R4          // U codes
	LSRW	$8, R2, R2
	LSLW	$24, R2, R2          // V codes
	LOADCONST
	ADD	$const_uOff, R0, R5
	MOVD	$2, R11

uvplane:
	SUB	$const_bps, R5, R8 // top row
	MOVD	R5, R9
	CMP	$1, R3
	BEQ	uvtm
	CMP	$2, R3
	BEQ	uvve
	CMP	$3, R3
	BEQ	uvhe
	CMP	$5, R3
	BHI	uvdc80
	CMP	$4, R3
	BEQ	uvdcnotop
	VLD1	(R8), [V0.B8]
	VUADDLV	V0.B8, V0
	VMOV	V0.H[0], R6
	CMP	$5, R3
	BEQ	uvdcnoleft
	SUMLEFT8
	ADD	R7, R6, R6
	ADD	$8, R6, R6
	LSR	$4, R6, R6
	B	uvfill

uvdcnotop:
	SUMLEFT8
	ADD	$4, R7, R6
	LSR	$3, R6, R6
	B	uvfill

uvdcnoleft:
	ADD	$4, R6, R6
	LSR	$3, R6, R6
	B	uvfill

uvdc80:
	MOVD	$0x80, R6

uvfill:
	VDUP	R6, V0.B8
	ST8x4
	ST8x4
	B	uvres

uvve:
	VLD1	(R8), [V0.B8]
	ST8x4
	ST8x4
	B	uvres

uvhe:
	SUB	$1, R5, R8
	HE8ROW
	HE8ROW
	HE8ROW
	HE8ROW
	HE8ROW
	HE8ROW
	HE8ROW
	HE8ROW
	B	uvres

uvtm:
	VLD1	(R8), [V0.B8]
	MOVBU	-1(R8), R6
	VDUP	R6, V1.H8
	VUXTL	V0.B8, V2.H8
	VSUB	V1.H8, V2.H8, V2.H8
	SUB	$1, R5, R8
	TM8ROW
	TM8ROW
	TM8ROW
	TM8ROW
	TM8ROW
	TM8ROW
	TM8ROW
	TM8ROW

uvres:
	CBZ	R4, uvnextplane
	MOVD	R14, R1
	MOVD	R5, R12
	MOVD	$4, R13

uvblk:
	LSRW	$30, R4, R7
	LSLW	$2, R4, R4
	CBZ	R7, uvnext
	CMP	$1, R7
	BEQ	uvdcres
	VLD1	(R1), [V16.H8, V17.H8]
	IDCT
	B	uvadd

uvdcres:
	DCRES

uvadd:
	ADDBLOCK(R12)

uvnext:
	ADD	$32, R1, R1
	ADD	$4, R12, R12
	SUB	$1, R13, R13
	CMP	$2, R13
	BNE	uvsamerow
	ADD	$(4*const_bps-8), R12, R12

uvsamerow:
	CBNZ	R13, uvblk

uvnextplane:
	ADD	$(4*16*2), R14, R14
	MOVD	R2, R4
	ADD	$(const_vOff-const_uOff), R5, R5
	SUB	$1, R11, R11
	CBNZ	R11, uvplane
	RET

// --- macroblock edges and row cache ---------------------------------------------

// ROT4(row, base, from) copies the 4 bytes at columns from..from+3 of row
// (relative to base) to columns -4..-1.
#define ROT4(row, base, from) \
	MOVWU	(base+(row)*const_bps+(from))(R0), R4 \
	MOVW	R4, (base+(row)*const_bps-4)(R0)

// func loadEdgesNEON(b *[yuvSize]byte, top *topSamples, tr, flags uint32)
//
// Mirrors loadEdgesRef (same flags) with straight-line loads and stores at
// constant offsets. Registers: R0 b, R1 top, R2 tr, R3 flags, R4-R7 data.
TEXT ·loadEdgesNEON(SB), NOSPLIT, $0-24
	MOVD	b+0(FP), R0
	MOVD	top+8(FP), R1
	MOVWU	tr+16(FP), R2
	MOVWU	flags+20(FP), R3
	TBZ	$0, R3, noRotate // edgeRotate
	ROT4(-1, const_yOff, 12)
	ROT4(0, const_yOff, 12)
	ROT4(1, const_yOff, 12)
	ROT4(2, const_yOff, 12)
	ROT4(3, const_yOff, 12)
	ROT4(4, const_yOff, 12)
	ROT4(5, const_yOff, 12)
	ROT4(6, const_yOff, 12)
	ROT4(7, const_yOff, 12)
	ROT4(8, const_yOff, 12)
	ROT4(9, const_yOff, 12)
	ROT4(10, const_yOff, 12)
	ROT4(11, const_yOff, 12)
	ROT4(12, const_yOff, 12)
	ROT4(13, const_yOff, 12)
	ROT4(14, const_yOff, 12)
	ROT4(15, const_yOff, 12)
	ROT4(-1, const_uOff, 4)
	ROT4(0, const_uOff, 4)
	ROT4(1, const_uOff, 4)
	ROT4(2, const_uOff, 4)
	ROT4(3, const_uOff, 4)
	ROT4(4, const_uOff, 4)
	ROT4(5, const_uOff, 4)
	ROT4(6, const_uOff, 4)
	ROT4(7, const_uOff, 4)
	ROT4(-1, const_vOff, 4)
	ROT4(0, const_vOff, 4)
	ROT4(1, const_vOff, 4)
	ROT4(2, const_vOff, 4)
	ROT4(3, const_vOff, 4)
	ROT4(4, const_vOff, 4)
	ROT4(5, const_vOff, 4)
	ROT4(6, const_vOff, 4)
	ROT4(7, const_vOff, 4)

noRotate:
	TBZ	$1, R3, noTop // edgeTop
	MOVD	(topSamples_y)(R1), R4
	MOVD	(topSamples_y+8)(R1), R5
	MOVD	(topSamples_u)(R1), R6
	MOVD	(topSamples_v)(R1), R7
	MOVD	R4, (const_yOff-const_bps)(R0)
	MOVD	R5, (const_yOff-const_bps+8)(R0)
	MOVD	R6, (const_uOff-const_bps)(R0)
	MOVD	R7, (const_vOff-const_bps)(R0)

noTop:
	TBZ	$3, R3, done // edgeI4
	TBZ	$2, R3, noTopRight // edgeTopRight
	MOVW	R2, (const_yOff-const_bps+16)(R0)

noTopRight:
	MOVWU	(const_yOff-const_bps+16)(R0), R4
	MOVW	R4, (const_yOff+3*const_bps+16)(R0)
	MOVW	R4, (const_yOff+7*const_bps+16)(R0)
	MOVW	R4, (const_yOff+11*const_bps+16)(R0)

done:
	RET

// func storeMBRowsNEON(b *[yuvSize]byte, top *topSamples, cy, cu, cv *byte, ys, uvs int)
//
// Mirrors storeMBRowsRef; the Go wrapper has checked that cy has
// 15*ys+16 bytes and cu, cv 7*uvs+8. Registers: R0 b, R1 top (may be nil),
// R2 cy, R3 cu, R4 cv, R5 ys, R6 uvs, R7 source rows, R8 scratch.
TEXT ·storeMBRowsNEON(SB), NOSPLIT, $0-56
	MOVD	b+0(FP), R0
	MOVD	top+8(FP), R1
	MOVD	cy+16(FP), R2
	MOVD	cu+24(FP), R3
	MOVD	cv+32(FP), R4
	MOVD	ys+40(FP), R5
	MOVD	uvs+48(FP), R6
	CBZ	R1, noStash
	MOVD	(const_yOff+15*const_bps)(R0), R7
	MOVD	(const_yOff+15*const_bps+8)(R0), R8
	MOVD	R7, (topSamples_y)(R1)
	MOVD	R8, (topSamples_y+8)(R1)
	MOVD	(const_uOff+7*const_bps)(R0), R7
	MOVD	R7, (topSamples_u)(R1)
	MOVD	(const_vOff+7*const_bps)(R0), R7
	MOVD	R7, (topSamples_v)(R1)

noStash:
	MOVD	$const_bps, R8
	ADD	$const_yOff, R0, R7
	VLD1.P	(R7)(R8), [V0.B16]
	VLD1.P	(R7)(R8), [V1.B16]
	VLD1.P	(R7)(R8), [V2.B16]
	VLD1.P	(R7)(R8), [V3.B16]
	VLD1.P	(R7)(R8), [V4.B16]
	VLD1.P	(R7)(R8), [V5.B16]
	VLD1.P	(R7)(R8), [V6.B16]
	VLD1.P	(R7)(R8), [V7.B16]
	VLD1.P	(R7)(R8), [V16.B16]
	VLD1.P	(R7)(R8), [V17.B16]
	VLD1.P	(R7)(R8), [V18.B16]
	VLD1.P	(R7)(R8), [V19.B16]
	VLD1.P	(R7)(R8), [V20.B16]
	VLD1.P	(R7)(R8), [V21.B16]
	VLD1.P	(R7)(R8), [V22.B16]
	VLD1	(R7), [V23.B16]
	VST1.P	[V0.B16], (R2)(R5)
	VST1.P	[V1.B16], (R2)(R5)
	VST1.P	[V2.B16], (R2)(R5)
	VST1.P	[V3.B16], (R2)(R5)
	VST1.P	[V4.B16], (R2)(R5)
	VST1.P	[V5.B16], (R2)(R5)
	VST1.P	[V6.B16], (R2)(R5)
	VST1.P	[V7.B16], (R2)(R5)
	VST1.P	[V16.B16], (R2)(R5)
	VST1.P	[V17.B16], (R2)(R5)
	VST1.P	[V18.B16], (R2)(R5)
	VST1.P	[V19.B16], (R2)(R5)
	VST1.P	[V20.B16], (R2)(R5)
	VST1.P	[V21.B16], (R2)(R5)
	VST1.P	[V22.B16], (R2)(R5)
	VST1	[V23.B16], (R2)

	// U and V rows 0-6 are the low halves of the 32 bytes at uOff + j*bps
	// (vOff = uOff + 16): one load per row. Row 7 is loaded as two halves,
	// as 32 bytes would run past the end of b.
	ADD	$const_uOff, R0, R7
	VLD1.P	const_bps(R7), [V0.B16, V1.B16]
	VLD1.P	const_bps(R7), [V2.B16, V3.B16]
	VLD1.P	const_bps(R7), [V4.B16, V5.B16]
	VLD1.P	const_bps(R7), [V6.B16, V7.B16]
	VLD1.P	const_bps(R7), [V16.B16, V17.B16]
	VLD1.P	const_bps(R7), [V18.B16, V19.B16]
	VLD1.P	const_bps(R7), [V20.B16, V21.B16]
	MOVD	$(const_vOff-const_uOff), R8
	VLD1.P	(R7)(R8), [V22.B8]
	VLD1	(R7), [V23.B8]
	VST1.P	[V0.B8], (R3)(R6)
	VST1.P	[V2.B8], (R3)(R6)
	VST1.P	[V4.B8], (R3)(R6)
	VST1.P	[V6.B8], (R3)(R6)
	VST1.P	[V16.B8], (R3)(R6)
	VST1.P	[V18.B8], (R3)(R6)
	VST1.P	[V20.B8], (R3)(R6)
	VST1	[V22.B8], (R3)
	VST1.P	[V1.B8], (R4)(R6)
	VST1.P	[V3.B8], (R4)(R6)
	VST1.P	[V5.B8], (R4)(R6)
	VST1.P	[V7.B8], (R4)(R6)
	VST1.P	[V17.B8], (R4)(R6)
	VST1.P	[V19.B8], (R4)(R6)
	VST1.P	[V21.B8], (R4)(R6)
	VST1	[V23.B8], (R4)
	RET
