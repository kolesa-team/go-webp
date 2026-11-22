// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//go:build amd64 && !purego

#include "go_asm.h"
#include "textflag.h"

// x86 macroblock reconstruction for the VP8 decoder: intra prediction and
// inverse transforms, one call per macroblock (reconI4*, reconI16*,
// reconUV*, AVX2 and SSE4.1 versions), the inverse WHT (transformWHTSSE41)
// and the per-macroblock edge setup and row cache copy (loadEdgesSSE2,
// storeMBRowsSSE2).
//
// Algorithm adapted from libwebp (BSD-3-Clause): dsp/dec.c,
// dsp/dec_sse2.c and dsp/dec_sse41.c (Transform_SSE2, TransformAC3, the
// 4x4, 8x8 and 16x16 intra predictors) and dec/frame_dec.c
// (ReconstructRow). Every kernel is bit-exact with its Go reference in
// dsp.go for every input:
//   - the inverse DCT is libwebp's Transform_SSE2 in 16-bit lanes (MUL1 and
//     MUL2 as PMULHW by 20091 and by 35468-65536 plus the input), with the
//     final sums saturated. That equals the reference whenever every
//     coefficient it uses lies in [-4095, 4095] (all sign patterns at the
//     bound and 20M random blocks were checked against a model; the
//     differential tests check the kernels). The kernels test that bound
//     with PABSW/PMAXUW on the masked coefficients and return false, before
//     anything of that block row is written, when it does not hold; the Go
//     wrapper then reruns the macroblock with the reference. Saturating the
//     final sums cannot change the result: the residual then exceeds 255 in
//     magnitude either way, so the pixel clips to the same bound.
//   - DC-only rows use (in[0] + 4) >> 3 with a saturating add, exact for any
//     input.
//   - Blocks are transformed with their coefficients masked by the 2-bit
//     transform code (reconMask): the full transform of in[0] alone is
//     TransformDC and of in[0], in[1], in[4] is TransformAC3, so one
//     transform serves every code, as in the arm64 kernels.
//
// The work buffer b (*[yuvSize]byte, stride const_bps) holds the luma block
// at const_yOff and the chroma blocks at const_uOff/const_vOff, each with
// its top row (and top-left/top-right samples) and left column of context.
// The kernels read and write in b only bytes the Go reference reads and
// writes. A kernel that returns false may already have written pixels of
// the block; the reference then recomputes every one of them from context
// (the row above, the left column, the top-right samples) that no kernel
// writes. coeffs
// (*[384]int16) is read in whole 32-byte blocks, modes (*[16]uint8) and top
// (*topSamples) element by element, and the row cache only within the bounds
// the Go wrapper checked. TestReconGuardPages and TestReconGlueGuardPages
// check this with every buffer placed against inaccessible pages.
//
// Register conventions: R14/R15 and BP are never touched; the AVX2 kernels
// end with VZEROUPPER on every path. Functions with a frame keep their
// scratch at 0(SP).

// --- constants -------------------------------------------------------------------

// MUL1/MUL2 multipliers for PMULHW: k1 = 20091, k2 = 35468 - 65536.
DATA reconK1<>+0(SB)/8, $0x4E7B4E7B4E7B4E7B
DATA reconK1<>+8(SB)/8, $0x4E7B4E7B4E7B4E7B
DATA reconK1<>+16(SB)/8, $0x4E7B4E7B4E7B4E7B
DATA reconK1<>+24(SB)/8, $0x4E7B4E7B4E7B4E7B
GLOBL reconK1<>(SB), RODATA|NOPTR, $32

DATA reconK2<>+0(SB)/8, $0x8A8C8A8C8A8C8A8C
DATA reconK2<>+8(SB)/8, $0x8A8C8A8C8A8C8A8C
DATA reconK2<>+16(SB)/8, $0x8A8C8A8C8A8C8A8C
DATA reconK2<>+24(SB)/8, $0x8A8C8A8C8A8C8A8C
GLOBL reconK2<>(SB), RODATA|NOPTR, $32

// The rounder of the second pass, 4 in each 16-bit lane.
DATA reconFour<>+0(SB)/8, $0x0004000400040004
DATA reconFour<>+8(SB)/8, $0x0004000400040004
DATA reconFour<>+16(SB)/8, $0x0004000400040004
DATA reconFour<>+24(SB)/8, $0x0004000400040004
GLOBL reconFour<>(SB), RODATA|NOPTR, $32

// reconRange: bits of max |coefficient| that must be clear for the 16-bit
// transform to be exact (|c| <= 4095).
DATA reconRange<>+0(SB)/8, $0xF000F000F000F000
DATA reconRange<>+8(SB)/8, $0xF000F000F000F000
DATA reconRange<>+16(SB)/8, $0xF000F000F000F000
DATA reconRange<>+24(SB)/8, $0xF000F000F000F000
GLOBL reconRange<>(SB), RODATA|NOPTR, $32

// reconMask: per transform code (0-3, 32 bytes each), the mask applied to
// in[0..15]: nothing, in[0], in[0], in[1], in[4], everything.
DATA reconMask<>+0(SB)/8, $0
DATA reconMask<>+8(SB)/8, $0
DATA reconMask<>+16(SB)/8, $0
DATA reconMask<>+24(SB)/8, $0
DATA reconMask<>+32(SB)/8, $0x000000000000FFFF
DATA reconMask<>+40(SB)/8, $0
DATA reconMask<>+48(SB)/8, $0
DATA reconMask<>+56(SB)/8, $0
DATA reconMask<>+64(SB)/8, $0x00000000FFFFFFFF
DATA reconMask<>+72(SB)/8, $0x000000000000FFFF
DATA reconMask<>+80(SB)/8, $0
DATA reconMask<>+88(SB)/8, $0
DATA reconMask<>+96(SB)/8, $-1
DATA reconMask<>+104(SB)/8, $-1
DATA reconMask<>+112(SB)/8, $-1
DATA reconMask<>+120(SB)/8, $-1
GLOBL reconMask<>(SB), RODATA|NOPTR, $128

// reconDCSpread: PSHUFB indices that copy 16-bit lane 0 to lanes 0-3 and
// lane 4 to lanes 4-7 (each 128-bit half).
DATA reconDCSpread<>+0(SB)/8, $0x0100010001000100
DATA reconDCSpread<>+8(SB)/8, $0x0908090809080908
DATA reconDCSpread<>+16(SB)/8, $0x0100010001000100
DATA reconDCSpread<>+24(SB)/8, $0x0908090809080908
GLOBL reconDCSpread<>(SB), RODATA|NOPTR, $32

// reconFours: PSHUFB indices selecting byte 4 everywhere.
DATA reconFours<>+0(SB)/8, $0x0404040404040404
DATA reconFours<>+8(SB)/8, $0x0404040404040404
DATA reconFours<>+16(SB)/8, $0x0404040404040404
DATA reconFours<>+24(SB)/8, $0x0404040404040404
GLOBL reconFours<>(SB), RODATA|NOPTR, $32

// reconPN: PSHUFB indices that turn the packed columns of two blocks,
// dwords [A c0, B c0, A c1, B c1] (bytes = rows 0..3), into [A c0, A c1,
// B c0, B c1] with each column's bytes reversed (rows 3..0): the 4x4 layout
// of the intra 4x4 kernels (see reconI4AVX2).
DATA reconPN<>+0(SB)/8, $0x08090A0B00010203
DATA reconPN<>+8(SB)/8, $0x0C0D0E0F04050607
DATA reconPN<>+16(SB)/8, $0x08090A0B00010203
DATA reconPN<>+24(SB)/8, $0x0C0D0E0F04050607
GLOBL reconPN<>(SB), RODATA|NOPTR, $32

// Intra 4x4 helpers (16 bytes each, see reconI4AVX2).
DATA reconOnes<>+0(SB)/8, $0x0101010101010101
DATA reconOnes<>+8(SB)/8, $0x0101010101010101
GLOBL reconOnes<>(SB), RODATA|NOPTR, $16

DATA reconC4096<>+0(SB)/8, $0x1000100010001000 // PMULHRSW by 4096: (x + 4) >> 3
DATA reconC4096<>+8(SB)/8, $0x1000100010001000
GLOBL reconC4096<>(SB), RODATA|NOPTR, $16

DATA reconIdxA<>+0(SB)/8, $0x8080800C08040080 // row 3 (bytes 0, 4, 8, 12) to bytes 1-4
DATA reconIdxA<>+8(SB)/8, $0x8080808080808080
GLOBL reconIdxA<>(SB), RODATA|NOPTR, $16

DATA reconIdxE<>+0(SB)/8, $0x0804008080808080 // row 3 to bytes 5-8
DATA reconIdxE<>+8(SB)/8, $0x808080808080800C
GLOBL reconIdxE<>(SB), RODATA|NOPTR, $16

DATA reconIdxX<>+0(SB)/8, $0x808080808080800C // pixel (3, 3) (byte 12) to byte 0
DATA reconIdxX<>+8(SB)/8, $0x8080808080808080
GLOBL reconIdxX<>(SB), RODATA|NOPTR, $16

DATA reconIdxPrev<>+0(SB)/8, $0x0605040302010000 // v[k-1], v[-1] = v[0]
DATA reconIdxPrev<>+8(SB)/8, $0x0E0D0C0B0A090807
GLOBL reconIdxPrev<>(SB), RODATA|NOPTR, $16

DATA reconIdxNext<>+0(SB)/8, $0x0807060504030201 // v[k+1], v[13..] = v[12]
DATA reconIdxNext<>+8(SB)/8, $0x0C0C0C0C0C0B0A09
GLOBL reconIdxNext<>(SB), RODATA|NOPTR, $16

DATA reconIdxDC<>+0(SB)/8, $0x0807060503020100 // L K J I A B C D
DATA reconIdxDC<>+8(SB)/8, $0x8080808080808080
GLOBL reconIdxDC<>(SB), RODATA|NOPTR, $16

DATA reconIdxTop<>+0(SB)/8, $0x0202020201010101 // top[x] at pixel (x, y)
DATA reconIdxTop<>+8(SB)/8, $0x0404040403030303
GLOBL reconIdxTop<>(SB), RODATA|NOPTR, $16

DATA reconIdxRows<>+0(SB)/8, $0x0E0A06020F0B0703 // 4x4 layout to row-major
DATA reconIdxRows<>+8(SB)/8, $0x0C0804000D090501
GLOBL reconIdxRows<>(SB), RODATA|NOPTR, $16

// reconI4Tab: for each 4x4 mode (64 bytes), the PSHUFB indices into m3,
// m2 and dc and the AND mask of tm whose OR is the prediction in the 4x4
// layout (see reconI4AVX2). Generated from the predictors of dsp.go and
// checked against them on random windows.
DATA reconI4Tab<>+0(SB)/8, $0x8080808080808080 // DC
DATA reconI4Tab<>+8(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+16(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+24(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+32(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+40(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+48(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+56(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+64(SB)/8, $0x8080808080808080 // TM
DATA reconI4Tab<>+72(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+80(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+88(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+96(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+104(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+112(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA reconI4Tab<>+120(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA reconI4Tab<>+128(SB)/8, $0x0606060605050505 // VE
DATA reconI4Tab<>+136(SB)/8, $0x0808080807070707
DATA reconI4Tab<>+144(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+152(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+160(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+168(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+176(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+184(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+192(SB)/8, $0x0302010003020100 // HE
DATA reconI4Tab<>+200(SB)/8, $0x0302010003020100
DATA reconI4Tab<>+208(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+216(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+224(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+232(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+240(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+248(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+256(SB)/8, $0x0504030204030201 // RD
DATA reconI4Tab<>+264(SB)/8, $0x0706050406050403
DATA reconI4Tab<>+272(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+280(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+288(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+296(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+304(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+312(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+320(SB)/8, $0x8005800480040302 // VR
DATA reconI4Tab<>+328(SB)/8, $0x8007800680068005
DATA reconI4Tab<>+336(SB)/8, $0x0680058005808080
DATA reconI4Tab<>+344(SB)/8, $0x0880078007800680
DATA reconI4Tab<>+352(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+360(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+368(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+376(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+384(SB)/8, $0x0708090A06070809 // LD
DATA reconI4Tab<>+392(SB)/8, $0x090A0B0C08090A0B
DATA reconI4Tab<>+400(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+408(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+416(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+424(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+432(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+440(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+448(SB)/8, $0x8007800880068007 // VL
DATA reconI4Tab<>+456(SB)/8, $0x80090A0B80088009
DATA reconI4Tab<>+464(SB)/8, $0x0780088006800780
DATA reconI4Tab<>+472(SB)/8, $0x0980808008800980
DATA reconI4Tab<>+480(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+488(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+496(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+504(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+512(SB)/8, $0x0403020180808080 // HD
DATA reconI4Tab<>+520(SB)/8, $0x0604030205808080
DATA reconI4Tab<>+528(SB)/8, $0x8080808004030201
DATA reconI4Tab<>+536(SB)/8, $0x8080808080040302
DATA reconI4Tab<>+544(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+552(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+560(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+568(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+576(SB)/8, $0x0201008080808080 // HU
DATA reconI4Tab<>+584(SB)/8, $0x0100808080808080
DATA reconI4Tab<>+592(SB)/8, $0x8080800003020100
DATA reconI4Tab<>+600(SB)/8, $0x8080000002010000
DATA reconI4Tab<>+608(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+616(SB)/8, $0x8080808080808080
DATA reconI4Tab<>+624(SB)/8, $0x0000000000000000
DATA reconI4Tab<>+632(SB)/8, $0x0000000000000000
GLOBL reconI4Tab<>(SB), RODATA|NOPTR, $640

// --- shared AVX2 pieces --------------------------------------------------------------

// CODEMASKS sets AX, CX, R11, R13 to the reconMask entries of the four
// transform codes in BL (block 0 in bits 6-7); R12 = reconMask.
#define CODEMASKS \
	MOVL	BX, AX \
	SHRL	$1, AX \
	ANDL	$0x60, AX \
	ADDQ	R12, AX \
	MOVL	BX, CX \
	SHLL	$1, CX \
	ANDL	$0x60, CX \
	ADDQ	R12, CX \
	MOVL	BX, R11 \
	SHLL	$3, R11 \
	ANDL	$0x60, R11 \
	ADDQ	R12, R11 \
	MOVL	BX, R13 \
	SHLL	$5, R13 \
	ANDL	$0x60, R13 \
	ADDQ	R12, R13

// LOAD4_AVX2 loads the four blocks at SI+o0..o3 masked by (m0)..(m3) into
// Y0-Y3 and sets Y13 to max |coefficient| (16-bit lanes). Clobbers Y4-Y7.
#define LOAD4_AVX2(o0, o1, o2, o3, m0, m1, m2, m3) \
	VMOVDQU	o0(SI), Y0 \
	VPAND	(m0), Y0, Y0 \
	VMOVDQU	o1(SI), Y1 \
	VPAND	(m1), Y1, Y1 \
	VMOVDQU	o2(SI), Y2 \
	VPAND	(m2), Y2, Y2 \
	VMOVDQU	o3(SI), Y3 \
	VPAND	(m3), Y3, Y3 \
	VPABSW	Y0, Y4 \
	VPABSW	Y1, Y5 \
	VPABSW	Y2, Y6 \
	VPABSW	Y3, Y7 \
	VPMAXUW	Y5, Y4, Y4 \
	VPMAXUW	Y7, Y6, Y6 \
	VPMAXUW	Y6, Y4, Y13

// DC4_AVX2 sets 16-bit lanes 0, 4, 8, 12 of Y0 to the DC-only residual
// sat(in[0] + 4) >> 3 of the blocks at SI+o0..o3 (masked by (m0)..(m3)).
// Clobbers Y1-Y3.
#define DC4_AVX2(o0, o1, o2, o3, m0, m1, m2, m3) \
	VMOVDQU	o0(SI), X0 \
	VPAND	(m0), X0, X0 \
	VMOVDQU	o1(SI), X1 \
	VPAND	(m1), X1, X1 \
	VMOVDQU	o2(SI), X2 \
	VPAND	(m2), X2, X2 \
	VMOVDQU	o3(SI), X3 \
	VPAND	(m3), X3, X3 \
	VPUNPCKLQDQ	X1, X0, X0 \
	VPUNPCKLQDQ	X3, X2, X2 \
	VINSERTI128	$1, X2, Y0, Y0 \
	VPADDSW	reconFour<>(SB), Y0, Y0 \
	VPSRAW	$3, Y0, Y0

// IDCT_PASS_AVX2: one pass on in0..in3 = Y0..Y3 (lane = column), result in
// Y0..Y3:
//   a = in0 + in2, b = in0 - in2
//   c = MUL2(in1) - MUL1(in3) = mulhi(in1, k2) - mulhi(in3, k1) + in1 - in3
//   d = MUL1(in1) + MUL2(in3) = mulhi(in1, k1) + mulhi(in3, k2) + in1 + in3
//   out = a + d, b + c, b - c, a - d (with ADD/SUB, saturating in pass 2)
// Clobbers Y4-Y9.
#define IDCT_PASS_AVX2(ADD, SUB) \
	VPADDW	Y2, Y0, Y4 \
	VPSUBW	Y2, Y0, Y5 \
	VPMULHW	reconK2<>(SB), Y1, Y6 \
	VPMULHW	reconK1<>(SB), Y3, Y7 \
	VPSUBW	Y3, Y1, Y8 \
	VPSUBW	Y7, Y6, Y6 \
	VPADDW	Y8, Y6, Y6 \
	VPMULHW	reconK1<>(SB), Y1, Y7 \
	VPMULHW	reconK2<>(SB), Y3, Y8 \
	VPADDW	Y3, Y1, Y9 \
	VPADDW	Y8, Y7, Y7 \
	VPADDW	Y9, Y7, Y7 \
	ADD	Y7, Y4, Y0 \
	ADD	Y6, Y5, Y1 \
	SUB	Y6, Y5, Y2 \
	SUB	Y7, Y4, Y3

// TRANSPOSE_AVX2: VP8Transpose_2_4x4_16b on Y0..Y3 in each 128-bit half.
// Clobbers Y4-Y11.
#define TRANSPOSE_AVX2 \
	VPUNPCKLWD	Y1, Y0, Y4 \
	VPUNPCKLWD	Y3, Y2, Y5 \
	VPUNPCKHWD	Y1, Y0, Y6 \
	VPUNPCKHWD	Y3, Y2, Y7 \
	VPUNPCKLDQ	Y5, Y4, Y8 \
	VPUNPCKLDQ	Y7, Y6, Y9 \
	VPUNPCKHDQ	Y5, Y4, Y10 \
	VPUNPCKHDQ	Y7, Y6, Y11 \
	VPUNPCKLQDQ	Y9, Y8, Y0 \
	VPUNPCKHQDQ	Y9, Y8, Y1 \
	VPUNPCKLQDQ	Y11, Y10, Y2 \
	VPUNPCKHQDQ	Y11, Y10, Y3

// IDCT4_AVX2 runs Transform_SSE2 on the four blocks in Y0-Y3 (as loaded:
// rows 0,1 | rows 2,3 of in[] each), two blocks per 128-bit half. The
// output is left column-major in Y0-Y3: Yk = column k of blocks [0 1 | 2 3],
// rows 0-3 in each group of four 16-bit lanes, the residual
// sat(x) >> 3. Clobbers Y4-Y11.
#define IDCT4_AVX2 \
	VPERM2I128	$0x20, Y2, Y0, Y4 /* [0: rows 0,1 | 2: rows 0,1] */ \
	VPERM2I128	$0x20, Y3, Y1, Y5 \
	VPERM2I128	$0x31, Y2, Y0, Y6 /* [0: rows 2,3 | 2: rows 2,3] */ \
	VPERM2I128	$0x31, Y3, Y1, Y7 \
	VPUNPCKLQDQ	Y5, Y4, Y0 /* in0 = row 0 of [0 1 | 2 3] */ \
	VPUNPCKHQDQ	Y5, Y4, Y1 \
	VPUNPCKLQDQ	Y7, Y6, Y2 \
	VPUNPCKHQDQ	Y7, Y6, Y3 \
	IDCT_PASS_AVX2(VPADDW, VPSUBW) \
	TRANSPOSE_AVX2 \
	VPADDW	reconFour<>(SB), Y0, Y0 \
	IDCT_PASS_AVX2(VPADDSW, VPSUBSW) \
	VPSRAW	$3, Y0, Y0 \
	VPSRAW	$3, Y1, Y1 \
	VPSRAW	$3, Y2, Y2 \
	VPSRAW	$3, Y3, Y3

// --- intra 4x4 (AVX2) -------------------------------------------------------------

// I4TOP_AVX2(c) sets X1 to X A B C D (bytes 0-4) of sub-block c >= 1 from
// the saved blocks above-left and above. Clobbers X2.
#define I4TOP_AVX2(c) \
	VMOVDQU	(16*(c))(R8), X1 \
	VPSHUFB	reconIdxA<>(SB), X1, X1 \
	VMOVDQU	(16*(c)-16)(R8), X2 \
	VPSHUFB	reconIdxX<>(SB), X2, X2 \
	VPOR	X2, X1, X1

// I4BLK_AVX2(c) predicts and reconstructs sub-block c of the block row
// from X0 (left) and X1 (top9), leaving the reconstruction in X0 and in the
// frame (16c(R9)) and storing it to b.
#define I4BLK_AVX2(c) \
	MOVBLZX	(c)(DX), AX \
	MOVL	$9, CX \
	CMPL	AX, CX \
	CMOVLHI	CX, AX \
	SHLL	$6, AX \
	VPALIGNR	$12, X0, X1, X2 \
	VPSHUFB	reconIdxPrev<>(SB), X2, X3 \
	VPSHUFB	reconIdxNext<>(SB), X2, X4 \
	VPAVGB	X2, X3, X5 \
	VPAVGB	X4, X3, X6 \
	VPXOR	X4, X3, X7 \
	VPAND	X14, X7, X7 \
	VPSUBUSB	X7, X6, X6 \
	VPAVGB	X2, X6, X6 \
	VPSHUFB	reconIdxDC<>(SB), X2, X8 \
	VPSADBW	X15, X8, X8 \
	VPMULHRSW	reconC4096<>(SB), X8, X8 \
	VPSHUFB	reconIdxTop<>(SB), X1, X9 \
	VPBROADCASTB	X1, X10 \
	VPSUBUSB	X10, X9, X13 \
	VPSUBUSB	X9, X10, X10 \
	VPSHUFD	$0xff, X0, X11 \
	VPADDUSB	X13, X11, X11 \
	VPSUBUSB	X10, X11, X11 \
	VPSHUFB	0(R12)(AX*1), X6, X12 \
	VPSHUFB	16(R12)(AX*1), X5, X13 \
	VPOR	X13, X12, X12 \
	VPSHUFB	32(R12)(AX*1), X8, X13 \
	VPOR	X13, X12, X12 \
	VPAND	48(R12)(AX*1), X11, X13 \
	VPOR	X13, X12, X12 \
	VPADDUSB	(32*(c))(SP), X12, X12 \
	VPSUBUSB	(32*(c)+16)(SP), X12, X0 \
	VMOVDQU	X0, (16*(c))(R9) \
	VPSHUFB	reconIdxRows<>(SB), X0, X12 \
	VMOVD	X12, (4*(c))(DI) \
	VPEXTRD	$1, X12, (4*(c)+const_bps)(DI) \
	VPEXTRD	$2, X12, (4*(c)+2*const_bps)(DI) \
	VPEXTRD	$3, X12, (4*(c)+3*const_bps)(DI)

// func reconI4AVX2(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) bool
//
// Mirrors reconI4Ref: for each sub-block n in raster order, predLuma4 with
// modes[n] (values above 9 select the default case, HU), then doTransform
// with the code bits>>30 (bits <<= 2).
//
// The 16 sub-blocks depend on each other through their predictions, so the
// kernel has no data-dependent branch per sub-block and keeps the chain from
// one sub-block to its right neighbour short (PALIGNR, PSHUFB, five ops of
// avg3, PSHUFB, two ORs, PADDUSB, PSUBUSB); a sub-block of a lower row
// depends on the row above through registers saved in the frame, never
// through a load of bytes written by several stores (which could not be
// forwarded).
//
// Layout. A 4x4 block lives in an xmm register with pixel (x, y) in byte
// 4x + 3 - y: column-major, each column bottom-up. So the right column of a
// reconstructed sub-block, its left neighbour's L K J I, is bytes 12-15.
//
// Per block row: the residuals of its four sub-blocks are computed first
// (one call of the 16-bit transform for all four, or the DC-only path, or
// nothing) and stored in the frame as their positive and negative parts
// clamped to [0, 255] (pos at 32c(SP), neg at 32c+16(SP), in the 4x4
// layout), so that clip8(pred + r) = PSUBUSB(PADDUSB(pred, pos), neg).
// Then each sub-block c is predicted from
//   top9 = [X A B C D E F G H ...], built from the saved reconstructions of
//          the row above (16c(R8): bytes 0, 4, 8, 12 are its bottom row,
//          byte 12 its bottom-right pixel), the left context column for X
//          when c = 0 and the top-right context for E..H when c = 3; for
//          the first row a pseudo row of saved blocks made from row -1;
//   left = the previous sub-block (X0), or for c = 0 the left context
//          column inserted at bytes 15 (I), 14, 13, 12 (L),
// as libwebp's VE4/LD4/VR4/VL4/RD4/HE4/HD4/HU4 do, with all ten modes at
// once:
//   v  = PALIGNR(top9, left, 12) = [L K J I X A B C D E F G H ...]
//   vp = v[k-1] (v[-1] = L), vn = v[k+1] (v[13..] = H)
//   m3 = avg3(vp, v, vn) (PAVGB, minus the lost bit, PAVGB, as AVG3 in
//        dec_sse2.c), m2 = avg2(vp, v) (so m2[0] = L)
//   dc = (A+B+C+D+I+J+K+L + 4) >> 3 (PSADBW, PMULHRSW by 4096)
//   tm = clip8(left[y] + top[x] - X) = PSUBUSB(PADDUSB(left, max(top-X,
//        0)), max(X-top, 0)), exact with saturating byte ops
// and the prediction of the mode is PSHUFB(m3) | PSHUFB(m2) | PSHUFB(dc) |
// (tm & mask) with the mode's row of reconI4Tab.
//
// Registers: DI luma row of the block row, SI its coefficients, BX bits
// (rotated), DX modes, R8/R9 saved reconstructions of the previous/current
// block row (128(SP), 192(SP), swapped per row), R10 row counter, R12 table
// base, AX/CX/R11/R13 scratch (mode offset, code masks). X0 left neighbour,
// X1 top9, X2 v, X3 vp, X4 vn, X5 m2, X6/X7 m3, X8 dc, X9/X10/X13 TM terms,
// X11 tm, X12 prediction, X14 = 0x01 bytes, X15 = 0; Y0-Y13 in the residual
// step.
TEXT ·reconI4AVX2(SB), NOSPLIT, $256-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	bits+16(FP), BX
	MOVQ	modes+24(FP), DX
	ADDQ	$const_yOff, DI
	LEAQ	128(SP), R8
	LEAQ	192(SP), R9
	VPXOR	X15, X15, X15
	VMOVDQU	reconOnes<>(SB), X14

	// Pseudo row above: row -1 as four saved blocks (bottom row in bytes
	// 0, 4, 8, 12).
	VPMOVZXBD	(-const_bps)(DI), X0
	VMOVDQU	X0, 0(R8)
	VPMOVZXBD	(-const_bps+4)(DI), X0
	VMOVDQU	X0, 16(R8)
	VPMOVZXBD	(-const_bps+8)(DI), X0
	VMOVDQU	X0, 32(R8)
	VPMOVZXBD	(-const_bps+12)(DI), X0
	VMOVDQU	X0, 48(R8)
	MOVL	$4, R10

i4avx2Row:
	ROLL	$8, BX
	TESTB	$0xff, BL
	JNZ	i4avx2Res
	VPXOR	X0, X0, X0
	VMOVDQU	X0, 0(SP)
	VMOVDQU	X0, 16(SP)
	VMOVDQU	X0, 32(SP)
	VMOVDQU	X0, 48(SP)
	VMOVDQU	X0, 64(SP)
	VMOVDQU	X0, 80(SP)
	VMOVDQU	X0, 96(SP)
	VMOVDQU	X0, 112(SP)
	JMP	i4avx2Pred

i4avx2Res:
	LEAQ	reconMask<>(SB), R12
	CODEMASKS
	TESTB	$0xaa, BL
	JNZ	i4avx2Full
	DC4_AVX2(0, 32, 64, 96, AX, CX, R11, R13)
	VPACKUSWB	Y0, Y0, Y1 // pos: byte 0 = block 0|2, byte 4 = block 1|3
	VPSUBW	Y0, Y15, Y0
	VPACKUSWB	Y0, Y0, Y0 // neg
	VPSHUFB	Y15, Y1, Y2
	VPSHUFB	reconFours<>(SB), Y1, Y1
	VPSHUFB	Y15, Y0, Y3
	VPSHUFB	reconFours<>(SB), Y0, Y0
	VMOVDQU	X2, 0(SP)
	VEXTRACTI128	$1, Y2, 64(SP)
	VMOVDQU	X1, 32(SP)
	VEXTRACTI128	$1, Y1, 96(SP)
	VMOVDQU	X3, 16(SP)
	VEXTRACTI128	$1, Y3, 80(SP)
	VMOVDQU	X0, 48(SP)
	VEXTRACTI128	$1, Y0, 112(SP)
	JMP	i4avx2Pred

i4avx2Full:
	LOAD4_AVX2(0, 32, 64, 96, AX, CX, R11, R13)
	VPTEST	reconRange<>(SB), Y13
	JNZ	i4avx2Fallback
	IDCT4_AVX2
	VPACKUSWB	Y1, Y0, Y4 // [0 c0, 1 c0, 0 c1, 1 c1 | 2, 3]
	VPACKUSWB	Y3, Y2, Y5 // [0 c2, 1 c2, 0 c3, 1 c3 | 2, 3]
	VPSHUFB	reconPN<>(SB), Y4, Y4
	VPSHUFB	reconPN<>(SB), Y5, Y5
	VPUNPCKLQDQ	Y5, Y4, Y6 // pos [0 | 2]
	VPUNPCKHQDQ	Y5, Y4, Y7 // pos [1 | 3]
	VPSUBW	Y0, Y15, Y0
	VPSUBW	Y1, Y15, Y1
	VPSUBW	Y2, Y15, Y2
	VPSUBW	Y3, Y15, Y3
	VPACKUSWB	Y1, Y0, Y4
	VPACKUSWB	Y3, Y2, Y5
	VPSHUFB	reconPN<>(SB), Y4, Y4
	VPSHUFB	reconPN<>(SB), Y5, Y5
	VPUNPCKLQDQ	Y5, Y4, Y8 // neg [0 | 2]
	VPUNPCKHQDQ	Y5, Y4, Y9 // neg [1 | 3]
	VMOVDQU	X6, 0(SP)
	VEXTRACTI128	$1, Y6, 64(SP)
	VMOVDQU	X7, 32(SP)
	VEXTRACTI128	$1, Y7, 96(SP)
	VMOVDQU	X8, 16(SP)
	VEXTRACTI128	$1, Y8, 80(SP)
	VMOVDQU	X9, 48(SP)
	VEXTRACTI128	$1, Y9, 112(SP)

i4avx2Pred:
	LEAQ	reconI4Tab<>(SB), R12

	// c = 0: X from the left context column, left from the context.
	VMOVDQU	0(R8), X1
	VPSHUFB	reconIdxA<>(SB), X1, X1
	VMOVDQU	16(R8), X3
	VPSHUFB	reconIdxE<>(SB), X3, X3
	VPOR	X3, X1, X1
	VPINSRB	$0, (-const_bps-1)(DI), X1, X1
	VPXOR	X0, X0, X0
	VPINSRB	$15, -1(DI), X0, X0
	VPINSRB	$14, (const_bps-1)(DI), X0, X0
	VPINSRB	$13, (2*const_bps-1)(DI), X0, X0
	VPINSRB	$12, (3*const_bps-1)(DI), X0, X0
	I4BLK_AVX2(0)

	I4TOP_AVX2(1)
	VMOVDQU	32(R8), X3
	VPSHUFB	reconIdxE<>(SB), X3, X3
	VPOR	X3, X1, X1
	I4BLK_AVX2(1)

	I4TOP_AVX2(2)
	VMOVDQU	48(R8), X3
	VPSHUFB	reconIdxE<>(SB), X3, X3
	VPOR	X3, X1, X1
	I4BLK_AVX2(2)

	// c = 3: E..H from the top-right context (columns 16-19 of the row
	// above, replicated by loadEdges).
	I4TOP_AVX2(3)
	VPMOVZXBD	(-const_bps+16)(DI), X3
	VPSHUFB	reconIdxE<>(SB), X3, X3
	VPOR	X3, X1, X1
	I4BLK_AVX2(3)

	ADDQ	$(4*const_bps), DI
	ADDQ	$128, SI
	ADDQ	$4, DX
	XCHGQ	R8, R9
	DECL	R10
	JNZ	i4avx2Row
	VZEROUPPER
	MOVB	$1, ret+32(FP)
	RET

i4avx2Fallback:
	VZEROUPPER
	MOVB	$0, ret+32(FP)
	RET

// --- 16x16 and chroma prediction helpers -------------------------------------------

// SUMLEFT16(base, dst) / SUMLEFT8: sum of the 16 (8) left samples of the
// block at base. Clobbers R11.
#define SUMLEFT8(base, dst) \
	MOVBLZX	-1(base), dst \
	MOVBLZX	(const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(2*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(3*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(4*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(5*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(6*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(7*const_bps-1)(base), R11 \
	ADDL	R11, dst

#define SUMLEFT16(base, dst) \
	SUMLEFT8(base, dst) \
	MOVBLZX	(8*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(9*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(10*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(11*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(12*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(13*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(14*const_bps-1)(base), R11 \
	ADDL	R11, dst \
	MOVBLZX	(15*const_bps-1)(base), R11 \
	ADDL	R11, dst

// --- intra 16x16 (AVX2) -------------------------------------------------------------

// ST16_AVX2(r) stores X0 to row r of the block at DI.
#define ST16_AVX2(r) VMOVDQU	X0, ((r)*const_bps)(DI)

#define ST16x16_AVX2 \
	ST16_AVX2(0) \
	ST16_AVX2(1) \
	ST16_AVX2(2) \
	ST16_AVX2(3) \
	ST16_AVX2(4) \
	ST16_AVX2(5) \
	ST16_AVX2(6) \
	ST16_AVX2(7) \
	ST16_AVX2(8) \
	ST16_AVX2(9) \
	ST16_AVX2(10) \
	ST16_AVX2(11) \
	ST16_AVX2(12) \
	ST16_AVX2(13) \
	ST16_AVX2(14) \
	ST16_AVX2(15)

// HE16_AVX2(r): row r = its left sample.
#define HE16_AVX2(r) \
	VPBROADCASTB	((r)*const_bps-1)(DI), X0 \
	VMOVDQU	X0, ((r)*const_bps)(DI)

// TM16_AVX2(r): row r = clip8(left + max(top-X, 0) - max(X-top, 0)) with
// the two differences in X3, X4.
#define TM16_AVX2(r) \
	VPBROADCASTB	((r)*const_bps-1)(DI), X0 \
	VPADDUSB	X3, X0, X0 \
	VPSUBUSB	X4, X0, X0 \
	VMOVDQU	X0, ((r)*const_bps)(DI)

// ADD16x2_AVX2(ta, tb, r) adds the 16-bit residual rows ta, tb to rows r,
// r+1 of the block at DI with clipping. Clobbers Y4, Y5.
#define ADD16x2_AVX2(ta, tb, r) \
	VPMOVZXBW	((r)*const_bps)(DI), Y4 \
	VPMOVZXBW	((r)*const_bps+const_bps)(DI), Y5 \
	VPADDW	ta, Y4, Y4 \
	VPADDW	tb, Y5, Y5 \
	VPACKUSWB	Y5, Y4, Y4 \
	VPERMQ	$0xD8, Y4, Y4 \
	VMOVDQU	X4, ((r)*const_bps)(DI) \
	VEXTRACTI128	$1, Y4, ((r)*const_bps+const_bps)(DI)

// func reconI16AVX2(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) bool
//
// Mirrors reconI16Ref: the predictor of mode (predLuma16, as libwebp's
// DC16/TM16/VE16/HE16 SSE2 versions; values outside 0-5 select the default
// case) is written to b, then each block row with a non-zero code gets the
// residuals of its four sub-blocks added (16-bit transform of all four, or
// the DC-only path), with ADD16x2.
//
// Registers: DI luma block, SI coefficients of the block row, BX bits
// (rotated), AX mode then scratch, CX/R11/R13 scratch and code masks, R10
// row counter, R12 = reconMask, X0-X4 prediction, Y0-Y13 residual, X15 = 0.
TEXT ·reconI16AVX2(SB), NOSPLIT, $0-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	bits+16(FP), BX
	MOVQ	mode+24(FP), AX
	ADDQ	$const_yOff, DI
	VPXOR	X15, X15, X15
	CMPQ	AX, $1
	JEQ	i16avx2TM
	CMPQ	AX, $2
	JEQ	i16avx2VE
	CMPQ	AX, $3
	JEQ	i16avx2HE
	CMPQ	AX, $0
	JEQ	i16avx2DC
	CMPQ	AX, $4
	JEQ	i16avx2DCNoTop
	CMPQ	AX, $5
	JEQ	i16avx2DCNoLeft
	MOVL	$0x80, AX
	JMP	i16avx2Fill

i16avx2DC:
	SUMLEFT16(DI, CX)
	VMOVDQU	(-const_bps)(DI), X0
	VPSADBW	X15, X0, X0
	VPEXTRW	$4, X0, AX
	ADDL	AX, CX
	VMOVD	X0, AX
	LEAL	16(AX)(CX*1), AX
	SHRL	$5, AX
	JMP	i16avx2Fill

i16avx2DCNoTop:
	SUMLEFT16(DI, AX)
	ADDL	$8, AX
	SHRL	$4, AX
	JMP	i16avx2Fill

i16avx2DCNoLeft:
	VMOVDQU	(-const_bps)(DI), X0
	VPSADBW	X15, X0, X0
	VPEXTRW	$4, X0, CX
	VMOVD	X0, AX
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX

i16avx2Fill:
	VMOVD	AX, X0
	VPBROADCASTB	X0, X0
	ST16x16_AVX2
	JMP	i16avx2Res

i16avx2VE:
	VMOVDQU	(-const_bps)(DI), X0
	ST16x16_AVX2
	JMP	i16avx2Res

i16avx2HE:
	HE16_AVX2(0)
	HE16_AVX2(1)
	HE16_AVX2(2)
	HE16_AVX2(3)
	HE16_AVX2(4)
	HE16_AVX2(5)
	HE16_AVX2(6)
	HE16_AVX2(7)
	HE16_AVX2(8)
	HE16_AVX2(9)
	HE16_AVX2(10)
	HE16_AVX2(11)
	HE16_AVX2(12)
	HE16_AVX2(13)
	HE16_AVX2(14)
	HE16_AVX2(15)
	JMP	i16avx2Res

i16avx2TM:
	VMOVDQU	(-const_bps)(DI), X1
	VPBROADCASTB	(-const_bps-1)(DI), X2
	VPSUBUSB	X2, X1, X3
	VPSUBUSB	X1, X2, X4
	TM16_AVX2(0)
	TM16_AVX2(1)
	TM16_AVX2(2)
	TM16_AVX2(3)
	TM16_AVX2(4)
	TM16_AVX2(5)
	TM16_AVX2(6)
	TM16_AVX2(7)
	TM16_AVX2(8)
	TM16_AVX2(9)
	TM16_AVX2(10)
	TM16_AVX2(11)
	TM16_AVX2(12)
	TM16_AVX2(13)
	TM16_AVX2(14)
	TM16_AVX2(15)

i16avx2Res:
	TESTL	BX, BX
	JZ	i16avx2Done
	LEAQ	reconMask<>(SB), R12
	MOVL	$4, R10

i16avx2Row:
	ROLL	$8, BX
	TESTB	$0xff, BL
	JZ	i16avx2Next
	CODEMASKS
	TESTB	$0xaa, BL
	JNZ	i16avx2Full
	DC4_AVX2(0, 32, 64, 96, AX, CX, R11, R13)
	VPSHUFB	reconDCSpread<>(SB), Y0, Y0
	ADD16x2_AVX2(Y0, Y0, 0)
	ADD16x2_AVX2(Y0, Y0, 2)
	JMP	i16avx2Next

i16avx2Full:
	LOAD4_AVX2(0, 32, 64, 96, AX, CX, R11, R13)
	VPTEST	reconRange<>(SB), Y13
	JNZ	i16avx2Fallback
	IDCT4_AVX2
	TRANSPOSE_AVX2
	ADD16x2_AVX2(Y0, Y1, 0)
	ADD16x2_AVX2(Y2, Y3, 2)

i16avx2Next:
	ADDQ	$(4*const_bps), DI
	ADDQ	$128, SI
	DECL	R10
	JNZ	i16avx2Row

i16avx2Done:
	VZEROUPPER
	MOVB	$1, ret+32(FP)
	RET

i16avx2Fallback:
	VZEROUPPER
	MOVB	$0, ret+32(FP)
	RET

// --- chroma (AVX2) -------------------------------------------------------------------

#define ST8_AVX2(base, r) VMOVQ	X0, ((r)*const_bps)(base)

#define ST8x8_AVX2(base) \
	ST8_AVX2(base, 0) \
	ST8_AVX2(base, 1) \
	ST8_AVX2(base, 2) \
	ST8_AVX2(base, 3) \
	ST8_AVX2(base, 4) \
	ST8_AVX2(base, 5) \
	ST8_AVX2(base, 6) \
	ST8_AVX2(base, 7)

#define HE8_AVX2(base, r) \
	VPBROADCASTB	((r)*const_bps-1)(base), X0 \
	VMOVQ	X0, ((r)*const_bps)(base)

#define HE8x8_AVX2(base) \
	HE8_AVX2(base, 0) \
	HE8_AVX2(base, 1) \
	HE8_AVX2(base, 2) \
	HE8_AVX2(base, 3) \
	HE8_AVX2(base, 4) \
	HE8_AVX2(base, 5) \
	HE8_AVX2(base, 6) \
	HE8_AVX2(base, 7)

#define TM8_AVX2(base, r) \
	VPBROADCASTB	((r)*const_bps-1)(base), X0 \
	VPADDUSB	X3, X0, X0 \
	VPSUBUSB	X4, X0, X0 \
	VMOVQ	X0, ((r)*const_bps)(base)

#define TM8x8_AVX2(base) \
	VMOVQ	(-const_bps)(base), X1 \
	VPBROADCASTB	(-const_bps-1)(base), X2 \
	VPSUBUSB	X2, X1, X3 \
	VPSUBUSB	X1, X2, X4 \
	TM8_AVX2(base, 0) \
	TM8_AVX2(base, 1) \
	TM8_AVX2(base, 2) \
	TM8_AVX2(base, 3) \
	TM8_AVX2(base, 4) \
	TM8_AVX2(base, 5) \
	TM8_AVX2(base, 6) \
	TM8_AVX2(base, 7)

// SUMTOP8_AVX2(base, dst): sum of the 8 top samples.
#define SUMTOP8_AVX2(base, dst) \
	VMOVQ	(-const_bps)(base), X0 \
	VPSADBW	X15, X0, X0 \
	VMOVD	X0, dst

// FILL8_AVX2(base, v): fill the 8x8 block with the byte in v (a GPR).
#define FILL8_AVX2(base, v) \
	VMOVD	v, X0 \
	VPBROADCASTB	X0, X0 \
	ST8x8_AVX2(base)

// ADDUV2_AVX2(ta, tb, r) adds the residual rows ta, tb ([U row | V row],
// 16-bit) to rows r, r+1 of the U (R8) and V (R9) blocks. Clobbers Y4, Y5.
#define ADDUV2_AVX2(ta, tb, r) \
	VMOVQ	((r)*const_bps)(R8), X4 \
	VPINSRQ	$1, ((r)*const_bps)(R9), X4, X4 \
	VMOVQ	((r)*const_bps+const_bps)(R8), X5 \
	VPINSRQ	$1, ((r)*const_bps+const_bps)(R9), X5, X5 \
	VPMOVZXBW	X4, Y4 \
	VPMOVZXBW	X5, Y5 \
	VPADDW	ta, Y4, Y4 \
	VPADDW	tb, Y5, Y5 \
	VPACKUSWB	Y5, Y4, Y4 \
	VEXTRACTI128	$1, Y4, X5 \
	VMOVQ	X4, ((r)*const_bps)(R8) \
	VPEXTRQ	$1, X4, ((r)*const_bps+const_bps)(R8) \
	VMOVQ	X5, ((r)*const_bps)(R9) \
	VPEXTRQ	$1, X5, ((r)*const_bps+const_bps)(R9)

// func reconUVAVX2(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int) bool
//
// Mirrors reconUVRef: the chroma predictor of mode (predChroma8, as
// libwebp's DC8uv/TM8uv/VE8uv/HE8uv; values outside 0-5 select the default
// case) on both planes, then the residuals. codes holds the transform code
// of every block of the U (bits 0-7) and V (bits 8-15) plane, the same for
// the four blocks of a plane (see uvPlaneCodes). The transform runs on
// blocks [U0 U1 | V0 V1] for rows 0-3 and [U2 U3 | V2 V3] for rows 4-7, so
// that each output row is [U row | V row].
//
// Registers: R8 U block, R9 V block, SI chroma coefficients, BX codes, AX
// mode then U mask, CX V mask, R11 scratch, R12 = reconMask, X0-X4
// prediction, Y0-Y13 residual, X15 = 0.
TEXT ·reconUVAVX2(SB), NOSPLIT, $0-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	codes+16(FP), BX
	MOVQ	mode+24(FP), AX
	LEAQ	const_uOff(DI), R8
	LEAQ	const_vOff(DI), R9
	VPXOR	X15, X15, X15
	CMPQ	AX, $1
	JEQ	uvavx2TM
	CMPQ	AX, $2
	JEQ	uvavx2VE
	CMPQ	AX, $3
	JEQ	uvavx2HE
	CMPQ	AX, $0
	JEQ	uvavx2DC
	CMPQ	AX, $4
	JEQ	uvavx2DCNoTop
	CMPQ	AX, $5
	JEQ	uvavx2DCNoLeft
	MOVL	$0x80, AX
	FILL8_AVX2(R8, AX)
	FILL8_AVX2(R9, AX)
	JMP	uvavx2Res

uvavx2DC:
	SUMLEFT8(R8, CX)
	SUMTOP8_AVX2(R8, AX)
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX
	FILL8_AVX2(R8, AX)
	SUMLEFT8(R9, CX)
	SUMTOP8_AVX2(R9, AX)
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX
	FILL8_AVX2(R9, AX)
	JMP	uvavx2Res

uvavx2DCNoTop:
	SUMLEFT8(R8, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_AVX2(R8, AX)
	SUMLEFT8(R9, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_AVX2(R9, AX)
	JMP	uvavx2Res

uvavx2DCNoLeft:
	SUMTOP8_AVX2(R8, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_AVX2(R8, AX)
	SUMTOP8_AVX2(R9, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_AVX2(R9, AX)
	JMP	uvavx2Res

uvavx2VE:
	VMOVQ	(-const_bps)(R8), X0
	ST8x8_AVX2(R8)
	VMOVQ	(-const_bps)(R9), X0
	ST8x8_AVX2(R9)
	JMP	uvavx2Res

uvavx2HE:
	HE8x8_AVX2(R8)
	HE8x8_AVX2(R9)
	JMP	uvavx2Res

uvavx2TM:
	TM8x8_AVX2(R8)
	TM8x8_AVX2(R9)

uvavx2Res:
	TESTL	BX, BX
	JZ	uvavx2Done
	ADDQ	$(16*16*2), SI
	LEAQ	reconMask<>(SB), R12
	MOVL	BX, AX
	ANDL	$3, AX
	SHLL	$5, AX
	ADDQ	R12, AX
	MOVL	BX, CX
	SHRL	$8, CX
	ANDL	$3, CX
	SHLL	$5, CX
	ADDQ	R12, CX
	TESTL	$0xaaaa, BX
	JNZ	uvavx2Full
	DC4_AVX2(0, 32, 128, 160, AX, AX, CX, CX)
	VPSHUFB	reconDCSpread<>(SB), Y0, Y0
	ADDUV2_AVX2(Y0, Y0, 0)
	ADDUV2_AVX2(Y0, Y0, 2)
	DC4_AVX2(64, 96, 192, 224, AX, AX, CX, CX)
	VPSHUFB	reconDCSpread<>(SB), Y0, Y0
	ADDUV2_AVX2(Y0, Y0, 4)
	ADDUV2_AVX2(Y0, Y0, 6)
	JMP	uvavx2Done

uvavx2Full:
	LOAD4_AVX2(0, 32, 128, 160, AX, AX, CX, CX)
	VPTEST	reconRange<>(SB), Y13
	JNZ	uvavx2Fallback
	IDCT4_AVX2
	TRANSPOSE_AVX2
	ADDUV2_AVX2(Y0, Y1, 0)
	ADDUV2_AVX2(Y2, Y3, 2)
	LOAD4_AVX2(64, 96, 192, 224, AX, AX, CX, CX)
	VPTEST	reconRange<>(SB), Y13
	JNZ	uvavx2Fallback
	IDCT4_AVX2
	TRANSPOSE_AVX2
	ADDUV2_AVX2(Y0, Y1, 4)
	ADDUV2_AVX2(Y2, Y3, 6)

uvavx2Done:
	VZEROUPPER
	MOVB	$1, ret+32(FP)
	RET

uvavx2Fallback:
	VZEROUPPER
	MOVB	$0, ret+32(FP)
	RET

// --- shared SSE4.1 pieces ------------------------------------------------------------
//
// The SSE4.1 kernels are the AVX2 ones with the 16-bit transform on two
// blocks per xmm register (Transform_SSE2 with do_two) and legacy SSE
// encodings only. Constants are used as memory operands (DATA symbols are
// aligned to their size up to 32 bytes); frame slots, the work buffer and
// the coefficients are loaded with MOVOU first.

// LOAD2_SSE loads the blocks at SI+o0 and SI+o1, masked by (m0) and (m1),
// as the rows in0..in3 = X0..X3 of Transform_SSE2 (block o0 in the low
// half), and sets X6 to max |coefficient|. Clobbers X4, X5, X7.
#define LOAD2_SSE(o0, o1, m0, m1) \
	MOVOU	o0(SI), X0 \
	PAND	(m0), X0 \
	MOVOU	(o0+16)(SI), X2 \
	PAND	16(m0), X2 \
	MOVOU	o1(SI), X4 \
	PAND	(m1), X4 \
	MOVOU	(o1+16)(SI), X5 \
	PAND	16(m1), X5 \
	PABSW	X0, X6 \
	PABSW	X2, X7 \
	PMAXUW	X7, X6 \
	PABSW	X4, X7 \
	PMAXUW	X7, X6 \
	PABSW	X5, X7 \
	PMAXUW	X7, X6 \
	MOVO	X0, X1 \
	PUNPCKLQDQ	X4, X0 \
	PUNPCKHQDQ	X4, X1 \
	MOVO	X2, X3 \
	PUNPCKLQDQ	X5, X2 \
	PUNPCKHQDQ	X5, X3

// DC2_SSE sets 16-bit lanes 0 and 4 of X0 to the DC-only residual
// sat(in[0] + 4) >> 3 of the blocks at SI+o0 and SI+o1 (masked by (m0),
// (m1)). Clobbers X1.
#define DC2_SSE(o0, o1, m0, m1) \
	MOVOU	o0(SI), X0 \
	PAND	(m0), X0 \
	MOVOU	o1(SI), X1 \
	PAND	(m1), X1 \
	PUNPCKLQDQ	X1, X0 \
	PADDSW	reconFour<>(SB), X0 \
	PSRAW	$3, X0

// IDCT_PASS_SSE: IDCT_PASS_AVX2 on X0..X3 with two-operand instructions.
// Clobbers X4-X8.
#define IDCT_PASS_SSE(ADD, SUB) \
	MOVO	X0, X4 \
	PADDW	X2, X4 \
	MOVO	X0, X5 \
	PSUBW	X2, X5 \
	MOVO	X1, X6 \
	PMULHW	reconK2<>(SB), X6 \
	MOVO	X3, X7 \
	PMULHW	reconK1<>(SB), X7 \
	PSUBW	X7, X6 \
	MOVO	X1, X7 \
	PSUBW	X3, X7 \
	PADDW	X7, X6 \
	MOVO	X1, X7 \
	PMULHW	reconK1<>(SB), X7 \
	MOVO	X3, X8 \
	PMULHW	reconK2<>(SB), X8 \
	PADDW	X3, X1 \
	PADDW	X8, X7 \
	PADDW	X1, X7 \
	MOVO	X4, X0 \
	ADD	X7, X0 \
	MOVO	X5, X1 \
	ADD	X6, X1 \
	MOVO	X5, X2 \
	SUB	X6, X2 \
	MOVO	X4, X3 \
	SUB	X7, X3

// TRANSPOSE_SSE: VP8Transpose_2_4x4_16b on X0..X3. Clobbers X4-X8.
#define TRANSPOSE_SSE \
	MOVO	X0, X4 \
	PUNPCKLWL	X1, X4 \
	MOVO	X2, X5 \
	PUNPCKLWL	X3, X5 \
	PUNPCKHWL	X1, X0 \
	PUNPCKHWL	X3, X2 \
	MOVO	X4, X6 \
	PUNPCKLLQ	X5, X6 \
	PUNPCKHLQ	X5, X4 \
	MOVO	X0, X7 \
	PUNPCKLLQ	X2, X7 \
	PUNPCKHLQ	X2, X0 \
	MOVO	X6, X8 \
	PUNPCKLQDQ	X7, X8 \
	PUNPCKHQDQ	X7, X6 \
	MOVO	X4, X2 \
	PUNPCKLQDQ	X0, X2 \
	PUNPCKHQDQ	X0, X4 \
	MOVO	X8, X0 \
	MOVO	X6, X1 \
	MOVO	X4, X3

// IDCT2_SSE: IDCT4_AVX2 for the two blocks in X0..X3 (from LOAD2_SSE):
// Xk = column k of blocks [o0 | o1]. Clobbers X4-X8.
#define IDCT2_SSE \
	IDCT_PASS_SSE(PADDW, PSUBW) \
	TRANSPOSE_SSE \
	PADDW	reconFour<>(SB), X0 \
	IDCT_PASS_SSE(PADDSW, PSUBSW) \
	PSRAW	$3, X0 \
	PSRAW	$3, X1 \
	PSRAW	$3, X2 \
	PSRAW	$3, X3

// ADD8x2_SSE(ta, tb, base, off) adds the 16-bit residual rows ta, tb to the
// 8 pixels at base+off and base+off+const_bps with clipping. Clobbers X4,
// X5.
#define ADD8x2_SSE(ta, tb, base, off) \
	PMOVZXBW	(off)(base), X4 \
	PMOVZXBW	((off)+const_bps)(base), X5 \
	PADDW	ta, X4 \
	PADDW	tb, X5 \
	PACKUSWB	X5, X4 \
	MOVQ	X4, (off)(base) \
	PEXTRQ	$1, X4, ((off)+const_bps)(base)

// --- intra 4x4 (SSE4.1) -------------------------------------------------------------

// I4PN_SSE(off) stores the positive and negative parts of the residual
// columns X0..X3 of two sub-blocks (from IDCT2_SSE) in the 4x4 layout at
// off(SP) (first block: pos, neg; second block: pos, neg). Clobbers X4-X6.
#define I4PN_SSE(off) \
	MOVO	X0, X4 \
	PACKUSWB	X1, X4 \
	MOVO	X2, X5 \
	PACKUSWB	X3, X5 \
	PSHUFB	reconPN<>(SB), X4 \
	PSHUFB	reconPN<>(SB), X5 \
	MOVO	X4, X6 \
	PUNPCKLQDQ	X5, X6 \
	PUNPCKHQDQ	X5, X4 \
	MOVOU	X6, (off)(SP) \
	MOVOU	X4, ((off)+32)(SP) \
	MOVO	X15, X4 \
	PSUBW	X0, X4 \
	MOVO	X15, X5 \
	PSUBW	X1, X5 \
	PACKUSWB	X5, X4 \
	MOVO	X15, X5 \
	PSUBW	X2, X5 \
	MOVO	X15, X6 \
	PSUBW	X3, X6 \
	PACKUSWB	X6, X5 \
	PSHUFB	reconPN<>(SB), X4 \
	PSHUFB	reconPN<>(SB), X5 \
	MOVO	X4, X6 \
	PUNPCKLQDQ	X5, X6 \
	PUNPCKHQDQ	X5, X4 \
	MOVOU	X6, ((off)+16)(SP) \
	MOVOU	X4, ((off)+48)(SP)

// I4DCPN_SSE(off) is I4PN_SSE for the DC-only residuals in lanes 0 and 4
// of X0 (from DC2_SSE). Clobbers X1, X2.
#define I4DCPN_SSE(off) \
	MOVO	X0, X1 \
	PACKUSWB	X1, X1 \
	MOVO	X1, X2 \
	PSHUFB	X15, X2 \
	PSHUFB	reconFours<>(SB), X1 \
	MOVOU	X2, (off)(SP) \
	MOVOU	X1, ((off)+32)(SP) \
	MOVO	X15, X1 \
	PSUBW	X0, X1 \
	PACKUSWB	X1, X1 \
	MOVO	X1, X2 \
	PSHUFB	X15, X2 \
	PSHUFB	reconFours<>(SB), X1 \
	MOVOU	X2, ((off)+16)(SP) \
	MOVOU	X1, ((off)+48)(SP)

// I4TOP_SSE(c): I4TOP_AVX2. Clobbers X2.
#define I4TOP_SSE(c) \
	MOVOU	(16*(c))(R8), X1 \
	PSHUFB	reconIdxA<>(SB), X1 \
	MOVOU	(16*(c)-16)(R8), X2 \
	PSHUFB	reconIdxX<>(SB), X2 \
	POR	X2, X1

// I4BLK_SSE(c): I4BLK_AVX2 with two-operand instructions.
#define I4BLK_SSE(c) \
	MOVBLZX	(c)(DX), AX \
	MOVL	$9, CX \
	CMPL	AX, CX \
	CMOVLHI	CX, AX \
	SHLL	$6, AX \
	MOVO	X1, X2 \
	PALIGNR	$12, X0, X2 \
	MOVO	X2, X3 \
	PSHUFB	reconIdxPrev<>(SB), X3 \
	MOVO	X2, X4 \
	PSHUFB	reconIdxNext<>(SB), X4 \
	MOVO	X3, X5 \
	PAVGB	X2, X5 \
	MOVO	X3, X6 \
	PAVGB	X4, X6 \
	MOVO	X3, X7 \
	PXOR	X4, X7 \
	PAND	X14, X7 \
	PSUBUSB	X7, X6 \
	PAVGB	X2, X6 \
	MOVO	X2, X8 \
	PSHUFB	reconIdxDC<>(SB), X8 \
	PSADBW	X15, X8 \
	PMULHRSW	reconC4096<>(SB), X8 \
	MOVO	X1, X9 \
	PSHUFB	reconIdxTop<>(SB), X9 \
	MOVO	X1, X10 \
	PSHUFB	X15, X10 \
	MOVO	X9, X13 \
	PSUBUSB	X10, X13 \
	PSUBUSB	X9, X10 \
	PSHUFD	$0xff, X0, X11 \
	PADDUSB	X13, X11 \
	PSUBUSB	X10, X11 \
	MOVO	X6, X12 \
	PSHUFB	0(R12)(AX*1), X12 \
	MOVO	X5, X13 \
	PSHUFB	16(R12)(AX*1), X13 \
	POR	X13, X12 \
	MOVO	X8, X13 \
	PSHUFB	32(R12)(AX*1), X13 \
	POR	X13, X12 \
	PAND	48(R12)(AX*1), X11 \
	POR	X11, X12 \
	MOVOU	(32*(c))(SP), X13 \
	PADDUSB	X13, X12 \
	MOVOU	(32*(c)+16)(SP), X13 \
	PSUBUSB	X13, X12 \
	MOVO	X12, X0 \
	MOVOU	X0, (16*(c))(R9) \
	PSHUFB	reconIdxRows<>(SB), X12 \
	MOVL	X12, (4*(c))(DI) \
	PEXTRD	$1, X12, (4*(c)+const_bps)(DI) \
	PEXTRD	$2, X12, (4*(c)+2*const_bps)(DI) \
	PEXTRD	$3, X12, (4*(c)+3*const_bps)(DI)

// func reconI4SSE41(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) bool
//
// reconI4AVX2 with the residuals of a block row computed two blocks at a
// time. Registers as in reconI4AVX2.
TEXT ·reconI4SSE41(SB), NOSPLIT, $256-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	bits+16(FP), BX
	MOVQ	modes+24(FP), DX
	ADDQ	$const_yOff, DI
	LEAQ	128(SP), R8
	LEAQ	192(SP), R9
	PXOR	X15, X15
	MOVOU	reconOnes<>(SB), X14
	PMOVZXBD	(-const_bps)(DI), X0
	MOVOU	X0, 0(R8)
	PMOVZXBD	(-const_bps+4)(DI), X0
	MOVOU	X0, 16(R8)
	PMOVZXBD	(-const_bps+8)(DI), X0
	MOVOU	X0, 32(R8)
	PMOVZXBD	(-const_bps+12)(DI), X0
	MOVOU	X0, 48(R8)
	MOVL	$4, R10

i4sseRow:
	ROLL	$8, BX
	TESTB	$0xff, BL
	JNZ	i4sseRes
	MOVOU	X15, 0(SP)
	MOVOU	X15, 16(SP)
	MOVOU	X15, 32(SP)
	MOVOU	X15, 48(SP)
	MOVOU	X15, 64(SP)
	MOVOU	X15, 80(SP)
	MOVOU	X15, 96(SP)
	MOVOU	X15, 112(SP)
	JMP	i4ssePred

i4sseRes:
	LEAQ	reconMask<>(SB), R12
	CODEMASKS
	TESTB	$0xaa, BL
	JNZ	i4sseFull
	DC2_SSE(0, 32, AX, CX)
	I4DCPN_SSE(0)
	DC2_SSE(64, 96, R11, R13)
	I4DCPN_SSE(64)
	JMP	i4ssePred

i4sseFull:
	LOAD2_SSE(0, 32, AX, CX)
	PTEST	reconRange<>(SB), X6
	JNZ	i4sseFallback
	IDCT2_SSE
	I4PN_SSE(0)
	LOAD2_SSE(64, 96, R11, R13)
	PTEST	reconRange<>(SB), X6
	JNZ	i4sseFallback
	IDCT2_SSE
	I4PN_SSE(64)

i4ssePred:
	LEAQ	reconI4Tab<>(SB), R12
	MOVOU	0(R8), X1
	PSHUFB	reconIdxA<>(SB), X1
	MOVOU	16(R8), X3
	PSHUFB	reconIdxE<>(SB), X3
	POR	X3, X1
	PINSRB	$0, (-const_bps-1)(DI), X1
	PXOR	X0, X0
	PINSRB	$15, -1(DI), X0
	PINSRB	$14, (const_bps-1)(DI), X0
	PINSRB	$13, (2*const_bps-1)(DI), X0
	PINSRB	$12, (3*const_bps-1)(DI), X0
	I4BLK_SSE(0)

	I4TOP_SSE(1)
	MOVOU	32(R8), X3
	PSHUFB	reconIdxE<>(SB), X3
	POR	X3, X1
	I4BLK_SSE(1)

	I4TOP_SSE(2)
	MOVOU	48(R8), X3
	PSHUFB	reconIdxE<>(SB), X3
	POR	X3, X1
	I4BLK_SSE(2)

	I4TOP_SSE(3)
	PMOVZXBD	(-const_bps+16)(DI), X3
	PSHUFB	reconIdxE<>(SB), X3
	POR	X3, X1
	I4BLK_SSE(3)

	ADDQ	$(4*const_bps), DI
	ADDQ	$128, SI
	ADDQ	$4, DX
	XCHGQ	R8, R9
	DECL	R10
	JNZ	i4sseRow
	MOVB	$1, ret+32(FP)
	RET

i4sseFallback:
	MOVB	$0, ret+32(FP)
	RET

// --- intra 16x16 (SSE4.1) -----------------------------------------------------------

// BCAST_SSE(src, x): x = the byte at src in every lane (X15 = 0). Clobbers
// R11.
#define BCAST_SSE(src, x) \
	MOVBLZX	src, R11 \
	MOVL	R11, x \
	PSHUFB	X15, x

#define ST16_SSE(r) MOVOU	X0, ((r)*const_bps)(DI)

#define ST16x16_SSE \
	ST16_SSE(0) \
	ST16_SSE(1) \
	ST16_SSE(2) \
	ST16_SSE(3) \
	ST16_SSE(4) \
	ST16_SSE(5) \
	ST16_SSE(6) \
	ST16_SSE(7) \
	ST16_SSE(8) \
	ST16_SSE(9) \
	ST16_SSE(10) \
	ST16_SSE(11) \
	ST16_SSE(12) \
	ST16_SSE(13) \
	ST16_SSE(14) \
	ST16_SSE(15)

#define HE16_SSE(r) \
	BCAST_SSE(((r)*const_bps-1)(DI), X0) \
	MOVOU	X0, ((r)*const_bps)(DI)

#define TM16_SSE(r) \
	BCAST_SSE(((r)*const_bps-1)(DI), X0) \
	PADDUSB	X3, X0 \
	PSUBUSB	X4, X0 \
	MOVOU	X0, ((r)*const_bps)(DI)

// func reconI16SSE41(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) bool
//
// reconI16AVX2 with the residuals of a block row computed and added two
// blocks (8 columns) at a time. Registers as in reconI16AVX2.
TEXT ·reconI16SSE41(SB), NOSPLIT, $0-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	bits+16(FP), BX
	MOVQ	mode+24(FP), AX
	ADDQ	$const_yOff, DI
	PXOR	X15, X15
	CMPQ	AX, $1
	JEQ	i16sseTM
	CMPQ	AX, $2
	JEQ	i16sseVE
	CMPQ	AX, $3
	JEQ	i16sseHE
	CMPQ	AX, $0
	JEQ	i16sseDC
	CMPQ	AX, $4
	JEQ	i16sseDCNoTop
	CMPQ	AX, $5
	JEQ	i16sseDCNoLeft
	MOVL	$0x80, AX
	JMP	i16sseFill

i16sseDC:
	SUMLEFT16(DI, CX)
	MOVOU	(-const_bps)(DI), X0
	PSADBW	X15, X0
	PEXTRW	$4, X0, AX
	ADDL	AX, CX
	MOVL	X0, AX
	LEAL	16(AX)(CX*1), AX
	SHRL	$5, AX
	JMP	i16sseFill

i16sseDCNoTop:
	SUMLEFT16(DI, AX)
	ADDL	$8, AX
	SHRL	$4, AX
	JMP	i16sseFill

i16sseDCNoLeft:
	MOVOU	(-const_bps)(DI), X0
	PSADBW	X15, X0
	PEXTRW	$4, X0, CX
	MOVL	X0, AX
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX

i16sseFill:
	MOVL	AX, X0
	PSHUFB	X15, X0
	ST16x16_SSE
	JMP	i16sseRes

i16sseVE:
	MOVOU	(-const_bps)(DI), X0
	ST16x16_SSE
	JMP	i16sseRes

i16sseHE:
	HE16_SSE(0)
	HE16_SSE(1)
	HE16_SSE(2)
	HE16_SSE(3)
	HE16_SSE(4)
	HE16_SSE(5)
	HE16_SSE(6)
	HE16_SSE(7)
	HE16_SSE(8)
	HE16_SSE(9)
	HE16_SSE(10)
	HE16_SSE(11)
	HE16_SSE(12)
	HE16_SSE(13)
	HE16_SSE(14)
	HE16_SSE(15)
	JMP	i16sseRes

i16sseTM:
	MOVOU	(-const_bps)(DI), X1
	BCAST_SSE((-const_bps-1)(DI), X2)
	MOVO	X1, X3
	PSUBUSB	X2, X3
	MOVO	X2, X4
	PSUBUSB	X1, X4
	TM16_SSE(0)
	TM16_SSE(1)
	TM16_SSE(2)
	TM16_SSE(3)
	TM16_SSE(4)
	TM16_SSE(5)
	TM16_SSE(6)
	TM16_SSE(7)
	TM16_SSE(8)
	TM16_SSE(9)
	TM16_SSE(10)
	TM16_SSE(11)
	TM16_SSE(12)
	TM16_SSE(13)
	TM16_SSE(14)
	TM16_SSE(15)

i16sseRes:
	TESTL	BX, BX
	JZ	i16sseDone
	LEAQ	reconMask<>(SB), R12
	MOVL	$4, R10

i16sseRow:
	ROLL	$8, BX
	TESTB	$0xff, BL
	JZ	i16sseNext
	CODEMASKS
	TESTB	$0xaa, BL
	JNZ	i16sseFull
	DC2_SSE(0, 32, AX, CX)
	PSHUFB	reconDCSpread<>(SB), X0
	ADD8x2_SSE(X0, X0, DI, 0)
	ADD8x2_SSE(X0, X0, DI, 2*const_bps)
	DC2_SSE(64, 96, R11, R13)
	PSHUFB	reconDCSpread<>(SB), X0
	ADD8x2_SSE(X0, X0, DI, 8)
	ADD8x2_SSE(X0, X0, DI, 2*const_bps+8)
	JMP	i16sseNext

i16sseFull:
	LOAD2_SSE(0, 32, AX, CX)
	PTEST	reconRange<>(SB), X6
	JNZ	i16sseFallback
	IDCT2_SSE
	TRANSPOSE_SSE
	ADD8x2_SSE(X0, X1, DI, 0)
	ADD8x2_SSE(X2, X3, DI, 2*const_bps)
	LOAD2_SSE(64, 96, R11, R13)
	PTEST	reconRange<>(SB), X6
	JNZ	i16sseFallback
	IDCT2_SSE
	TRANSPOSE_SSE
	ADD8x2_SSE(X0, X1, DI, 8)
	ADD8x2_SSE(X2, X3, DI, 2*const_bps+8)

i16sseNext:
	ADDQ	$(4*const_bps), DI
	ADDQ	$128, SI
	DECL	R10
	JNZ	i16sseRow

i16sseDone:
	MOVB	$1, ret+32(FP)
	RET

i16sseFallback:
	MOVB	$0, ret+32(FP)
	RET

// --- chroma (SSE4.1) -------------------------------------------------------------------

#define ST8_SSE(base, r) MOVQ	X0, ((r)*const_bps)(base)

#define ST8x8_SSE(base) \
	ST8_SSE(base, 0) \
	ST8_SSE(base, 1) \
	ST8_SSE(base, 2) \
	ST8_SSE(base, 3) \
	ST8_SSE(base, 4) \
	ST8_SSE(base, 5) \
	ST8_SSE(base, 6) \
	ST8_SSE(base, 7)

#define HE8_SSE(base, r) \
	BCAST_SSE(((r)*const_bps-1)(base), X0) \
	MOVQ	X0, ((r)*const_bps)(base)

#define HE8x8_SSE(base) \
	HE8_SSE(base, 0) \
	HE8_SSE(base, 1) \
	HE8_SSE(base, 2) \
	HE8_SSE(base, 3) \
	HE8_SSE(base, 4) \
	HE8_SSE(base, 5) \
	HE8_SSE(base, 6) \
	HE8_SSE(base, 7)

#define TM8_SSE(base, r) \
	BCAST_SSE(((r)*const_bps-1)(base), X0) \
	PADDUSB	X3, X0 \
	PSUBUSB	X4, X0 \
	MOVQ	X0, ((r)*const_bps)(base)

#define TM8x8_SSE(base) \
	MOVQ	(-const_bps)(base), X1 \
	BCAST_SSE((-const_bps-1)(base), X2) \
	MOVO	X1, X3 \
	PSUBUSB	X2, X3 \
	MOVO	X2, X4 \
	PSUBUSB	X1, X4 \
	TM8_SSE(base, 0) \
	TM8_SSE(base, 1) \
	TM8_SSE(base, 2) \
	TM8_SSE(base, 3) \
	TM8_SSE(base, 4) \
	TM8_SSE(base, 5) \
	TM8_SSE(base, 6) \
	TM8_SSE(base, 7)

#define SUMTOP8_SSE(base, dst) \
	MOVQ	(-const_bps)(base), X0 \
	PSADBW	X15, X0 \
	MOVL	X0, dst

#define FILL8_SSE(base, v) \
	MOVL	v, X0 \
	PSHUFB	X15, X0 \
	ST8x8_SSE(base)

// UVFULL_SSE(o0, o1, m, base, r) adds the residuals of the two blocks at
// SI+o0, SI+o1 (masked by (m)) to rows r..r+3 of the 8x8 block at base
// (falling back if a coefficient is out of range); UVDC_SSE the DC-only
// residuals. Clobbers X0-X8.
#define UVFULL_SSE(o0, o1, m, base, r) \
	LOAD2_SSE(o0, o1, m, m) \
	PTEST	reconRange<>(SB), X6 \
	JNZ	uvsseFallback \
	IDCT2_SSE \
	TRANSPOSE_SSE \
	ADD8x2_SSE(X0, X1, base, (r)*const_bps) \
	ADD8x2_SSE(X2, X3, base, (r)*const_bps+2*const_bps)

#define UVDC_SSE(o0, o1, m, base, r) \
	DC2_SSE(o0, o1, m, m) \
	PSHUFB	reconDCSpread<>(SB), X0 \
	ADD8x2_SSE(X0, X0, base, (r)*const_bps) \
	ADD8x2_SSE(X0, X0, base, (r)*const_bps+2*const_bps)

// func reconUVSSE41(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int) bool
//
// reconUVAVX2 with the residuals computed and added per plane two blocks
// at a time. Registers as in reconUVAVX2.
TEXT ·reconUVSSE41(SB), NOSPLIT, $0-33
	MOVQ	b+0(FP), DI
	MOVQ	coeffs+8(FP), SI
	MOVL	codes+16(FP), BX
	MOVQ	mode+24(FP), AX
	LEAQ	const_uOff(DI), R8
	LEAQ	const_vOff(DI), R9
	PXOR	X15, X15
	CMPQ	AX, $1
	JEQ	uvsseTM
	CMPQ	AX, $2
	JEQ	uvsseVE
	CMPQ	AX, $3
	JEQ	uvsseHE
	CMPQ	AX, $0
	JEQ	uvsseDC
	CMPQ	AX, $4
	JEQ	uvsseDCNoTop
	CMPQ	AX, $5
	JEQ	uvsseDCNoLeft
	MOVL	$0x80, AX
	FILL8_SSE(R8, AX)
	FILL8_SSE(R9, AX)
	JMP	uvsseRes

uvsseDC:
	SUMLEFT8(R8, CX)
	SUMTOP8_SSE(R8, AX)
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX
	FILL8_SSE(R8, AX)
	SUMLEFT8(R9, CX)
	SUMTOP8_SSE(R9, AX)
	LEAL	8(AX)(CX*1), AX
	SHRL	$4, AX
	FILL8_SSE(R9, AX)
	JMP	uvsseRes

uvsseDCNoTop:
	SUMLEFT8(R8, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_SSE(R8, AX)
	SUMLEFT8(R9, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_SSE(R9, AX)
	JMP	uvsseRes

uvsseDCNoLeft:
	SUMTOP8_SSE(R8, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_SSE(R8, AX)
	SUMTOP8_SSE(R9, AX)
	ADDL	$4, AX
	SHRL	$3, AX
	FILL8_SSE(R9, AX)
	JMP	uvsseRes

uvsseVE:
	MOVQ	(-const_bps)(R8), X0
	ST8x8_SSE(R8)
	MOVQ	(-const_bps)(R9), X0
	ST8x8_SSE(R9)
	JMP	uvsseRes

uvsseHE:
	HE8x8_SSE(R8)
	HE8x8_SSE(R9)
	JMP	uvsseRes

uvsseTM:
	TM8x8_SSE(R8)
	TM8x8_SSE(R9)

uvsseRes:
	TESTL	BX, BX
	JZ	uvsseDone
	ADDQ	$(16*16*2), SI
	LEAQ	reconMask<>(SB), R12
	MOVL	BX, AX
	ANDL	$3, AX
	SHLL	$5, AX
	ADDQ	R12, AX
	MOVL	BX, CX
	SHRL	$8, CX
	ANDL	$3, CX
	SHLL	$5, CX
	ADDQ	R12, CX
	TESTL	$0xaaaa, BX
	JNZ	uvsseFull
	UVDC_SSE(0, 32, AX, R8, 0)
	UVDC_SSE(64, 96, AX, R8, 4)
	UVDC_SSE(128, 160, CX, R9, 0)
	UVDC_SSE(192, 224, CX, R9, 4)
	JMP	uvsseDone

uvsseFull:
	UVFULL_SSE(0, 32, AX, R8, 0)
	UVFULL_SSE(64, 96, AX, R8, 4)
	UVFULL_SSE(128, 160, CX, R9, 0)
	UVFULL_SSE(192, 224, CX, R9, 4)

uvsseDone:
	MOVB	$1, ret+32(FP)
	RET

uvsseFallback:
	MOVB	$0, ret+32(FP)
	RET

// --- inverse WHT (SSE4.1) -----------------------------------------------------------

// func transformWHTSSE41(in *[16]int16, out *[384]int16)
//
// Mirrors transformWHTRef (libwebp's TransformWHT): a vertical pass on the
// four input rows (lane = column, 32-bit), a 4x4 transpose, a horizontal
// pass with the +3 rounder, then >> 3; the low 16 bits of each 32-bit lane
// (PEXTRW) are int16(x >> 3) for any sign. In 32-bit lanes, so any int16
// input gives the reference's result. Output (row i, column k) goes to the
// DC of block 4*i + k, out[(4*i+k)*16].
// Registers: SI in, DI out, AX scratch, X0-X7.
TEXT ·transformWHTSSE41(SB), NOSPLIT, $0-16
	MOVQ	in+0(FP), SI
	MOVQ	out+8(FP), DI
	PMOVSXWD	0(SI), X0  // in[0..3]
	PMOVSXWD	8(SI), X1  // in[4..7]
	PMOVSXWD	16(SI), X2 // in[8..11]
	PMOVSXWD	24(SI), X3 // in[12..15]
	MOVO	X0, X4
	PADDL	X3, X4 // a0 = in0 + in12
	MOVO	X1, X5
	PADDL	X2, X5 // a1 = in4 + in8
	PSUBL	X2, X1 // a2 = in4 - in8
	PSUBL	X3, X0 // a3 = in0 - in12
	MOVO	X4, X2
	PADDL	X5, X2 // tmp[0..3] = a0 + a1
	PSUBL	X5, X4 // tmp[8..11] = a0 - a1
	MOVO	X0, X3
	PADDL	X1, X3 // tmp[4..7] = a3 + a2
	PSUBL	X1, X0 // tmp[12..15] = a3 - a2

	// Transpose (rows X2, X3, X4, X0) so that lane i holds row i of tmp.
	MOVO	X2, X5
	PUNPCKLLQ	X3, X5 // t00 t10 t01 t11
	PUNPCKHLQ	X3, X2 // t02 t12 t03 t13
	MOVO	X4, X6
	PUNPCKLLQ	X0, X6 // t20 t30 t21 t31
	PUNPCKHLQ	X0, X4 // t22 t32 t23 t33
	MOVO	X5, X0
	PUNPCKLQDQ	X6, X0 // tmp[4i+0]
	PUNPCKHQDQ	X6, X5 // tmp[4i+1]
	MOVO	X2, X1
	PUNPCKLQDQ	X4, X1 // tmp[4i+2]
	PUNPCKHQDQ	X4, X2 // tmp[4i+3]

	MOVL	$3, AX
	MOVQ	AX, X7
	PSHUFD	$0, X7, X7
	PADDL	X7, X0 // dc = tmp[4i] + 3
	MOVO	X0, X3
	PADDL	X2, X3 // a0 = dc + tmp[4i+3]
	PSUBL	X2, X0 // a3 = dc - tmp[4i+3]
	MOVO	X5, X4
	PADDL	X1, X4 // a1 = tmp[4i+1] + tmp[4i+2]
	PSUBL	X1, X5 // a2 = tmp[4i+1] - tmp[4i+2]
	MOVO	X3, X1
	PADDL	X4, X1 // column 0: a0 + a1
	PSUBL	X4, X3 // column 2: a0 - a1
	MOVO	X0, X2
	PADDL	X5, X2 // column 1: a3 + a2
	PSUBL	X5, X0 // column 3: a3 - a2
	PSRAL	$3, X1
	PSRAL	$3, X2
	PSRAL	$3, X3
	PSRAL	$3, X0

	// Lane i of column k goes to out[(4i + k)*16], byte offset 128i + 32k.
	PEXTRW	$0, X1, AX
	MOVW	AX, 0(DI)
	PEXTRW	$0, X2, AX
	MOVW	AX, 32(DI)
	PEXTRW	$0, X3, AX
	MOVW	AX, 64(DI)
	PEXTRW	$0, X0, AX
	MOVW	AX, 96(DI)
	PEXTRW	$2, X1, AX
	MOVW	AX, 128(DI)
	PEXTRW	$2, X2, AX
	MOVW	AX, 160(DI)
	PEXTRW	$2, X3, AX
	MOVW	AX, 192(DI)
	PEXTRW	$2, X0, AX
	MOVW	AX, 224(DI)
	PEXTRW	$4, X1, AX
	MOVW	AX, 256(DI)
	PEXTRW	$4, X2, AX
	MOVW	AX, 288(DI)
	PEXTRW	$4, X3, AX
	MOVW	AX, 320(DI)
	PEXTRW	$4, X0, AX
	MOVW	AX, 352(DI)
	PEXTRW	$6, X1, AX
	MOVW	AX, 384(DI)
	PEXTRW	$6, X2, AX
	MOVW	AX, 416(DI)
	PEXTRW	$6, X3, AX
	MOVW	AX, 448(DI)
	PEXTRW	$6, X0, AX
	MOVW	AX, 480(DI)
	RET

// --- edge setup and row cache copy (SSE2) -------------------------------------------

// ROT4(row, base, from) copies the 4 bytes at columns from..from+3 of row
// (relative to base) to columns -4..-1.
#define ROT4(row, base, from) \
	MOVL	((base)+(row)*const_bps+(from))(DI), AX \
	MOVL	AX, ((base)+(row)*const_bps-4)(DI)

// func loadEdgesSSE2(b *[yuvSize]byte, top *topSamples, tr, flags uint32)
//
// Mirrors loadEdgesRef (same flags) with straight-line loads and stores at
// constant offsets. Registers: DI b, SI top, DX tr, BX flags, AX/CX data.
TEXT ·loadEdgesSSE2(SB), NOSPLIT, $0-24
	MOVQ	b+0(FP), DI
	MOVQ	top+8(FP), SI
	MOVL	tr+16(FP), DX
	MOVL	flags+20(FP), BX
	TESTL	$1, BX // edgeRotate
	JZ	ledgeNoRotate
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

ledgeNoRotate:
	TESTL	$2, BX // edgeTop
	JZ	ledgeNoTop
	MOVOU	(topSamples_y)(SI), X0
	MOVQ	(topSamples_u)(SI), AX
	MOVQ	(topSamples_v)(SI), CX
	MOVOU	X0, (const_yOff-const_bps)(DI)
	MOVQ	AX, (const_uOff-const_bps)(DI)
	MOVQ	CX, (const_vOff-const_bps)(DI)

ledgeNoTop:
	TESTL	$8, BX // edgeI4
	JZ	ledgeDone
	TESTL	$4, BX // edgeTopRight
	JZ	ledgeNoTopRight
	MOVL	DX, (const_yOff-const_bps+16)(DI)

ledgeNoTopRight:
	MOVL	(const_yOff-const_bps+16)(DI), AX
	MOVL	AX, (const_yOff+3*const_bps+16)(DI)
	MOVL	AX, (const_yOff+7*const_bps+16)(DI)
	MOVL	AX, (const_yOff+11*const_bps+16)(DI)

ledgeDone:
	RET

// func storeMBRowsSSE2(b *[yuvSize]byte, top *topSamples, cy, cu, cv *byte, ys, uvs int)
//
// Mirrors storeMBRowsRef; the Go wrapper has checked that cy has
// 15*ys+16 bytes and cu, cv 7*uvs+8. All rows are loaded before the first
// store, as the reference reads b only. Registers: DI b, SI top (may be
// nil), AX cy, BX cu, CX cv, DX ys, R8 uvs, R9-R13 scratch, X0-X15 rows.
TEXT ·storeMBRowsSSE2(SB), NOSPLIT, $0-56
	MOVQ	b+0(FP), DI
	MOVQ	top+8(FP), SI
	MOVQ	cy+16(FP), AX
	MOVQ	cu+24(FP), BX
	MOVQ	cv+32(FP), CX
	MOVQ	ys+40(FP), DX
	MOVQ	uvs+48(FP), R8
	TESTQ	SI, SI
	JZ	smbNoStash
	MOVOU	(const_yOff+15*const_bps)(DI), X0
	MOVQ	(const_uOff+7*const_bps)(DI), R9
	MOVQ	(const_vOff+7*const_bps)(DI), R10
	MOVOU	X0, (topSamples_y)(SI)
	MOVQ	R9, (topSamples_u)(SI)
	MOVQ	R10, (topSamples_v)(SI)

smbNoStash:
	MOVOU	(const_yOff+0*const_bps)(DI), X0
	MOVOU	(const_yOff+1*const_bps)(DI), X1
	MOVOU	(const_yOff+2*const_bps)(DI), X2
	MOVOU	(const_yOff+3*const_bps)(DI), X3
	MOVOU	(const_yOff+4*const_bps)(DI), X4
	MOVOU	(const_yOff+5*const_bps)(DI), X5
	MOVOU	(const_yOff+6*const_bps)(DI), X6
	MOVOU	(const_yOff+7*const_bps)(DI), X7
	MOVOU	(const_yOff+8*const_bps)(DI), X8
	MOVOU	(const_yOff+9*const_bps)(DI), X9
	MOVOU	(const_yOff+10*const_bps)(DI), X10
	MOVOU	(const_yOff+11*const_bps)(DI), X11
	MOVOU	(const_yOff+12*const_bps)(DI), X12
	MOVOU	(const_yOff+13*const_bps)(DI), X13
	MOVOU	(const_yOff+14*const_bps)(DI), X14
	MOVOU	(const_yOff+15*const_bps)(DI), X15
	MOVOU	X0, (AX)
	ADDQ	DX, AX
	MOVOU	X1, (AX)
	ADDQ	DX, AX
	MOVOU	X2, (AX)
	ADDQ	DX, AX
	MOVOU	X3, (AX)
	ADDQ	DX, AX
	MOVOU	X4, (AX)
	ADDQ	DX, AX
	MOVOU	X5, (AX)
	ADDQ	DX, AX
	MOVOU	X6, (AX)
	ADDQ	DX, AX
	MOVOU	X7, (AX)
	ADDQ	DX, AX
	MOVOU	X8, (AX)
	ADDQ	DX, AX
	MOVOU	X9, (AX)
	ADDQ	DX, AX
	MOVOU	X10, (AX)
	ADDQ	DX, AX
	MOVOU	X11, (AX)
	ADDQ	DX, AX
	MOVOU	X12, (AX)
	ADDQ	DX, AX
	MOVOU	X13, (AX)
	ADDQ	DX, AX
	MOVOU	X14, (AX)
	ADDQ	DX, AX
	MOVOU	X15, (AX)

	// U row j and V row j are 16 bytes apart: one 16-byte load holds U row
	// j and the 8 bytes after it, so load the halves separately.
	MOVQ	(const_uOff+0*const_bps)(DI), X0
	MOVQ	(const_uOff+1*const_bps)(DI), X1
	MOVQ	(const_uOff+2*const_bps)(DI), X2
	MOVQ	(const_uOff+3*const_bps)(DI), X3
	MOVQ	(const_uOff+4*const_bps)(DI), X4
	MOVQ	(const_uOff+5*const_bps)(DI), X5
	MOVQ	(const_uOff+6*const_bps)(DI), X6
	MOVQ	(const_uOff+7*const_bps)(DI), X7
	MOVQ	(const_vOff+0*const_bps)(DI), X8
	MOVQ	(const_vOff+1*const_bps)(DI), X9
	MOVQ	(const_vOff+2*const_bps)(DI), X10
	MOVQ	(const_vOff+3*const_bps)(DI), X11
	MOVQ	(const_vOff+4*const_bps)(DI), X12
	MOVQ	(const_vOff+5*const_bps)(DI), X13
	MOVQ	(const_vOff+6*const_bps)(DI), X14
	MOVQ	(const_vOff+7*const_bps)(DI), X15
	MOVQ	X0, (BX)
	MOVQ	X8, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X1, (BX)
	MOVQ	X9, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X2, (BX)
	MOVQ	X10, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X3, (BX)
	MOVQ	X11, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X4, (BX)
	MOVQ	X12, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X5, (BX)
	MOVQ	X13, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X6, (BX)
	MOVQ	X14, (CX)
	ADDQ	R8, BX
	ADDQ	R8, CX
	MOVQ	X7, (BX)
	MOVQ	X15, (CX)
	RET
