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

// AVX2 kernels of the VP8 encoder (see dsp_amd64.go for the Go wrappers
// and dsp_amd64.s for the SSE4.1 kernels they extend). They use the VEX
// encoding only and end with VZEROUPPER. Every kernel is bit-exact with
// the Go code of dsp.go; the arithmetic is the one of the SSE4.1 kernels,
// on 256-bit registers whose two 128-bit lanes hold two block pairs (or
// the two halves of one block for the quantizer). A pair table with an odd
// number of pairs ends with its last pair in both lanes, of which only
// lane 0 is stored.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/enc_sse2.c (the
// SSE2 transforms, quantizer and distortion metrics, here on 256-bit
// registers).

#include "go_asm.h"
#include "textflag.h"

// 32-byte copies of the fTransform constants of dsp_amd64.s (memory
// operands of 256-bit instructions).
DATA ·yk88p+0(SB)/8, $0x0008000800080008
DATA ·yk88p+8(SB)/8, $0x0008000800080008
DATA ·yk88p+16(SB)/8, $0x0008000800080008
DATA ·yk88p+24(SB)/8, $0x0008000800080008
GLOBL ·yk88p(SB), RODATA|NOPTR, $32
DATA ·yk88m+0(SB)/8, $0xfff80008fff80008
DATA ·yk88m+8(SB)/8, $0xfff80008fff80008
DATA ·yk88m+16(SB)/8, $0xfff80008fff80008
DATA ·yk88m+24(SB)/8, $0xfff80008fff80008
GLOBL ·yk88m(SB), RODATA|NOPTR, $32
DATA ·yk5352p+0(SB)/8, $0x08a914e808a914e8
DATA ·yk5352p+8(SB)/8, $0x08a914e808a914e8
DATA ·yk5352p+16(SB)/8, $0x08a914e808a914e8
DATA ·yk5352p+24(SB)/8, $0x08a914e808a914e8
GLOBL ·yk5352p(SB), RODATA|NOPTR, $32
DATA ·yk5352m+0(SB)/8, $0xeb1808a9eb1808a9
DATA ·yk5352m+8(SB)/8, $0xeb1808a9eb1808a9
DATA ·yk5352m+16(SB)/8, $0xeb1808a9eb1808a9
DATA ·yk5352m+24(SB)/8, $0xeb1808a9eb1808a9
GLOBL ·yk5352m(SB), RODATA|NOPTR, $32
DATA ·yk1812+0(SB)/8, $0x0000071400000714
DATA ·yk1812+8(SB)/8, $0x0000071400000714
DATA ·yk1812+16(SB)/8, $0x0000071400000714
DATA ·yk1812+24(SB)/8, $0x0000071400000714
GLOBL ·yk1812(SB), RODATA|NOPTR, $32
DATA ·yk937+0(SB)/8, $0x000003a9000003a9
DATA ·yk937+8(SB)/8, $0x000003a9000003a9
DATA ·yk937+16(SB)/8, $0x000003a9000003a9
DATA ·yk937+24(SB)/8, $0x000003a9000003a9
GLOBL ·yk937(SB), RODATA|NOPTR, $32
DATA ·yk2217p+0(SB)/8, $0x14e808a914e808a9
DATA ·yk2217p+8(SB)/8, $0x14e808a914e808a9
DATA ·yk2217p+16(SB)/8, $0x14e808a914e808a9
DATA ·yk2217p+24(SB)/8, $0x14e808a914e808a9
GLOBL ·yk2217p(SB), RODATA|NOPTR, $32
DATA ·yk2217m+0(SB)/8, $0x08a9eb1808a9eb18
DATA ·yk2217m+8(SB)/8, $0x08a9eb1808a9eb18
DATA ·yk2217m+16(SB)/8, $0x08a9eb1808a9eb18
DATA ·yk2217m+24(SB)/8, $0x08a9eb1808a9eb18
GLOBL ·yk2217m(SB), RODATA|NOPTR, $32
DATA ·yk12000+0(SB)/8, $0x00012ee000012ee0
DATA ·yk12000+8(SB)/8, $0x00012ee000012ee0
DATA ·yk12000+16(SB)/8, $0x00012ee000012ee0
DATA ·yk12000+24(SB)/8, $0x00012ee000012ee0
GLOBL ·yk12000(SB), RODATA|NOPTR, $32
DATA ·yk51000+0(SB)/8, $0x0000c7380000c738
DATA ·yk51000+8(SB)/8, $0x0000c7380000c738
DATA ·yk51000+16(SB)/8, $0x0000c7380000c738
DATA ·yk51000+24(SB)/8, $0x0000c7380000c738
GLOBL ·yk51000(SB), RODATA|NOPTR, $32
DATA ·yk7+0(SB)/8, $0x0007000700070007
DATA ·yk7+8(SB)/8, $0x0007000700070007
DATA ·yk7+16(SB)/8, $0x0007000700070007
DATA ·yk7+24(SB)/8, $0x0007000700070007
GLOBL ·yk7(SB), RODATA|NOPTR, $32

// FT_PASS1_Y and FT_PASS2_Y are FT_PASS1 and FT_PASS2 of dsp_amd64.s in
// three-operand form, on one block per 128-bit lane.
#define FT_PASS1_Y(I01, I23, O01, O32, T0, T1) \
	VPSHUFHW	$0xb1, I01, I01 \
	VPSHUFHW	$0xb1, I23, I23 \
	VPUNPCKLQDQ	I23, I01, T0 \
	VPUNPCKHQDQ	I23, I01, I01 \
	VPADDW	I01, T0, T1 \
	VPSUBW	I01, T0, T0 \
	VPMADDWD	·yk88p(SB), T1, I01 \
	VPMADDWD	·yk88m(SB), T1, T1 \
	VPACKSSDW	T1, I01, I01 \
	VPMADDWD	·yk5352p(SB), T0, I23 \
	VPMADDWD	·yk5352m(SB), T0, T0 \
	VPADDD	·yk1812(SB), I23, I23 \
	VPADDD	·yk937(SB), T0, T0 \
	VPSRAD	$9, I23, I23 \
	VPSRAD	$9, T0, T0 \
	VPACKSSDW	T0, I23, I23 \
	VPUNPCKLWD	I23, I01, T0 \
	VPUNPCKHWD	I23, I01, T1 \
	VPUNPCKLDQ	T1, T0, O01 \
	VPUNPCKHDQ	T1, T0, T0 \
	VPSHUFD	$0x4e, T0, O32

#define FT_PASS2_Y(V01, V32, T0, T1, T2, T3) \
	VPSUBW	V32, V01, T0 \
	VPADDW	V32, V01, V01 \
	VPUNPCKHQDQ	T0, T0, T1 \
	VPUNPCKLWD	T0, T1, T1 \
	VPMADDWD	·yk2217p(SB), T1, T2 \
	VPMADDWD	·yk2217m(SB), T1, T1 \
	VPADDD	·yk12000(SB), T2, T2 \
	VPADDD	·yk51000(SB), T1, T1 \
	VPSRAD	$16, T2, T2 \
	VPSRAD	$16, T1, T1 \
	VPACKSSDW	T2, T2, T2 \
	VPACKSSDW	T1, T1, T1 \
	VPXOR	T3, T3, T3 \
	VPCMPEQW	T3, T0, T0 \
	VPADDW	T0, T2, T2 \
	VPUNPCKHQDQ	V01, V01, T3 \
	VPADDW	·yk7(SB), V01, V01 \
	VPSUBW	T3, V01, V32 \
	VPADDW	T3, V01, V01 \
	VPSRAW	$4, V01, V01 \
	VPSRAW	$4, V32, V32 \
	VPUNPCKLQDQ	T2, V01, V01 \
	VPUNPCKLQDQ	T1, V32, V32

// ROW2_Y loads row off of two pairs as 16 words (pair k from P0, pair k+1
// from P1, one per lane).
#define ROW2_Y(P0, P1, off, X, Y) \
	VMOVQ	off(P0), X \
	VPINSRQ	$1, off(P1), X, X \
	VPMOVZXBW	X, Y

// FT_BODY_Y transforms the two pairs whose src rows start at R9 (pair k)
// and R11 (pair k+1) and ref rows at R10 and R12: on return Y6/Y7 hold
// out[0..7]/out[8..15] of the left blocks and Y1/Y3 those of the right
// blocks, pair k in lane 0 and pair k+1 in lane 1.
#define FT_BODY_Y \
	ROW2_Y(R9, R11, 0, X0, Y0) \
	ROW2_Y(R10, R12, 0, X4, Y4) \
	VPSUBW	Y4, Y0, Y0 \
	ROW2_Y(R9, R11, 32, X1, Y1) \
	ROW2_Y(R10, R12, 32, X4, Y4) \
	VPSUBW	Y4, Y1, Y1 \
	ROW2_Y(R9, R11, 64, X2, Y2) \
	ROW2_Y(R10, R12, 64, X4, Y4) \
	VPSUBW	Y4, Y2, Y2 \
	ROW2_Y(R9, R11, 96, X3, Y3) \
	ROW2_Y(R10, R12, 96, X4, Y4) \
	VPSUBW	Y4, Y3, Y3 \
	VPUNPCKLDQ	Y1, Y0, Y4 \
	VPUNPCKHDQ	Y1, Y0, Y0 \
	VPUNPCKLDQ	Y3, Y2, Y5 \
	VPUNPCKHDQ	Y3, Y2, Y2 \
	FT_PASS1_Y(Y4, Y5, Y6, Y7, Y8, Y9) \
	FT_PASS1_Y(Y0, Y2, Y1, Y3, Y8, Y9) \
	FT_PASS2_Y(Y6, Y7, Y8, Y9, Y10, Y11) \
	FT_PASS2_Y(Y1, Y3, Y8, Y9, Y10, Y11)

// ---------------------------------------------------------------------------
// func fTransformPairsAVX2(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)
//
// fTransformPairsSSE41 two pairs at a time: the rows of pair k and pair
// k+1 are packed into the two lanes (VMOVQ, VPINSRQ, VPMOVZXBW), so one
// pass of FT_PASS1_Y/FT_PASS2_Y transforms four blocks; libwebp's
// FTransform2_SSE2 on 256-bit registers.
//
// Registers: SI src, DI ref, R8 table, CX pairs left, DX out, R9/R10 and
// R11/R12 the src/ref rows of the two pairs.
TEXT ·fTransformPairsAVX2(SB), NOSPLIT, $0-40
	MOVQ	src+0(FP), SI
	MOVQ	ref+8(FP), DI
	MOVQ	off+16(FP), R8
	MOVQ	n+24(FP), CX
	MOVQ	out+32(FP), DX
	TESTQ	CX, CX
	JZ	fydone

fyloop:
	MOVWLZX	(R8), AX
	MOVWLZX	2(R8), BX
	LEAQ	(SI)(AX*1), R9
	LEAQ	(DI)(BX*1), R10
	CMPQ	CX, $1
	JEQ	fyone
	MOVWLZX	4(R8), AX
	MOVWLZX	6(R8), BX
	LEAQ	(SI)(AX*1), R11
	LEAQ	(DI)(BX*1), R12
	FT_BODY_Y
	VMOVDQU	X6, (DX)
	VMOVDQU	X7, 16(DX)
	VMOVDQU	X1, 32(DX)
	VMOVDQU	X3, 48(DX)
	VEXTRACTI128	$1, Y6, 64(DX)
	VEXTRACTI128	$1, Y7, 80(DX)
	VEXTRACTI128	$1, Y1, 96(DX)
	VEXTRACTI128	$1, Y3, 112(DX)
	ADDQ	$8, R8
	ADDQ	$128, DX
	SUBQ	$2, CX
	JNZ	fyloop
	JMP	fydone

fyone:
	MOVQ	R9, R11
	MOVQ	R10, R12
	FT_BODY_Y
	VMOVDQU	X6, (DX)
	VMOVDQU	X7, 16(DX)
	VMOVDQU	X1, 32(DX)
	VMOVDQU	X3, 48(DX)

fydone:
	VZEROUPPER
	RET

// Zigzag shuffles of quantizeBlocksAVX2: [zzLo | zzHi] on the levels, and
// [zz8 | zz7] on the levels with their lanes swapped (see the SSE4.1
// constants of dsp_amd64.s).
DATA ·yzzA+0(SB)/8, $0x8080090803020100
DATA ·yzzA+8(SB)/8, $0x0d0c070605040b0a
DATA ·yzzA+16(SB)/8, $0x05040b0a09080302
DATA ·yzzA+24(SB)/8, $0x0f0e0d0c07068080
GLOBL ·yzzA(SB), RODATA|NOPTR, $32
DATA ·yzzB+0(SB)/8, $0x0100808080808080
DATA ·yzzB+8(SB)/8, $0x8080808080808080
DATA ·yzzB+16(SB)/8, $0x8080808080808080
DATA ·yzzB+24(SB)/8, $0x8080808080800f0e
GLOBL ·yzzB(SB), RODATA|NOPTR, $32
DATA ·yk2047+0(SB)/8, $0x07ff07ff07ff07ff
DATA ·yk2047+8(SB)/8, $0x07ff07ff07ff07ff
DATA ·yk2047+16(SB)/8, $0x07ff07ff07ff07ff
DATA ·yk2047+24(SB)/8, $0x07ff07ff07ff07ff
GLOBL ·yk2047(SB), RODATA|NOPTR, $32

// ---------------------------------------------------------------------------
// func quantizeBlocksAVX2(in, out *[16]int16, n int, m *matrix, info *uint16) uint32
//
// quantizeBlocksSSE41 with one block per 256-bit register. The 32-bit
// products (VPUNPCKLWD/VPUNPCKHWD within lanes) come out as coefficients
// [0..3 | 8..11] and [4..7 | 12..15], so the bias is permuted the same way
// (VPERM2I128), and VPACKUSDW restores the natural order. The zigzag
// order takes two in-lane shuffles, one of them on the lane-swapped levels
// (VPERMQ). VPMOVMSKB gives two mask bits per level: with m the mask of
// the non-zero levels, the last index plus one is (BSR(m) + 1) >> 1 and
// the AC count POPCNT(m & ^3) >> 1.
//
// Registers: DI in, SI out, CX blocks left, DX matrix, R8 info, R9 nz
// mask, R10 current bit, R11 -1, Y8 iq, Y9 sharpen, Y10 q, Y11/Y12 bias,
// Y13 2047, Y14/Y15 zigzag shuffles.
TEXT ·quantizeBlocksAVX2(SB), NOSPLIT, $0-44
	MOVQ	in+0(FP), DI
	MOVQ	out+8(FP), SI
	MOVQ	n+16(FP), CX
	MOVQ	m+24(FP), DX
	MOVQ	info+32(FP), R8
	VMOVDQU	matrix_iq16(DX), Y8
	VMOVDQU	matrix_sharpen16(DX), Y9
	VMOVDQU	matrix_q16(DX), Y10
	VMOVDQU	matrix_bias(DX), Y0
	VMOVDQU	matrix_bias+32(DX), Y1
	VPERM2I128	$0x20, Y1, Y0, Y11
	VPERM2I128	$0x31, Y1, Y0, Y12
	VMOVDQU	·yk2047(SB), Y13
	VMOVDQU	·yzzA(SB), Y14
	VMOVDQU	·yzzB(SB), Y15
	XORL	R9, R9
	MOVL	$1, R10
	MOVL	$-1, R11
	TESTQ	CX, CX
	JZ	qydone

qyloop:
	VMOVDQU	(DI), Y0
	VPSRAW	$15, Y0, Y1
	VPABSW	Y0, Y0
	VPADDW	Y9, Y0, Y0
	VPMULHUW	Y8, Y0, Y2
	VPMULLW	Y8, Y0, Y0
	VPUNPCKLWD	Y2, Y0, Y3
	VPUNPCKHWD	Y2, Y0, Y4
	VPADDD	Y11, Y3, Y3
	VPADDD	Y12, Y4, Y4
	VPSRLD	$17, Y3, Y3
	VPSRLD	$17, Y4, Y4
	VPACKUSDW	Y4, Y3, Y0
	VPMINSW	Y13, Y0, Y0
	VPXOR	Y1, Y0, Y0
	VPSUBW	Y1, Y0, Y0 // levels
	VPMULLW	Y10, Y0, Y2
	VMOVDQU	Y2, (DI)
	VPERMQ	$0x4e, Y0, Y3
	VPSHUFB	Y14, Y0, Y0
	VPSHUFB	Y15, Y3, Y3
	VPOR	Y3, Y0, Y0 // zigzag order
	VMOVDQU	Y0, (SI)
	VPXOR	Y2, Y2, Y2
	VPCMPEQW	Y2, Y0, Y0
	VPMOVMSKB	Y0, AX
	XORL	BX, BX
	XORL	$-1, AX // non-zero levels, two bits each
	CMOVLNE	R10, BX
	ORL	BX, R9
	TESTQ	R8, R8
	JZ	qynext
	BSRL	AX, BX
	CMOVLEQ	R11, BX
	INCL	BX
	SHRL	$1, BX // last + 1
	ANDL	$-4, AX
	POPCNTL	AX, AX
	SHLL	$7, AX // (count / 2) << 8
	ORL	BX, AX
	MOVW	AX, (R8)
	ADDQ	$2, R8

qynext:
	ADDQ	$32, DI
	ADDQ	$32, SI
	SHLL	$1, R10
	DECQ	CX
	JNZ	qyloop

qydone:
	MOVL	R9, ret+40(FP)
	VZEROUPPER
	RET

// TRANSPOSE_2_4x4_Y is TRANSPOSE_2_4x4 of dsp_amd64.s in three-operand
// form (two pairs, one per lane); I0..I3 are destroyed.
#define TRANSPOSE_2_4x4_Y(I0, I1, I2, I3, O0, O1, O2, O3) \
	VPUNPCKLWD	I1, I0, O0 \
	VPUNPCKHWD	I1, I0, I0 \
	VPUNPCKLWD	I3, I2, O1 \
	VPUNPCKHWD	I3, I2, I2 \
	VPUNPCKLDQ	O1, O0, I1 \
	VPUNPCKHDQ	O1, O0, O0 \
	VPUNPCKLDQ	I2, I0, I3 \
	VPUNPCKHDQ	I2, I0, I0 \
	VPUNPCKLQDQ	I0, O0, O2 \
	VPUNPCKHQDQ	I0, O0, O3 \
	VPUNPCKLQDQ	I3, I1, O0 \
	VPUNPCKHQDQ	I3, I1, O1

// HADAMARD4_Y is HADAMARD4 of dsp_amd64.s in three-operand form.
#define HADAMARD4_Y(I0, I1, I2, I3, O0, O1, O2, O3) \
	VPADDW	I2, I0, O0 \
	VPSUBW	I2, I0, I0 \
	VPADDW	I3, I1, O1 \
	VPSUBW	I3, I1, I1 \
	VPSUBW	O1, O0, O3 \
	VPADDW	O1, O0, O0 \
	VPADDW	I1, I0, O1 \
	VPSUBW	I1, I0, O2

// ---------------------------------------------------------------------------
// func tTransformPairsAVX2(a *byte, off *[8][2]uint16, n int, out *int32)
//
// tTransformPairsSSE41 two pairs at a time (one per lane). VPHADDD leaves
// [A, B, A, B] per lane; VPERMQ gathers pair k's [A, B] and pair k+1's
// into the low 16 bytes.
//
// Registers: SI a, R8 table, CX pairs left, DX out, R9/R11 the rows of
// the two pairs, Y12-Y15 weights.
TEXT ·tTransformPairsAVX2(SB), NOSPLIT, $0-32
	MOVQ	a+0(FP), SI
	MOVQ	off+8(FP), R8
	MOVQ	n+16(FP), CX
	MOVQ	out+24(FP), DX
	VBROADCASTI128	·ttW+0(SB), Y12
	VBROADCASTI128	·ttW+16(SB), Y13
	VBROADCASTI128	·ttW+32(SB), Y14
	VBROADCASTI128	·ttW+48(SB), Y15
	TESTQ	CX, CX
	JZ	tydone

tyloop:
	MOVWLZX	(R8), AX
	LEAQ	(SI)(AX*1), R9
	MOVQ	R9, R11
	CMPQ	CX, $1
	JEQ	tyrows
	MOVWLZX	4(R8), AX
	LEAQ	(SI)(AX*1), R11

tyrows:
	ROW2_Y(R9, R11, 0, X0, Y0)
	ROW2_Y(R9, R11, 32, X1, Y1)
	ROW2_Y(R9, R11, 64, X2, Y2)
	ROW2_Y(R9, R11, 96, X3, Y3)
	HADAMARD4_Y(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
	TRANSPOSE_2_4x4_Y(Y4, Y5, Y6, Y7, Y0, Y1, Y2, Y3)
	HADAMARD4_Y(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
	VPABSW	Y4, Y4
	VPABSW	Y5, Y5
	VPABSW	Y6, Y6
	VPABSW	Y7, Y7
	VPMADDWD	Y12, Y4, Y4
	VPMADDWD	Y13, Y5, Y5
	VPMADDWD	Y14, Y6, Y6
	VPMADDWD	Y15, Y7, Y7
	VPADDD	Y5, Y4, Y4
	VPADDD	Y7, Y6, Y6
	VPADDD	Y6, Y4, Y4
	VPHADDD	Y4, Y4, Y4
	CMPQ	CX, $1
	JEQ	tylast
	VPERMQ	$0x08, Y4, Y4
	VMOVDQU	X4, (DX)
	ADDQ	$8, R8
	ADDQ	$16, DX
	SUBQ	$2, CX
	JNZ	tyloop
	JMP	tydone

tylast:
	VMOVQ	X4, (DX)

tydone:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// func sse16xNAVX2(a, b *byte, rows int) int
//
// sse16xNSSE41 with one row per 256-bit register (VPMOVZXBW of 16 bytes),
// two accumulators.
//
// Registers: SI a, DI b, CX rows left, Y4/Y5 accumulators.
TEXT ·sse16xNAVX2(SB), NOSPLIT, $0-32
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	rows+16(FP), CX
	VPXOR	Y4, Y4, Y4
	VPXOR	Y5, Y5, Y5
	CMPQ	CX, $2
	JLT	eytail

eyloop:
	VPMOVZXBW	(SI), Y0
	VPMOVZXBW	(DI), Y1
	VPSUBW	Y1, Y0, Y0
	VPMADDWD	Y0, Y0, Y0
	VPADDD	Y0, Y4, Y4
	VPMOVZXBW	32(SI), Y2
	VPMOVZXBW	32(DI), Y3
	VPSUBW	Y3, Y2, Y2
	VPMADDWD	Y2, Y2, Y2
	VPADDD	Y2, Y5, Y5
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$2, CX
	CMPQ	CX, $2
	JGE	eyloop

eytail:
	TESTQ	CX, CX
	JZ	eydone
	VPMOVZXBW	(SI), Y0
	VPMOVZXBW	(DI), Y1
	VPSUBW	Y1, Y0, Y0
	VPMADDWD	Y0, Y0, Y0
	VPADDD	Y0, Y4, Y4

eydone:
	VPADDD	Y5, Y4, Y4
	VEXTRACTI128	$1, Y4, X5
	VPADDD	X5, X4, X4
	VPSHUFD	$0x4e, X4, X5
	VPADDD	X5, X4, X4
	VPSHUFD	$0xb1, X4, X5
	VPADDD	X5, X4, X4
	VMOVD	X4, AX
	MOVQ	AX, ret+24(FP)
	VZEROUPPER
	RET

// IT_PASS_Y is IT_PASS of dsp_amd64.s in three-operand form.
#define IT_PASS_Y(I0, I1, I2, I3, O0, O1, O2, O3, K1, K2) \
	VPADDW	I2, I0, O0 \
	VPSUBW	I2, I0, I0 \
	VPMULHW	K2, I1, O1 \
	VPMULHW	K1, I3, O2 \
	VPSUBW	O2, O1, O1 \
	VPSUBW	I3, I1, O2 \
	VPADDW	O2, O1, O1 \
	VPMULHW	K1, I1, O2 \
	VPMULHW	K2, I3, O3 \
	VPADDW	O3, O2, O2 \
	VPADDW	I3, I1, I1 \
	VPADDW	I1, O2, O2 \
	VPSUBW	O2, O0, O3 \
	VPADDW	O2, O0, O0 \
	VPSUBW	O1, I0, O2 \
	VPADDW	I0, O1, O1

// ITRANS_ROW_Y adds row R (pair k in lane 0, pair k+1 in lane 1) to the
// reference rows at off(R9) and off(R11), clips to bytes (VPACKUSWB) and
// stores lane 0 at off(R10); lane 1 is left in the low 8 bytes of XL.
#define ITRANS_ROW_Y(R, XR, off, XT, YT, XL) \
	ROW2_Y(R9, R11, off, XT, YT) \
	VPADDW	YT, R, R \
	VPACKUSWB	R, R, R \
	VMOVQ	XR, off(R10) \
	VEXTRACTI128	$1, R, XL

// ---------------------------------------------------------------------------
// func iTransformPairsAVX2(ref *byte, in *[16]int16, dst *byte, off *[2]uint16, n int) int
//
// iTransformPairsSSE41 two pairs at a time (pair k in lane 0, pair k+1 in
// lane 1), with the same contract: it stops before a pair holding a
// coefficient with |in| > 2047 and returns the number of pairs done. The
// bound is checked per lane; when only pair k+1 fails it (or pair k is the
// last one), the iteration stores pair k alone.
//
// Registers: SI ref, DX in, DI dst, R8 table, CX pairs, R13 pairs done,
// BX pairs of this iteration, R9/R10 and R11/R12 the ref/dst rows of the
// two pairs, Y12 k1, Y13 k2, Y14 4, Y15 2047.
TEXT ·iTransformPairsAVX2(SB), NOSPLIT, $0-48
	MOVQ	ref+0(FP), SI
	MOVQ	in+8(FP), DX
	MOVQ	dst+16(FP), DI
	MOVQ	off+24(FP), R8
	MOVQ	n+32(FP), CX
	XORQ	R13, R13
	VBROADCASTI128	·ik1(SB), Y12
	VBROADCASTI128	·ik2(SB), Y13
	VBROADCASTI128	·ik4(SB), Y14
	VMOVDQU	·yk2047(SB), Y15

iyloop:
	MOVQ	CX, BX
	SUBQ	R13, BX
	JZ	iydone
	VMOVDQU	(DX), X0   // pair k: A in[0..7]
	VMOVDQU	16(DX), X1 // A in[8..15]
	VMOVDQU	32(DX), X2 // B in[0..7]
	VMOVDQU	48(DX), X3 // B in[8..15]
	MOVQ	$64, AX    // offset of pair k+1 (0: pair k again)
	CMPQ	BX, $1
	JNE	iytwo
	XORQ	AX, AX

iytwo:
	VINSERTI128	$1, (DX)(AX*1), Y0, Y0
	VINSERTI128	$1, 16(DX)(AX*1), Y1, Y1
	VINSERTI128	$1, 32(DX)(AX*1), Y2, Y2
	VINSERTI128	$1, 48(DX)(AX*1), Y3, Y3
	VPABSW	Y0, Y4
	VPABSW	Y1, Y5
	VPMAXUW	Y5, Y4, Y4
	VPABSW	Y2, Y5
	VPMAXUW	Y5, Y4, Y4
	VPABSW	Y3, Y5
	VPMAXUW	Y5, Y4, Y4
	VPSUBUSW	Y15, Y4, Y4
	VPTEST	X4, X4
	JNZ	iydone // pair k fails the bound
	VEXTRACTI128	$1, Y4, X4
	VPTEST	X4, X4
	JZ	iyok
	MOVQ	$1, BX // pair k+1 fails it: pair k alone

iyok:
	CMPQ	BX, $2
	JLT	iypairs
	MOVQ	$2, BX

iypairs:
	VPUNPCKLQDQ	Y2, Y0, Y4 // in0
	VPUNPCKHQDQ	Y2, Y0, Y0 // in1
	VPUNPCKLQDQ	Y3, Y1, Y5 // in2
	VPUNPCKHQDQ	Y3, Y1, Y1 // in3
	IT_PASS_Y(Y4, Y0, Y5, Y1, Y6, Y7, Y8, Y9, Y12, Y13)
	TRANSPOSE_2_4x4_Y(Y6, Y7, Y8, Y9, Y0, Y1, Y2, Y3)
	VPADDW	Y14, Y0, Y0
	IT_PASS_Y(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y12, Y13)
	VPSRAW	$3, Y4, Y4
	VPSRAW	$3, Y5, Y5
	VPSRAW	$3, Y6, Y6
	VPSRAW	$3, Y7, Y7
	TRANSPOSE_2_4x4_Y(Y4, Y5, Y6, Y7, Y0, Y1, Y2, Y3)

	MOVWLZX	(R8), AX
	LEAQ	(SI)(AX*1), R9
	MOVWLZX	2(R8), AX
	LEAQ	(DI)(AX*1), R10
	MOVQ	R9, R11
	MOVQ	R10, R12
	CMPQ	BX, $1
	JEQ	iyrows
	MOVWLZX	4(R8), AX
	LEAQ	(SI)(AX*1), R11
	MOVWLZX	6(R8), AX
	LEAQ	(DI)(AX*1), R12

iyrows:
	ITRANS_ROW_Y(Y0, X0, 0, X4, Y4, X8)
	ITRANS_ROW_Y(Y1, X1, 32, X4, Y4, X9)
	ITRANS_ROW_Y(Y2, X2, 64, X4, Y4, X10)
	ITRANS_ROW_Y(Y3, X3, 96, X4, Y4, X11)
	CMPQ	BX, $1
	JEQ	iynext
	VMOVQ	X8, (R12)
	VMOVQ	X9, 32(R12)
	VMOVQ	X10, 64(R12)
	VMOVQ	X11, 96(R12)

iynext:
	ADDQ	BX, R13
	SHLQ	$2, BX
	ADDQ	BX, R8
	SHLQ	$4, BX
	ADDQ	BX, DX
	JMP	iyloop

iydone:
	MOVQ	R13, ret+40(FP)
	VZEROUPPER
	RET
