//go:build amd64 && !purego

// SSE4.1 and AVX2 kernels of the VP8 encoder (see dsp_amd64.go for the Go
// wrappers, which do all the bounds checking and pick the level, and
// dsp_batch.go for the reference semantics). Every kernel is bit-exact
// with the Go code of dsp.go. Algorithms adapted from libwebp's
// src/dsp/enc_sse2.c and enc_sse41.c (BSD-3-Clause).
//
// All 4x4 blocks use a row stride of bps = 32 bytes. Pair tables
// (pairTable.off) hold two uint16 offsets per pair of horizontally
// adjacent blocks: one 8-byte row load covers a row of both blocks.
//
// The SSE4.1 kernels use the legacy SSE encoding only (they run with the
// upper halves of the ymm registers clean: every AVX function, here and in
// the Go runtime, ends with VZEROUPPER); the AVX2 kernels use the VEX
// encoding only and end with VZEROUPPER. No kernel touches R14, R15 or BP.
// Memory operands of legacy SSE instructions are 16-byte RODATA constants
// (the linker aligns 16-byte and larger symbols to at least 16 bytes); all
// other loads and stores are unaligned (MOVOU, MOVQ, PMOVZXBW).

#include "go_asm.h"
#include "textflag.h"

// ---------------------------------------------------------------------------
// Constants of fTransform (FTransformPass1_SSE2 / FTransformPass2_SSE2).
// Word lanes are listed low to high in the comments.

// [8, 8, ...] and [8, -8, ...]: (a0 + a1) * 8 and (a0 - a1) * 8 by PMADDWD
DATA ·fk88p+0(SB)/8, $0x0008000800080008
DATA ·fk88p+8(SB)/8, $0x0008000800080008
GLOBL ·fk88p(SB), RODATA|NOPTR, $16
DATA ·fk88m+0(SB)/8, $0xfff80008fff80008
DATA ·fk88m+8(SB)/8, $0xfff80008fff80008
GLOBL ·fk88m(SB), RODATA|NOPTR, $16
// [5352, 2217, ...] on [a3, a2] pairs: a3*5352 + a2*2217
DATA ·fk5352p+0(SB)/8, $0x08a914e808a914e8
DATA ·fk5352p+8(SB)/8, $0x08a914e808a914e8
GLOBL ·fk5352p(SB), RODATA|NOPTR, $16
// [2217, -5352, ...] on [a3, a2] pairs: a3*2217 - a2*5352
DATA ·fk5352m+0(SB)/8, $0xeb1808a9eb1808a9
DATA ·fk5352m+8(SB)/8, $0xeb1808a9eb1808a9
GLOBL ·fk5352m(SB), RODATA|NOPTR, $16
DATA ·fk1812+0(SB)/8, $0x0000071400000714
DATA ·fk1812+8(SB)/8, $0x0000071400000714
GLOBL ·fk1812(SB), RODATA|NOPTR, $16
DATA ·fk937+0(SB)/8, $0x000003a9000003a9
DATA ·fk937+8(SB)/8, $0x000003a9000003a9
GLOBL ·fk937(SB), RODATA|NOPTR, $16
// [2217, 5352, ...] on [a2, a3] pairs: a2*2217 + a3*5352
DATA ·fk2217p+0(SB)/8, $0x14e808a914e808a9
DATA ·fk2217p+8(SB)/8, $0x14e808a914e808a9
GLOBL ·fk2217p(SB), RODATA|NOPTR, $16
// [-5352, 2217, ...] on [a2, a3] pairs: a3*2217 - a2*5352
DATA ·fk2217m+0(SB)/8, $0x08a9eb1808a9eb18
DATA ·fk2217m+8(SB)/8, $0x08a9eb1808a9eb18
GLOBL ·fk2217m(SB), RODATA|NOPTR, $16
// 12000 + (1 << 16) and 51000 (int32 lanes)
DATA ·fk12000+0(SB)/8, $0x00012ee000012ee0
DATA ·fk12000+8(SB)/8, $0x00012ee000012ee0
GLOBL ·fk12000(SB), RODATA|NOPTR, $16
DATA ·fk51000+0(SB)/8, $0x0000c7380000c738
DATA ·fk51000+8(SB)/8, $0x0000c7380000c738
GLOBL ·fk51000(SB), RODATA|NOPTR, $16
DATA ·fk7+0(SB)/8, $0x0007000700070007
DATA ·fk7+8(SB)/8, $0x0007000700070007
GLOBL ·fk7(SB), RODATA|NOPTR, $16

// FT_PASS1 is the first (horizontal) pass of fTransform on one block
// (FTransformPass1_SSE2). I01 = [r0c0 r0c1 r1c0 r1c1 r0c2 r0c3 r1c2 r1c3]
// and I23 (rows 2, 3) hold the differences src - ref (destroyed); the
// output is O01 = [t row 0 | t row 1] and O32 = [t row 3 | t row 2] with
//   t0 = (a0 + a1) * 8, t2 = (a0 - a1) * 8,
//   t1 = (a2*2217 + a3*5352 + 1812) >> 9, t3 = (a3*2217 - a2*5352 + 937) >> 9
// (PMADDWD on the [a0, a1] and [a3, a2] pairs; |t| <= 8160, so PACKSSDW is
// exact).
#define FT_PASS1(I01, I23, O01, O32, T0, T1, T2) \
	PSHUFHW	$0xb1, I01, I01 \
	PSHUFHW	$0xb1, I23, I23 \
	MOVO	I01, T0 \
	PUNPCKLQDQ	I23, T0 \
	PUNPCKHQDQ	I23, I01 \
	MOVO	T0, T1 \
	PADDW	I01, T0 \
	PSUBW	I01, T1 \
	MOVO	T0, I01 \
	PMADDWL	·fk88p(SB), T0 \
	PMADDWL	·fk88m(SB), I01 \
	PACKSSLW	I01, T0 \
	MOVO	T1, I23 \
	PMADDWL	·fk5352p(SB), T1 \
	PMADDWL	·fk5352m(SB), I23 \
	PADDL	·fk1812(SB), T1 \
	PADDL	·fk937(SB), I23 \
	PSRAL	$9, T1 \
	PSRAL	$9, I23 \
	PACKSSLW	I23, T1 \
	MOVO	T0, T2 \
	PUNPCKLWL	T1, T0 \
	PUNPCKHWL	T1, T2 \
	MOVO	T0, O01 \
	PUNPCKLLQ	T2, O01 \
	PUNPCKHLQ	T2, T0 \
	PSHUFD	$0x4e, T0, O32

// FT_PASS2 is the second (vertical) pass (FTransformPass2_SSE2): from
// V01 = [t row 0 | t row 1] and V32 = [t row 3 | t row 2] it leaves
// out[0..7] in V01 and out[8..15] in V32:
//   out0 = (a0 + a1 + 7) >> 4, out2 = (a0 - a1 + 7) >> 4,
//   out1 = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3 != 0)
//          (the constant holds 12000 + 65536; the a3 == 0 mask subtracts
//          the 1 back),
//   out3 = (a3*2217 - a2*5352 + 51000) >> 16.
// All values fit their 16-bit lanes (|a0 + a1 + 7| <= 32647).
#define FT_PASS2(V01, V32, T0, T1, T2, T3) \
	MOVO	V01, T0 \
	PSUBW	V32, T0 \
	PADDW	V32, V01 \
	MOVO	T0, T1 \
	PUNPCKHQDQ	T1, T1 \
	PUNPCKLWL	T0, T1 \
	MOVO	T1, T2 \
	PMADDWL	·fk2217p(SB), T1 \
	PMADDWL	·fk2217m(SB), T2 \
	PADDL	·fk12000(SB), T1 \
	PADDL	·fk51000(SB), T2 \
	PSRAL	$16, T1 \
	PSRAL	$16, T2 \
	PACKSSLW	T1, T1 \
	PACKSSLW	T2, T2 \
	PXOR	T3, T3 \
	PCMPEQW	T3, T0 \
	PADDW	T0, T1 \
	MOVO	V01, T3 \
	PUNPCKHQDQ	T3, T3 \
	PADDW	·fk7(SB), V01 \
	MOVO	V01, V32 \
	PADDW	T3, V01 \
	PSUBW	T3, V32 \
	PSRAW	$4, V01 \
	PSRAW	$4, V32 \
	PUNPCKLQDQ	T1, V01 \
	PUNPCKLQDQ	T2, V32

// ---------------------------------------------------------------------------
// func fTransformPairsSSE41(src, ref *byte, off *[8][2]uint16, n int, out *[16]int16)
//
// fTransform (dsp.go) of the n block pairs of a pair table; libwebp's
// FTransform2_SSE2. Each row load (PMOVZXBW of 8 bytes) covers a row of
// both blocks of the pair; the differences are regrouped per block
// (PUNPCKLDQ/PUNPCKHDQ) and each block goes through FT_PASS1 and FT_PASS2.
//
// Registers: SI src, DI ref, R8 table, CX pairs left, DX out, R9/R10 the
// pair's src/ref pointers, X0-X11 data.
TEXT ·fTransformPairsSSE41(SB), NOSPLIT, $0-40
	MOVQ	src+0(FP), SI
	MOVQ	ref+8(FP), DI
	MOVQ	off+16(FP), R8
	MOVQ	n+24(FP), CX
	MOVQ	out+32(FP), DX
	TESTQ	CX, CX
	JZ	fdone

floop:
	MOVWLZX	(R8), AX
	MOVWLZX	2(R8), BX
	ADDQ	$4, R8
	LEAQ	(SI)(AX*1), R9
	LEAQ	(DI)(BX*1), R10
	PMOVZXBW	(R9), X0
	PMOVZXBW	(R10), X4
	PSUBW	X4, X0
	PMOVZXBW	32(R9), X1
	PMOVZXBW	32(R10), X4
	PSUBW	X4, X1
	PMOVZXBW	64(R9), X2
	PMOVZXBW	64(R10), X4
	PSUBW	X4, X2
	PMOVZXBW	96(R9), X3
	PMOVZXBW	96(R10), X4
	PSUBW	X4, X3
	MOVO	X0, X4
	PUNPCKLLQ	X1, X4 // block A rows 0-1
	PUNPCKHLQ	X1, X0 // block B rows 0-1
	MOVO	X2, X5
	PUNPCKLLQ	X3, X5 // block A rows 2-3
	PUNPCKHLQ	X3, X2 // block B rows 2-3
	FT_PASS1(X4, X5, X6, X7, X8, X9, X10)
	FT_PASS1(X0, X2, X1, X3, X8, X9, X10)
	FT_PASS2(X6, X7, X8, X9, X10, X11)
	FT_PASS2(X1, X3, X8, X9, X10, X11)
	MOVOU	X6, (DX)
	MOVOU	X7, 16(DX)
	MOVOU	X1, 32(DX)
	MOVOU	X3, 48(DX)
	ADDQ	$64, DX
	DECQ	CX
	JNZ	floop

fdone:
	RET

// TRANSPOSE_2_4x4 transposes the two 4x4 int16 matrices held side by side
// in I0..I3 (I_r = [A row r | B row r], destroyed) into O0..O3
// (O_c = [A column c | B column c]); libwebp's VP8Transpose_2_4x4_16b.
#define TRANSPOSE_2_4x4(I0, I1, I2, I3, O0, O1, O2, O3) \
	MOVO	I0, O0 \
	PUNPCKLWL	I1, O0 \
	PUNPCKHWL	I1, I0 \
	MOVO	I2, O1 \
	PUNPCKLWL	I3, O1 \
	PUNPCKHWL	I3, I2 \
	MOVO	O0, I1 \
	PUNPCKLLQ	O1, I1 \
	PUNPCKHLQ	O1, O0 \
	MOVO	I0, I3 \
	PUNPCKLLQ	I2, I3 \
	PUNPCKHLQ	I2, I0 \
	MOVO	O0, O2 \
	PUNPCKLQDQ	I0, O2 \
	MOVO	O0, O3 \
	PUNPCKHQDQ	I0, O3 \
	MOVO	I1, O0 \
	PUNPCKLQDQ	I3, O0 \
	MOVO	I1, O1 \
	PUNPCKHQDQ	I3, O1

// ---------------------------------------------------------------------------
// Constants of iTransform: k1 = 20091 and k2 = 35468 - 65536 = -30068, so
// that mul1(x) = ((x*20091) >> 16) + x = PMULHW(x, k1) + x and
// mul2(x) = (x*35468) >> 16 = PMULHW(x, k2) + x (exact for any int16 x).
DATA ·ik1+0(SB)/8, $0x4e7b4e7b4e7b4e7b
DATA ·ik1+8(SB)/8, $0x4e7b4e7b4e7b4e7b
GLOBL ·ik1(SB), RODATA|NOPTR, $16
DATA ·ik2+0(SB)/8, $0x8a8c8a8c8a8c8a8c
DATA ·ik2+8(SB)/8, $0x8a8c8a8c8a8c8a8c
GLOBL ·ik2(SB), RODATA|NOPTR, $16
DATA ·ik4+0(SB)/8, $0x0004000400040004
DATA ·ik4+8(SB)/8, $0x0004000400040004
GLOBL ·ik4(SB), RODATA|NOPTR, $16
DATA ·k2047w+0(SB)/8, $0x07ff07ff07ff07ff
DATA ·k2047w+8(SB)/8, $0x07ff07ff07ff07ff
GLOBL ·k2047w(SB), RODATA|NOPTR, $16

// IT_PASS is one pass of the inverse DCT on 8 lanes (two blocks), as in
// ITransform_Two_SSE2: from I0..I3 (destroyed; for the vertical pass the
// coefficient rows, for the horizontal pass the columns with 4 already
// added to I0) it computes
//   a = I0 + I2, b = I0 - I2,
//   c = mul2(I1) - mul1(I3) = PMULHW(I1, k2) - PMULHW(I3, k1) + I1 - I3,
//   d = mul1(I1) + mul2(I3) = PMULHW(I1, k1) + PMULHW(I3, k2) + I1 + I3,
// and O0..O3 = a + d, b + c, b - c, a - d. K1 and K2 hold k1 and k2.
#define IT_PASS(I0, I1, I2, I3, O0, O1, O2, O3, K1, K2) \
	MOVO	I0, O0 \
	PADDW	I2, O0 \
	PSUBW	I2, I0 \
	MOVO	I1, O1 \
	PMULHW	K2, O1 \
	MOVO	I3, O2 \
	PMULHW	K1, O2 \
	PSUBW	O2, O1 \
	MOVO	I1, O2 \
	PSUBW	I3, O2 \
	PADDW	O2, O1 \
	MOVO	I1, O2 \
	PMULHW	K1, O2 \
	MOVO	I3, O3 \
	PMULHW	K2, O3 \
	PADDW	O3, O2 \
	PADDW	I3, I1 \
	PADDW	I1, O2 \
	MOVO	O0, O3 \
	PSUBW	O2, O3 \
	PADDW	O2, O0 \
	MOVO	I0, O2 \
	PSUBW	O1, O2 \
	PADDW	I0, O1

// ITRANS_STORE_ROW adds row R (8 int16: the row of both blocks) to the 8
// reference bytes at off(R9), clips to bytes (PACKUSWB) and stores them at
// off(R10).
#define ITRANS_STORE_ROW(R, off, T) \
	PMOVZXBW	off(R9), T \
	PADDW	T, R \
	PACKUSWB	R, R \
	MOVQ	R, off(R10)

// ---------------------------------------------------------------------------
// func iTransformPairsSSE41(ref *byte, in *[16]int16, dst *byte, off *[2]uint16, n int) int
//
// iTransform (dsp.go) of the blocks of n consecutive pairs of a pair table
// (off points to the first pair's offsets, in to its first block):
// dst = clip8(ref + IDCT(in)); libwebp's ITransform_Two_SSE2.
//
// The Go code computes in int32. When every coefficient of the pair has
// |in| <= 2047, no intermediate value exceeds 14.8*2047+4 < 2^15, so the
// 16-bit lanes give the same results (see the NEON kernel); the kernel
// checks that bound per pair (PABSW reads -32768 as 32768, PMAXUW,
// PSUBUSW, PTEST) and returns the number of pairs done when a pair fails
// it: the wrapper runs that pair in Go and calls the kernel again.
//
// Registers: SI ref, DX in, DI dst, R8 table, CX pairs, R11 pairs done,
// R9/R10 the pair's ref/dst pointers, X12 k1, X13 k2, X14 4, X0-X11 data.
TEXT ·iTransformPairsSSE41(SB), NOSPLIT, $0-48
	MOVQ	ref+0(FP), SI
	MOVQ	in+8(FP), DX
	MOVQ	dst+16(FP), DI
	MOVQ	off+24(FP), R8
	MOVQ	n+32(FP), CX
	XORQ	R11, R11
	MOVOU	·ik1(SB), X12
	MOVOU	·ik2(SB), X13
	MOVOU	·ik4(SB), X14

iloop:
	CMPQ	R11, CX
	JEQ	idone
	MOVOU	(DX), X0   // A in[0..7]
	MOVOU	16(DX), X1 // A in[8..15]
	MOVOU	32(DX), X2 // B in[0..7]
	MOVOU	48(DX), X3 // B in[8..15]
	PABSW	X0, X4
	PABSW	X1, X5
	PMAXUW	X5, X4
	PABSW	X2, X5
	PMAXUW	X5, X4
	PABSW	X3, X5
	PMAXUW	X5, X4
	PSUBUSW	·k2047w(SB), X4
	PTEST	X4, X4
	JNZ	idone
	MOVO	X0, X4
	PUNPCKLQDQ	X2, X4 // in0 = [A in0-3 | B in0-3]
	PUNPCKHQDQ	X2, X0 // in1 = [A in4-7 | B in4-7]
	MOVO	X1, X5
	PUNPCKLQDQ	X3, X5 // in2
	PUNPCKHQDQ	X3, X1 // in3

	// vertical pass, lanes = columns; transpose to lanes = rows
	IT_PASS(X4, X0, X5, X1, X6, X7, X8, X9, X12, X13)
	TRANSPOSE_2_4x4(X6, X7, X8, X9, X0, X1, X2, X3)

	// horizontal pass, >> 3, transpose back to rows
	PADDW	X14, X0
	IT_PASS(X0, X1, X2, X3, X4, X5, X6, X7, X12, X13)
	PSRAW	$3, X4
	PSRAW	$3, X5
	PSRAW	$3, X6
	PSRAW	$3, X7
	TRANSPOSE_2_4x4(X4, X5, X6, X7, X0, X1, X2, X3)

	MOVWLZX	(R8), AX
	MOVWLZX	2(R8), BX
	ADDQ	$4, R8
	LEAQ	(SI)(AX*1), R9
	LEAQ	(DI)(BX*1), R10
	ITRANS_STORE_ROW(X0, 0, X4)
	ITRANS_STORE_ROW(X1, 32, X4)
	ITRANS_STORE_ROW(X2, 64, X4)
	ITRANS_STORE_ROW(X3, 96, X4)
	ADDQ	$64, DX
	INCQ	R11
	JMP	iloop

idone:
	MOVQ	R11, ret+40(FP)
	RET

// ---------------------------------------------------------------------------
// Zigzag shuffles of quantizeBlocksSSE41 (libwebp's DoQuantizeBlock_SSE41):
// out[0..7] = levels 0 1 4 8 5 2 3 6 and out[8..15] = 9 12 13 10 7 11 14 15;
// zzLo/zzHi pick the levels of the same register, zz8/zz7 the one level
// (8 and 7) that crosses over.
DATA ·zzLo+0(SB)/8, $0x8080090803020100
DATA ·zzLo+8(SB)/8, $0x0d0c070605040b0a
GLOBL ·zzLo(SB), RODATA|NOPTR, $16
DATA ·zz8+0(SB)/8, $0x0100808080808080
DATA ·zz8+8(SB)/8, $0x8080808080808080
GLOBL ·zz8(SB), RODATA|NOPTR, $16
DATA ·zzHi+0(SB)/8, $0x05040b0a09080302
DATA ·zzHi+8(SB)/8, $0x0f0e0d0c07068080
GLOBL ·zzHi(SB), RODATA|NOPTR, $16
DATA ·zz7+0(SB)/8, $0x8080808080808080
DATA ·zz7+8(SB)/8, $0x8080808080800f0e
GLOBL ·zz7(SB), RODATA|NOPTR, $16

// ---------------------------------------------------------------------------
// func quantizeBlocksSSE41(in, out *[16]int16, n int, m *matrix, info *uint16) uint32
//
// quantizeBlock (dsp.go) on n consecutive blocks, with the 16-bit
// parameters of the matrix (vecOK, see prepareVec); libwebp's
// DoQuantizeBlock_SSE41. Per block, two vectors of 8 coefficients:
//   sign  = x >> 15                       (PSRAW)
//   coeff = |x| + sharpen                 (PABSW, PADDW; uint16, no overflow)
//   level = min((coeff*iq + bias) >> 17, 2047)
//           (PMULHUW/PMULLW give the exact 32-bit product, PADDD wraps
//           like the uint32 Go code, PSRLD, PACKUSDW: the result is
//           < 2^15, PMINSW)
//   level = (level ^ sign) - sign
//   in    = level * q                     (PMULLW, low 16 bits as in Go)
//   out   = zigzag(level)                 (PSHUFB, POR)
// then the summary levelInfoOf(out) (dsp_batch.go): PACKSSWB keeps the
// non-zero levels non-zero, PCMPEQB/PMOVMSKB give the mask m of the
// non-zero levels; BSR(m) + 1 (0 if m = 0, by CMOV) is the last index plus
// one, POPCNT(m & 0xfffe) the AC count. It is stored to info[b] unless
// info is nil. Bit b of the result is set if block b has a non-zero level.
//
// Registers: DI in, SI out, CX blocks left, DX matrix, R8 info, R9 nz
// mask, R10 current bit, R11 -1, X8/X9 iq, X10/X11 sharpen, X12/X13 q,
// X14 2047, X15 zero.
TEXT ·quantizeBlocksSSE41(SB), NOSPLIT, $0-44
	MOVQ	in+0(FP), DI
	MOVQ	out+8(FP), SI
	MOVQ	n+16(FP), CX
	MOVQ	m+24(FP), DX
	MOVQ	info+32(FP), R8
	MOVOU	matrix_iq16(DX), X8
	MOVOU	matrix_iq16+16(DX), X9
	MOVOU	matrix_sharpen16(DX), X10
	MOVOU	matrix_sharpen16+16(DX), X11
	MOVOU	matrix_q16(DX), X12
	MOVOU	matrix_q16+16(DX), X13
	MOVOU	·k2047w(SB), X14
	PXOR	X15, X15
	XORL	R9, R9
	MOVL	$1, R10
	MOVL	$-1, R11
	TESTQ	CX, CX
	JZ	qdone

qloop:
	MOVOU	(DI), X0
	MOVOU	16(DI), X1
	MOVO	X0, X2
	PSRAW	$15, X2
	MOVO	X1, X3
	PSRAW	$15, X3
	PABSW	X0, X0
	PABSW	X1, X1
	PADDW	X10, X0
	PADDW	X11, X1
	MOVO	X0, X4
	PMULHUW	X8, X4
	PMULLW	X8, X0
	MOVO	X0, X5
	PUNPCKLWL	X4, X0 // coeff*iq, lanes 0-3
	PUNPCKHWL	X4, X5 // lanes 4-7
	MOVO	X1, X6
	PMULHUW	X9, X6
	PMULLW	X9, X1
	MOVO	X1, X7
	PUNPCKLWL	X6, X1 // lanes 8-11
	PUNPCKHWL	X6, X7 // lanes 12-15
	MOVOU	matrix_bias(DX), X4
	PADDL	X4, X0
	MOVOU	matrix_bias+16(DX), X4
	PADDL	X4, X5
	MOVOU	matrix_bias+32(DX), X4
	PADDL	X4, X1
	MOVOU	matrix_bias+48(DX), X4
	PADDL	X4, X7
	PSRLL	$17, X0
	PSRLL	$17, X5
	PSRLL	$17, X1
	PSRLL	$17, X7
	PACKUSDW	X5, X0
	PACKUSDW	X7, X1
	PMINSW	X14, X0
	PMINSW	X14, X1
	PXOR	X2, X0
	PSUBW	X2, X0 // levels 0-7
	PXOR	X3, X1
	PSUBW	X3, X1 // levels 8-15
	MOVO	X0, X4
	PMULLW	X12, X4
	MOVOU	X4, (DI)
	MOVO	X1, X5
	PMULLW	X13, X5
	MOVOU	X5, 16(DI)
	MOVO	X0, X4
	PSHUFB	·zzLo(SB), X4
	MOVO	X1, X5
	PSHUFB	·zz8(SB), X5
	POR	X5, X4 // out[0..7]
	PSHUFB	·zzHi(SB), X1
	PSHUFB	·zz7(SB), X0
	POR	X0, X1 // out[8..15]
	MOVOU	X4, (SI)
	MOVOU	X1, 16(SI)
	PACKSSWB	X1, X4
	PCMPEQB	X15, X4
	PMOVMSKB	X4, AX
	XORL	BX, BX
	XORL	$0xffff, AX // non-zero levels
	CMOVLNE	R10, BX
	ORL	BX, R9
	TESTQ	R8, R8
	JZ	qnext
	BSRL	AX, BX
	CMOVLEQ	R11, BX
	INCL	BX // last + 1
	ANDL	$0xfffe, AX
	POPCNTL	AX, AX
	SHLL	$8, AX
	ORL	BX, AX
	MOVW	AX, (R8)
	ADDQ	$2, R8

qnext:
	ADDQ	$32, DI
	ADDQ	$32, SI
	SHLL	$1, R10
	DECQ	CX
	JNZ	qloop

qdone:
	MOVL	R9, ret+40(FP)
	RET

// HADAMARD4 is one pass of the 4-point Hadamard transform of tTransform on
// 8 lanes: a0 = I0 + I2, a1 = I1 + I3, a2 = I1 - I3, a3 = I0 - I2, and
// O0..O3 = a0 + a1, a3 + a2, a3 - a2, a0 - a1 (I0..I3 destroyed).
#define HADAMARD4(I0, I1, I2, I3, O0, O1, O2, O3) \
	MOVO	I0, O0 \
	PADDW	I2, O0 \
	PSUBW	I2, I0 \
	MOVO	I1, O1 \
	PADDW	I3, O1 \
	PSUBW	I3, I1 \
	MOVO	O0, O3 \
	PSUBW	O1, O3 \
	PADDW	O1, O0 \
	MOVO	I0, O1 \
	PADDW	I1, O1 \
	MOVO	I0, O2 \
	PSUBW	I1, O2

// Weights of tTransform (kWeightY) for the horizontal coefficient h of
// each vertical index v, i.e. row h of the (symmetric) weight matrix,
// duplicated for the two blocks of a pair.
DATA ·ttW+0(SB)/8, $0x0009001400200026
DATA ·ttW+8(SB)/8, $0x0009001400200026
DATA ·ttW+16(SB)/8, $0x00070011001c0020
DATA ·ttW+24(SB)/8, $0x00070011001c0020
DATA ·ttW+32(SB)/8, $0x0004000a00110014
DATA ·ttW+40(SB)/8, $0x0004000a00110014
DATA ·ttW+48(SB)/8, $0x0002000400070009
DATA ·ttW+56(SB)/8, $0x0002000400070009
GLOBL ·ttW(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// func tTransformPairsSSE41(a *byte, off *[8][2]uint16, n int, out *int32)
//
// tTransform(a, &kWeightY) (dsp.go) of the 2n blocks of the n pairs of a
// pair table (first offsets only); libwebp's TTransform_SSE41 for one
// input. The Hadamard transform has no rounding, so the vertical pass can
// run first (lanes = columns): after a transpose the horizontal pass works
// lane-wise, and coefficient (vertical v, horizontal h) sits in lane v of
// register h, with weight kWeightY[4v+h] = kWeightY[4h+v]. The weighted
// sum of |coeff| (<= 4080) is exact in PMADDWD's 32-bit lanes; PHADDD
// leaves [A, B, A, B].
//
// Registers: SI a, R8 table, CX pairs left, DX out, R9 pair pointer,
// X8-X11 weights.
TEXT ·tTransformPairsSSE41(SB), NOSPLIT, $0-32
	MOVQ	a+0(FP), SI
	MOVQ	off+8(FP), R8
	MOVQ	n+16(FP), CX
	MOVQ	out+24(FP), DX
	MOVOU	·ttW+0(SB), X8
	MOVOU	·ttW+16(SB), X9
	MOVOU	·ttW+32(SB), X10
	MOVOU	·ttW+48(SB), X11
	TESTQ	CX, CX
	JZ	tdone

tloop:
	MOVWLZX	(R8), AX
	ADDQ	$4, R8
	LEAQ	(SI)(AX*1), R9
	PMOVZXBW	(R9), X0
	PMOVZXBW	32(R9), X1
	PMOVZXBW	64(R9), X2
	PMOVZXBW	96(R9), X3
	HADAMARD4(X0, X1, X2, X3, X4, X5, X6, X7)
	TRANSPOSE_2_4x4(X4, X5, X6, X7, X0, X1, X2, X3)
	HADAMARD4(X0, X1, X2, X3, X4, X5, X6, X7)
	PABSW	X4, X4
	PABSW	X5, X5
	PABSW	X6, X6
	PABSW	X7, X7
	PMADDWL	X8, X4
	PMADDWL	X9, X5
	PMADDWL	X10, X6
	PMADDWL	X11, X7
	PADDL	X5, X4
	PADDL	X7, X6
	PADDL	X6, X4
	PHADDD	X4, X4
	MOVQ	X4, (DX)
	ADDQ	$8, DX
	DECQ	CX
	JNZ	tloop

tdone:
	RET

// SSE_ROW8 accumulates into ACC the squared differences of the 8 bytes at
// off(R9) and off(R10) (one row of both blocks of a pair), as two int32
// sums per block (PMADDWD of the int16 differences with themselves).
#define SSE_ROW8(off, ACC, T0, T1) \
	PMOVZXBW	off(R9), T0 \
	PMOVZXBW	off(R10), T1 \
	PSUBW	T1, T0 \
	PMADDWL	T0, T0 \
	PADDL	T0, ACC

// ---------------------------------------------------------------------------
// func sse4x4PairsSSE41(a, b *byte, off *[8][2]uint16, n int, out *int32)
//
// sse4x4 (dsp.go) of the 2n blocks of the n pairs of a pair table:
// PMOVZXBW, PSUBW, PMADDWD (squares <= 65025, sums exact in int32) into
// [A, A, B, B] partial sums, PHADDD leaves [A, B, A, B]; libwebp's
// SSE4x4_SSE2 on a pair.
//
// Registers: SI a, DI b, R8 table, CX pairs left, DX out, R9/R10 pair
// pointers.
TEXT ·sse4x4PairsSSE41(SB), NOSPLIT, $0-40
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	off+16(FP), R8
	MOVQ	n+24(FP), CX
	MOVQ	out+32(FP), DX
	TESTQ	CX, CX
	JZ	sdone

sloop:
	MOVWLZX	(R8), AX
	MOVWLZX	2(R8), BX
	ADDQ	$4, R8
	LEAQ	(SI)(AX*1), R9
	LEAQ	(DI)(BX*1), R10
	PMOVZXBW	(R9), X0
	PMOVZXBW	(R10), X1
	PSUBW	X1, X0
	PMADDWL	X0, X0
	SSE_ROW8(32, X0, X2, X3)
	SSE_ROW8(64, X0, X2, X3)
	SSE_ROW8(96, X0, X2, X3)
	PHADDD	X0, X0
	MOVQ	X0, (DX)
	ADDQ	$8, DX
	DECQ	CX
	JNZ	sloop

sdone:
	RET

// ---------------------------------------------------------------------------
// func sse16xNSSE41(a, b *byte, rows int) int
//
// Sum of squared differences of a 16-pixel wide block of the given number
// of rows (sse16x16, sse16x8 of dsp.go); libwebp's SSE_16xN_SSE2 with
// PMOVZXBW loads: per row two PSUBW, two PMADDWD into two accumulators,
// then a horizontal sum (at most 256 * 65025, exact in int32).
//
// Registers: SI a, DI b, CX rows left, X4/X5 accumulators.
TEXT ·sse16xNSSE41(SB), NOSPLIT, $0-32
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	rows+16(FP), CX
	PXOR	X4, X4
	PXOR	X5, X5
	TESTQ	CX, CX
	JZ	edone

eloop:
	PMOVZXBW	(SI), X0
	PMOVZXBW	(DI), X1
	PSUBW	X1, X0
	PMADDWL	X0, X0
	PADDL	X0, X4
	PMOVZXBW	8(SI), X2
	PMOVZXBW	8(DI), X3
	PSUBW	X3, X2
	PMADDWL	X2, X2
	PADDL	X2, X5
	ADDQ	$32, SI
	ADDQ	$32, DI
	DECQ	CX
	JNZ	eloop

edone:
	PADDL	X5, X4
	PSHUFD	$0x4e, X4, X5
	PADDL	X5, X4
	PSHUFD	$0xb1, X4, X5
	PADDL	X5, X4
	MOVL	X4, AX
	MOVQ	AX, ret+24(FP)
	RET

// ---------------------------------------------------------------------------
// Constants of the predictors: bytes of 1, the low bit of each byte, the
// HE indexes of the chroma rows ([0 x8, 8 x8]), the spreading of two qword
// DCs to [dc0 x8, dc1 x8], and 8 in each qword.
DATA ·kOnes+0(SB)/8, $0x0101010101010101
DATA ·kOnes+8(SB)/8, $0x0101010101010101
GLOBL ·kOnes(SB), RODATA|NOPTR, $16
DATA ·c8HEIdx+0(SB)/8, $0x0000000000000000
DATA ·c8HEIdx+8(SB)/8, $0x0808080808080808
GLOBL ·c8HEIdx(SB), RODATA|NOPTR, $16
DATA ·k8q+0(SB)/8, $8
DATA ·k8q+8(SB)/8, $8
GLOBL ·k8q(SB), RODATA|NOPTR, $16
// word splat of byte 0: [0, 0x80] x8
DATA ·wSplat0+0(SB)/8, $0x8000800080008000
DATA ·wSplat0+8(SB)/8, $0x8000800080008000
GLOBL ·wSplat0(SB), RODATA|NOPTR, $16

// ---------------------------------------------------------------------------
// func predLuma16SSE41(p, left, top *byte)
//
// predLuma16 (dsp.go) of a macroblock with both neighbours: the DC, TM, VE
// and HE 16x16 predictions at p+i16DC16 (= p), p+i16TM16, p+i16VE16 and
// p+i16HE16. left[0] is the top-left sample, left[1..16] the left column,
// top[0..15] the row above. libwebp's DC16, TM16, VE16 and HE16 (SSE2):
//   DC: (sum(top) + sum(left) + 16) >> 5   (PSADBW)
//   TM: clip8(left[y] - tl + top[x]): top - tl as int16, plus the row's
//       left sample, PACKUSWB (the sum is in [-255, 510]: exact)
//   VE: top on every row; HE: PSHUFB of left[y] on row y (the index
//       vector X9 counts up by one per row).
//
// Registers: DI output row, CX rows left, X0 top, X1 left[1..16], X2 DC
// row, X3/X4 top - tl (int16), X7 zero, X8 ones, X9 HE index.
TEXT ·predLuma16SSE41(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), DI
	MOVQ	left+8(FP), SI
	MOVQ	top+16(FP), DX
	MOVOU	(DX), X0
	MOVOU	1(SI), X1
	MOVBLZX	(SI), AX
	PXOR	X7, X7
	MOVO	X0, X2
	PSADBW	X7, X2
	MOVO	X1, X3
	PSADBW	X7, X3
	PADDQ	X3, X2
	PSHUFD	$0x4e, X2, X3
	PADDQ	X3, X2
	MOVL	X2, BX
	ADDL	$16, BX
	SHRL	$5, BX
	MOVL	BX, X2
	PSHUFB	X7, X2 // DC row
	PMOVZXBW	X0, X3
	MOVO	X0, X4
	PUNPCKHBW	X7, X4
	MOVL	AX, X5
	PSHUFB	·wSplat0(SB), X5
	PSUBW	X5, X3 // top[0..7] - tl
	PSUBW	X5, X4 // top[8..15] - tl
	MOVOU	·kOnes(SB), X8
	PXOR	X9, X9
	MOVL	$16, CX

p16loop:
	MOVOU	X2, (DI)
	MOVOU	X0, 512(DI)
	MOVO	X1, X5
	PSHUFB	X9, X5 // left[y] x16
	MOVOU	X5, 528(DI)
	PMOVZXBW	X5, X6
	MOVO	X6, X5
	PADDW	X3, X6
	PADDW	X4, X5
	PACKUSWB	X5, X6
	MOVOU	X6, 16(DI)
	PADDB	X8, X9
	ADDQ	$32, DI
	DECL	CX
	JNZ	p16loop
	RET

// ---------------------------------------------------------------------------
// func predChroma8SSE41(p, uLeft, vLeft, uvTop *byte)
//
// predChroma8 (dsp.go) of a macroblock with both neighbours, U and V side
// by side: the DC, TM, VE and HE 8x8 predictions at p (= c8DC8), p+16,
// p+8*bps and p+8*bps+16, each row being [U 8 | V 8]. uLeft/vLeft hold the
// top-left sample then the 8 left samples, uvTop the 8 U then 8 V samples
// above. Same formulas as predLuma16SSE41 with DC = (sum + 8) >> 4 per
// plane (PSADBW gives the U and V sums in the two qwords).
//
// Registers: DI output row, CX rows left, X0 uvTop, X1 [uLeft | vLeft],
// X2 DC row, X3/X4 top - tl (int16, U and V), X7 zero, X8 HE index,
// X9 ones.
TEXT ·predChroma8SSE41(SB), NOSPLIT, $0-32
	MOVQ	p+0(FP), DI
	MOVQ	uLeft+8(FP), SI
	MOVQ	vLeft+16(FP), DX
	MOVQ	uvTop+24(FP), R8
	MOVOU	(R8), X0
	MOVQ	1(SI), X1
	PINSRQ	$1, 1(DX), X1
	PXOR	X7, X7
	MOVO	X0, X2
	PSADBW	X7, X2
	MOVO	X1, X3
	PSADBW	X7, X3
	PADDQ	X3, X2
	PADDQ	·k8q(SB), X2
	PSRLQ	$4, X2
	PSHUFB	·c8HEIdx(SB), X2 // [dcU x8 | dcV x8]
	MOVBLZX	(SI), AX
	MOVBLZX	(DX), BX
	PMOVZXBW	X0, X3
	MOVO	X0, X4
	PUNPCKHBW	X7, X4
	MOVL	AX, X5
	PSHUFB	·wSplat0(SB), X5
	PSUBW	X5, X3 // uTop - uTL
	MOVL	BX, X5
	PSHUFB	·wSplat0(SB), X5
	PSUBW	X5, X4 // vTop - vTL
	MOVOU	·c8HEIdx(SB), X8
	MOVOU	·kOnes(SB), X9
	MOVL	$8, CX

c8loop:
	MOVOU	X2, (DI)
	MOVOU	X0, 256(DI)
	MOVO	X1, X5
	PSHUFB	X8, X5 // [uLeft[y] x8 | vLeft[y] x8]
	MOVOU	X5, 272(DI)
	PMOVZXBW	X5, X6
	PUNPCKHBW	X7, X5
	PADDW	X3, X6
	PADDW	X4, X5
	PACKUSWB	X5, X6
	MOVOU	X6, 16(DI)
	PADDB	X9, X8
	ADDQ	$32, DI
	DECL	CX
	JNZ	c8loop
	RET

// Source shuffles of predLuma4SSE41 (dsp_arm64.s i4SrcIdx): the extended
// samples S' = [L L K J I X A B C D E F G H H 0] from the 16 loaded bytes,
// the TM left rows [I x4 J x4 K x4 L x4] and top columns [A B C D] x4, the
// 8 samples summed for DC, and X as words.
DATA ·i4Src+0(SB)/8, $0x0605040302010000
DATA ·i4Src+8(SB)/8, $0x800c0c0b0a090807
DATA ·i4Src+16(SB)/8, $0x0303030304040404
DATA ·i4Src+24(SB)/8, $0x0101010102020202
DATA ·i4Src+32(SB)/8, $0x0908070609080706
DATA ·i4Src+40(SB)/8, $0x0908070609080706
DATA ·i4Src+48(SB)/8, $0x0908070604030201
DATA ·i4Src+56(SB)/8, $0x8080808080808080
DATA ·i4Src+64(SB)/8, $0x8005800580058005
DATA ·i4Src+72(SB)/8, $0x8005800580058005
GLOBL ·i4Src(SB), RODATA|NOPTR, $80

// Output shuffles of predLuma4SSE41, generated from dsp_arm64.s i4OutIdx
// (an index t*16 + i there selects byte i of table t of [S', A3, P, TM]):
// for each 16-byte output, one PSHUFB per table it reads, the bytes of the
// other tables zeroed (0x80).
DATA ·i4Shuf+0(SB)/8, $0x808080800f0f0f0f
DATA ·i4Shuf+8(SB)/8, $0x8080808080808080
DATA ·i4Shuf+16(SB)/8, $0x8080808080808080
DATA ·i4Shuf+24(SB)/8, $0x0404040409080706
DATA ·i4Shuf+32(SB)/8, $0x0302010080808080
DATA ·i4Shuf+40(SB)/8, $0x8080808080808080
DATA ·i4Shuf+48(SB)/8, $0x8080808008070605
DATA ·i4Shuf+56(SB)/8, $0x808080800a090807
DATA ·i4Shuf+64(SB)/8, $0x0807060580808080
DATA ·i4Shuf+72(SB)/8, $0x0908070680808080
DATA ·i4Shuf+80(SB)/8, $0x808080800f0f0f0f
DATA ·i4Shuf+88(SB)/8, $0x8080808080808080
DATA ·i4Shuf+96(SB)/8, $0x8080808080808080
DATA ·i4Shuf+104(SB)/8, $0x0303030309080706
DATA ·i4Shuf+112(SB)/8, $0x0706050480808080
DATA ·i4Shuf+120(SB)/8, $0x8080808080808080
DATA ·i4Shuf+128(SB)/8, $0x0807060507060504
DATA ·i4Shuf+136(SB)/8, $0x0a0908070b0a0908
DATA ·i4Shuf+144(SB)/8, $0x808080800f0f0f0f
DATA ·i4Shuf+152(SB)/8, $0x8080808080808080
DATA ·i4Shuf+160(SB)/8, $0x8080808080808080
DATA ·i4Shuf+168(SB)/8, $0x0202020209080706
DATA ·i4Shuf+176(SB)/8, $0x0b0a090880808080
DATA ·i4Shuf+184(SB)/8, $0x8080808080808080
DATA ·i4Shuf+192(SB)/8, $0x8080800406050403
DATA ·i4Shuf+200(SB)/8, $0x0b8080800c0b0a09
DATA ·i4Shuf+208(SB)/8, $0x0706058080808080
DATA ·i4Shuf+216(SB)/8, $0x8009080780808080
DATA ·i4Shuf+224(SB)/8, $0x808080800f0f0f0f
DATA ·i4Shuf+232(SB)/8, $0x8080808080808080
DATA ·i4Shuf+240(SB)/8, $0x8080808080808080
DATA ·i4Shuf+248(SB)/8, $0x0101010109080706
DATA ·i4Shuf+256(SB)/8, $0x0f0e0d0c80808080
DATA ·i4Shuf+264(SB)/8, $0x8080808080808080
DATA ·i4Shuf+272(SB)/8, $0x0706050305040302
DATA ·i4Shuf+280(SB)/8, $0x0c0a09080d0c0b0a
DATA ·i4Shuf+288(SB)/8, $0x0280038007060580
DATA ·i4Shuf+296(SB)/8, $0x0180028005800480
DATA ·i4Shuf+304(SB)/8, $0x8002800380808004
DATA ·i4Shuf+312(SB)/8, $0x8001800280048003
DATA ·i4Shuf+320(SB)/8, $0x0101808080808080
DATA ·i4Shuf+328(SB)/8, $0x0101010180808080
DATA ·i4Shuf+336(SB)/8, $0x8080018004800380
DATA ·i4Shuf+344(SB)/8, $0x8080808003800280
DATA ·i4Shuf+352(SB)/8, $0x8080800180038002
DATA ·i4Shuf+360(SB)/8, $0x8080808080028001
GLOBL ·i4Shuf(SB), RODATA|NOPTR, $368

// I4_OUT1/2/3 build one 16-byte output of predLuma4SSE41 into X4 from
// one, two or three of the tables X0-X3 (shuffle offsets o1..o3 into
// ·i4Shuf).
#define I4_OUT1(t1, o1) \
	MOVO	t1, X4 \
	PSHUFB	·i4Shuf+o1(SB), X4

#define I4_OUT2(t1, o1, t2, o2) \
	I4_OUT1(t1, o1) \
	MOVO	t2, X5 \
	PSHUFB	·i4Shuf+o2(SB), X5 \
	POR	X5, X4

#define I4_OUT3(t1, o1, t2, o2, t3, o3) \
	I4_OUT2(t1, o1, t2, o2) \
	MOVO	t3, X5 \
	PSHUFB	·i4Shuf+o3(SB), X5 \
	POR	X5, X4

// ---------------------------------------------------------------------------
// func predLuma4SSE41(p, s *byte)
//
// predLuma4 (dsp.go): the 10 intra4 predictions of a sub-block, written at
// p = i4DC4 in the prediction buffer (modes 0-7 side by side on rows 0-3,
// modes 8-9 on rows 4-7). s points to the 13 boundary samples
// L K J I X A B C D E F G H (16 bytes are loaded). Every output byte is a
// lookup into one of four 16-byte tables (as in the NEON kernel):
//   S'  = L, the 13 samples, H, DC    (edges replicated, as kll and ghh
//                                      need)
//   A3  = avg3 of each sample with its neighbours: PAVGB(prev, next) minus
//         the lost low bit gives floor((prev + next) / 2), then PAVGB with
//         the sample: ((a+c)>>1 + b + 1) >> 1 = (a + 2b + c + 2) >> 2
//   P   = avg2 of each sample and the next one (PAVGB)
//   TM  = clip8(left - X + top) on 16-bit lanes (PACKUSWB; exact)
// and DC = (sum of A..D and I..L + 4) >> 3 (PSADBW). PSHUFB reads one table
// at a time, so an output reading k tables takes k shuffles and k-1 ORs.
//
// Registers: DI p, SI s, X0 S', X1 A3, X2 P, X3 TM, X4/X5 outputs.
TEXT ·predLuma4SSE41(SB), NOSPLIT, $0-16
	MOVQ	p+0(FP), DI
	MOVQ	s+8(FP), SI
	MOVOU	(SI), X0
	PSHUFB	·i4Src+0(SB), X0 // S' (DC still 0)
	MOVO	X0, X1
	PSLLO	$1, X1 // previous sample
	MOVO	X0, X2
	PSRLO	$1, X2 // next sample
	MOVO	X1, X3
	PAVGB	X2, X3
	PXOR	X2, X1
	PAND	·kOnes(SB), X1
	PSUBB	X1, X3 // floor((prev + next) / 2)
	PAVGB	X0, X3
	MOVO	X3, X1 // A3
	PAVGB	X0, X2 // P

	// TM: left rows and top columns as 16-bit, minus X, clipped
	PXOR	X7, X7
	MOVO	X0, X4
	PSHUFB	·i4Src+16(SB), X4 // left rows
	MOVO	X0, X5
	PSHUFB	·i4Src+32(SB), X5 // top columns
	MOVO	X0, X6
	PSHUFB	·i4Src+64(SB), X6 // X as words
	PMOVZXBW	X5, X3
	PUNPCKHBW	X7, X5
	PSUBW	X6, X3
	PSUBW	X6, X5
	PMOVZXBW	X4, X6
	PUNPCKHBW	X7, X4
	PADDW	X6, X3
	PADDW	X4, X5
	PACKUSWB	X5, X3 // TM

	// DC into S'[15]
	MOVO	X0, X4
	PSHUFB	·i4Src+48(SB), X4
	PSADBW	X7, X4
	MOVL	X4, AX
	ADDL	$4, AX
	SHRL	$3, AX
	PINSRB	$15, AX, X0

	I4_OUT3(X0, 0, X1, 16, X3, 32)
	MOVOU	X4, (DI)
	I4_OUT2(X1, 48, X2, 64)
	MOVOU	X4, 16(DI)
	I4_OUT3(X0, 80, X1, 96, X3, 112)
	MOVOU	X4, 32(DI)
	I4_OUT1(X1, 128)
	MOVOU	X4, 48(DI)
	I4_OUT3(X0, 144, X1, 160, X3, 176)
	MOVOU	X4, 64(DI)
	I4_OUT2(X1, 192, X2, 208)
	MOVOU	X4, 80(DI)
	I4_OUT3(X0, 224, X1, 240, X3, 256)
	MOVOU	X4, 96(DI)
	I4_OUT1(X1, 272)
	MOVOU	X4, 112(DI)
	I4_OUT2(X1, 288, X2, 304)
	MOVQ	X4, 128(DI)
	MOVHPS	X4, 160(DI)
	I4_OUT3(X0, 320, X1, 336, X2, 352)
	MOVQ	X4, 192(DI)
	MOVHPS	X4, 224(DI)
	RET

DATA ·k31w+0(SB)/8, $0x001f001f001f001f
DATA ·k31w+8(SB)/8, $0x001f001f001f001f
GLOBL ·k31w(SB), RODATA|NOPTR, $16

// ---------------------------------------------------------------------------
// func zeroBinStatsSSE41(c *[16]int16, n int) (count0, maxBin int)
//
// zeroBinStatsRef (dsp.go) of n <= 16 blocks: the histogram bins
// min(|v| >> 3, 31) of 16 coefficients (PABSW, PSRLW, PMINUW, PACKUSWB to
// bytes; PABSW(-32768) reads as 32768 unsigned, as |v| in Go), their
// running maximum (PMAXUB) and the per-lane count of bin 0 (PCMPEQB masks
// subtracted, at most n <= 16 per byte lane), reduced at the end (PSADBW,
// PMAXUB folds).
//
// Registers: SI c, CX blocks left, X2 maximum bins, X3 bin-0 counts,
// X6 31, X7 zero.
TEXT ·zeroBinStatsSSE41(SB), NOSPLIT, $0-32
	MOVQ	c+0(FP), SI
	MOVQ	n+8(FP), CX
	MOVOU	·k31w(SB), X6
	PXOR	X7, X7
	PXOR	X2, X2
	PXOR	X3, X3
	TESTQ	CX, CX
	JZ	zdone

zloop:
	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	PABSW	X0, X0
	PABSW	X1, X1
	PSRLW	$3, X0
	PSRLW	$3, X1
	PMINUW	X6, X0
	PMINUW	X6, X1
	PACKUSWB	X1, X0
	PMAXUB	X0, X2
	PCMPEQB	X7, X0
	PSUBB	X0, X3
	ADDQ	$32, SI
	DECQ	CX
	JNZ	zloop

zdone:
	PSADBW	X7, X3
	PSHUFD	$0x4e, X3, X4
	PADDQ	X4, X3
	MOVQ	X3, AX
	PSHUFD	$0x4e, X2, X4
	PMAXUB	X4, X2
	PSHUFD	$0xb1, X2, X4
	PMAXUB	X4, X2
	MOVO	X2, X4
	PSRLL	$16, X4
	PMAXUB	X4, X2
	MOVO	X2, X4
	PSRLW	$8, X4
	PMAXUB	X4, X2
	MOVL	X2, BX
	MOVBQZX	BX, BX
	MOVQ	AX, count0+16(FP)
	MOVQ	BX, maxBin+24(FP)
	RET
