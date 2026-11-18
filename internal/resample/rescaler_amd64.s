//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 and AVX2 row kernels of the rescaler. Algorithm adapted from
// libwebp (BSD-3-Clause): src/dsp/rescaler.c (the plain C kernels, which
// these reproduce bit for bit), src/dsp/rescaler_sse2.c (the export
// kernels, the 4-channel shrink) and src/dsp/alpha_processing_sse2.c
// (MultARGBRow_SSE2). The SSE4.1 kernels use legacy encodings only, the
// AVX2 ones VEX encodings only, ending with VZEROUPPER.
//
// Shrinking, every destination pixel x of a row is computed from the
// column table (see initCols): its span of cnt source pixels, of which B is
// the last, and na, the weight of B that belongs to pixel x+1:
//
//	sum   = carry + (sum of the span)
//	fr    = B * na                       (32-bit, wrapping)
//	frow  = sum*xSub - fr; acc += frow   (32-bit, wrapping)
//	carry = (fr*fx + 2^31) >> 32         (MULT_FIX, 64-bit product)
//
// carry is 0 for the first pixel of a row and only depends on the previous
// pixel's fr, so the pixels are independent of each other once the spans
// are summed.

// rnd31<> holds 2^31 in four 64-bit lanes (the MULT_FIX rounder).
DATA rnd31<>+0x00(SB)/8, $0x80000000
DATA rnd31<>+0x08(SB)/8, $0x80000000
DATA rnd31<>+0x10(SB)/8, $0x80000000
DATA rnd31<>+0x18(SB)/8, $0x80000000
GLOBL rnd31<>(SB), RODATA|NOPTR, $32

// c255<> holds 255 in eight 32-bit lanes (clip8).
DATA c255<>+0x00(SB)/8, $0x000000ff000000ff
DATA c255<>+0x08(SB)/8, $0x000000ff000000ff
DATA c255<>+0x10(SB)/8, $0x000000ff000000ff
DATA c255<>+0x18(SB)/8, $0x000000ff000000ff
GLOBL c255<>(SB), RODATA|NOPTR, $32

// rot1<> are the VPERMD indices rotating eight words up by one lane.
DATA rot1<>+0x00(SB)/8, $0x0000000000000007
DATA rot1<>+0x08(SB)/8, $0x0000000200000001
DATA rot1<>+0x10(SB)/8, $0x0000000400000003
DATA rot1<>+0x18(SB)/8, $0x0000000600000005
GLOBL rot1<>(SB), RODATA|NOPTR, $32

// shr4Mask<> + 16*h keeps the first h pixels (4h bytes) of 16.
DATA shr4Mask<>+0x00(SB)/8, $0
DATA shr4Mask<>+0x08(SB)/8, $0
DATA shr4Mask<>+0x10(SB)/8, $0x00000000ffffffff
DATA shr4Mask<>+0x18(SB)/8, $0
DATA shr4Mask<>+0x20(SB)/8, $0xffffffffffffffff
DATA shr4Mask<>+0x28(SB)/8, $0
DATA shr4Mask<>+0x30(SB)/8, $0xffffffffffffffff
DATA shr4Mask<>+0x38(SB)/8, $0x00000000ffffffff
GLOBL shr4Mask<>(SB), RODATA|NOPTR, $64

// shr1Mask<> + 16*h keeps the first h bytes of 16.
DATA shr1Mask<>+0x00(SB)/8, $0
DATA shr1Mask<>+0x08(SB)/8, $0
DATA shr1Mask<>+0x10(SB)/8, $0x00000000000000ff
DATA shr1Mask<>+0x18(SB)/8, $0
DATA shr1Mask<>+0x20(SB)/8, $0x000000000000ffff
DATA shr1Mask<>+0x28(SB)/8, $0
DATA shr1Mask<>+0x30(SB)/8, $0x0000000000ffffff
DATA shr1Mask<>+0x38(SB)/8, $0
DATA shr1Mask<>+0x40(SB)/8, $0x00000000ffffffff
DATA shr1Mask<>+0x48(SB)/8, $0
DATA shr1Mask<>+0x50(SB)/8, $0x000000ffffffffff
DATA shr1Mask<>+0x58(SB)/8, $0
DATA shr1Mask<>+0x60(SB)/8, $0x0000ffffffffffff
DATA shr1Mask<>+0x68(SB)/8, $0
DATA shr1Mask<>+0x70(SB)/8, $0x00ffffffffffffff
DATA shr1Mask<>+0x78(SB)/8, $0
DATA shr1Mask<>+0x80(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0x88(SB)/8, $0
DATA shr1Mask<>+0x90(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0x98(SB)/8, $0x00000000000000ff
DATA shr1Mask<>+0xa0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xa8(SB)/8, $0x000000000000ffff
DATA shr1Mask<>+0xb0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xb8(SB)/8, $0x0000000000ffffff
DATA shr1Mask<>+0xc0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xc8(SB)/8, $0x00000000ffffffff
DATA shr1Mask<>+0xd0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xd8(SB)/8, $0x000000ffffffffff
DATA shr1Mask<>+0xe0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xe8(SB)/8, $0x0000ffffffffffff
DATA shr1Mask<>+0xf0(SB)/8, $0xffffffffffffffff
DATA shr1Mask<>+0xf8(SB)/8, $0x00ffffffffffffff
GLOBL shr1Mask<>(SB), RODATA|NOPTR, $256

// MULTFIX_SSE(x, k, t) sets t to MULT_FIX(x, k) = (x*k + 2^31) >> 32 per
// 32-bit lane, leaving x shifted right by 32 bits per 64-bit lane. X14
// holds rnd31<>.
#define MULTFIX_SSE(x, k, t) \
	MOVO    x, t      \
	PMULULQ k, t      \
	PSRLQ   $32, x    \
	PMULULQ k, x      \
	PADDQ   X14, t    \
	PADDQ   X14, x    \
	PSRLQ   $32, t    \
	PBLENDW $0xcc, x, t

// MULTFIX_AVX(x, k, t, u) is MULTFIX_SSE in VEX form (x preserved, u
// scratch), with Y14 holding rnd31<>; x, k, t, u are all X or all Y
// registers.
#define MULTFIX_AVX(x, k, t, u) \
	VPMULUDQ k, x, t     \
	VPSRLQ   $32, x, u   \
	VPMULUDQ k, u, u     \
	VPADDQ   Y14, t, t   \
	VPADDQ   Y14, u, u   \
	VPSRLQ   $32, t, t   \
	VPBLENDD $0xaa, u, t, t

// func shrinkRow4SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)
//
// WebPRescalerImportRowShrink_C for 4 channels, one destination pixel (4
// lanes, one per channel) per iteration, like RescalerImportRowShrink_SSE2
// but driven by the column table and exact for any na. The span is summed
// in 16-bit lanes (exact for cnt <= 511): first a masked 16-byte load for
// its cnt&3 leading pixels, then cnt>>2 full 16-byte chunks. Lane i<4
// collects channel i of the even pixels of a chunk, lane 4+i of the odd
// ones.
//
// DI frow, DX acc (nil: no accumulation), SI src, BX cols, CX pixels left,
// R10 shr4Mask, R8 cnt then chunks, R9 na, R11 cnt&3, R12 mask offset.
// X0 carry, X1/X2 source bytes, X3 16-bit sums, X4 B, X5 sum then frow,
// X6 fr, X7 acc, X8 carry out, X12 xSub, X13 fx, X14 rnd31, X15 zero.
TEXT ·shrinkRow4SSE41(SB), NOSPLIT, $0-48
	MOVQ   frow+0(FP), DI
	MOVQ   acc+8(FP), DX
	MOVQ   src+16(FP), SI
	MOVQ   cols+24(FP), BX
	MOVQ   n+32(FP), CX
	TESTQ  CX, CX
	JZ     s4done
	MOVL   xSub+40(FP), AX
	MOVQ   AX, X12
	PSHUFD $0, X12, X12
	MOVL   fx+44(FP), AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13
	MOVOU  rnd31<>(SB), X14
	PXOR   X15, X15
	PXOR   X0, X0
	LEAQ   shr4Mask<>(SB), R10

s4loop:
	MOVL  (BX), R8
	MOVL  4(BX), R9
	ADDQ  $8, BX
	MOVQ  R8, R11
	ANDQ  $3, R11
	SHRQ  $2, R8
	MOVQ  R11, R12
	SHLQ  $4, R12

	// Leading cnt&3 pixels: masked 16-byte load.
	MOVOU     (SI), X1
	MOVOU     (R10)(R12*1), X2
	PAND      X2, X1
	PMOVZXBW  X1, X3
	PUNPCKHBW X15, X1
	PADDW     X1, X3
	LEAQ      (SI)(R11*4), SI
	TESTQ     R8, R8
	JZ        s4last

s4chunk:
	MOVOU     (SI), X1
	PMOVZXBW  X1, X2
	PUNPCKHBW X15, X1
	PADDW     X2, X3
	PADDW     X1, X3
	ADDQ      $16, SI
	DECQ      R8
	JNZ       s4chunk

s4last:
	// B: the last pixel of the span, widened to 4 words.
	PMOVZXBD  -4(SI), X4
	PMOVZXWD  X3, X5
	PUNPCKHWL X15, X3
	PADDL     X3, X5
	PADDL     X0, X5
	MOVQ      R9, X6
	PSHUFD    $0, X6, X6
	PMULLD    X4, X6
	PMULLD    X12, X5
	PSUBL     X6, X5
	MOVOU     X5, (DI)
	ADDQ      $16, DI
	TESTQ     DX, DX
	JZ        s4carry
	MOVOU     (DX), X7
	PADDL     X5, X7
	MOVOU     X7, (DX)
	ADDQ      $16, DX

s4carry:
	MULTFIX_SSE(X6, X13, X0)
	DECQ CX
	JNZ  s4loop

s4done:
	RET

// func shrinkRow1SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)
//
// WebPRescalerImportRowShrink_C for 1 channel and spans of any length, one
// destination pixel per iteration: the span is summed with PSADBW, first a
// masked 16-byte load for its cnt&15 leading bytes, then cnt>>4 full
// 16-byte chunks (64-bit sums, exact for any cnt). The rest is the scalar
// math above.
//
// DI frow, DX acc (nil: no accumulation), SI src, BX cols, CX pixels left,
// R10 shr1Mask, AX cnt then chunks, R11 na, R8 cnt&15 then sum and frow,
// R9 mask offset then B and fr, R12 xSub, R13 carry.
// X1 sums, X2 source bytes and mask, X15 zero.
TEXT ·shrinkRow1SSE41(SB), NOSPLIT, $0-48
	MOVQ  frow+0(FP), DI
	MOVQ  acc+8(FP), DX
	MOVQ  src+16(FP), SI
	MOVQ  cols+24(FP), BX
	MOVQ  n+32(FP), CX
	TESTQ CX, CX
	JZ    s1done
	MOVL  xSub+40(FP), R12
	XORL  R13, R13
	LEAQ  shr1Mask<>(SB), R10
	PXOR  X15, X15

s1loop:
	MOVL  (BX), AX
	MOVL  4(BX), R11
	ADDQ  $8, BX
	MOVQ  AX, R8
	ANDQ  $15, R8
	SHRQ  $4, AX
	MOVQ  R8, R9
	SHLQ  $4, R9
	MOVOU (SI), X1
	MOVOU (R10)(R9*1), X2
	PAND  X2, X1
	PSADBW X15, X1
	ADDQ  R8, SI
	TESTQ AX, AX
	JZ    s1last

s1chunk:
	MOVOU  (SI), X2
	PSADBW X15, X2
	PADDQ  X2, X1
	ADDQ   $16, SI
	DECQ   AX
	JNZ    s1chunk

s1last:
	PSHUFD  $0x4e, X1, X2
	PADDQ   X2, X1
	MOVQ    X1, R8
	ADDL    R13, R8
	MOVBLZX -1(SI), R9
	IMULL   R11, R9
	IMULL   R12, R8
	SUBL    R9, R8
	MOVL    R8, (DI)
	ADDQ    $4, DI
	TESTQ   DX, DX
	JZ      s1carry
	ADDL    R8, (DX)
	ADDQ    $4, DX

s1carry:
	// carry = (fr*fx + 2^31) >> 32; fr and fx are below 2^32, so the
	// sum fits 64 bits and the low half of the signed product is exact.
	MOVL  R9, R13
	MOVL  fx+44(FP), R8
	IMULQ R8, R13
	MOVL  $0x80000000, R8
	ADDQ  R8, R13
	SHRQ  $32, R13
	DECQ  CX
	JNZ   s1loop

s1done:
	RET

// The 1-channel record kernels: 8 destination pixels per record (see
// buildShrink1PRecords and buildShrink1SRecords in rescaler_amd64.go).
// PSHUFB gathers each span from its 16-byte window, right-aligned in a
// zeroed 8- or 16-byte block, so that PSADBW sums it and its last pixel
// (B) sits in the top byte of the block. The math is then done for 4
// (SSE4.1) or 8 (AVX2) pixels at once; the carry of pixel j is MULT_FIX of
// pixel j-1's fr, so the carries computed for a group are shifted up by
// one lane, the previous group's last one entering lane 0.
//
// Registers of both: DI frow, DX acc (nil: no accumulation), SI src, BX
// records, CX records left, AX/R8/R9/R10 window offsets. X0 sums then
// frow, X4 B, X6 fr, X7 carries out, X9 carries in, X11 previous carries
// out, X12 xSub, X13 fx, X14 rnd31, X15 zero (Y for AVX2, where Y10 holds
// rot1 and Y11 the previous carries out rotated).

// SH1_TAIL_SSE completes a group of 4 pixels from the sums (X0), B (X4)
// and the record's na at na(BX); skip is a fresh label.
#define SH1_TAIL_SSE(na, skip) \
	MOVOU   na(BX), X6         \
	PMULLD  X4, X6             \
	MOVO    X6, X5             \
	MULTFIX_SSE(X5, X13, X7)   \
	MOVO    X7, X9             \
	PALIGNR $12, X11, X9       \
	MOVO    X7, X11            \
	PADDL   X9, X0             \
	PMULLD  X12, X0            \
	PSUBL   X6, X0             \
	MOVOU   X0, (DI)           \
	ADDQ    $16, DI            \
	TESTQ   DX, DX             \
	JZ      skip               \
	MOVOU   (DX), X10          \
	PADDL   X0, X10            \
	MOVOU   X10, (DX)          \
	ADDQ    $16, DX            \
skip:

// SH1_SETUP_SSE sets the constants of a record kernel up from AX (xSub)
// and R8 (fx).
#define SH1_SETUP_SSE \
	MOVQ   AX, X12          \
	PSHUFD $0, X12, X12     \
	MOVQ   R8, X13          \
	PSHUFD $0, X13, X13     \
	MOVOU  rnd31<>(SB), X14 \
	PXOR   X15, X15         \
	PXOR   X11, X11

// SH1P_GROUP_SSE does a group of 4 pixels of a P record: the PSHUFB
// indices of its pairs at ia/ib, their window offsets at oa/ob.
#define SH1P_GROUP_SSE(ia, oa, ib, ob, na, skip) \
	MOVL   oa(BX), AX           \
	MOVL   ob(BX), R8           \
	MOVOU  (SI)(AX*1), X0       \
	MOVOU  (SI)(R8*1), X1       \
	MOVOU  ia(BX), X2           \
	PSHUFB X2, X0               \
	MOVOU  ib(BX), X3           \
	PSHUFB X3, X1               \
	MOVO   X0, X4               \
	PSRLQ  $56, X4              \
	MOVO   X1, X5               \
	PSRLQ  $56, X5              \
	SHUFPS $0x88, X5, X4        \
	PSADBW X15, X0              \
	PSADBW X15, X1              \
	SHUFPS $0x88, X1, X0        \
	SH1_TAIL_SSE(na, skip)

// func shrink1PSSE41(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)
TEXT ·shrink1PSSE41(SB), NOSPLIT, $0-48
	MOVQ frow+0(FP), DI
	MOVQ acc+8(FP), DX
	MOVQ src+16(FP), SI
	MOVQ rec+24(FP), BX
	MOVQ blocks+32(FP), CX
	MOVL xSub+40(FP), AX
	MOVL fx+44(FP), R8
	SH1_SETUP_SSE
	TESTQ CX, CX
	JZ    p1done

p1loop:
	SH1P_GROUP_SSE(0, 96, 32, 104, 64, p1skip0)
	SH1P_GROUP_SSE(16, 100, 48, 108, 80, p1skip1)
	ADDQ $128, BX
	DECQ CX
	JNZ  p1loop

p1done:
	RET

// SH1S_GROUP_SSE does a group of 4 pixels of an S record: the PSHUFB
// indices of pixel j at i<j>, its window offset at o<j>.
#define SH1S_GROUP_SSE(i0, i1, i2, i3, o0, o1, o2, o3, na, skip) \
	MOVL       o0(BX), AX     \
	MOVL       o1(BX), R8     \
	MOVL       o2(BX), R9     \
	MOVL       o3(BX), R10    \
	MOVOU      (SI)(AX*1), X0 \
	MOVOU      i0(BX), X4     \
	PSHUFB     X4, X0         \
	MOVOU      (SI)(R8*1), X1 \
	MOVOU      i1(BX), X4     \
	PSHUFB     X4, X1         \
	MOVOU      (SI)(R9*1), X2 \
	MOVOU      i2(BX), X4     \
	PSHUFB     X4, X2         \
	MOVOU      (SI)(R10*1), X3 \
	MOVOU      i3(BX), X4     \
	PSHUFB     X4, X3         \
	MOVO       X0, X4         \
	PUNPCKHLQ  X1, X4         \
	MOVO       X2, X5         \
	PUNPCKHLQ  X3, X5         \
	PUNPCKHQDQ X5, X4         \
	PSRLL      $24, X4        \
	PSADBW     X15, X0        \
	PSADBW     X15, X1        \
	PSADBW     X15, X2        \
	PSADBW     X15, X3        \
	SHUFPS     $0x88, X1, X0  \
	SHUFPS     $0x88, X3, X2  \
	MOVO       X0, X5         \
	SHUFPS     $0x88, X2, X0  \
	SHUFPS     $0xdd, X2, X5  \
	PADDL      X5, X0         \
	SH1_TAIL_SSE(na, skip)

// func shrink1SSSE41(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)
TEXT ·shrink1SSSE41(SB), NOSPLIT, $0-48
	MOVQ frow+0(FP), DI
	MOVQ acc+8(FP), DX
	MOVQ src+16(FP), SI
	MOVQ rec+24(FP), BX
	MOVQ blocks+32(FP), CX
	MOVL xSub+40(FP), AX
	MOVL fx+44(FP), R8
	SH1_SETUP_SSE
	TESTQ CX, CX
	JZ    q1done

q1loop:
	SH1S_GROUP_SSE(0, 32, 64, 96, 160, 164, 168, 172, 128, q1skip0)
	SH1S_GROUP_SSE(16, 48, 80, 112, 176, 180, 184, 188, 144, q1skip1)
	ADDQ $192, BX
	DECQ CX
	JNZ  q1loop

q1done:
	RET

// SH1_SETUP_AVX sets the constants of an AVX2 record kernel up from AX
// (xSub) and R8 (fx).
#define SH1_SETUP_AVX \
	VMOVQ        AX, X12          \
	VPBROADCASTD X12, Y12         \
	VMOVQ        R8, X13          \
	VPBROADCASTD X13, Y13         \
	VMOVDQU      rnd31<>(SB), Y14 \
	VMOVDQU      rot1<>(SB), Y10  \
	VPXOR        Y15, Y15, Y15    \
	VPXOR        Y11, Y11, Y11

// SH1_TAIL_AVX completes a block of 8 pixels from the sums (Y0), B (Y4)
// and the record's na at na(BX); skip is a fresh label.
#define SH1_TAIL_AVX(na, skip) \
	VPMULLD  na(BX), Y4, Y6       \
	MULTFIX_AVX(Y6, Y13, Y7, Y8)  \
	VPERMD   Y7, Y10, Y9          \
	VPBLENDD $0x01, Y11, Y9, Y7   \
	VMOVDQA  Y9, Y11              \
	VPADDD   Y7, Y0, Y0           \
	VPMULLD  Y12, Y0, Y0          \
	VPSUBD   Y6, Y0, Y0           \
	VMOVDQU  Y0, (DI)             \
	ADDQ     $32, DI              \
	TESTQ    DX, DX               \
	JZ       skip                 \
	VPADDD   (DX), Y0, Y0         \
	VMOVDQU  Y0, (DX)             \
	ADDQ     $32, DX              \
skip:

// func shrink1PAVX2(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)
//
// Y0 holds the A windows of both groups (one per 128-bit lane), Y1 the B
// windows, so that each lane does what SH1P_GROUP_SSE does.
TEXT ·shrink1PAVX2(SB), NOSPLIT, $0-48
	MOVQ frow+0(FP), DI
	MOVQ acc+8(FP), DX
	MOVQ src+16(FP), SI
	MOVQ rec+24(FP), BX
	MOVQ blocks+32(FP), CX
	MOVL xSub+40(FP), AX
	MOVL fx+44(FP), R8
	SH1_SETUP_AVX
	TESTQ CX, CX
	JZ    p2done

p2loop:
	MOVL        96(BX), AX
	MOVL        100(BX), R8
	MOVL        104(BX), R9
	MOVL        108(BX), R10
	VMOVDQU     (SI)(AX*1), X0
	VINSERTI128 $1, (SI)(R8*1), Y0, Y0
	VMOVDQU     (SI)(R9*1), X1
	VINSERTI128 $1, (SI)(R10*1), Y1, Y1
	VPSHUFB     (BX), Y0, Y0
	VPSHUFB     32(BX), Y1, Y1
	VPSRLQ      $56, Y0, Y4
	VPSRLQ      $56, Y1, Y5
	VSHUFPS     $0x88, Y5, Y4, Y4
	VPSADBW     Y15, Y0, Y0
	VPSADBW     Y15, Y1, Y1
	VSHUFPS     $0x88, Y1, Y0, Y0
	SH1_TAIL_AVX(64, p2skip)
	ADDQ        $128, BX
	DECQ        CX
	JNZ         p2loop

p2done:
	VZEROUPPER
	RET

// SH1S_LOAD_AVX loads the windows of pixels j and j+4 of an S record into
// the two lanes of y and gathers their spans (indices at i(BX)).
#define SH1S_LOAD_AVX(oj, oj4, i, x, y) \
	MOVL        oj(BX), AX              \
	MOVL        oj4(BX), R8             \
	VMOVDQU     (SI)(AX*1), x           \
	VINSERTI128 $1, (SI)(R8*1), y, y    \
	VPSHUFB     i(BX), y, y

// func shrink1SAVX2(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)
//
// Yk holds the windows of pixels k and k+4 (k < 4), so that lane 0 does
// pixels 0-3 and lane 1 pixels 4-7 like SH1S_GROUP_SSE.
TEXT ·shrink1SAVX2(SB), NOSPLIT, $0-48
	MOVQ frow+0(FP), DI
	MOVQ acc+8(FP), DX
	MOVQ src+16(FP), SI
	MOVQ rec+24(FP), BX
	MOVQ blocks+32(FP), CX
	MOVL xSub+40(FP), AX
	MOVL fx+44(FP), R8
	SH1_SETUP_AVX
	TESTQ CX, CX
	JZ    q2done

q2loop:
	SH1S_LOAD_AVX(160, 176, 0, X0, Y0)
	SH1S_LOAD_AVX(164, 180, 32, X1, Y1)
	SH1S_LOAD_AVX(168, 184, 64, X2, Y2)
	SH1S_LOAD_AVX(172, 188, 96, X3, Y3)
	VPUNPCKHDQ  Y1, Y0, Y4
	VPUNPCKHDQ  Y3, Y2, Y5
	VPUNPCKHQDQ Y5, Y4, Y4
	VPSRLD      $24, Y4, Y4
	VPSADBW     Y15, Y0, Y0
	VPSADBW     Y15, Y1, Y1
	VPSADBW     Y15, Y2, Y2
	VPSADBW     Y15, Y3, Y3
	VSHUFPS     $0x88, Y1, Y0, Y0
	VSHUFPS     $0x88, Y3, Y2, Y2
	VSHUFPS     $0xdd, Y2, Y0, Y5
	VSHUFPS     $0x88, Y2, Y0, Y0
	VPADDD      Y5, Y0, Y0
	SH1_TAIL_AVX(128, q2skip)
	ADDQ        $192, BX
	DECQ        CX
	JNZ         q2loop

q2done:
	VZEROUPPER
	RET

// func expandRow4SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xAdd uint32)
//
// WebPRescalerImportRowExpand_C for 4 channels and a source at least two
// pixels wide, one destination pixel per iteration:
//
//	frow = right*xAdd + (left - right)*w; acc += frow   (32-bit, wrapping)
//
// where left/right are the 8 bytes at source pixel cols[2x] and w is
// cols[2x+1].
//
// DI frow, DX acc (nil: no accumulation), SI src, BX cols, CX pixels left,
// R8 left pixel, R9 w. X0 left then the result, X1 right, X2 w, X3 acc,
// X12 xAdd.
TEXT ·expandRow4SSE41(SB), NOSPLIT, $0-44
	MOVQ   frow+0(FP), DI
	MOVQ   acc+8(FP), DX
	MOVQ   src+16(FP), SI
	MOVQ   cols+24(FP), BX
	MOVQ   n+32(FP), CX
	TESTQ  CX, CX
	JZ     e4done
	MOVL   xAdd+40(FP), AX
	MOVQ   AX, X12
	PSHUFD $0, X12, X12

e4loop:
	MOVL     (BX), R8
	MOVL     4(BX), R9
	ADDQ     $8, BX
	PMOVZXBD (SI)(R8*4), X0
	PMOVZXBD 4(SI)(R8*4), X1
	PSUBL    X1, X0
	MOVQ     R9, X2
	PSHUFD   $0, X2, X2
	PMULLD   X2, X0
	PMULLD   X12, X1
	PADDL    X1, X0
	MOVOU    X0, (DI)
	ADDQ     $16, DI
	TESTQ    DX, DX
	JZ       e4next
	MOVOU    (DX), X3
	PADDL    X0, X3
	MOVOU    X3, (DX)
	ADDQ     $16, DX

e4next:
	DECQ CX
	JNZ  e4loop

e4done:
	RET

// Export kernels: 8 values per iteration, n > 0 and a multiple of 8, like
// libwebp's RescalerExportRowExpand/Shrink_SSE2. MULT_FIX(v, scale) =
// (v*scale + 2^31) >> 32 is two PMULUDQ (even and odd lanes); clip8 of the
// unsigned result is PMINUD with 255 before the saturating packs (libwebp
// packs signed, which differs for results of 2^31 and more).

// EXP_STORE_SSE stores clip8 of X0 and X1 (8 words) at DI and advances DI.
// X15 holds c255<>.
#define EXP_STORE_SSE \
	PMINUD   X15, X0 \
	PMINUD   X15, X1 \
	PACKUSDW X1, X0  \
	PACKUSWB X0, X0  \
	MOVQ     X0, (DI) \
	ADDQ     $8, DI

// EXP_STORE_AVX stores clip8 of Y0 (8 words) at DI and advances DI. Y15
// holds c255<>.
#define EXP_STORE_AVX \
	VPMINUD      Y15, Y0, Y0 \
	VEXTRACTI128 $1, Y0, X1  \
	VPACKUSDW    X1, X0, X0  \
	VPACKUSWB    X0, X0, X0  \
	VMOVQ        X0, (DI)    \
	ADDQ         $8, DI

// func exportShrinkSSE41(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)
//
// WebPRescalerExportRowShrink_C with yscale != 0:
//
//	fr = (frow*yscale) >> 32; dst = clip8(MULT_FIX(irow - fr, fxy)); irow = fr
//
// DI dst, SI irow, BX frow, CX values left. X0-X3 values, X4-X7 products,
// X12 yscale, X13 fxy, X14 rnd31, X15 c255.
TEXT ·exportShrinkSSE41(SB), NOSPLIT, $0-40
	MOVQ   dst+0(FP), DI
	MOVQ   irow+8(FP), SI
	MOVQ   frow+16(FP), BX
	MOVQ   n+24(FP), CX
	MOVL   yscale+32(FP), AX
	MOVQ   AX, X12
	PSHUFD $0, X12, X12
	MOVL   fxy+36(FP), AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13
	MOVOU  rnd31<>(SB), X14
	MOVOU  c255<>(SB), X15

exsloop:
	// fr = (frow*yscale) >> 32 per lane, in X4 and X5.
	MOVOU   (BX), X0
	MOVOU   16(BX), X1
	MOVO    X0, X4
	PMULULQ X12, X4
	PSRLQ   $32, X0
	PMULULQ X12, X0
	PSRLQ   $32, X4
	PBLENDW $0xcc, X0, X4
	MOVO    X1, X5
	PMULULQ X12, X5
	PSRLQ   $32, X1
	PMULULQ X12, X1
	PSRLQ   $32, X5
	PBLENDW $0xcc, X1, X5
	MOVOU   (SI), X2
	MOVOU   16(SI), X3
	MOVOU   X4, (SI)
	MOVOU   X5, 16(SI)
	PSUBL   X4, X2
	PSUBL   X5, X3
	MULTFIX_SSE(X2, X13, X0)
	MULTFIX_SSE(X3, X13, X1)
	EXP_STORE_SSE
	ADDQ    $32, SI
	ADDQ    $32, BX
	SUBQ    $8, CX
	JNZ     exsloop
	RET

// func exportShrinkAVX2(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)
TEXT ·exportShrinkAVX2(SB), NOSPLIT, $0-40
	MOVQ         dst+0(FP), DI
	MOVQ         irow+8(FP), SI
	MOVQ         frow+16(FP), BX
	MOVQ         n+24(FP), CX
	MOVL         yscale+32(FP), AX
	VMOVQ        AX, X12
	VPBROADCASTD X12, Y12
	MOVL         fxy+36(FP), AX
	VMOVQ        AX, X13
	VPBROADCASTD X13, Y13
	VMOVDQU      rnd31<>(SB), Y14
	VMOVDQU      c255<>(SB), Y15

exs2loop:
	VMOVDQU  (BX), Y2
	VPMULUDQ Y12, Y2, Y4
	VPSRLQ   $32, Y2, Y5
	VPMULUDQ Y12, Y5, Y5
	VPSRLQ   $32, Y4, Y4
	VPBLENDD $0xaa, Y5, Y4, Y4
	VMOVDQU  (SI), Y3
	VMOVDQU  Y4, (SI)
	VPSUBD   Y4, Y3, Y3
	MULTFIX_AVX(Y3, Y13, Y0, Y5)
	EXP_STORE_AVX
	ADDQ     $32, SI
	ADDQ     $32, BX
	SUBQ     $8, CX
	JNZ      exs2loop
	VZEROUPPER
	RET

// func exportShrinkZeroSSE41(dst *byte, irow *uint32, n int, fxy uint32)
//
// WebPRescalerExportRowShrink_C with yscale == 0:
// dst = clip8(MULT_FIX(irow, fxy)); irow = 0.
TEXT ·exportShrinkZeroSSE41(SB), NOSPLIT, $0-28
	MOVQ   dst+0(FP), DI
	MOVQ   irow+8(FP), SI
	MOVQ   n+16(FP), CX
	MOVL   fxy+24(FP), AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13
	MOVOU  rnd31<>(SB), X14
	MOVOU  c255<>(SB), X15
	PXOR   X8, X8

exzloop:
	MOVOU (SI), X2
	MOVOU 16(SI), X3
	MOVOU X8, (SI)
	MOVOU X8, 16(SI)
	MULTFIX_SSE(X2, X13, X0)
	MULTFIX_SSE(X3, X13, X1)
	EXP_STORE_SSE
	ADDQ  $32, SI
	SUBQ  $8, CX
	JNZ   exzloop
	RET

// func exportShrinkZeroAVX2(dst *byte, irow *uint32, n int, fxy uint32)
TEXT ·exportShrinkZeroAVX2(SB), NOSPLIT, $0-28
	MOVQ         dst+0(FP), DI
	MOVQ         irow+8(FP), SI
	MOVQ         n+16(FP), CX
	MOVL         fxy+24(FP), AX
	VMOVQ        AX, X13
	VPBROADCASTD X13, Y13
	VMOVDQU      rnd31<>(SB), Y14
	VMOVDQU      c255<>(SB), Y15
	VPXOR        Y8, Y8, Y8

exz2loop:
	VMOVDQU (SI), Y3
	VMOVDQU Y8, (SI)
	MULTFIX_AVX(Y3, Y13, Y0, Y5)
	EXP_STORE_AVX
	ADDQ    $32, SI
	SUBQ    $8, CX
	JNZ     exz2loop
	VZEROUPPER
	RET

// func exportExpandSSE41(dst *byte, frow *uint32, n int, fy uint32)
//
// WebPRescalerExportRowExpand_C with y_accum == 0:
// dst = clip8(MULT_FIX(frow, fy)).
TEXT ·exportExpandSSE41(SB), NOSPLIT, $0-28
	MOVQ   dst+0(FP), DI
	MOVQ   frow+8(FP), BX
	MOVQ   n+16(FP), CX
	MOVL   fy+24(FP), AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13
	MOVOU  rnd31<>(SB), X14
	MOVOU  c255<>(SB), X15

exeloop:
	MOVOU (BX), X2
	MOVOU 16(BX), X3
	MULTFIX_SSE(X2, X13, X0)
	MULTFIX_SSE(X3, X13, X1)
	EXP_STORE_SSE
	ADDQ  $32, BX
	SUBQ  $8, CX
	JNZ   exeloop
	RET

// func exportExpandAVX2(dst *byte, frow *uint32, n int, fy uint32)
TEXT ·exportExpandAVX2(SB), NOSPLIT, $0-28
	MOVQ         dst+0(FP), DI
	MOVQ         frow+8(FP), BX
	MOVQ         n+16(FP), CX
	MOVL         fy+24(FP), AX
	VMOVQ        AX, X13
	VPBROADCASTD X13, Y13
	VMOVDQU      rnd31<>(SB), Y14
	VMOVDQU      c255<>(SB), Y15

exe2loop:
	VMOVDQU (BX), Y3
	MULTFIX_AVX(Y3, Y13, Y0, Y5)
	EXP_STORE_AVX
	ADDQ    $32, BX
	SUBQ    $8, CX
	JNZ     exe2loop
	VZEROUPPER
	RET

// EXM_MIX_SSE(f, g, t, u) sets t to (a*f + b*g + 2^31) >> 32 per lane
// (X11 a, X12 b), clobbering f, g and u.
#define EXM_MIX_SSE(f, g, t, u) \
	MOVO    f, t       \
	PMULULQ X11, t     \
	MOVO    g, u       \
	PMULULQ X12, u     \
	PADDQ   u, t       \
	PSRLQ   $32, f     \
	PMULULQ X11, f     \
	PSRLQ   $32, g     \
	PMULULQ X12, g     \
	PADDQ   g, f       \
	PADDQ   X14, t     \
	PADDQ   X14, f     \
	PSRLQ   $32, t     \
	PBLENDW $0xcc, f, t

// func exportExpandMixSSE41(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)
//
// WebPRescalerExportRowExpand_C with y_accum != 0:
//
//	j = (a*frow + b*irow + 2^31) >> 32; dst = clip8(MULT_FIX(j, fy))
//
// a + b == 2^32, so the 64-bit sum cannot overflow.
// DI dst, BX frow, SI irow, CX values left. X0-X7 values and products,
// X11 a, X12 b, X13 fy, X14 rnd31, X15 c255.
TEXT ·exportExpandMixSSE41(SB), NOSPLIT, $0-44
	MOVQ   dst+0(FP), DI
	MOVQ   frow+8(FP), BX
	MOVQ   irow+16(FP), SI
	MOVQ   n+24(FP), CX
	MOVL   a+32(FP), AX
	MOVQ   AX, X11
	PSHUFD $0, X11, X11
	MOVL   b+36(FP), AX
	MOVQ   AX, X12
	PSHUFD $0, X12, X12
	MOVL   fy+40(FP), AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13
	MOVOU  rnd31<>(SB), X14
	MOVOU  c255<>(SB), X15

exmloop:
	MOVOU (BX), X0
	MOVOU (SI), X1
	MOVOU 16(BX), X2
	MOVOU 16(SI), X3
	EXM_MIX_SSE(X0, X1, X4, X5)
	EXM_MIX_SSE(X2, X3, X6, X7)
	MULTFIX_SSE(X4, X13, X0)
	MULTFIX_SSE(X6, X13, X1)
	EXP_STORE_SSE
	ADDQ  $32, BX
	ADDQ  $32, SI
	SUBQ  $8, CX
	JNZ   exmloop
	RET

// func exportExpandMixAVX2(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)
TEXT ·exportExpandMixAVX2(SB), NOSPLIT, $0-44
	MOVQ         dst+0(FP), DI
	MOVQ         frow+8(FP), BX
	MOVQ         irow+16(FP), SI
	MOVQ         n+24(FP), CX
	MOVL         a+32(FP), AX
	VMOVQ        AX, X11
	VPBROADCASTD X11, Y11
	MOVL         b+36(FP), AX
	VMOVQ        AX, X12
	VPBROADCASTD X12, Y12
	MOVL         fy+40(FP), AX
	VMOVQ        AX, X13
	VPBROADCASTD X13, Y13
	VMOVDQU      rnd31<>(SB), Y14
	VMOVDQU      c255<>(SB), Y15

exm2loop:
	VMOVDQU  (BX), Y2
	VMOVDQU  (SI), Y3
	VPMULUDQ Y11, Y2, Y4
	VPMULUDQ Y12, Y3, Y5
	VPADDQ   Y5, Y4, Y4
	VPSRLQ   $32, Y2, Y2
	VPSRLQ   $32, Y3, Y3
	VPMULUDQ Y11, Y2, Y2
	VPMULUDQ Y12, Y3, Y3
	VPADDQ   Y3, Y2, Y2
	VPADDQ   Y14, Y4, Y4
	VPADDQ   Y14, Y2, Y2
	VPSRLQ   $32, Y4, Y4
	VPBLENDD $0xaa, Y2, Y4, Y4
	MULTFIX_AVX(Y4, Y13, Y0, Y5)
	EXP_STORE_AVX
	ADDQ     $32, BX
	ADDQ     $32, SI
	SUBQ     $8, CX
	JNZ      exm2loop
	VZEROUPPER
	RET

// Premultiply kernels: WebPMultARGBRow(..., inverse=0), like libwebp's
// MultARGBRow_SSE2. For every channel value x of a pixel with alpha a,
// libwebp's C code computes (x * (a * KINV_255) + 2^23) >> 24 (and leaves
// a = 255 pixels alone, zeroes a = 0 ones); for all 8-bit x and a that
// equals
//
//	(u * 0x0101) >> 16, u = x*a + 128
//
// (checked exhaustively by TestPremultiplyExhaustive). Alpha itself is
// multiplied by 255, which the same formula maps back to a. u fits 16 bits.

// pmAlpha<> are PSHUFB indices broadcasting each pixel's alpha to its
// color bytes (zero in its alpha byte), twice; pmOne<> has 0xff in the
// alpha bytes, twice.
DATA pmAlpha<>+0x00(SB)/8, $0x8007070780030303
DATA pmAlpha<>+0x08(SB)/8, $0x800f0f0f800b0b0b
DATA pmAlpha<>+0x10(SB)/8, $0x8007070780030303
DATA pmAlpha<>+0x18(SB)/8, $0x800f0f0f800b0b0b
GLOBL pmAlpha<>(SB), RODATA|NOPTR, $32

DATA pmOne<>+0x00(SB)/8, $0xff000000ff000000
DATA pmOne<>+0x08(SB)/8, $0xff000000ff000000
DATA pmOne<>+0x10(SB)/8, $0xff000000ff000000
DATA pmOne<>+0x18(SB)/8, $0xff000000ff000000
GLOBL pmOne<>(SB), RODATA|NOPTR, $32

// w128<> holds 128 and w0101<> 0x0101 in sixteen 16-bit lanes.
DATA w128<>+0x00(SB)/8, $0x0080008000800080
DATA w128<>+0x08(SB)/8, $0x0080008000800080
DATA w128<>+0x10(SB)/8, $0x0080008000800080
DATA w128<>+0x18(SB)/8, $0x0080008000800080
GLOBL w128<>(SB), RODATA|NOPTR, $32

DATA w0101<>+0x00(SB)/8, $0x0101010101010101
DATA w0101<>+0x08(SB)/8, $0x0101010101010101
DATA w0101<>+0x10(SB)/8, $0x0101010101010101
DATA w0101<>+0x18(SB)/8, $0x0101010101010101
GLOBL w0101<>(SB), RODATA|NOPTR, $32

// func premultiplySSE41(row *byte, n int)
//
// Four pixels per iteration, n > 0 and a multiple of 16.
// DI row, CX bytes left. X0/X2 pixels, X1/X3 multipliers, X8 pmAlpha,
// X9 pmOne, X10 w128, X11 w0101, X15 zero.
TEXT ·premultiplySSE41(SB), NOSPLIT, $0-16
	MOVQ  row+0(FP), DI
	MOVQ  n+8(FP), CX
	MOVOU pmAlpha<>(SB), X8
	MOVOU pmOne<>(SB), X9
	MOVOU w128<>(SB), X10
	MOVOU w0101<>(SB), X11
	PXOR  X15, X15

pm1loop:
	MOVOU     (DI), X0
	MOVO      X0, X1
	PSHUFB    X8, X1
	POR       X9, X1
	PMOVZXBW  X0, X2
	PUNPCKHBW X15, X0
	PMOVZXBW  X1, X3
	PUNPCKHBW X15, X1
	PMULLW    X3, X2
	PMULLW    X1, X0
	PADDW     X10, X2
	PADDW     X10, X0
	PMULHUW   X11, X2
	PMULHUW   X11, X0
	PACKUSWB  X0, X2
	MOVOU     X2, (DI)
	ADDQ      $16, DI
	SUBQ      $16, CX
	JNZ       pm1loop
	RET

// func premultiplyAVX2(row *byte, n int)
//
// Eight pixels per iteration (the unpack and pack instructions work within
// 128-bit lanes, so the order is kept), then four if n is an odd multiple
// of 16.
TEXT ·premultiplyAVX2(SB), NOSPLIT, $0-16
	MOVQ    row+0(FP), DI
	MOVQ    n+8(FP), CX
	VMOVDQU pmAlpha<>(SB), Y8
	VMOVDQU pmOne<>(SB), Y9
	VMOVDQU w128<>(SB), Y10
	VMOVDQU w0101<>(SB), Y11
	VPXOR   Y15, Y15, Y15
	CMPQ    CX, $32
	JB      pm2tail

pm2loop:
	VMOVDQU    (DI), Y0
	VPSHUFB    Y8, Y0, Y1
	VPOR       Y9, Y1, Y1
	VPUNPCKLBW Y15, Y0, Y2
	VPUNPCKHBW Y15, Y0, Y0
	VPUNPCKLBW Y15, Y1, Y3
	VPUNPCKHBW Y15, Y1, Y1
	VPMULLW    Y3, Y2, Y2
	VPMULLW    Y1, Y0, Y0
	VPADDW     Y10, Y2, Y2
	VPADDW     Y10, Y0, Y0
	VPMULHUW   Y11, Y2, Y2
	VPMULHUW   Y11, Y0, Y0
	VPACKUSWB  Y0, Y2, Y2
	VMOVDQU    Y2, (DI)
	ADDQ       $32, DI
	SUBQ       $32, CX
	CMPQ       CX, $32
	JAE        pm2loop

pm2tail:
	TESTQ      CX, CX
	JZ         pm2done
	VMOVDQU    (DI), X0
	VPSHUFB    X8, X0, X1
	VPOR       X9, X1, X1
	VPUNPCKLBW X15, X0, X2
	VPUNPCKHBW X15, X0, X0
	VPUNPCKLBW X15, X1, X3
	VPUNPCKHBW X15, X1, X1
	VPMULLW    X3, X2, X2
	VPMULLW    X1, X0, X0
	VPADDW     X10, X2, X2
	VPADDW     X10, X0, X0
	VPMULHUW   X11, X2, X2
	VPMULHUW   X11, X0, X0
	VPACKUSWB  X0, X2, X2
	VMOVDQU    X2, (DI)

pm2done:
	VZEROUPPER
	RET

// func unpremultiplySSE41(row *byte, n int, table *[256]uint32)
//
// WebPMultARGBRow(..., inverse=1) on n bytes (n > 0, a multiple of 16),
// four pixels per iteration: every color byte x of a pixel with alpha a
// becomes ((x*s + 2^23) mod 2^32) >> 24 with s = table[a] = (255<<24)/a
// (libwebp's Mult with GetScale, 32-bit wrapping arithmetic), and a is
// kept. a = 255 (s = 2^24) and a = 0 (s = 0) need no special case. The
// scales are looked up with scalar loads (gathers are slow on many CPUs).
//
// DI row, CX bytes left, R8 table, AX/BX/DX/R9 alphas then scales.
// X0 pixels, X1 scales, X2-X5 one pixel's channels each, X6 scratch,
// X13 2^23, X14 pmOne (the alpha bytes).
TEXT ·unpremultiplySSE41(SB), NOSPLIT, $0-24
	MOVQ   row+0(FP), DI
	MOVQ   n+8(FP), CX
	MOVQ   table+16(FP), R8
	MOVOU  pmOne<>(SB), X14
	MOVL   $0x800000, AX
	MOVQ   AX, X13
	PSHUFD $0, X13, X13

um1loop:
	// Four opaque pixels stay as they are: skip them when no alpha bit
	// is clear (PANDN X14, X6 is X6 = ^X6 & X14).
	MOVOU    (DI), X0
	MOVO     X0, X6
	PANDN    X14, X6
	PTEST    X6, X6
	JZ       um1next
	MOVBLZX  3(DI), AX
	MOVBLZX  7(DI), BX
	MOVBLZX  11(DI), DX
	MOVBLZX  15(DI), R9
	MOVL     (R8)(AX*4), AX
	MOVL     (R8)(BX*4), BX
	MOVL     (R8)(DX*4), DX
	MOVL     (R8)(R9*4), R9
	MOVQ     AX, X1
	PINSRD   $1, BX, X1
	PINSRD   $2, DX, X1
	PINSRD   $3, R9, X1
	PMOVZXBD 0(DI), X2
	PMOVZXBD 4(DI), X3
	PMOVZXBD 8(DI), X4
	PMOVZXBD 12(DI), X5
	PSHUFD   $0x00, X1, X6
	PMULLD   X6, X2
	PSHUFD   $0x55, X1, X6
	PMULLD   X6, X3
	PSHUFD   $0xaa, X1, X6
	PMULLD   X6, X4
	PSHUFD   $0xff, X1, X6
	PMULLD   X6, X5
	PADDL    X13, X2
	PADDL    X13, X3
	PADDL    X13, X4
	PADDL    X13, X5
	PSRLL    $24, X2
	PSRLL    $24, X3
	PSRLL    $24, X4
	PSRLL    $24, X5
	PACKUSDW X3, X2
	PACKUSDW X5, X4
	PACKUSWB X4, X2
	PAND     X14, X0
	MOVO     X14, X6
	PANDN    X2, X6
	POR      X0, X6
	MOVOU    X6, (DI)

um1next:
	ADDQ     $16, DI
	SUBQ     $16, CX
	JNZ      um1loop
	RET
