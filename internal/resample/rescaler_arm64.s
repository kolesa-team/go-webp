//go:build arm64 && !purego

#include "textflag.h"

// NEON row kernels of the rescaler. Algorithm adapted from libwebp
// (BSD-3-Clause): src/dsp/rescaler.c (the plain C kernels, which these
// reproduce bit for bit; libwebp's own rescaler_neon.c rounds
// differently) and src/dsp/rescaler_sse2.c (the 4-channel shrink).
//
// Instructions the Go assembler does not know are emitted with WORD; each
// macro names the instruction, the encodings were checked against clang.

// ADDHN Vd.2S, Vn.2D, Vm.2D: Vd.S[i] = (Vn.D[i] + Vm.D[i]) >> 32, upper half cleared.
#define ADDHN_2S(d, n, m) WORD $(0x0EA04000 | ((m)<<16) | ((n)<<5) | (d))
// ADDHN2 Vd.4S, Vn.2D, Vm.2D: as ADDHN into lanes 2 and 3.
#define ADDHN2_4S(d, n, m) WORD $(0x4EA04000 | ((m)<<16) | ((n)<<5) | (d))

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

// func shrinkRow4NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)
//
// WebPRescalerImportRowShrink_C for 4 channels, one destination pixel (4
// lanes, one per channel) per iteration, driven by the column table:
//
//	sum  = carry + (sum of the cnt source pixels of the span)
//	B    = last source pixel of the span
//	fr   = B * na                       (32-bit, wrapping)
//	frow = sum*xSub - fr; acc += frow   (32-bit, wrapping)
//	carry = (fr*fx + 2^31) >> 32        (MULT_FIX, 64-bit product)
//
// The span is summed in 16-bit lanes (exact for cnt <= 511): first a
// masked 16-byte load for its cnt&3 leading pixels, then cnt>>2 full
// 16-byte chunks. Lane i<4 collects channel i of the even pixels of a
// chunk, lane 4+i of the odd ones.
//
// R0 frow, R1 acc (nil: no accumulation), R2 src, R3 cols, R4 pixels left,
// R7 shr4Mask, R8 cnt, R9 na, R10 cnt&3, R11 chunks, R12 scratch.
// V0 carry, V1 source bytes, V2 mask, V3 16-bit sums, V4 B, V5 sum,
// V6 fr, V7 frow, V8 acc, V9/V10 fr*fx, V16 xSub, V17 fx, V18 2^31 (2D),
// V19 TBL indices widening the 4 bytes of B to 4 words.
TEXT ·shrinkRow4NEON(SB), NOSPLIT, $0-48
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  cols+24(FP), R3
	MOVD  n+32(FP), R4
	MOVWU xSub+40(FP), R5
	MOVWU fx+44(FP), R6
	CBZ   R4, done4
	MOVD  $shr4Mask<>(SB), R7
	VDUP  R5, V16.S4
	VDUP  R6, V17.S4
	MOVD  $0x80000000, R8
	VDUP  R8, V18.D2
	VMOVQ $0xffffff01ffffff00, $0xffffff03ffffff02, V19
	VEOR  V0.B16, V0.B16, V0.B16

loop4:
	MOVWU (R3), R8
	MOVWU 4(R3), R9
	ADD   $8, R3, R3
	AND   $3, R8, R10
	LSR   $2, R8, R11

	// Leading cnt&3 pixels: masked 16-byte load.
	VLD1    (R2), [V1.B16]
	ADD     R10<<4, R7, R12
	VLD1    (R12), [V2.B16]
	VAND    V2.B16, V1.B16, V1.B16
	VUXTL   V1.B8, V3.H8
	VUADDW2 V1.B16, V3.H8, V3.H8
	ADD     R10<<2, R2, R2
	CBZ     R11, last4

chunk4:
	VLD1.P  16(R2), [V1.B16]
	VUADDW  V1.B8, V3.H8, V3.H8
	VUADDW2 V1.B16, V3.H8, V3.H8
	SUBS    $1, R11, R11
	BNE     chunk4

last4:
	// B: the last pixel of the span, widened to 4 words.
	SUB  $4, R2, R12
	VLD1 (R12), V4.S[0]
	VTBL V19.B16, [V4.B16], V4.B16

	VUADDW  V3.H4, V0.S4, V5.S4
	VUADDW2 V3.H8, V5.S4, V5.S4
	VDUP    R9, V6.S4
	VMUL    V6.S4, V4.S4, V6.S4
	VMUL    V16.S4, V5.S4, V7.S4
	VSUB    V6.S4, V7.S4, V7.S4
	VST1.P  [V7.S4], 16(R0)
	CBZ     R1, carry4
	VLD1    (R1), [V8.S4]
	VADD    V7.S4, V8.S4, V8.S4
	VST1.P  [V8.S4], 16(R1)

carry4:
	VUMULL  V17.S2, V6.S2, V9.D2
	VUMULL2 V17.S4, V6.S4, V10.D2
	ADDHN_2S(0, 9, 18)
	ADDHN2_4S(0, 10, 18)
	SUBS    $1, R4, R4
	BNE     loop4

done4:
	RET

// UADDLP Vd.8H, Vn.16B: Vd.H[i] = Vn.B[2i] + Vn.B[2i+1].
#define UADDLP_8H(d, n) WORD $(0x6E202800 | ((n)<<5) | (d))
// UADALP Vd.8H, Vn.16B: Vd.H[i] += Vn.B[2i] + Vn.B[2i+1].
#define UADALP_8H(d, n) WORD $(0x6E206800 | ((n)<<5) | (d))

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

// func shrinkRow1NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)
//
// WebPRescalerImportRowShrink_C for 1 channel and long spans (more than
// 16 pixels; up to 2047), one destination pixel per iteration: the span is
// summed with a masked 16-byte load for its cnt&15 leading bytes, then
// cnt>>4 full 16-byte chunks, pairwise into 16-bit lanes (UADDLP/UADALP,
// exact for cnt <= 2047), then across the vector (UADDLV). The rest is
// the scalar math of shrinkRow4NEON:
//
//	frow = (carry + sum)*xSub - B*na; acc += frow; carry = MULT_FIX(B*na, fx)
//
// R0 frow, R1 acc (nil: no accumulation), R2 src, R3 cols, R4 pixels left,
// R5 xSub, R6 fx, R7 shr1Mask, R8 cnt, R9 na, R10 cnt&15, R11 chunks,
// R12 scratch, R13 sum/frow, R14 B/fr, R15 carry, R19 acc, R20 2^31.
// V1 source bytes, V2 mask, V3 16-bit sums.
TEXT ·shrinkRow1NEON(SB), NOSPLIT, $0-48
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  cols+24(FP), R3
	MOVD  n+32(FP), R4
	MOVWU xSub+40(FP), R5
	MOVWU fx+44(FP), R6
	CBZ   R4, done1
	MOVD  $shr1Mask<>(SB), R7
	MOVD  $0, R15
	MOVD  $0x80000000, R20

loop1:
	MOVWU (R3), R8
	MOVWU 4(R3), R9
	ADD   $8, R3, R3
	AND   $15, R8, R10
	LSR   $4, R8, R11

	VLD1 (R2), [V1.B16]
	ADD  R10<<4, R7, R12
	VLD1 (R12), [V2.B16]
	VAND V2.B16, V1.B16, V1.B16
	UADDLP_8H(3, 1)
	ADD  R10, R2, R2
	CBZ  R11, last1

chunk1:
	VLD1.P 16(R2), [V1.B16]
	UADALP_8H(3, 1)
	SUBS   $1, R11, R11
	BNE    chunk1

last1:
	VUADDLV V3.H8, V3
	VMOV    V3.S[0], R13
	MOVBU   -1(R2), R14
	ADDW    R13, R15, R13
	MULW    R9, R14, R14
	MULW    R5, R13, R13
	SUBW    R14, R13, R13
	MOVW.P  R13, 4(R0)
	CBZ     R1, carry1
	MOVWU   (R1), R19
	ADDW    R13, R19, R19
	MOVW.P  R19, 4(R1)

carry1:
	UMULL R6, R14, R15
	ADD   R20, R15, R15
	LSR   $32, R15, R15
	SUBS  $1, R4, R4
	BNE   loop1

done1:
	RET

// shrinkRow1T{16,32,64}NEON: WebPRescalerImportRowShrink_C for 1 channel
// and spans of at most 16 pixels, four destination pixels per iteration
// (one per 32-bit lane), driven by 96-byte records built by initArch:
//
//	[ 0:64) per destination pixel j, 16 TBL indices into the window
//	        selecting its span (0xff, i.e. zero, past its end)
//	[64:80) TBL indices widening the last pixel of each span (B) to a word
//	[80:96) the four -accum values (na)
//
// The window is the 16, 32 or 64 source bytes starting at the group's
// first span; it advances by the group's span total (index 76 + 1). The
// four spans gathered by TBL are summed pairwise (UADDLP) and then across
// pixels with three ADDPs, all in 16-bit lanes (at most 16*255). The
// carry of pixel j is MULT_FIX of pixel j-1's B*na: the four of a group
// are computed together and shifted by one lane (EXT) with the last one
// of the previous group.
//
// R0 frow, R1 acc (nil: no accumulation), R2 window, R3 records, R4
// groups left, R5 xSub, R6 fx, R8 advance. V0-V3 indices then spans,
// V4 B indices then B, V5 na, V6 fr, V7 acc, V9/V10 fr*fx, V11 carries
// out, V12 previous carries out, V13 carries in, V16 xSub, V17 fx,
// V18 2^31 (2D), V20-V23 window.

// SH1T_SETUP sets the constants up once the arguments are loaded.
#define SH1T_SETUP \
	VDUP  R5, V16.S4 \
	VDUP  R6, V17.S4 \
	MOVD  $0x80000000, R8 \
	VDUP  R8, V18.D2 \
	VEOR  V12.B16, V12.B16, V12.B16

// SH1T_RECORD loads the next record and advances the window pointer
// (after the window itself was loaded).
#define SH1T_RECORD \
	VLD1.P 64(R3), [V0.B16, V1.B16, V2.B16, V3.B16] \
	VLD1.P 32(R3), [V4.B16, V5.B16] \
	MOVBU  -20(R3), R8 \
	ADD    $1, R8, R8 \
	ADD    R8, R2, R2

// SH1T_MATH sums the gathered spans (V0-V3) and stores four frow values.
#define SH1T_MATH(skip) \
	UADDLP_8H(0, 0) \
	UADDLP_8H(1, 1) \
	UADDLP_8H(2, 2) \
	UADDLP_8H(3, 3) \
	VADDP   V1.H8, V0.H8, V0.H8 \
	VADDP   V3.H8, V2.H8, V2.H8 \
	VADDP   V2.H8, V0.H8, V0.H8 \
	VADDP   V0.H8, V0.H8, V0.H8 \
	VUXTL   V0.H4, V0.S4 \
	VMUL    V5.S4, V4.S4, V6.S4 \
	VUMULL  V17.S2, V6.S2, V9.D2 \
	VUMULL2 V17.S4, V6.S4, V10.D2 \
	ADDHN_2S(11, 9, 18) \
	ADDHN2_4S(11, 10, 18) \
	VEXT    $12, V11.B16, V12.B16, V13.B16 \
	VMOV    V11.B16, V12.B16 \
	VADD    V13.S4, V0.S4, V0.S4 \
	VMUL    V16.S4, V0.S4, V0.S4 \
	VSUB    V6.S4, V0.S4, V0.S4 \
	VST1.P  [V0.S4], 16(R0) \
	CBZ     R1, skip \
	VLD1    (R1), [V7.S4] \
	VADD    V0.S4, V7.S4, V7.S4 \
	VST1.P  [V7.S4], 16(R1) \
skip:

// func shrinkRow1T16NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)
TEXT ·shrinkRow1T16NEON(SB), NOSPLIT, $0-48
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  tab+24(FP), R3
	MOVD  groups+32(FP), R4
	MOVWU xSub+40(FP), R5
	MOVWU fx+44(FP), R6
	SH1T_SETUP
	CBZ R4, t16done

t16loop:
	VLD1 (R2), [V20.B16]
	SH1T_RECORD
	VTBL V0.B16, [V20.B16], V0.B16
	VTBL V1.B16, [V20.B16], V1.B16
	VTBL V2.B16, [V20.B16], V2.B16
	VTBL V3.B16, [V20.B16], V3.B16
	VTBL V4.B16, [V20.B16], V4.B16
	SH1T_MATH(t16skip)
	SUBS $1, R4, R4
	BNE  t16loop

t16done:
	RET

// func shrinkRow1T32NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)
TEXT ·shrinkRow1T32NEON(SB), NOSPLIT, $0-48
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  tab+24(FP), R3
	MOVD  groups+32(FP), R4
	MOVWU xSub+40(FP), R5
	MOVWU fx+44(FP), R6
	SH1T_SETUP
	CBZ R4, t32done

t32loop:
	VLD1 (R2), [V20.B16, V21.B16]
	SH1T_RECORD
	VTBL V0.B16, [V20.B16, V21.B16], V0.B16
	VTBL V1.B16, [V20.B16, V21.B16], V1.B16
	VTBL V2.B16, [V20.B16, V21.B16], V2.B16
	VTBL V3.B16, [V20.B16, V21.B16], V3.B16
	VTBL V4.B16, [V20.B16, V21.B16], V4.B16
	SH1T_MATH(t32skip)
	SUBS $1, R4, R4
	BNE  t32loop

t32done:
	RET

// func shrinkRow1T64NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)
TEXT ·shrinkRow1T64NEON(SB), NOSPLIT, $0-48
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  tab+24(FP), R3
	MOVD  groups+32(FP), R4
	MOVWU xSub+40(FP), R5
	MOVWU fx+44(FP), R6
	SH1T_SETUP
	CBZ R4, t64done

t64loop:
	VLD1 (R2), [V20.B16, V21.B16, V22.B16, V23.B16]
	SH1T_RECORD
	VTBL V0.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V0.B16
	VTBL V1.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V1.B16
	VTBL V2.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V2.B16
	VTBL V3.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V3.B16
	VTBL V4.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V4.B16
	SH1T_MATH(t64skip)
	SUBS $1, R4, R4
	BNE  t64loop

t64done:
	RET
