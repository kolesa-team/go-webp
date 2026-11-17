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
