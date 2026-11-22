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

// Export kernels: 8 values per iteration, n > 0 and a multiple of 8.
// MULT_FIX(v, scale) = (v*scale + 2^31) >> 32 is UMULL/UMULL2 then ADDHN
// with 2^31; clip8 of the unsigned result is two saturating narrows
// (UQXTN to 16 then 8 bits). V18 holds 2^31 in both 64-bit lanes.

// EXP_SETUP2(r) loads 2^31 into V18 using scratch register r.
#define EXP_SETUP2(r) \
	MOVD $0x80000000, r \
	VDUP r, V18.D2

// EXP_MULTFIX(vs, vk, vlo, vhi, t, lo, hi) sets register t (4S) to
// MULT_FIX(vs, vk) per lane, with vlo/vhi (numbered lo/hi) as 2D scratch.
#define EXP_MULTFIX(vs, vk, vlo, vhi, t, lo, hi) \
	VUMULL  vk.S2, vs.S2, vlo.D2 \
	VUMULL2 vk.S4, vs.S4, vhi.D2 \
	ADDHN_2S(t, lo, 18) \
	ADDHN2_4S(t, hi, 18)

// EXP_CLIP_STORE stores clip8 of V12 and V13 (8 words) at R0.
#define EXP_CLIP_STORE \
	VUQXTN  V12.S4, V14.H4 \
	VUQXTN2 V13.S4, V14.H8 \
	VUQXTN  V14.H8, V15.B8 \
	VST1.P  [V15.B8], 8(R0)

// func exportShrinkNEON(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)
//
// WebPRescalerExportRowShrink_C with yscale != 0:
//
//	fr = (frow*yscale) >> 32; dst = clip8(MULT_FIX(irow - fr, fxy)); irow = fr
//
// R0 dst, R1 irow, R2 frow, R3 values left. V0/V1 frow, V2/V3 irow,
// V4-V7 products, V4/V5 fr, V8-V11 products, V12/V13 results, V16
// yscale, V17 fxy.
TEXT ·exportShrinkNEON(SB), NOSPLIT, $0-40
	MOVD  dst+0(FP), R0
	MOVD  irow+8(FP), R1
	MOVD  frow+16(FP), R2
	MOVD  n+24(FP), R3
	MOVWU yscale+32(FP), R4
	MOVWU fxy+36(FP), R5
	VDUP  R4, V16.S4
	VDUP  R5, V17.S4
	EXP_SETUP2(R6)

exs:
	VLD1.P  32(R2), [V0.S4, V1.S4]
	VLD1    (R1), [V2.S4, V3.S4]
	VUMULL  V16.S2, V0.S2, V4.D2
	VUMULL2 V16.S4, V0.S4, V5.D2
	VUMULL  V16.S2, V1.S2, V6.D2
	VUMULL2 V16.S4, V1.S4, V7.D2
	VUZP2   V5.S4, V4.S4, V4.S4
	VUZP2   V7.S4, V6.S4, V5.S4
	VSUB    V4.S4, V2.S4, V2.S4
	VSUB    V5.S4, V3.S4, V3.S4
	VST1.P  [V4.S4, V5.S4], 32(R1)
	EXP_MULTFIX(V2, V17, V8, V9, 12, 8, 9)
	EXP_MULTFIX(V3, V17, V10, V11, 13, 10, 11)
	EXP_CLIP_STORE
	SUBS    $8, R3, R3
	BNE     exs
	RET

// func exportShrinkZeroNEON(dst *byte, irow *uint32, n int, fxy uint32)
//
// WebPRescalerExportRowShrink_C with yscale == 0:
// dst = clip8(MULT_FIX(irow, fxy)); irow = 0.
TEXT ·exportShrinkZeroNEON(SB), NOSPLIT, $0-28
	MOVD  dst+0(FP), R0
	MOVD  irow+8(FP), R1
	MOVD  n+16(FP), R3
	MOVWU fxy+24(FP), R5
	VDUP  R5, V17.S4
	EXP_SETUP2(R6)
	VEOR  V20.B16, V20.B16, V20.B16
	VEOR  V21.B16, V21.B16, V21.B16

exz:
	VLD1   (R1), [V2.S4, V3.S4]
	VST1.P [V20.S4, V21.S4], 32(R1)
	EXP_MULTFIX(V2, V17, V8, V9, 12, 8, 9)
	EXP_MULTFIX(V3, V17, V10, V11, 13, 10, 11)
	EXP_CLIP_STORE
	SUBS   $8, R3, R3
	BNE    exz
	RET

// func exportExpandNEON(dst *byte, frow *uint32, n int, fy uint32)
//
// WebPRescalerExportRowExpand_C with y_accum == 0:
// dst = clip8(MULT_FIX(frow, fy)).
TEXT ·exportExpandNEON(SB), NOSPLIT, $0-28
	MOVD  dst+0(FP), R0
	MOVD  frow+8(FP), R2
	MOVD  n+16(FP), R3
	MOVWU fy+24(FP), R5
	VDUP  R5, V17.S4
	EXP_SETUP2(R6)

exe:
	VLD1.P 32(R2), [V2.S4, V3.S4]
	EXP_MULTFIX(V2, V17, V8, V9, 12, 8, 9)
	EXP_MULTFIX(V3, V17, V10, V11, 13, 10, 11)
	EXP_CLIP_STORE
	SUBS   $8, R3, R3
	BNE    exe
	RET

// func exportExpandMixNEON(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)
//
// WebPRescalerExportRowExpand_C with y_accum != 0:
//
//	j = (a*frow + b*irow + 2^31) >> 32; dst = clip8(MULT_FIX(j, fy))
//
// a + b == 2^32, so the 64-bit sum (UMULL then UMLAL) cannot overflow.
// R0 dst, R1 irow, R2 frow, R3 values left. V0/V1 frow, V2/V3 irow,
// V4-V7 sums, V8-V11 products, V12/V13 results, V16 a, V17 fy, V19 b.
TEXT ·exportExpandMixNEON(SB), NOSPLIT, $0-44
	MOVD  dst+0(FP), R0
	MOVD  frow+8(FP), R2
	MOVD  irow+16(FP), R1
	MOVD  n+24(FP), R3
	MOVWU a+32(FP), R4
	MOVWU b+36(FP), R5
	MOVWU fy+40(FP), R6
	VDUP  R4, V16.S4
	VDUP  R5, V19.S4
	VDUP  R6, V17.S4
	EXP_SETUP2(R7)

exm:
	VLD1.P  32(R2), [V0.S4, V1.S4]
	VLD1.P  32(R1), [V2.S4, V3.S4]
	VUMULL  V16.S2, V0.S2, V4.D2
	VUMLAL  V19.S2, V2.S2, V4.D2
	VUMULL2 V16.S4, V0.S4, V5.D2
	VUMLAL2 V19.S4, V2.S4, V5.D2
	VUMULL  V16.S2, V1.S2, V6.D2
	VUMLAL  V19.S2, V3.S2, V6.D2
	VUMULL2 V16.S4, V1.S4, V7.D2
	VUMLAL2 V19.S4, V3.S4, V7.D2
	ADDHN_2S(0, 4, 18)
	ADDHN2_4S(0, 5, 18)
	ADDHN_2S(1, 6, 18)
	ADDHN2_4S(1, 7, 18)
	EXP_MULTFIX(V0, V17, V8, V9, 12, 8, 9)
	EXP_MULTFIX(V1, V17, V10, V11, 13, 10, 11)
	EXP_CLIP_STORE
	SUBS    $8, R3, R3
	BNE     exm
	RET

// func expandRow4NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xAdd uint32)
//
// WebPRescalerImportRowExpand_C for 4 channels and a source at least two
// pixels wide, one destination pixel per iteration:
//
//	frow = right*xAdd + (left - right)*w; acc += frow   (32-bit, wrapping)
//
// where left/right are the 8 bytes at source pixel cols[2x] and w is
// cols[2x+1].
//
// R0 frow, R1 acc (nil: no accumulation), R2 src, R3 cols, R4 pixels left,
// R8 left pixel, R9 w, R10 its address. V0 the two pixels, V1 left then
// the result, V2 right, V3 w, V4 acc, V16 xAdd.
TEXT ·expandRow4NEON(SB), NOSPLIT, $0-44
	MOVD  frow+0(FP), R0
	MOVD  acc+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  cols+24(FP), R3
	MOVD  n+32(FP), R4
	MOVWU xAdd+40(FP), R5
	CBZ   R4, edone
	VDUP  R5, V16.S4

eloop:
	MOVWU  (R3), R8
	MOVWU  4(R3), R9
	ADD    $8, R3, R3
	ADD    R8<<2, R2, R10
	VLD1   (R10), V0.D[0]
	VUXTL  V0.B8, V0.H8
	VUXTL  V0.H4, V1.S4
	VUXTL2 V0.H8, V2.S4
	VSUB   V2.S4, V1.S4, V1.S4
	VDUP   R9, V3.S4
	VMUL   V3.S4, V1.S4, V1.S4
	VMLA   V16.S4, V2.S4, V1.S4
	VST1.P [V1.S4], 16(R0)
	CBZ    R1, enext
	VLD1   (R1), [V4.S4]
	VADD   V1.S4, V4.S4, V4.S4
	VST1.P [V4.S4], 16(R1)

enext:
	SUBS $1, R4, R4
	BNE  eloop

edone:
	RET

// func premultiplyNEON(row *byte, n int)
//
// WebPMultARGBRow(..., inverse=0) on n bytes (n > 0, a multiple of 16),
// four RGBA pixels per iteration. For every channel value x of a pixel
// with alpha a, libwebp computes (x * (a * KINV_255) + 2^23) >> 24 (and
// leaves a = 255 pixels alone, zeroes a = 0 ones); for all 8-bit x and a
// that equals
//
//	(u + (u >> 8)) >> 8, u = x*a + 128
//
// (checked exhaustively by TestPremultiplyExhaustive). Alpha itself is
// multiplied by 255, which the same formula maps back to a. u fits 16 bits.
//
// R0 row, R1 bytes left. V0 pixels, V1 multipliers, V2/V3 products,
// V4 result, V16 TBL indices broadcasting alpha, V17 255 in the alpha
// bytes, V18 128 per 16-bit lane.
TEXT ·premultiplyNEON(SB), NOSPLIT, $0-16
	MOVD  row+0(FP), R0
	MOVD  n+8(FP), R1
	CBZ   R1, pdone
	VMOVQ $0xff070707ff030303, $0xff0f0f0fff0b0b0b, V16
	VMOVQ $0xff000000ff000000, $0xff000000ff000000, V17
	MOVD  $128, R2
	VDUP  R2, V18.H8

ploop:
	VLD1    (R0), [V0.B16]
	VTBL    V16.B16, [V0.B16], V1.B16
	VORR    V17.B16, V1.B16, V1.B16
	VUMULL  V1.B8, V0.B8, V2.H8
	VUMULL2 V1.B16, V0.B16, V3.H8
	VADD    V18.H8, V2.H8, V2.H8
	VADD    V18.H8, V3.H8, V3.H8
	VUSRA   $8, V2.H8, V2.H8
	VUSRA   $8, V3.H8, V3.H8
	VSHRN   $8, V2.H8, V4.B8
	VSHRN2  $8, V3.H8, V4.B16
	VST1.P  [V4.B16], 16(R0)
	SUBS    $16, R1, R1
	BNE     ploop

pdone:
	RET
