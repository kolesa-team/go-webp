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

// NEON kernels of the VP8L inverse transforms and output conversion.
// Algorithms adapted from libwebp (BSD-3-Clause), src/dsp/lossless_neon.c;
// the Go references in transform.go and pixels.go are the specification.
//
// Pixels are ARGB words, so in memory each pixel is the bytes B, G, R, A.
// Every function processes exact pixel counts: vector loops of 4 or 16
// pixels, then one pixel at a time in the low 32 bits of a vector register
// (FMOVS loads zero the rest of the register), so nothing is read or
// written outside the slices of the Go wrappers (dsp_arm64.go).
//
// The Go assembler lacks some instructions; they are encoded with WORD
// (encodings checked against clang -arch arm64 and objdump).

#include "textflag.h"

// UABD Vd.16B, Vn.16B, Vm.16B
#define UABD_16B(d, n, m) WORD $(0x6E207400 | ((m)<<16) | ((n)<<5) | (d))
// UADDLP Vd.8H, Vn.16B
#define UADDLP_8H(d, n) WORD $(0x6E202800 | ((n)<<5) | (d))
// UADDLP Vd.4S, Vn.8H
#define UADDLP_4S(d, n) WORD $(0x6E602800 | ((n)<<5) | (d))

// VTBL indices (an index of 0xff yields 0).
//
// swzRGBA: B,G,R,A to R,G,B,A in each pixel.
DATA ·swzRGBA+0(SB)/8, $0x0704050603000102
DATA ·swzRGBA+8(SB)/8, $0x0f0c0d0e0b08090a
GLOBL ·swzRGBA(SB), RODATA|NOPTR, $16

// greenBR: the green byte at the blue and red positions, 0 elsewhere.
DATA ·greenBR+0(SB)/8, $0xff05ff05ff01ff01
DATA ·greenBR+8(SB)/8, $0xff0dff0dff09ff09
GLOBL ·greenBR(SB), RODATA|NOPTR, $16

// greenHi: G<<8 in both 16-bit halves of each pixel.
DATA ·greenHi+0(SB)/8, $0x05ff05ff01ff01ff
DATA ·greenHi+8(SB)/8, $0x0dff0dff09ff09ff
GLOBL ·greenHi(SB), RODATA|NOPTR, $16

// rep8: bytes 0 and 1, eight times each.
DATA ·rep8+0(SB)/8, $0
DATA ·rep8+8(SB)/8, $0x0101010101010101
GLOBL ·rep8(SB), RODATA|NOPTR, $16

// bitMask: bit k in lanes k and k+8.
DATA ·bitMask+0(SB)/8, $0x8040201008040201
DATA ·bitMask+8(SB)/8, $0x8040201008040201
GLOBL ·bitMask(SB), RODATA|NOPTR, $16

// func emitRGBAAsm(src *uint32, dst *byte, n int)
//
// dst[4i:4i+4] = R, G, B, A of src[i] for i < n (libwebp
// VP8LConvertBGRAToRGBA_NEON, with a byte shuffle instead of LD4/ST4).
//
// R0 src, R1 dst, R2 pixels left; V31 swzRGBA; V0-V3 pixels.
TEXT ·emitRGBAAsm(SB), NOSPLIT, $0-24
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD n+16(FP), R2
	MOVD $·swzRGBA(SB), R3
	VLD1 (R3), [V31.B16]
	CMP  $16, R2
	BLT  rgba4

rgba16:
	VLD1.P 64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VTBL   V31.B16, [V0.B16], V0.B16
	VTBL   V31.B16, [V1.B16], V1.B16
	VTBL   V31.B16, [V2.B16], V2.B16
	VTBL   V31.B16, [V3.B16], V3.B16
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	SUB    $16, R2, R2
	CMP    $16, R2
	BGE    rgba16

rgba4:
	CMP    $4, R2
	BLT    rgba1
	VLD1.P 16(R0), [V0.B16]
	VTBL   V31.B16, [V0.B16], V0.B16
	VST1.P [V0.B16], 16(R1)
	SUB    $4, R2, R2
	B      rgba4

rgba1:
	CBZ     R2, rgbadone
	FMOVS.P 4(R0), F0
	VTBL    V31.B16, [V0.B16], V0.B16
	FMOVS.P F0, 4(R1)
	SUB     $1, R2, R2
	B       rgba1

rgbadone:
	RET

// func emitAddGreenAsm(src *uint32, dst *byte, n int)
//
// emitRGBAAsm after adding green to red and blue (libwebp
// VP8LAddGreenToBlueAndRed_NEON fused with the conversion). The swizzle
// maps the green copies at the red and blue positions onto each other, so
// out = swz(p) + greenBR(p): two independent shuffles and an add.
//
// R0 src, R1 dst, R2 pixels left; V31 swzRGBA, V30 greenBR;
// V0-V3 pixels, V4-V7 greens.
TEXT ·emitAddGreenAsm(SB), NOSPLIT, $0-24
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD n+16(FP), R2
	MOVD $·swzRGBA(SB), R3
	VLD1 (R3), [V31.B16]
	MOVD $·greenBR(SB), R3
	VLD1 (R3), [V30.B16]
	CMP  $16, R2
	BLT  ag4

ag16:
	VLD1.P 64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VTBL   V30.B16, [V0.B16], V4.B16
	VTBL   V30.B16, [V1.B16], V5.B16
	VTBL   V30.B16, [V2.B16], V6.B16
	VTBL   V30.B16, [V3.B16], V7.B16
	VTBL   V31.B16, [V0.B16], V0.B16
	VTBL   V31.B16, [V1.B16], V1.B16
	VTBL   V31.B16, [V2.B16], V2.B16
	VTBL   V31.B16, [V3.B16], V3.B16
	VADD   V4.B16, V0.B16, V0.B16
	VADD   V5.B16, V1.B16, V1.B16
	VADD   V6.B16, V2.B16, V2.B16
	VADD   V7.B16, V3.B16, V3.B16
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	SUB    $16, R2, R2
	CMP    $16, R2
	BGE    ag16

ag4:
	CMP    $4, R2
	BLT    ag1
	VLD1.P 16(R0), [V0.B16]
	VTBL   V30.B16, [V0.B16], V4.B16
	VTBL   V31.B16, [V0.B16], V0.B16
	VADD   V4.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R1)
	SUB    $4, R2, R2
	B      ag4

ag1:
	CBZ     R2, agdone
	FMOVS.P 4(R0), F0
	VTBL    V30.B16, [V0.B16], V4.B16
	VTBL    V31.B16, [V0.B16], V0.B16
	VADD    V4.B16, V0.B16, V0.B16
	FMOVS.P F0, 4(R1)
	SUB     $1, R2, R2
	B       ag1

agdone:
	RET

// func addGreenAsm(in, out *uint32, n int)
//
// out[i] = in[i] with green added to red and blue (libwebp
// VP8LAddGreenToBlueAndRed_NEON). in may alias out: each vector is loaded
// before it is stored.
//
// R0 in, R1 out, R2 pixels left; V30 greenBR; V0-V3 pixels, V4-V7 greens.
TEXT ·addGreenAsm(SB), NOSPLIT, $0-24
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD n+16(FP), R2
	MOVD $·greenBR(SB), R3
	VLD1 (R3), [V30.B16]
	CMP  $16, R2
	BLT  ga4

ga16:
	VLD1.P 64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VTBL   V30.B16, [V0.B16], V4.B16
	VTBL   V30.B16, [V1.B16], V5.B16
	VTBL   V30.B16, [V2.B16], V6.B16
	VTBL   V30.B16, [V3.B16], V7.B16
	VADD   V4.B16, V0.B16, V0.B16
	VADD   V5.B16, V1.B16, V1.B16
	VADD   V6.B16, V2.B16, V2.B16
	VADD   V7.B16, V3.B16, V3.B16
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	SUB    $16, R2, R2
	CMP    $16, R2
	BGE    ga16

ga4:
	CMP    $4, R2
	BLT    ga1
	VLD1.P 16(R0), [V0.B16]
	VTBL   V30.B16, [V0.B16], V4.B16
	VADD   V4.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R1)
	SUB    $4, R2, R2
	B      ga4

ga1:
	CBZ     R2, gadone
	FMOVS.P 4(R0), F0
	VTBL    V30.B16, [V0.B16], V4.B16
	VADD    V4.B16, V0.B16, V0.B16
	FMOVS.P F0, 4(R1)
	SUB     $1, R2, R2
	B       ga1

gadone:
	RET

// func lookupBytesAsm(pal *[256]byte, in, out *byte, n int)
//
// out[i] = pal[in[i]] for i < n, n a multiple of 16 (the 8-bit alpha
// palette, libwebp MapAlpha). The 256-byte table lives in V16-V31: TBL
// looks up the indices 0-63 in V16-V19 (others give 0), then TBX the
// indices 64-127, 128-191 and 192-255, each rebased into 0-63, in the next
// four registers (TBX leaves out-of-range lanes unchanged).
//
// R0 pal, R1 in, R2 out, R3 blocks of 16 left; V4-V6 64, 128, 192;
// V0, V8-V10 indices, V2 result (and V1, V11-V13, V3 for the second
// block of an iteration).
TEXT ·lookupBytesAsm(SB), NOSPLIT, $0-32
	MOVD   pal+0(FP), R0
	MOVD   in+8(FP), R1
	MOVD   out+16(FP), R2
	MOVD   n+24(FP), R3
	VLD1.P 64(R0), [V16.B16, V17.B16, V18.B16, V19.B16]
	VLD1.P 64(R0), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 64(R0), [V24.B16, V25.B16, V26.B16, V27.B16]
	VLD1   (R0), [V28.B16, V29.B16, V30.B16, V31.B16]
	VMOVI  $64, V4.B16
	VMOVI  $128, V5.B16
	VMOVI  $192, V6.B16
	LSR    $4, R3, R3
	TBZ    $0, R3, lb32

	// An odd block first, then two blocks per iteration.
	VLD1.P 16(R1), [V0.B16]
	VSUB   V4.B16, V0.B16, V8.B16
	VSUB   V5.B16, V0.B16, V9.B16
	VSUB   V6.B16, V0.B16, V10.B16
	VTBL   V0.B16, [V16.B16, V17.B16, V18.B16, V19.B16], V2.B16
	VTBX   V8.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V2.B16
	VTBX   V9.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V2.B16
	VTBX   V10.B16, [V28.B16, V29.B16, V30.B16, V31.B16], V2.B16
	VST1.P [V2.B16], 16(R2)
	SUB    $1, R3, R3

lb32:
	CBZ    R3, lbdone
	VLD1.P 32(R1), [V0.B16, V1.B16]
	VSUB   V4.B16, V0.B16, V8.B16
	VSUB   V5.B16, V0.B16, V9.B16
	VSUB   V6.B16, V0.B16, V10.B16
	VSUB   V4.B16, V1.B16, V11.B16
	VSUB   V5.B16, V1.B16, V12.B16
	VSUB   V6.B16, V1.B16, V13.B16
	VTBL   V0.B16, [V16.B16, V17.B16, V18.B16, V19.B16], V2.B16
	VTBL   V1.B16, [V16.B16, V17.B16, V18.B16, V19.B16], V3.B16
	VTBX   V8.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V2.B16
	VTBX   V11.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V3.B16
	VTBX   V9.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V2.B16
	VTBX   V12.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V3.B16
	VTBX   V10.B16, [V28.B16, V29.B16, V30.B16, V31.B16], V2.B16
	VTBX   V13.B16, [V28.B16, V29.B16, V30.B16, V31.B16], V3.B16
	VST1.P [V2.B16, V3.B16], 32(R2)
	SUB    $2, R3, R3
	B      lb32

lbdone:
	RET

// CCOLOR undoes the cross-color transform on the pixels in V0 (4, or one
// in the low word) into V2 (libwebp VP8LTransformColorInverse_NEON, with
// MUL and SSHR instead of SQDMULH). Per pixel, in 16-bit lanes
// [low half, high half] = [G<<8|B, A<<8|R]:
//   V1 = [Gs, Gs] (sign-extended green) * [g2b, g2r] >> 5 = [dB, dR]
//   V2 = bytes of V0 + V1: B' = B+dB at byte 0, R' = R+dR at byte 2
//   V3 = sign-extended R' (32-bit) * [r2b, 0] >> 5 = [dB2, 0]
//   V2 = V2 + V3 bytewise, then G and A restored from V0.
// The products fit in 16 bits (|s8 * s8| <= 2^14) and the arithmetic
// shift matches colorTransformDelta; only the low byte of each delta
// matters. Uses V31 greenHi, V30 0xff00ff00, V20 [g2b, g2r], V21 [r2b, 0].
#define CCOLOR \
	VTBL  V31.B16, [V0.B16], V1.B16 \
	VSSHR $8, V1.H8, V1.H8          \
	VMUL  V20.H8, V1.H8, V1.H8      \
	VSSHR $5, V1.H8, V1.H8          \
	VADD  V1.B16, V0.B16, V2.B16    \
	VSHL  $8, V2.S4, V3.S4          \
	VSSHR $24, V3.S4, V3.S4         \
	VMUL  V21.H8, V3.H8, V3.H8      \
	VSSHR $5, V3.H8, V3.H8          \
	VADD  V3.B16, V2.B16, V2.B16    \
	VBIT  V30.B16, V0.B16, V2.B16

// func crossColorRowAsm(mults, src, dst *uint32, width, bits int)
//
// Undoes the cross-color transform on one row of width pixels, the tile
// of 1<<bits pixels at x using the multipliers mults[x>>bits] (g2r in byte
// 0, g2b in byte 1, r2b in byte 2). src may alias dst.
//
// R0 mults, R1 src, R2 dst, R3 pixels left in the row, R5 tile width,
// R6 pixels left in the tile, R7 multipliers, R8/R9 temporaries.
TEXT ·crossColorRowAsm(SB), NOSPLIT, $0-40
	MOVD  mults+0(FP), R0
	MOVD  src+8(FP), R1
	MOVD  dst+16(FP), R2
	MOVD  width+24(FP), R3
	MOVD  bits+32(FP), R4
	MOVD  $1, R5
	LSL   R4, R5, R5
	MOVD  $·greenHi(SB), R8
	VLD1  (R8), [V31.B16]
	MOVW  $0xff00ff00, R8
	VDUP  R8, V30.S4

cctile:
	CBZ     R3, ccdone
	MOVWU.P 4(R0), R7
	SBFX    $8, R7, $8, R8  // g2b
	SBFX    $0, R7, $8, R9  // g2r
	AND     $0xffff, R8, R8
	ORR     R9<<16, R8, R8
	VDUP    R8, V20.S4      // [g2b, g2r]
	SBFX    $16, R7, $8, R8 // r2b
	AND     $0xffff, R8, R8
	VDUP    R8, V21.S4      // [r2b, 0]
	CMP     R5, R3
	CSEL    LT, R3, R5, R6  // pixels in this tile
	SUB     R6, R3, R3

cc4:
	CMP    $4, R6
	BLT    cc1
	VLD1.P 16(R1), [V0.B16]
	CCOLOR
	VST1.P [V2.B16], 16(R2)
	SUB    $4, R6, R6
	B      cc4

cc1:
	CBZ     R6, cctile
	FMOVS.P 4(R1), F0
	CCOLOR
	FMOVS.P F2, 4(R2)
	SUB     $1, R6, R6
	B       cc1

ccdone:
	RET

// Predictor mode 11 (libwebp Select): out = withT = residual + T if
// pl = sum over the channels of |L - TL| is at most pt = sum |T - TL|,
// else withL = residual + L, with L the left output pixel.
//
// pt, withT and q = sum |withT - T| come from vector code (SELECTV). The
// serial part runs in scalar registers, and computes the pl of the next
// pixel (whose TL is this T) for both outcomes before the decision: q if
// this pixel takes withT, sum |withL - T| (SADT) otherwise, picked by the
// same flags as the output. The work on withL starts as soon as L is
// known, so that the dependency from one decision to the next is two
// instructions (CSEL pl, CMP), and from L to the decision after next
// eleven: about 5.5 cycles per pixel instead of 7.
//
// Registers: R15 = L, R26 = pl of the current pixel, R14 residual,
// R8 withT, R7 pt, R25 q, R19 T, R24 withL; R16, R17, R20-R23 temporaries.

// SADT(p) sets R20 = sum over the channels of |p - T|, T in R19 (R16, R17,
// R21-R23 are clobbered).
#define SADT(p) \
	ANDW  $0xff, R19, R20    \
	UBFX  $8, R19, $8, R21   \
	UBFX  $16, R19, $8, R22  \
	LSRW  $24, R19, R23      \
	LSRW  $8, p, R16         \
	LSRW  $16, p, R17        \
	SUBS  p.UXTB, R20, R20   \
	CNEG  MI, R20, R20       \
	SUBS  R16.UXTB, R21, R21 \
	CNEG  MI, R21, R21       \
	SUBS  R17.UXTB, R22, R22 \
	CNEG  MI, R22, R22       \
	SUBS  p>>24, R23, R23    \
	CNEG  MI, R23, R23       \
	ADD   R21, R20, R20      \
	ADD   R23, R22, R22      \
	ADD   R22, R20, R20

// SELECT(k) does the pixel in lane k of the vectors of SELECTV, T at
// k*4(R12), and stores it at k*4(R11). Pixel addition modulo 256 per
// channel is ((a & 0x7f..) + (b & 0x7f..)) ^ ((a ^ b) & 0x80..).
#define SELECT(k) \
	VMOV  V3.S[k], R14          \
	VMOV  V5.S[k], R8           \
	VMOV  V4.S[k], R7           \
	VMOV  V6.S[k], R25          \
	MOVWU (k*4)(R12), R19       \
	ANDW  $0x7f7f7f7f, R15, R24 \
	ANDW  $0x7f7f7f7f, R14, R16 \
	ADDW  R16, R24, R24         \
	EORW  R14, R15, R16         \
	ANDW  $0x80808080, R16, R16 \
	EORW  R16, R24, R24         \
	SADT(R24)                   \
	CMP   R7, R26               \
	CSEL  LS, R8, R24, R15      \
	CSEL  LS, R25, R20, R26     \
	MOVW  R15, (k*4)(R11)

// SELECTV computes, for the pixels in V1 (T), V2 (TL) and V3 (residuals),
// V4 = pt, V5 = withT and V6 = q per pixel.
#define SELECTV \
	UABD_16B(4, 1, 2) /* UABD V4.16B, V1.16B, V2.16B */ \
	UADDLP_8H(4, 4)   /* UADDLP V4.8H, V4.16B */        \
	UADDLP_4S(4, 4)   /* UADDLP V4.4S, V4.8H */         \
	VADD  V1.B16, V3.B16, V5.B16                        \
	UABD_16B(6, 5, 1) /* UABD V6.16B, V5.16B, V1.16B */ \
	UADDLP_8H(6, 6)   /* UADDLP V6.8H, V6.16B */        \
	UADDLP_4S(6, 6)   /* UADDLP V6.4S, V6.8H */

// func predictRowAsm(in, out, up, modes *uint32, x, width, bits int) int
//
// Undoes the predictor transform on the pixels x to width-1 of a row
// (x >= 1; out[0:x] done), the tile of 1<<bits pixels at x using the mode
// in bits 8-11 of modes[x>>bits]. up is the previous output row extended
// by out[0] (width+1 pixels). in may alias out. It returns width, or the
// first pixel of a tile in mode 12 or 13, which the caller does in Go.
// libwebp: VP8LPredictorsAdd_NEON (lossless_neon.c), modes 0-11.
//
//   0, 14, 15  in + 0xff000000                vector
//   1          L: prefix sums of 4 pixels      vector, serial carry
//   2, 3, 4    in + T, TR, TL                  vector
//   8, 9       in + avg(TL, T), avg(T, TR)     vector (UHADD)
//   5, 6, 7    in + avg(avg(L, TR), T), avg(L, TL), avg(L, T)
//   10         in + avg(avg(L, TL), avg(T, TR))
//              one pixel at a time in the low word of V0 (L)
//   11         Select: vector |T - TL| sums, scalar serial part (SELECT)
//
// R0 in, R1 out, R2 up, R3 modes, R4 x (start of the next tile), R5 width,
// R6 bits, R9 pixels left in the tile, R10 &in[x], R11 &out[x],
// R12 &up[x], R13 &up[x-1] or &up[x+1], R14 &up[x+1]; V30 0xff000000,
// V29 zero. Mode 11 uses R7, R8, R14-R17, R19-R26.
TEXT ·predictRowAsm(SB), NOSPLIT, $0-64
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD up+16(FP), R2
	MOVD modes+24(FP), R3
	MOVD x+32(FP), R4
	MOVD width+40(FP), R5
	MOVD bits+48(FP), R6
	MOVW $0xff000000, R7
	VDUP R7, V30.S4
	VEOR V29.B16, V29.B16, V29.B16

prtile:
	CMP   R5, R4
	BGE   prdone
	LSR   R6, R4, R7
	MOVWU (R3)(R7<<2), R8
	UBFX  $8, R8, $4, R8   // mode
	ADD   $1, R7, R7
	LSL   R6, R7, R7
	CMP   R5, R7
	CSEL  GT, R5, R7, R7   // end of the tile
	CMP   $12, R8
	BEQ   prgo
	CMP   $13, R8
	BEQ   prgo
	SUB   R4, R7, R9
	ADD   R4<<2, R0, R10
	ADD   R4<<2, R1, R11
	ADD   R4<<2, R2, R12
	MOVD  R7, R4
	CMP   $1, R8
	BEQ   pr1
	CMP   $11, R8
	BEQ   pr11
	CMP   $2, R8
	BEQ   pr2
	CMP   $7, R8
	BEQ   pr7
	CMP   $4, R8
	BEQ   pr4
	CMP   $5, R8
	BEQ   pr5
	CMP   $3, R8
	BEQ   pr3
	CMP   $6, R8
	BEQ   pr6
	CMP   $8, R8
	BEQ   pr8
	CMP   $9, R8
	BEQ   pr9
	CMP   $10, R8
	BEQ   pr10
	B     pr0              // 0, 14, 15

prgo:
	MOVD R4, ret+56(FP)
	RET

prdone:
	MOVD R5, ret+56(FP)
	RET

	// Mode 0 (and 14, 15): in + 0xff000000.
pr0:
	CMP    $4, R9
	BLT    pr0s
	VLD1.P 16(R10), [V0.B16]
	VADD   V30.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R11)
	SUB    $4, R9, R9
	B      pr0

pr0s:
	CBZ     R9, prtile
	FMOVS.P 4(R10), F0
	VADD    V30.B16, V0.B16, V0.B16
	FMOVS.P F0, 4(R11)
	SUB     $1, R9, R9
	B       pr0s

	// Mode 1, L: per 4 pixels a b c d, the prefix sums
	// a, a+b, a+b+c, a+b+c+d from two shifted adds, plus L in all lanes;
	// the new L is lane 3. V0 = L (all lanes). With 8 pixels, the second
	// group adds the total of the first off the serial path, which is one
	// add and one lane broadcast per 8 pixels.
pr1:
	FMOVS -4(R11), F0
	VDUP  V0.S[0], V0.S4

pr1v8:
	CMP    $8, R9
	BLT    pr1v
	VLD1.P 32(R10), [V1.B16, V2.B16]
	VEXT   $12, V1.B16, V29.B16, V3.B16
	VEXT   $12, V2.B16, V29.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VEXT   $8, V1.B16, V29.B16, V3.B16
	VEXT   $8, V2.B16, V29.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VDUP   V1.S[3], V3.S4
	VADD   V3.B16, V2.B16, V2.B16
	VADD   V0.B16, V1.B16, V1.B16
	VADD   V0.B16, V2.B16, V2.B16
	VDUP   V2.S[3], V0.S4
	VST1.P [V1.B16, V2.B16], 32(R11)
	SUB    $8, R9, R9
	B      pr1v8

pr1v:
	CMP    $4, R9
	BLT    pr1s
	VLD1.P 16(R10), [V1.B16]
	VEXT   $12, V1.B16, V29.B16, V2.B16 // 0, a, b, c
	VADD   V2.B16, V1.B16, V1.B16
	VEXT   $8, V1.B16, V29.B16, V2.B16  // 0, 0, a, a+b
	VADD   V2.B16, V1.B16, V1.B16
	VADD   V0.B16, V1.B16, V1.B16
	VDUP   V1.S[3], V0.S4
	VST1.P [V1.B16], 16(R11)
	SUB    $4, R9, R9
	B      pr1v

pr1s:
	CBZ     R9, prtile
	FMOVS.P 4(R10), F1
	VADD    V1.B16, V0.B16, V0.B16
	FMOVS.P F0, 4(R11)
	SUB     $1, R9, R9
	B       pr1s

	// Modes 2, 3, 4: in + the top pixel at R13.
pr2:
	MOVD R12, R13
	B    prtop

pr3:
	ADD $4, R12, R13
	B   prtop

pr4:
	SUB $4, R12, R13

prtop:
	CMP    $4, R9
	BLT    prtops
	VLD1.P 16(R10), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VADD   V1.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R11)
	SUB    $4, R9, R9
	B      prtop

prtops:
	CBZ     R9, prtile
	FMOVS.P 4(R10), F0
	FMOVS.P 4(R13), F1
	VADD    V1.B16, V0.B16, V0.B16
	FMOVS.P F0, 4(R11)
	SUB     $1, R9, R9
	B       prtops

	// Modes 8, 9: in + avg(the top pixels at R13 and R14).
pr8:
	SUB  $4, R12, R13
	MOVD R12, R14
	B    pravg

pr9:
	MOVD R12, R13
	ADD  $4, R12, R14

pravg:
	CMP    $4, R9
	BLT    pravgs
	VLD1.P 16(R10), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VLD1.P 16(R14), [V2.B16]
	VUHADD V2.B16, V1.B16, V1.B16
	VADD   V1.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R11)
	SUB    $4, R9, R9
	B      pravg

pravgs:
	CBZ     R9, prtile
	FMOVS.P 4(R10), F0
	FMOVS.P 4(R13), F1
	FMOVS.P 4(R14), F2
	VUHADD  V2.B8, V1.B8, V1.B8
	VADD    V1.B8, V0.B8, V0.B8
	FMOVS.P F0, 4(R11)
	SUB     $1, R9, R9
	B       pravgs

	// Modes 5, 6, 7, 10, one pixel at a time: V0 = L, V1 = T, V2 = TR,
	// V3 = residual, V4 = TL.
pr5:
	FMOVS -4(R11), F0

pr5l:
	FMOVS   4(R12), F2
	FMOVS.P 4(R12), F1
	FMOVS.P 4(R10), F3
	VUHADD  V2.B8, V0.B8, V0.B8
	VUHADD  V1.B8, V0.B8, V0.B8
	VADD    V3.B8, V0.B8, V0.B8
	FMOVS.P F0, 4(R11)
	SUBS    $1, R9, R9
	BNE     pr5l
	B       prtile

pr6:
	FMOVS -4(R11), F0
	SUB   $4, R12, R13

pr6l:
	FMOVS.P 4(R13), F4
	FMOVS.P 4(R10), F3
	VUHADD  V4.B8, V0.B8, V0.B8
	VADD    V3.B8, V0.B8, V0.B8
	FMOVS.P F0, 4(R11)
	SUBS    $1, R9, R9
	BNE     pr6l
	B       prtile

pr7:
	FMOVS -4(R11), F0

pr7l:
	FMOVS.P 4(R12), F1
	FMOVS.P 4(R10), F3
	VUHADD  V1.B8, V0.B8, V0.B8
	VADD    V3.B8, V0.B8, V0.B8
	FMOVS.P F0, 4(R11)
	SUBS    $1, R9, R9
	BNE     pr7l
	B       prtile

pr10:
	FMOVS -4(R11), F0
	SUB   $4, R12, R13

pr10l:
	FMOVS.P 4(R13), F4
	FMOVS   4(R12), F2
	FMOVS.P 4(R12), F1
	FMOVS.P 4(R10), F3
	VUHADD  V2.B8, V1.B8, V1.B8 // avg(T, TR), off the serial path
	VUHADD  V4.B8, V0.B8, V0.B8
	VUHADD  V1.B8, V0.B8, V0.B8
	VADD    V3.B8, V0.B8, V0.B8
	FMOVS.P F0, 4(R11)
	SUBS    $1, R9, R9
	BNE     pr10l
	B       prtile

	// Mode 11, Select: 4 pixels at a time, then one at a time. First
	// pl of the first pixel, from L and TL in memory.
pr11:
	MOVWU -4(R11), R15
	SUB   $4, R12, R13
	MOVWU (R13), R19
	SADT(R15)
	MOVD  R20, R26

pr11v:
	CMP  $4, R9
	BLT  pr11s
	VLD1 (R12), [V1.B16]
	VLD1 (R13), [V2.B16]
	VLD1 (R10), [V3.B16]
	SELECTV
	SELECT(0)
	SELECT(1)
	SELECT(2)
	SELECT(3)
	ADD  $16, R10, R10
	ADD  $16, R11, R11
	ADD  $16, R12, R12
	ADD  $16, R13, R13
	SUB  $4, R9, R9
	B    pr11v

pr11s:
	CBZ   R9, prtile
	FMOVS (R12), F1
	FMOVS (R13), F2
	FMOVS (R10), F3
	SELECTV
	SELECT(0)
	ADD   $4, R10, R10
	ADD   $4, R11, R11
	ADD   $4, R12, R12
	ADD   $4, R13, R13
	SUB   $1, R9, R9
	B     pr11s

// Bundled palette indices (color indexing with at most 16 colors): each
// packed byte holds 2, 4 or 8 indices of 4, 2 or 1 bits, the first in the
// low bits. The UNPACK macros turn the 16 packed bytes in V0 into byte
// vectors of indices, in pixel order.

// UNPACK4BIT: V3, V4 = the 32 indices (V28 = 0x0f).
#define UNPACK4BIT \
	VAND  V28.B16, V0.B16, V1.B16 \
	VUSHR $4, V0.B16, V2.B16      \
	VZIP1 V2.B16, V1.B16, V3.B16  \
	VZIP2 V2.B16, V1.B16, V4.B16

// UNPACK2BIT: V1-V4 = the 64 indices (V28 = 3): the four fields a, b, c,
// d of each byte, interleaved by bytes (ab, cd) and then by 16-bit pairs.
#define UNPACK2BIT \
	VAND  V28.B16, V0.B16, V1.B16 \
	VUSHR $2, V0.B16, V2.B16      \
	VAND  V28.B16, V2.B16, V2.B16 \
	VUSHR $4, V0.B16, V3.B16      \
	VAND  V28.B16, V3.B16, V3.B16 \
	VUSHR $6, V0.B16, V4.B16      \
	VZIP1 V2.B16, V1.B16, V5.B16  \
	VZIP2 V2.B16, V1.B16, V6.B16  \
	VZIP1 V4.B16, V3.B16, V7.B16  \
	VZIP2 V4.B16, V3.B16, V8.B16  \
	VZIP1 V7.H8, V5.H8, V1.H8     \
	VZIP2 V7.H8, V5.H8, V2.H8     \
	VZIP1 V8.H8, V6.H8, V3.H8     \
	VZIP2 V8.H8, V6.H8, V4.H8

// BITMASK(v) sets v to 0xff for the set bits of the next two packed bytes
// of V0 (8 lanes each, bit k in lane k), and moves V24 to the next two
// bytes (V24 indices, V25 = 2, V26 bitMask).
#define BITMASK(v) \
	VTBL   V24.B16, [V0.B16], v \
	VCMTST V26.B16, v, v        \
	VADD   V25.B16, V24.B16, V24.B16

// func expandBundled8Asm(pal *[256]byte, src, dst *byte, n, bits int)
//
// dst[k<<bits+j] = pal[index j of src[k]] for k < n (a multiple of 16),
// bits 1 to 3 (the 8-bit alpha path, libwebp MapAlpha after
// VP8LColorIndexInverseTransformAlpha's unpacking): one TBL per 16 output
// bytes from the palette in V16, or for 1-bit indices a select between
// the two colors (V17, V18).
//
// R0 pal, R1 src, R2 dst, R3 blocks of 16 packed bytes left, R4 bits.
TEXT ·expandBundled8Asm(SB), NOSPLIT, $0-40
	MOVD pal+0(FP), R0
	MOVD src+8(FP), R1
	MOVD dst+16(FP), R2
	MOVD n+24(FP), R3
	MOVD bits+32(FP), R4
	VLD1 (R0), [V16.B16]
	LSR  $4, R3, R3
	CBZ  R3, e8done
	CMP  $2, R4
	BEQ  e8bits2
	BGT  e8bits3
	VMOVI $15, V28.B16

e8bits1:
	VLD1.P 16(R1), [V0.B16]
	UNPACK4BIT
	VTBL   V3.B16, [V16.B16], V3.B16
	VTBL   V4.B16, [V16.B16], V4.B16
	VST1.P [V3.B16, V4.B16], 32(R2)
	SUBS   $1, R3, R3
	BNE    e8bits1
	RET

e8bits2:
	VMOVI $3, V28.B16

e8bits2l:
	VLD1.P 16(R1), [V0.B16]
	UNPACK2BIT
	VTBL   V1.B16, [V16.B16], V1.B16
	VTBL   V2.B16, [V16.B16], V2.B16
	VTBL   V3.B16, [V16.B16], V3.B16
	VTBL   V4.B16, [V16.B16], V4.B16
	VST1.P [V1.B16, V2.B16, V3.B16, V4.B16], 64(R2)
	SUBS   $1, R3, R3
	BNE    e8bits2l
	RET

e8bits3:
	VDUP  V16.B[0], V17.B16
	VDUP  V16.B[1], V18.B16
	MOVD  $·rep8(SB), R5
	VLD1  (R5), [V27.B16]
	MOVD  $·bitMask(SB), R5
	VLD1  (R5), [V26.B16]
	VMOVI $2, V25.B16

e8bits3l:
	VLD1.P 16(R1), [V0.B16]
	VMOV   V27.B16, V24.B16
	BITMASK(V1.B16)
	BITMASK(V2.B16)
	BITMASK(V3.B16)
	BITMASK(V4.B16)
	BITMASK(V5.B16)
	BITMASK(V6.B16)
	BITMASK(V7.B16)
	BITMASK(V8.B16)
	VBSL   V17.B16, V18.B16, V1.B16
	VBSL   V17.B16, V18.B16, V2.B16
	VBSL   V17.B16, V18.B16, V3.B16
	VBSL   V17.B16, V18.B16, V4.B16
	VBSL   V17.B16, V18.B16, V5.B16
	VBSL   V17.B16, V18.B16, V6.B16
	VBSL   V17.B16, V18.B16, V7.B16
	VBSL   V17.B16, V18.B16, V8.B16
	VST1.P [V1.B16, V2.B16, V3.B16, V4.B16], 64(R2)
	VST1.P [V5.B16, V6.B16, V7.B16, V8.B16], 64(R2)
	SUBS   $1, R3, R3
	BNE    e8bits3l

e8done:
	RET

// RGBA16(v) stores the RGBA colors of the 16 indices in v: one TBL per
// channel plane (V16-V19 = R, G, B, A of the palette), interleaved by ST4.
#define RGBA16(v) \
	VTBL   v, [V16.B16], V20.B16 \
	VTBL   v, [V17.B16], V21.B16 \
	VTBL   v, [V18.B16], V22.B16 \
	VTBL   v, [V19.B16], V23.B16 \
	VST4.P [V20.B16, V21.B16, V22.B16, V23.B16], 64(R2)

// func expandBundledRGBAAsm(pal *[256]uint32, src, dst *byte, n, bits int)
//
// dst[4(k<<bits+j):][:4] = the RGBA word pal[index j of src[k]] for k < n
// (a multiple of 16), bits 1 or 2: the emitPalette of the byte path
// (libwebp VP8LColorIndexInverseTransform + VP8LConvertBGRAToRGBA). LD4
// splits the first 16 palette words into channel planes, so that each
// channel of 16 pixels is one TBL. (1-bit indices are left to the Go
// table of pixel pairs, which a NEON version beat by only 1.2x: the stores
// dominate.)
//
// R0 pal, R1 src, R2 dst, R3 blocks of 16 packed bytes left, R4 bits.
TEXT ·expandBundledRGBAAsm(SB), NOSPLIT, $0-40
	MOVD pal+0(FP), R0
	MOVD src+8(FP), R1
	MOVD dst+16(FP), R2
	MOVD n+24(FP), R3
	MOVD bits+32(FP), R4
	VLD4 (R0), [V16.B16, V17.B16, V18.B16, V19.B16]
	LSR  $4, R3, R3
	CBZ  R3, ecdone
	CMP  $2, R4
	BEQ  ecbits2
	VMOVI $15, V28.B16

ecbits1:
	VLD1.P 16(R1), [V0.B16]
	UNPACK4BIT
	RGBA16(V3.B16)
	RGBA16(V4.B16)
	SUBS   $1, R3, R3
	BNE    ecbits1
	RET

ecbits2:
	VMOVI $3, V28.B16

ecbits2l:
	VLD1.P 16(R1), [V0.B16]
	UNPACK2BIT
	RGBA16(V1.B16)
	RGBA16(V2.B16)
	RGBA16(V3.B16)
	RGBA16(V4.B16)
	SUBS   $1, R3, R3
	BNE    ecbits2l

ecdone:
	RET
