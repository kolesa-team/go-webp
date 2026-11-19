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

// NEON YUV->RGBA conversion with libwebp's fancy upsampler, adapted from
// libwebp (BSD-3-Clause) src/dsp/upsampling_neon.c (UPSAMPLE_16PIXELS,
// CONVERT8) and bit-exact with the Go code in yuv.go (yuvPix,
// upsampleUV, sampleRow).
//
// Instructions the Go assembler does not know are emitted with WORD; each
// macro below names the instruction it encodes. The encodings were checked
// against clang's assembler.

// SQDMULH Vd.8H, Vn.8H, Vm.8H
#define SQDMULH_8H(d, n, m) WORD $(0x4E60B400 | ((m)<<16) | ((n)<<5) | (d))
// SQSHRUN Vd.8B, Vn.8H, #sh (lower half; clears the upper half)
#define SQSHRUN_8B(d, n, sh) WORD $(0x2F008400 | ((16-(sh))<<16) | ((n)<<5) | (d))
// SQSHRUN2 Vd.16B, Vn.8H, #sh (upper half)
#define SQSHRUN2_16B(d, n, sh) WORD $(0x6F008400 | ((16-(sh))<<16) | ((n)<<5) | (d))
// UADDL Vd.8H, Vn.8B, Vm.8B
#define UADDL_8H(d, n, m) WORD $(0x2E200000 | ((m)<<16) | ((n)<<5) | (d))

// Constant registers shared by both functions (libwebp's kCoeffs1 and
// rounders, VP8YuvToRgb in 14-bit fixed point):
//   V24 = 19077   V25 = 26149   V26 = 6419   V27 = 13320   V28 = 282
//   V29 = -14234  V30 = 8708    V31 = -17685 (all int16 x8)
//   V23 = 0xff bytes (opaque alpha)
#define YUV_CONSTS \
	MOVD $19077, R20 \
	VDUP R20, V24.H8 \
	MOVD $26149, R20 \
	VDUP R20, V25.H8 \
	MOVD $6419, R20 \
	VDUP R20, V26.H8 \
	MOVD $13320, R20 \
	VDUP R20, V27.H8 \
	MOVD $282, R20 \
	VDUP R20, V28.H8 \
	MOVD $-14234, R20 \
	VDUP R20, V29.H8 \
	MOVD $8708, R20 \
	VDUP R20, V30.H8 \
	MOVD $-17685, R20 \
	VDUP R20, V31.H8 \
	VMOVI $0xff, V23.B16

// CONV8 is libwebp's CONVERT8 on 8 pixels whose samples were widened and
// shifted left by 7 into V0 (y), V1 (u) and V2 (v), as int16:
//   y1 = MultHi(y, 19077)            = SQDMULH(y<<7, 19077)
//   R  = y1 + MultHi(v, 26149) - 14234
//   G  = y1 - MultHi(u, 6419) - MultHi(v, 13320) + 8708
//   B  = y1 + MultHi(u, 33050) - 17685, with 33050 = 32768 + 282, so
//        MultHi(u, 33050) = (u<<7) + SQDMULH(u<<7, 282).
// The saturating sums cannot overflow except in B's last addition, which
// only saturates values that clip to 255 anyway; that is why u<<7 is added
// last, as in libwebp. Leaves R in V6, G in V3 and B in V0 (int16, still
// <<6); clobbers V0-V6.
#define CONV8 \
	SQDMULH_8H(0, 0, 24) \
	SQDMULH_8H(3, 2, 25) \
	SQDMULH_8H(4, 1, 26) \
	SQDMULH_8H(2, 2, 27) \
	SQDMULH_8H(5, 1, 28) \
	VSQADD V29.H8, V0.H8, V6.H8 \
	VSQADD V3.H8, V6.H8, V6.H8 \
	VSQADD V30.H8, V0.H8, V3.H8 \
	VSQADD V2.H8, V4.H8, V4.H8 \
	VSQSUB V4.H8, V3.H8, V3.H8 \
	VSQADD V31.H8, V0.H8, V0.H8 \
	VSQADD V5.H8, V0.H8, V0.H8 \
	VSQADD V1.H8, V0.H8, V0.H8

// CONV16 converts the 16 pixels with luma YY, chroma UU and VV (16 bytes
// each) to R, G, B bytes in V8, V9 and V10; VP8Clip8 is the saturating
// narrowing shift SQSHRUN #6. Clobbers V0-V6.
#define CONV16(YY, UU, VV) \
	VUSHLL $7, YY.B8, V0.H8 \
	VUSHLL $7, UU.B8, V1.H8 \
	VUSHLL $7, VV.B8, V2.H8 \
	CONV8 \
	SQSHRUN_8B(8, 6, 6) \
	SQSHRUN_8B(9, 3, 6) \
	SQSHRUN_8B(10, 0, 6) \
	VUSHLL2 $7, YY.B16, V0.H8 \
	VUSHLL2 $7, UU.B16, V1.H8 \
	VUSHLL2 $7, VV.B16, V2.H8 \
	CONV8 \
	SQSHRUN2_16B(8, 6, 6) \
	SQSHRUN2_16B(9, 3, 6) \
	SQSHRUN2_16B(10, 0, 6)

// UPSAMPLE16(TOP, CUR, OT, OB) is libwebp's UPSAMPLE_16PIXELS: from the 9
// chroma samples at TOP and CUR (pointer registers) it computes
// the 16 full resolution samples of the luma row next to TOP (into OT) and
// of the luma row next to CUR (into OB). With a = top[i], b = top[i+1],
// c = cur[i], d = cur[i+1]:
//   diag1 = (a + 3b + 3c + d) >> 3,  diag2 = (3a + b + c + 3d) >> 3
//   OT = zip(rhadd(a, diag1), rhadd(b, diag2))
//   OB = zip(rhadd(c, diag2), rhadd(d, diag1))
// which equals upsampleUV's (diag + x + 1) >> 1 with its +8 rounding
// folded into the rounding halving add. Clobbers V0-V7.
#define UPSAMPLE16(TOP, CUR, OT, OB) \
	FMOVD (TOP), F0 \
	FMOVD 1(TOP), F1 \
	FMOVD (CUR), F2 \
	FMOVD 1(CUR), F3 \
	UADDL_8H(4, 0, 3) \
	UADDL_8H(5, 1, 2) \
	VADD V5.H8, V4.H8, V6.H8 \
	VSHL $1, V4.H8, V7.H8 \
	VADD V6.H8, V7.H8, V7.H8 \
	VSHL $1, V5.H8, V4.H8 \
	VADD V6.H8, V4.H8, V4.H8 \
	VSHRN $3, V7.H8, V7.B8 \
	VSHRN $3, V4.H8, V4.B8 \
	VURHADD V4.B8, V0.B8, V0.B8 \
	VURHADD V7.B8, V1.B8, V1.B8 \
	VURHADD V7.B8, V2.B8, V2.B8 \
	VURHADD V4.B8, V3.B8, V3.B8 \
	VZIP1 V1.B16, V0.B16, OT.B16 \
	VZIP1 V3.B16, V2.B16, OB.B16

// func upsampleRowsNEON(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA *byte, uvW int)
//
// Mirrors libwebp's UpsampleRgbaLinePair_NEON without its first and last
// pixels: converts output pixels [1, 2*uvW-1) of the top row and, when
// botY != nil, of the bottom row. The chroma rows hold uvW >= 9 samples.
// Blocks of 16 pixels start at chroma index j = 0, 8, 16, ...; the last
// block starts at uvW-9 and may overlap the previous one (it rewrites the
// same values), so nothing outside the rows is read or written. topA/botA
// are alpha rows, or nil for opaque output.
//
// Registers: R0-R9 arguments, R10 last block start, R11 block start j,
// R12-R13 scratch pointers, R14 = 2j+1 (first output pixel of the block).
// V16/V17 top/bottom U, V18/V19 top/bottom V, V20 luma, V8-V11 R,G,B,A.
TEXT ·upsampleRowsNEON(SB), NOSPLIT, $0-88
	MOVD topY+0(FP), R0
	MOVD botY+8(FP), R1
	MOVD topU+16(FP), R2
	MOVD topV+24(FP), R3
	MOVD curU+32(FP), R4
	MOVD curV+40(FP), R5
	MOVD topDst+48(FP), R6
	MOVD botDst+56(FP), R7
	MOVD topA+64(FP), R8
	MOVD botA+72(FP), R9
	MOVD uvW+80(FP), R10
	SUB $9, R10, R10
	YUV_CONSTS
	MOVD $0, R11

ublock:
	CMP R10, R11
	CSEL GT, R10, R11, R11 // clamp the last block to uvW-9
	ADD R11, R2, R12
	ADD R11, R4, R13
	UPSAMPLE16(R12, R13, V16, V17)
	ADD R11, R3, R12
	ADD R11, R5, R13
	UPSAMPLE16(R12, R13, V18, V19)
	LSL $1, R11, R14
	ADD $1, R14, R14

	// top row
	ADD R14, R0, R12
	VLD1 (R12), [V20.B16]
	VMOV V23.B16, V11.B16
	CBZ R8, utopconv
	ADD R14, R8, R12
	VLD1 (R12), [V11.B16]

utopconv:
	CONV16(V20, V16, V18)
	ADD R14<<2, R6, R12
	VST4 [V8.B16, V9.B16, V10.B16, V11.B16], (R12)

	// bottom row
	CBZ R1, unext
	ADD R14, R1, R12
	VLD1 (R12), [V20.B16]
	VMOV V23.B16, V11.B16
	CBZ R9, ubotconv
	ADD R14, R9, R12
	VLD1 (R12), [V11.B16]

ubotconv:
	CONV16(V20, V17, V19)
	ADD R14<<2, R7, R12
	VST4 [V8.B16, V9.B16, V10.B16, V11.B16], (R12)

unext:
	CMP R10, R11
	BEQ udone
	ADD $8, R11, R11
	B ublock

udone:
	RET

// func sampleRowNEON(y, u, v, a, dst *byte, w int)
//
// Point-sampled conversion (libwebp's YuvToRgbaRow / sampleRow): pixel x
// uses chroma sample x/2. Converts pixels [0, w) with w even and >= 16, in
// blocks of 16 pixels starting at even x; the last block starts at w-16
// and may overlap the previous one. a is the alpha row or nil (opaque).
//
// Registers: R0-R5 arguments (R5 becomes the last block start), R6 block
// start x, R7-R8 scratch pointers. V16/V18 duplicated U/V, V20 luma,
// V8-V11 R,G,B,A.
TEXT ·sampleRowNEON(SB), NOSPLIT, $0-48
	MOVD y+0(FP), R0
	MOVD u+8(FP), R1
	MOVD v+16(FP), R2
	MOVD a+24(FP), R3
	MOVD dst+32(FP), R4
	MOVD w+40(FP), R5
	SUB $16, R5, R5
	YUV_CONSTS
	VMOV V23.B16, V11.B16
	MOVD $0, R6

sblock:
	CMP R5, R6
	CSEL GT, R5, R6, R6 // clamp the last block to w-16
	ADD R6>>1, R1, R7
	VLD1 (R7), [V16.B8]
	VZIP1 V16.B16, V16.B16, V16.B16
	ADD R6>>1, R2, R7
	VLD1 (R7), [V18.B8]
	VZIP1 V18.B16, V18.B16, V18.B16
	ADD R6, R0, R7
	VLD1 (R7), [V20.B16]
	CBZ R3, sconv
	ADD R6, R3, R7
	VLD1 (R7), [V11.B16]

sconv:
	CONV16(V20, V16, V18)
	ADD R6<<2, R4, R8
	VST4 [V8.B16, V9.B16, V10.B16, V11.B16], (R8)
	CMP R5, R6
	BEQ sdone
	ADD $16, R6, R6
	B sblock

sdone:
	RET
