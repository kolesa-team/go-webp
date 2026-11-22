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

// NEON YUV->RGBA row conversion for the scaled Y'CbCr path (exportYUV),
// adapted from libwebp (BSD-3-Clause) src/dsp/upsampling_neon.c (CONVERT8)
// and bit-exact with yuvToRGBARowGo. internal/vp8/yuv_arm64.s uses the same
// conversion inside its upsampling kernels; see the comments there.
//
// Instructions the Go assembler does not know are emitted with WORD; each
// macro names the instruction it encodes. The encodings were checked
// against clang's assembler.

// SQDMULH Vd.8H, Vn.8H, Vm.8H
#define SQDMULH_8H(d, n, m) WORD $(0x4E60B400 | ((m)<<16) | ((n)<<5) | (d))
// SQSHRUN Vd.8B, Vn.8H, #sh (lower half; clears the upper half)
#define SQSHRUN_8B(d, n, sh) WORD $(0x2F008400 | ((16-(sh))<<16) | ((n)<<5) | (d))
// SQSHRUN2 Vd.16B, Vn.8H, #sh (upper half)
#define SQSHRUN2_16B(d, n, sh) WORD $(0x6F008400 | ((16-(sh))<<16) | ((n)<<5) | (d))

// V24-V31 = 19077, 26149, 6419, 13320, 282, -14234, 8708, -17685 (int16)
// and V23 = 0xff bytes (opaque alpha).
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

// CONV8: libwebp's CONVERT8 on y<<7, u<<7, v<<7 (int16) in V0, V1, V2;
// leaves R in V6, G in V3, B in V0 (<<6); u<<7 is added to B last so that
// the only possible saturation clips to 255 anyway. Clobbers V0-V6.
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

// CONV16 converts the 16 pixels with luma YY and chroma UU, VV to R, G, B
// bytes in V8, V9, V10 (VP8Clip8 = SQSHRUN #6). Clobbers V0-V6.
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

// func yuvToRGBARowNEON(y, u, v, a, dst *byte, n int)
//
// Converts pixels [0, n), n >= 16, in blocks of 16; the last block starts
// at n-16 and may overlap the previous one (rewriting the same values), so
// nothing outside the rows is accessed. a is the alpha row or nil.
//
// Registers: R0-R4 arguments, R5 last block start, R6 block start, R7-R8
// scratch pointers; V20 luma, V16 U, V18 V, V8-V11 R,G,B,A.
TEXT ·yuvToRGBARowNEON(SB), NOSPLIT, $0-48
	MOVD y+0(FP), R0
	MOVD u+8(FP), R1
	MOVD v+16(FP), R2
	MOVD a+24(FP), R3
	MOVD dst+32(FP), R4
	MOVD n+40(FP), R5
	SUB $16, R5, R5
	YUV_CONSTS
	VMOV V23.B16, V11.B16
	MOVD $0, R6

block:
	CMP R5, R6
	CSEL GT, R5, R6, R6 // clamp the last block to n-16
	ADD R6, R0, R7
	VLD1 (R7), [V20.B16]
	ADD R6, R1, R7
	VLD1 (R7), [V16.B16]
	ADD R6, R2, R7
	VLD1 (R7), [V18.B16]
	CBZ R3, conv
	ADD R6, R3, R7
	VLD1 (R7), [V11.B16]

conv:
	CONV16(V20, V16, V18)
	ADD R6<<2, R4, R8
	VST4 [V8.B16, V9.B16, V10.B16, V11.B16], (R8)
	CMP R5, R6
	BEQ done
	ADD $16, R6, R6
	B block

done:
	RET
