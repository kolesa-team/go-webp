//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 and AVX2 YUV->RGBA row conversion for the scaled Y'CbCr path
// (exportYUV), adapted from libwebp (BSD-3-Clause) src/dsp/yuv_sse2.c
// (ConvertYUV444ToRGB_SSE2) and bit-exact with yuvToRGBARowGo. With the
// samples in the high byte of 16-bit lanes (x<<8), PMULHUW by k computes
// (x*k)>>8, so
//
//	yh = (y*19077)>>8
//	R  = yh + (v*26149)>>8 - 14234                 in [-14234, 30815]
//	G  = yh - (u*6419)>>8 - (v*13320)>>8 + 8708    in [-10953, 27710]
//	B  = yh + (u*33050)>>8 - 17685                 (unsigned, saturating)
//
// stay inside 16 bits. VP8Clip8 (x>>6 clamped to [0, 255]) is an
// arithmetic shift by 6 then PACKUSWB; B, computed with unsigned
// saturating arithmetic, is 0 where the exact value is negative (which
// VP8Clip8 maps to 0 as well) and shifted logically.

DATA yuvK<>+0x00(SB)/8, $0x4a854a854a854a85 // 19077
DATA yuvK<>+0x08(SB)/8, $0x4a854a854a854a85
DATA yuvK<>+0x10(SB)/8, $0x4a854a854a854a85
DATA yuvK<>+0x18(SB)/8, $0x4a854a854a854a85
DATA yuvK<>+0x20(SB)/8, $0x6625662566256625 // 26149
DATA yuvK<>+0x28(SB)/8, $0x6625662566256625
DATA yuvK<>+0x30(SB)/8, $0x6625662566256625
DATA yuvK<>+0x38(SB)/8, $0x6625662566256625
DATA yuvK<>+0x40(SB)/8, $0x379a379a379a379a // 14234
DATA yuvK<>+0x48(SB)/8, $0x379a379a379a379a
DATA yuvK<>+0x50(SB)/8, $0x379a379a379a379a
DATA yuvK<>+0x58(SB)/8, $0x379a379a379a379a
DATA yuvK<>+0x60(SB)/8, $0x1913191319131913 // 6419
DATA yuvK<>+0x68(SB)/8, $0x1913191319131913
DATA yuvK<>+0x70(SB)/8, $0x1913191319131913
DATA yuvK<>+0x78(SB)/8, $0x1913191319131913
DATA yuvK<>+0x80(SB)/8, $0x3408340834083408 // 13320
DATA yuvK<>+0x88(SB)/8, $0x3408340834083408
DATA yuvK<>+0x90(SB)/8, $0x3408340834083408
DATA yuvK<>+0x98(SB)/8, $0x3408340834083408
DATA yuvK<>+0xa0(SB)/8, $0x2204220422042204 // 8708
DATA yuvK<>+0xa8(SB)/8, $0x2204220422042204
DATA yuvK<>+0xb0(SB)/8, $0x2204220422042204
DATA yuvK<>+0xb8(SB)/8, $0x2204220422042204
DATA yuvK<>+0xc0(SB)/8, $0x811a811a811a811a // 33050
DATA yuvK<>+0xc8(SB)/8, $0x811a811a811a811a
DATA yuvK<>+0xd0(SB)/8, $0x811a811a811a811a
DATA yuvK<>+0xd8(SB)/8, $0x811a811a811a811a
DATA yuvK<>+0xe0(SB)/8, $0x4515451545154515 // 17685
DATA yuvK<>+0xe8(SB)/8, $0x4515451545154515
DATA yuvK<>+0xf0(SB)/8, $0x4515451545154515
DATA yuvK<>+0xf8(SB)/8, $0x4515451545154515
GLOBL yuvK<>(SB), RODATA|NOPTR, $256

// CONV_SSE(Y, U, V, T, T2) converts 8 pixels, y<<8, u<<8, v<<8 in the
// 16-bit lanes of Y, U, V, leaving R (shifted, signed) in T, G in T2 and
// B in U. Clobbers Y and V. X8-X15 hold the constants 19077, 26149, 14234,
// 6419, 13320, 8708, 33050, 17685.
#define CONV_SSE(Y, U, V, T, T2) \
	PMULHUW X8, Y   \
	MOVO    V, T    \
	PMULHUW X9, T   \
	PADDW   Y, T    \
	PSUBW   X10, T  \
	PMULHUW X12, V  \
	MOVO    U, T2   \
	PMULHUW X11, T2 \
	PADDW   T2, V   \
	MOVO    Y, T2   \
	PADDW   X13, T2 \
	PSUBW   V, T2   \
	PMULHUW X14, U  \
	PADDUSW Y, U    \
	PSUBUSW X15, U  \
	PSRAW   $6, T   \
	PSRAW   $6, T2  \
	PSRLW   $6, U

// func yuvToRGBARowSSE41(y, u, v, a, dst *byte, n int)
//
// Converts pixels [0, n), n >= 16, in blocks of 16; the last block starts
// at n-16 and may overlap the previous one (rewriting the same values), so
// nothing outside the rows is accessed. a is the alpha row or nil.
//
// AX y, BX u, CX v, DX a, DI dst, SI last block start, R8 block start,
// R9 output address. X5-X7 the source bytes, X0-X7 work registers.
TEXT ·yuvToRGBARowSSE41(SB), NOSPLIT, $0-48
	MOVQ  y+0(FP), AX
	MOVQ  u+8(FP), BX
	MOVQ  v+16(FP), CX
	MOVQ  a+24(FP), DX
	MOVQ  dst+32(FP), DI
	MOVQ  n+40(FP), SI
	SUBQ  $16, SI
	LEAQ  yuvK<>(SB), R9
	MOVOU 0x00(R9), X8
	MOVOU 0x20(R9), X9
	MOVOU 0x40(R9), X10
	MOVOU 0x60(R9), X11
	MOVOU 0x80(R9), X12
	MOVOU 0xa0(R9), X13
	MOVOU 0xc0(R9), X14
	MOVOU 0xe0(R9), X15
	XORQ  R8, R8

y1block:
	CMPQ    R8, SI
	CMOVQGT SI, R8 // clamp the last block to n-16
	MOVOU   (AX)(R8*1), X5
	MOVOU   (BX)(R8*1), X6
	MOVOU   (CX)(R8*1), X7

	// Pixels 0-7: R in X3, G in X4, B in X1.
	PXOR      X0, X0
	PUNPCKLBW X5, X0
	PXOR      X1, X1
	PUNPCKLBW X6, X1
	PXOR      X2, X2
	PUNPCKLBW X7, X2
	CONV_SSE(X0, X1, X2, X3, X4)

	// Pixels 8-15: R in X6, G in X7, B in X2.
	PXOR      X0, X0
	PUNPCKHBW X5, X0
	PXOR      X2, X2
	PUNPCKHBW X6, X2
	PXOR      X5, X5
	PUNPCKHBW X7, X5
	CONV_SSE(X0, X2, X5, X6, X7)

	PACKUSWB X6, X3
	PACKUSWB X7, X4
	PACKUSWB X2, X1
	PCMPEQB  X0, X0
	TESTQ    DX, DX
	JZ       y1store
	MOVOU    (DX)(R8*1), X0

y1store:
	// Interleave R, G, B, A (X3, X4, X1, X0) into 16 RGBA pixels.
	MOVO      X3, X2
	PUNPCKLBW X4, X3
	PUNPCKHBW X4, X2
	MOVO      X1, X5
	PUNPCKLBW X0, X1
	PUNPCKHBW X0, X5
	MOVO      X3, X6
	PUNPCKLWL X1, X3
	PUNPCKHWL X1, X6
	MOVO      X2, X7
	PUNPCKLWL X5, X2
	PUNPCKHWL X5, X7
	LEAQ      (DI)(R8*4), R9
	MOVOU     X3, 0(R9)
	MOVOU     X6, 16(R9)
	MOVOU     X2, 32(R9)
	MOVOU     X7, 48(R9)
	CMPQ      R8, SI
	JEQ       y1done
	ADDQ      $16, R8
	JMP       y1block

y1done:
	RET

// CONV_AVX(Y, U, V, R, G, B) is CONV_SSE for 16 pixels in VEX form, with
// the constants as memory operands at R9 (yuvK<>); Y, U, V are clobbered.
#define CONV_AVX(Y, U, V, R, G, B) \
	VPMULHUW 0x00(R9), Y, Y \
	VPMULHUW 0x20(R9), V, R \
	VPADDW   Y, R, R        \
	VPSUBW   0x40(R9), R, R \
	VPMULHUW 0x80(R9), V, V \
	VPMULHUW 0x60(R9), U, G \
	VPADDW   V, G, G        \
	VPADDW   0xa0(R9), Y, V \
	VPSUBW   G, V, G        \
	VPMULHUW 0xc0(R9), U, U \
	VPADDUSW Y, U, U        \
	VPSUBUSW 0xe0(R9), U, B \
	VPSRAW   $6, R, R       \
	VPSRAW   $6, G, G       \
	VPSRLW   $6, B, B

// func yuvToRGBARowAVX2(y, u, v, a, dst *byte, n int)
//
// yuvToRGBARowSSE41 in blocks of 32 pixels, n >= 32. The unpack and pack
// instructions work within 128-bit lanes: the R, G, B, A bytes come back
// in order, the interleaved RGBA words hold pixels 0-3 and 16-19 (and so
// on) and are stored one 128-bit lane at a time.
//
// AX y, BX u, CX v, DX a, DI dst, SI last block start, R8 block start,
// R9 yuvK<>, R10 output address. Y15 zero.
TEXT ·yuvToRGBARowAVX2(SB), NOSPLIT, $0-48
	MOVQ  y+0(FP), AX
	MOVQ  u+8(FP), BX
	MOVQ  v+16(FP), CX
	MOVQ  a+24(FP), DX
	MOVQ  dst+32(FP), DI
	MOVQ  n+40(FP), SI
	SUBQ  $32, SI
	MOVQ  $yuvK<>(SB), R9
	VPXOR Y15, Y15, Y15
	XORQ  R8, R8

y2block:
	CMPQ    R8, SI
	CMOVQGT SI, R8 // clamp the last block to n-32
	VMOVDQU (AX)(R8*1), Y12
	VMOVDQU (BX)(R8*1), Y13
	VMOVDQU (CX)(R8*1), Y14

	VPUNPCKLBW Y12, Y15, Y0
	VPUNPCKLBW Y13, Y15, Y1
	VPUNPCKLBW Y14, Y15, Y2
	CONV_AVX(Y0, Y1, Y2, Y3, Y4, Y5)
	VPUNPCKHBW Y12, Y15, Y0
	VPUNPCKHBW Y13, Y15, Y1
	VPUNPCKHBW Y14, Y15, Y2
	CONV_AVX(Y0, Y1, Y2, Y6, Y7, Y8)
	VPACKUSWB  Y6, Y3, Y3
	VPACKUSWB  Y7, Y4, Y4
	VPACKUSWB  Y8, Y5, Y5
	VPCMPEQB   Y6, Y6, Y6
	TESTQ      DX, DX
	JZ         y2store
	VMOVDQU    (DX)(R8*1), Y6

y2store:
	VPUNPCKLBW Y4, Y3, Y0
	VPUNPCKHBW Y4, Y3, Y1
	VPUNPCKLBW Y6, Y5, Y2
	VPUNPCKHBW Y6, Y5, Y7
	VPUNPCKLWD Y2, Y0, Y3
	VPUNPCKHWD Y2, Y0, Y4
	VPUNPCKLWD Y7, Y1, Y5
	VPUNPCKHWD Y7, Y1, Y6
	LEAQ         (DI)(R8*4), R10
	VMOVDQU      X3, 0(R10)
	VMOVDQU      X4, 16(R10)
	VMOVDQU      X5, 32(R10)
	VMOVDQU      X6, 48(R10)
	VEXTRACTI128 $1, Y3, 64(R10)
	VEXTRACTI128 $1, Y4, 80(R10)
	VEXTRACTI128 $1, Y5, 96(R10)
	VEXTRACTI128 $1, Y6, 112(R10)
	CMPQ         R8, SI
	JEQ          y2done
	ADDQ         $32, R8
	JMP          y2block

y2done:
	VZEROUPPER
	RET
