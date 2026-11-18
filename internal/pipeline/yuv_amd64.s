//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 and AVX2 YUV->RGBA row conversion for the scaled Y'CbCr path
// (exportYUV), adapted from libwebp (BSD-3-Clause) src/dsp/yuv_sse2.c
// (ConvertYUV444ToRGB_SSE2, PackAndStore4_SSE2, YuvToRgbaRow_SSE2) and
// bit-exact with yuvToRGBARowGo. internal/vp8/yuv_amd64.s uses the same
// conversion inside its upsampling kernels; see the comments there for the
// arithmetic (samples<<8 in 16-bit lanes, PMULHUW = libwebp's MultHi).
//
// The SSE4.1 kernel uses legacy SSE encodings only; the AVX2 one uses VEX
// encodings only and ends with VZEROUPPER.

DATA yuvK19077<>+0(SB)/8, $0x4A854A854A854A85
DATA yuvK19077<>+8(SB)/8, $0x4A854A854A854A85
DATA yuvK19077<>+16(SB)/8, $0x4A854A854A854A85
DATA yuvK19077<>+24(SB)/8, $0x4A854A854A854A85
GLOBL yuvK19077<>(SB), RODATA|NOPTR, $32

DATA yuvK26149<>+0(SB)/8, $0x6625662566256625
DATA yuvK26149<>+8(SB)/8, $0x6625662566256625
DATA yuvK26149<>+16(SB)/8, $0x6625662566256625
DATA yuvK26149<>+24(SB)/8, $0x6625662566256625
GLOBL yuvK26149<>(SB), RODATA|NOPTR, $32

DATA yuvK14234<>+0(SB)/8, $0x379A379A379A379A
DATA yuvK14234<>+8(SB)/8, $0x379A379A379A379A
DATA yuvK14234<>+16(SB)/8, $0x379A379A379A379A
DATA yuvK14234<>+24(SB)/8, $0x379A379A379A379A
GLOBL yuvK14234<>(SB), RODATA|NOPTR, $32

DATA yuvK33050<>+0(SB)/8, $0x811A811A811A811A
DATA yuvK33050<>+8(SB)/8, $0x811A811A811A811A
DATA yuvK33050<>+16(SB)/8, $0x811A811A811A811A
DATA yuvK33050<>+24(SB)/8, $0x811A811A811A811A
GLOBL yuvK33050<>(SB), RODATA|NOPTR, $32

DATA yuvK17685<>+0(SB)/8, $0x4515451545154515
DATA yuvK17685<>+8(SB)/8, $0x4515451545154515
DATA yuvK17685<>+16(SB)/8, $0x4515451545154515
DATA yuvK17685<>+24(SB)/8, $0x4515451545154515
GLOBL yuvK17685<>(SB), RODATA|NOPTR, $32

DATA yuvK6419<>+0(SB)/8, $0x1913191319131913
DATA yuvK6419<>+8(SB)/8, $0x1913191319131913
DATA yuvK6419<>+16(SB)/8, $0x1913191319131913
DATA yuvK6419<>+24(SB)/8, $0x1913191319131913
GLOBL yuvK6419<>(SB), RODATA|NOPTR, $32

DATA yuvK13320<>+0(SB)/8, $0x3408340834083408
DATA yuvK13320<>+8(SB)/8, $0x3408340834083408
DATA yuvK13320<>+16(SB)/8, $0x3408340834083408
DATA yuvK13320<>+24(SB)/8, $0x3408340834083408
GLOBL yuvK13320<>(SB), RODATA|NOPTR, $32

DATA yuvK8708<>+0(SB)/8, $0x2204220422042204
DATA yuvK8708<>+8(SB)/8, $0x2204220422042204
DATA yuvK8708<>+16(SB)/8, $0x2204220422042204
DATA yuvK8708<>+24(SB)/8, $0x2204220422042204
GLOBL yuvK8708<>(SB), RODATA|NOPTR, $32

DATA yuvFF<>+0(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+16(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+24(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL yuvFF<>(SB), RODATA|NOPTR, $32

// CONV8_SSE is ConvertYUV444ToRGB_SSE2 on 8 pixels whose samples<<8 are in
// YW, UW, VW (words). Leaves R in RW, G in YW and B in UW (int16, >> 6);
// clobbers VW and TW. The constants are 32-byte aligned (the linker aligns
// data symbols to their size, up to 32), as legacy SSE memory operands
// require.
#define CONV8_SSE(YW, UW, VW, RW, TW) \
	PMULHUW yuvK19077<>(SB), YW \
	MOVO VW, RW \
	PMULHUW yuvK26149<>(SB), RW \
	PADDW YW, RW \
	PSUBW yuvK14234<>(SB), RW \
	MOVO UW, TW \
	PMULHUW yuvK6419<>(SB), TW \
	PMULHUW yuvK13320<>(SB), VW \
	PADDW VW, TW \
	PMULHUW yuvK33050<>(SB), UW \
	PADDUSW YW, UW \
	PSUBUSW yuvK17685<>(SB), UW \
	PADDW yuvK8708<>(SB), YW \
	PSUBW TW, YW \
	PSRAW $6, RW \
	PSRAW $6, YW \
	PSRLW $6, UW

// CONV16_SSE converts the 16 pixels with luma X8, chroma X9 (U) and X10
// (V) to R, G, B bytes in X6, X0 and X2. Clobbers X0-X8.
#define CONV16_SSE \
	PXOR X0, X0 \
	PUNPCKLBW X8, X0 \
	PXOR X1, X1 \
	PUNPCKHBW X8, X1 \
	PXOR X2, X2 \
	PUNPCKLBW X9, X2 \
	PXOR X3, X3 \
	PUNPCKHBW X9, X3 \
	PXOR X4, X4 \
	PUNPCKLBW X10, X4 \
	PXOR X5, X5 \
	PUNPCKHBW X10, X5 \
	CONV8_SSE(X0, X2, X4, X6, X7) \
	CONV8_SSE(X1, X3, X5, X8, X7) \
	PACKUSWB X8, X6 \
	PACKUSWB X1, X0 \
	PACKUSWB X3, X2

// STORE16_SSE interleaves R (X6), G (X0), B (X2) and A (X11) into 16 RGBA
// pixels at (R11)(CX*4). Clobbers X0-X3, X6, X7.
#define STORE16_SSE \
	MOVO X6, X7 \
	PUNPCKLBW X0, X6 \
	PUNPCKHBW X0, X7 \
	MOVO X2, X3 \
	PUNPCKLBW X11, X2 \
	PUNPCKHBW X11, X3 \
	MOVO X6, X0 \
	PUNPCKLWL X2, X6 \
	PUNPCKHWL X2, X0 \
	MOVO X7, X1 \
	PUNPCKLWL X3, X7 \
	PUNPCKHWL X3, X1 \
	MOVOU X6, (R11)(CX*4) \
	MOVOU X0, 16(R11)(CX*4) \
	MOVOU X7, 32(R11)(CX*4) \
	MOVOU X1, 48(R11)(CX*4)

// func yuvToRGBARowSSE41(y, u, v, a, dst *byte, n int)
//
// Converts pixels [0, n), n >= 16, in blocks of 16; the last block starts
// at n-16 and may overlap the previous one (rewriting the same values), so
// nothing outside the rows is accessed. a is the alpha row or nil.
//
// Registers: R10 y, SI u, DI v, R12 a, R11 dst, BX last block start, CX
// block start.
TEXT ·yuvToRGBARowSSE41(SB), NOSPLIT, $0-48
	MOVQ y+0(FP), R10
	MOVQ u+8(FP), SI
	MOVQ v+16(FP), DI
	MOVQ a+24(FP), R12
	MOVQ dst+32(FP), R11
	MOVQ n+40(FP), BX
	SUBQ $16, BX
	XORQ CX, CX

sseloop:
	CMPQ CX, BX
	CMOVQGT BX, CX // clamp the last block to n-16
	MOVOU (R10)(CX*1), X8
	MOVOU (SI)(CX*1), X9
	MOVOU (DI)(CX*1), X10
	CONV16_SSE
	MOVOU yuvFF<>(SB), X11
	TESTQ R12, R12
	JZ 2(PC)
	MOVOU (R12)(CX*1), X11
	STORE16_SSE
	CMPQ CX, BX
	JEQ ssedone
	ADDQ $16, CX
	JMP sseloop

ssedone:
	RET

// CONV16_AVX is CONV8_SSE on 16 pixels (ymm, VEX encoded).
#define CONV16_AVX(YW, UW, VW, RW, TW) \
	VPMULHUW yuvK19077<>(SB), YW, YW \
	VPMULHUW yuvK26149<>(SB), VW, RW \
	VPADDW YW, RW, RW \
	VPSUBW yuvK14234<>(SB), RW, RW \
	VPMULHUW yuvK6419<>(SB), UW, TW \
	VPMULHUW yuvK13320<>(SB), VW, VW \
	VPADDW VW, TW, TW \
	VPMULHUW yuvK33050<>(SB), UW, UW \
	VPADDUSW YW, UW, UW \
	VPSUBUSW yuvK17685<>(SB), UW, UW \
	VPADDW yuvK8708<>(SB), YW, YW \
	VPSUBW TW, YW, YW \
	VPSRAW $6, RW, RW \
	VPSRAW $6, YW, YW \
	VPSRLW $6, UW, UW

// CONV32_AVX converts the 32 pixels with luma Y8 and chroma Y10 (U), Y11
// (V) to R, G, B bytes in Y6, Y0, Y2 (pixel order; the in-lane unpacks and
// PACKUSWB cancel out). Y15 must be zero. Clobbers Y0-Y8.
#define CONV32_AVX \
	VPUNPCKLBW Y8, Y15, Y0 \
	VPUNPCKHBW Y8, Y15, Y1 \
	VPUNPCKLBW Y10, Y15, Y2 \
	VPUNPCKHBW Y10, Y15, Y3 \
	VPUNPCKLBW Y11, Y15, Y4 \
	VPUNPCKHBW Y11, Y15, Y5 \
	CONV16_AVX(Y0, Y2, Y4, Y6, Y7) \
	CONV16_AVX(Y1, Y3, Y5, Y8, Y7) \
	VPACKUSWB Y8, Y6, Y6 \
	VPACKUSWB Y1, Y0, Y0 \
	VPACKUSWB Y3, Y2, Y2

// STORE32_AVX interleaves R (Y6), G (Y0), B (Y2) and A (Y9) into 32 RGBA
// pixels at (R11)(CX*4); VPERM2I128 puts the in-lane results in order.
// Clobbers Y0-Y7.
#define STORE32_AVX \
	VPUNPCKLBW Y0, Y6, Y1 \
	VPUNPCKHBW Y0, Y6, Y3 \
	VPUNPCKLBW Y9, Y2, Y4 \
	VPUNPCKHBW Y9, Y2, Y5 \
	VPUNPCKLWD Y4, Y1, Y0 \
	VPUNPCKHWD Y4, Y1, Y6 \
	VPUNPCKLWD Y5, Y3, Y2 \
	VPUNPCKHWD Y5, Y3, Y7 \
	VPERM2I128 $0x20, Y6, Y0, Y1 \
	VPERM2I128 $0x20, Y7, Y2, Y3 \
	VPERM2I128 $0x31, Y6, Y0, Y4 \
	VPERM2I128 $0x31, Y7, Y2, Y5 \
	VMOVDQU Y1, (R11)(CX*4) \
	VMOVDQU Y3, 32(R11)(CX*4) \
	VMOVDQU Y4, 64(R11)(CX*4) \
	VMOVDQU Y5, 96(R11)(CX*4)

// func yuvToRGBARowAVX2(y, u, v, a, dst *byte, n int)
//
// yuvToRGBARowSSE41 with blocks of 32 pixels (n >= 32; the last block
// starts at n-32).
//
// Registers: R10 y, SI u, DI v, R12 a, R11 dst, BX last block start, CX
// block start, Y15 = 0.
TEXT ·yuvToRGBARowAVX2(SB), NOSPLIT, $0-48
	MOVQ y+0(FP), R10
	MOVQ u+8(FP), SI
	MOVQ v+16(FP), DI
	MOVQ a+24(FP), R12
	MOVQ dst+32(FP), R11
	MOVQ n+40(FP), BX
	SUBQ $32, BX
	VPXOR Y15, Y15, Y15
	XORQ CX, CX

avxloop:
	CMPQ CX, BX
	CMOVQGT BX, CX // clamp the last block to n-32
	VMOVDQU (R10)(CX*1), Y8
	VMOVDQU (SI)(CX*1), Y10
	VMOVDQU (DI)(CX*1), Y11
	CONV32_AVX
	VMOVDQU yuvFF<>(SB), Y9
	TESTQ R12, R12
	JZ 2(PC)
	VMOVDQU (R12)(CX*1), Y9
	STORE32_AVX
	CMPQ CX, BX
	JEQ avxdone
	ADDQ $32, CX
	JMP avxloop

avxdone:
	VZEROUPPER
	RET
