//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 and AVX2 YUV->RGBA conversion with libwebp's fancy upsampler,
// adapted from libwebp (BSD-3-Clause) src/dsp/upsampling_sse2.c
// (UPSAMPLE_32PIXELS, GET_M) and src/dsp/yuv_sse2.c
// (ConvertYUV444ToRGB_SSE2, PackAndStore4_SSE2), bit-exact with the Go
// code in yuv.go (yuvPix, upsampleUV, sampleRowGo).
//
// The conversion works on 16-bit lanes holding sample<<8 (the byte
// unpacked into the high half of a word), so that PMULHUW by a 14-bit
// constant k gives libwebp's MultHi(sample, k) = (sample*k) >> 8 exactly:
//   y1 = MultHi(y, 19077)
//   R  = (y1 + MultHi(v, 26149) - 14234) >> 6           (arithmetic)
//   G  = (y1 - MultHi(u, 6419) - MultHi(v, 13320) + 8708) >> 6
//   B  = sat0(y1 + MultHi(u, 33050) - 17685) >> 6       (logical)
// R and G stay within int16 (wrapping adds give the exact value); B may
// exceed 32767, so it is formed with unsigned saturating arithmetic and a
// logical shift, as in libwebp. PACKUSWB then clamps to [0, 255], which is
// VP8Clip8.
//
// The SSE4.1 kernels use legacy SSE encodings only; the AVX2 ones use VEX
// encodings only and end with VZEROUPPER.

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

DATA yuvOne<>+0(SB)/8, $0x0101010101010101
DATA yuvOne<>+8(SB)/8, $0x0101010101010101
DATA yuvOne<>+16(SB)/8, $0x0101010101010101
DATA yuvOne<>+24(SB)/8, $0x0101010101010101
GLOBL yuvOne<>(SB), RODATA|NOPTR, $32

DATA yuvFF<>+0(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+16(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA yuvFF<>+24(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL yuvFF<>(SB), RODATA|NOPTR, $32

// ---- SSE4.1 ---------------------------------------------------------------

// CONV8_SSE is ConvertYUV444ToRGB_SSE2 on 8 pixels whose samples<<8 are in
// YW, UW, VW (words). Leaves R in RW, G in YW and B in UW (int16, already
// >> 6); clobbers VW and TW. The 32-byte constants are 32-byte aligned (the
// linker aligns data symbols to their size, up to 32), as the legacy SSE
// memory operands require.
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
// pixels at DOFF(R11)(CX*4). Clobbers X0-X3, X6, X7.
#define STORE16_SSE(DOFF) \
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
	MOVOU X6, DOFF(R11)(CX*4) \
	MOVOU X0, (DOFF+16)(R11)(CX*4) \
	MOVOU X7, (DOFF+32)(R11)(CX*4) \
	MOVOU X1, (DOFF+48)(R11)(CX*4)

// ALPHA16_SSE loads the 16 alpha bytes at H(R12)(CX*1) into X11, or 0xff
// when R12 is nil.
#define ALPHA16_SSE(H) \
	MOVOU yuvFF<>(SB), X11 \
	TESTQ R12, R12 \
	JZ 2(PC) \
	MOVOU H(R12)(CX*1), X11

// UPS32_SSE is libwebp's UPSAMPLE_32PIXELS: from the 17 chroma samples at
// TOP and CUR (pointer registers) it computes the 32 full resolution
// samples of the luma row next to TOP (stored at TOFF(SP)) and of the luma
// row next to CUR (at BOFF(SP)). With a = top[i], b = top[i+1],
// c = cur[i], d = cur[i+1], it forms diag1 = (a + 3b + 3c + d) >> 3 and
// diag2 = (3a + b + c + 3d) >> 3 exactly with byte averages and an lsb
// correction (k = (a+b+c+d) >> 2 first), then
//   top = zip(avg(a, diag1), avg(b, diag2))
//   bot = zip(avg(c, diag2), avg(d, diag1))
// which equals upsampleUV's (diag + x + 1) >> 1 with its +8 rounding (the
// identity was checked on all 2^32 (a, b, c, d)). X12 holds 0x01 bytes.
// Clobbers X0-X11.
#define UPS32_SSE(TOP, CUR, TOFF, BOFF) \
	MOVOU (TOP), X0 \
	MOVOU 1(TOP), X1 \
	MOVOU (CUR), X2 \
	MOVOU 1(CUR), X3 \
	MOVO X0, X4 \
	PAVGB X3, X4 \
	MOVO X1, X5 \
	PAVGB X2, X5 \
	MOVO X4, X6 \
	PXOR X5, X6 \
	MOVO X0, X7 \
	PXOR X3, X7 \
	MOVO X1, X8 \
	PXOR X2, X8 \
	MOVO X7, X9 \
	POR X8, X9 \
	POR X6, X9 \
	PAND X12, X9 \
	MOVO X4, X10 \
	PAVGB X5, X10 \
	PSUBB X9, X10 \
	PAND X6, X8 \
	MOVO X10, X11 \
	PXOR X5, X11 \
	POR X11, X8 \
	PAND X12, X8 \
	PAVGB X10, X5 \
	PSUBB X8, X5 \
	PAND X6, X7 \
	MOVO X10, X11 \
	PXOR X4, X11 \
	POR X11, X7 \
	PAND X12, X7 \
	PAVGB X10, X4 \
	PSUBB X7, X4 \
	PAVGB X5, X0 \
	PAVGB X4, X1 \
	PAVGB X4, X2 \
	PAVGB X5, X3 \
	MOVO X0, X6 \
	PUNPCKLBW X1, X0 \
	PUNPCKHBW X1, X6 \
	MOVOU X0, TOFF(SP) \
	MOVOU X6, (TOFF+16)(SP) \
	MOVO X2, X7 \
	PUNPCKLBW X3, X2 \
	PUNPCKHBW X3, X7 \
	MOVOU X2, BOFF(SP) \
	MOVOU X7, (BOFF+16)(SP)

// ROW16_SSE converts the 16 pixels at CX+H of the row with luma R10, alpha
// R12 (or nil) and output R11, with the upsampled chroma at UOFF+H(SP) and
// VOFF+H(SP). H4 = 4*H.
#define ROW16_SSE(H, H4, UOFF, VOFF) \
	MOVOU H(R10)(CX*1), X8 \
	MOVOU (UOFF+H)(SP), X9 \
	MOVOU (VOFF+H)(SP), X10 \
	CONV16_SSE \
	ALPHA16_SSE(H) \
	STORE16_SSE(H4)

// func upsampleRowsSSE41(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA *byte, uvW int)
//
// Mirrors libwebp's UpsampleRgbaLinePair_SSE2 without its first and last
// pixels: converts output pixels [1, 2*uvW-1) of the top row and, when
// botY != nil, of the bottom row. The chroma rows hold uvW >= 17 samples.
// Blocks of 32 pixels start at chroma index j = 0, 16, 32, ...; the last
// block starts at uvW-17 and may overlap the previous one (it rewrites the
// same values), so nothing outside the rows is read or written. topA/botA
// are alpha rows, or nil for opaque output.
//
// Registers: SI/DI/R8/R9 topU/topV/curU/curV, BX last block start, AX
// block start j, CX = 2j+1 (first output pixel of the block), R10/R11/R12
// luma/output/alpha row of the current row (chroma pointers while
// upsampling), X12 = 0x01 bytes. The frame holds the upsampled chroma of
// the block: top U at 0, top V at 32, bottom U at 64, bottom V at 96.
TEXT ·upsampleRowsSSE41(SB), NOSPLIT, $128-88
	MOVQ topU+16(FP), SI
	MOVQ topV+24(FP), DI
	MOVQ curU+32(FP), R8
	MOVQ curV+40(FP), R9
	MOVQ uvW+80(FP), BX
	SUBQ $17, BX
	MOVOU yuvOne<>(SB), X12
	XORQ AX, AX

sseblock:
	CMPQ AX, BX
	CMOVQGT BX, AX // clamp the last block to uvW-17
	LEAQ (SI)(AX*1), R10
	LEAQ (R8)(AX*1), R11
	UPS32_SSE(R10, R11, 0, 64)
	LEAQ (DI)(AX*1), R10
	LEAQ (R9)(AX*1), R11
	UPS32_SSE(R10, R11, 32, 96)
	LEAQ 1(AX)(AX*1), CX

	MOVQ topY+0(FP), R10
	MOVQ topDst+48(FP), R11
	MOVQ topA+64(FP), R12
	ROW16_SSE(0, 0, 0, 32)
	ROW16_SSE(16, 64, 0, 32)

	MOVQ botY+8(FP), R10
	TESTQ R10, R10
	JZ ssenext
	MOVQ botDst+56(FP), R11
	MOVQ botA+72(FP), R12
	ROW16_SSE(0, 0, 64, 96)
	ROW16_SSE(16, 64, 64, 96)

ssenext:
	CMPQ AX, BX
	JEQ ssedone
	ADDQ $16, AX
	JMP sseblock

ssedone:
	RET

// func sampleRowSSE41(y, u, v, a, dst *byte, w int)
//
// Point-sampled conversion (libwebp's YuvToRgbaRow_SSE2): pixel x uses
// chroma sample x/2. Converts pixels [0, w) with w even and >= 16, in
// blocks of 16 pixels starting at even x; the last block starts at w-16
// and may overlap the previous one. a is the alpha row or nil (opaque).
//
// Registers: R10 y, SI u, DI v, R12 a, R11 dst, BX last block start, CX
// block start x, DX = x/2.
TEXT ·sampleRowSSE41(SB), NOSPLIT, $0-48
	MOVQ y+0(FP), R10
	MOVQ u+8(FP), SI
	MOVQ v+16(FP), DI
	MOVQ a+24(FP), R12
	MOVQ dst+32(FP), R11
	MOVQ w+40(FP), BX
	SUBQ $16, BX
	XORQ CX, CX

ssloop:
	CMPQ CX, BX
	CMOVQGT BX, CX // clamp the last block to w-16
	MOVQ CX, DX
	SHRQ $1, DX
	MOVOU (R10)(CX*1), X8
	MOVQ (SI)(DX*1), X9
	PUNPCKLBW X9, X9
	MOVQ (DI)(DX*1), X10
	PUNPCKLBW X10, X10
	CONV16_SSE
	ALPHA16_SSE(0)
	STORE16_SSE(0)
	CMPQ CX, BX
	JEQ ssdone
	ADDQ $16, CX
	JMP ssloop

ssdone:
	RET

// ---- AVX2 -----------------------------------------------------------------

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

// CONV32_AVX converts the 32 pixels with luma Y8 and chroma UT, VT (32
// bytes each, in pixel order) to R, G, B bytes in Y6, Y0, Y2 (pixel
// order). The unpacks work within 128-bit lanes, so the words hold pixels
// 0-7|16-23 (Y0, Y2, Y4) and 8-15|24-31 (Y1, Y3, Y5), and PACKUSWB puts
// them back in pixel order. Y15 must be zero. Clobbers Y0-Y8.
#define CONV32_AVX(UT, VT) \
	VPUNPCKLBW Y8, Y15, Y0 \
	VPUNPCKHBW Y8, Y15, Y1 \
	VPUNPCKLBW UT, Y15, Y2 \
	VPUNPCKHBW UT, Y15, Y3 \
	VPUNPCKLBW VT, Y15, Y4 \
	VPUNPCKHBW VT, Y15, Y5 \
	CONV16_AVX(Y0, Y2, Y4, Y6, Y7) \
	CONV16_AVX(Y1, Y3, Y5, Y8, Y7) \
	VPACKUSWB Y8, Y6, Y6 \
	VPACKUSWB Y1, Y0, Y0 \
	VPACKUSWB Y3, Y2, Y2

// ALPHA32_AVX loads the 32 alpha bytes at (R12)(CX*1) into Y9, or 0xff
// when R12 is nil.
#define ALPHA32_AVX \
	VMOVDQU yuvFF<>(SB), Y9 \
	TESTQ R12, R12 \
	JZ 2(PC) \
	VMOVDQU (R12)(CX*1), Y9

// STORE32_AVX interleaves R (Y6), G (Y0), B (Y2) and A (Y9) into 32 RGBA
// pixels at (R11)(CX*4): the in-lane unpacks leave pixels 0-3|16-19,
// 4-7|20-23, 8-11|24-27 and 12-15|28-31, which VPERM2I128 puts in order.
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

// UPS32_AVX is UPS32_SSE on U and V at once: U in the low and V in the
// high 128-bit lane, from the chroma samples at R10 (top U), R13 (top V),
// R11 (cur U) and DX (cur V). Leaves the 32 upsampled samples of the top
// row in Y10 (U) and Y11 (V) and of the bottom row in Y12 (U) and Y13 (V),
// in pixel order. Y14 holds 0x01 bytes. Clobbers Y0-Y9.
#define UPS32_AVX \
	VMOVDQU (R10), X0 \
	VINSERTI128 $1, (R13), Y0, Y0 \
	VMOVDQU 1(R10), X1 \
	VINSERTI128 $1, 1(R13), Y1, Y1 \
	VMOVDQU (R11), X2 \
	VINSERTI128 $1, (DX), Y2, Y2 \
	VMOVDQU 1(R11), X3 \
	VINSERTI128 $1, 1(DX), Y3, Y3 \
	VPAVGB Y3, Y0, Y4 \
	VPAVGB Y2, Y1, Y5 \
	VPXOR Y5, Y4, Y6 \
	VPXOR Y3, Y0, Y7 \
	VPXOR Y2, Y1, Y8 \
	VPOR Y8, Y7, Y9 \
	VPOR Y6, Y9, Y9 \
	VPAND Y14, Y9, Y9 \
	VPAVGB Y5, Y4, Y10 \
	VPSUBB Y9, Y10, Y10 \
	VPAND Y6, Y8, Y8 \
	VPXOR Y5, Y10, Y11 \
	VPOR Y11, Y8, Y8 \
	VPAND Y14, Y8, Y8 \
	VPAVGB Y5, Y10, Y5 \
	VPSUBB Y8, Y5, Y5 \
	VPAND Y6, Y7, Y7 \
	VPXOR Y4, Y10, Y11 \
	VPOR Y11, Y7, Y7 \
	VPAND Y14, Y7, Y7 \
	VPAVGB Y4, Y10, Y4 \
	VPSUBB Y7, Y4, Y4 \
	VPAVGB Y5, Y0, Y0 \
	VPAVGB Y4, Y1, Y1 \
	VPAVGB Y4, Y2, Y2 \
	VPAVGB Y5, Y3, Y3 \
	VPUNPCKLBW Y1, Y0, Y4 \
	VPUNPCKHBW Y1, Y0, Y5 \
	VPERM2I128 $0x20, Y5, Y4, Y10 \
	VPERM2I128 $0x31, Y5, Y4, Y11 \
	VPUNPCKLBW Y3, Y2, Y4 \
	VPUNPCKHBW Y3, Y2, Y5 \
	VPERM2I128 $0x20, Y5, Y4, Y12 \
	VPERM2I128 $0x31, Y5, Y4, Y13

// func upsampleRowsAVX2(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA *byte, uvW int)
//
// The same as upsampleRowsSSE41 (same blocks of 32 pixels, uvW >= 17),
// with U and V upsampled together and 32 pixels converted at once.
//
// Registers: SI/DI/R8/R9 topU/topV/curU/curV, BX last block start, AX
// block start j, CX = 2j+1, R10/R11/R12 luma/output/alpha row (R10, R13,
// R11, DX chroma pointers while upsampling), Y14 = 0x01 bytes, Y15 = 0.
TEXT ·upsampleRowsAVX2(SB), NOSPLIT, $0-88
	MOVQ topU+16(FP), SI
	MOVQ topV+24(FP), DI
	MOVQ curU+32(FP), R8
	MOVQ curV+40(FP), R9
	MOVQ uvW+80(FP), BX
	SUBQ $17, BX
	VMOVDQU yuvOne<>(SB), Y14
	VPXOR Y15, Y15, Y15
	XORQ AX, AX

avxblock:
	CMPQ AX, BX
	CMOVQGT BX, AX // clamp the last block to uvW-17
	LEAQ (SI)(AX*1), R10
	LEAQ (DI)(AX*1), R13
	LEAQ (R8)(AX*1), R11
	LEAQ (R9)(AX*1), DX
	UPS32_AVX
	LEAQ 1(AX)(AX*1), CX

	MOVQ topY+0(FP), R10
	MOVQ topDst+48(FP), R11
	MOVQ topA+64(FP), R12
	VMOVDQU (R10)(CX*1), Y8
	CONV32_AVX(Y10, Y11)
	ALPHA32_AVX
	STORE32_AVX

	MOVQ botY+8(FP), R10
	TESTQ R10, R10
	JZ avxnext
	MOVQ botDst+56(FP), R11
	MOVQ botA+72(FP), R12
	VMOVDQU (R10)(CX*1), Y8
	CONV32_AVX(Y12, Y13)
	ALPHA32_AVX
	STORE32_AVX

avxnext:
	CMPQ AX, BX
	JEQ avxdone
	ADDQ $16, AX
	JMP avxblock

avxdone:
	VZEROUPPER
	RET

// func sampleRowAVX2(y, u, v, a, dst *byte, w int)
//
// sampleRowSSE41 with blocks of 32 pixels (w even and >= 32; the last
// block starts at w-32). The 16 chroma samples of a block are widened to
// words and doubled into bytes (u | u<<8) to get one sample per pixel.
//
// Registers: R10 y, SI u, DI v, R12 a, R11 dst, BX last block start, CX
// block start x, DX = x/2, Y15 = 0.
TEXT ·sampleRowAVX2(SB), NOSPLIT, $0-48
	MOVQ y+0(FP), R10
	MOVQ u+8(FP), SI
	MOVQ v+16(FP), DI
	MOVQ a+24(FP), R12
	MOVQ dst+32(FP), R11
	MOVQ w+40(FP), BX
	SUBQ $32, BX
	VPXOR Y15, Y15, Y15
	XORQ CX, CX

saloop:
	CMPQ CX, BX
	CMOVQGT BX, CX // clamp the last block to w-32
	MOVQ CX, DX
	SHRQ $1, DX
	VPMOVZXBW (SI)(DX*1), Y10
	VPSLLW $8, Y10, Y9
	VPOR Y9, Y10, Y10
	VPMOVZXBW (DI)(DX*1), Y11
	VPSLLW $8, Y11, Y9
	VPOR Y9, Y11, Y11
	VMOVDQU (R10)(CX*1), Y8
	CONV32_AVX(Y10, Y11)
	ALPHA32_AVX
	STORE32_AVX
	CMPQ CX, BX
	JEQ sadone
	ADDQ $32, CX
	JMP saloop

sadone:
	VZEROUPPER
	RET
