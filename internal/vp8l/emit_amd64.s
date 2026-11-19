//go:build amd64 && !purego

// SSE4.1 and AVX2 kernels of the output conversion, the inverse
// subtract-green and the inverse cross-color transforms. Algorithms adapted
// from libwebp (BSD-3-Clause), src/dsp/lossless_sse2.c, lossless_sse41.c and
// lossless_avx2.c; the Go references in transform.go and pixels.go are the
// specification.
//
// Pixels are ARGB words, so in memory each pixel is the bytes B, G, R, A.
// Every function processes exact pixel counts: vector loops of 16/32, 4 or
// 8 pixels, then one pixel at a time in the low 32 bits of a vector
// register (MOVL/VMOVD loads zero the rest), so nothing is read or written
// outside the slices of the Go wrappers (dsp_amd64.go). The AVX2 functions
// use VEX encodings only and end with VZEROUPPER.

#include "textflag.h"

// PSHUFB indices (bit 7 set yields 0).
//
// swzRGBA: B,G,R,A to R,G,B,A in each pixel.
DATA swzRGBA<>+0(SB)/8, $0x0704050603000102
DATA swzRGBA<>+8(SB)/8, $0x0f0c0d0e0b08090a
GLOBL swzRGBA<>(SB), RODATA|NOPTR, $16

// greenBR: the green byte at the blue and red positions, 0 elsewhere.
DATA greenBR<>+0(SB)/8, $0x8005800580018001
DATA greenBR<>+8(SB)/8, $0x800d800d80098009
GLOBL greenBR<>(SB), RODATA|NOPTR, $16

// ccGreen: G<<8 in both 16-bit halves of each pixel.
DATA ccGreen<>+0(SB)/8, $0x0580058001800180
DATA ccGreen<>+8(SB)/8, $0x0d800d8009800980
GLOBL ccGreen<>(SB), RODATA|NOPTR, $16

// ccRed: R<<8 in the low 16-bit half of each pixel, 0 in the high half.
DATA ccRed<>+0(SB)/8, $0x8080068080800280
DATA ccRed<>+8(SB)/8, $0x80800e8080800a80
GLOBL ccRed<>(SB), RODATA|NOPTR, $16

// ccMaskAG: 0xff in the alpha and green bytes (PBLENDVB keeps them).
DATA ccMaskAG<>+0(SB)/8, $0xff00ff00ff00ff00
DATA ccMaskAG<>+8(SB)/8, $0xff00ff00ff00ff00
GLOBL ccMaskAG<>(SB), RODATA|NOPTR, $16

// func emitRGBASSE41(src *uint32, dst *byte, n int)
//
// dst[4i:4i+4] = R, G, B, A of src[i] for i < n (libwebp
// VP8LConvertBGRAToRGBA_SSE2, with one PSHUFB per 4 pixels).
//
// SI src, DI dst, CX pixels left; X7 swzRGBA; X0-X3 pixels.
TEXT ·emitRGBASSE41(SB), NOSPLIT, $0-24
	MOVQ  src+0(FP), SI
	MOVQ  dst+8(FP), DI
	MOVQ  n+16(FP), CX
	MOVOU swzRGBA<>(SB), X7
	CMPQ  CX, $16
	JB    rgba4

rgba16:
	MOVOU  (SI), X0
	MOVOU  16(SI), X1
	MOVOU  32(SI), X2
	MOVOU  48(SI), X3
	PSHUFB X7, X0
	PSHUFB X7, X1
	PSHUFB X7, X2
	PSHUFB X7, X3
	MOVOU  X0, (DI)
	MOVOU  X1, 16(DI)
	MOVOU  X2, 32(DI)
	MOVOU  X3, 48(DI)
	ADDQ   $64, SI
	ADDQ   $64, DI
	SUBQ   $16, CX
	CMPQ   CX, $16
	JAE    rgba16

rgba4:
	CMPQ   CX, $4
	JB     rgba1
	MOVOU  (SI), X0
	PSHUFB X7, X0
	MOVOU  X0, (DI)
	ADDQ   $16, SI
	ADDQ   $16, DI
	SUBQ   $4, CX
	JMP    rgba4

rgba1:
	TESTQ  CX, CX
	JZ     rgbadone
	MOVL   (SI), X0
	PSHUFB X7, X0
	MOVL   X0, (DI)
	ADDQ   $4, SI
	ADDQ   $4, DI
	DECQ   CX
	JMP    rgba1

rgbadone:
	RET

// func emitRGBAAVX2(src *uint32, dst *byte, n int)
//
// emitRGBASSE41 with 8 pixels per vector (libwebp
// ConvertBGRAToRGBA_AVX2).
//
// SI src, DI dst, CX pixels left; Y7 swzRGBA; Y0-Y3 pixels.
TEXT ·emitRGBAAVX2(SB), NOSPLIT, $0-24
	MOVQ           src+0(FP), SI
	MOVQ           dst+8(FP), DI
	MOVQ           n+16(FP), CX
	VBROADCASTI128 swzRGBA<>(SB), Y7
	CMPQ           CX, $32
	JB             rgba8

rgba32:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 64(SI), Y2
	VMOVDQU 96(SI), Y3
	VPSHUFB Y7, Y0, Y0
	VPSHUFB Y7, Y1, Y1
	VPSHUFB Y7, Y2, Y2
	VPSHUFB Y7, Y3, Y3
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ    $128, SI
	ADDQ    $128, DI
	SUBQ    $32, CX
	CMPQ    CX, $32
	JAE     rgba32

rgba8:
	CMPQ    CX, $8
	JB      rgba4
	VMOVDQU (SI), Y0
	VPSHUFB Y7, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     rgba8

rgba4:
	CMPQ    CX, $4
	JB      rgba1
	VMOVDQU (SI), X0
	VPSHUFB X7, X0, X0
	VMOVDQU X0, (DI)
	ADDQ    $16, SI
	ADDQ    $16, DI
	SUBQ    $4, CX

rgba1:
	TESTQ   CX, CX
	JZ      rgbadone
	VMOVD   (SI), X0
	VPSHUFB X7, X0, X0
	VMOVD   X0, (DI)
	ADDQ    $4, SI
	ADDQ    $4, DI
	DECQ    CX
	JMP     rgba1

rgbadone:
	VZEROUPPER
	RET

// func emitAddGreenSSE41(src *uint32, dst *byte, n int)
//
// emitRGBASSE41 after adding green to red and blue (libwebp
// AddGreenToBlueAndRed_SSE2 fused with the conversion). The swizzle maps
// the green copies at the red and blue positions onto each other, so
// out = swz(p) + greenBR(p): two independent shuffles and an add.
//
// SI src, DI dst, CX pixels left; X7 swzRGBA, X6 greenBR; X0-X3 pixels,
// X4, X5 greens.
TEXT ·emitAddGreenSSE41(SB), NOSPLIT, $0-24
	MOVQ  src+0(FP), SI
	MOVQ  dst+8(FP), DI
	MOVQ  n+16(FP), CX
	MOVOU swzRGBA<>(SB), X7
	MOVOU greenBR<>(SB), X6
	CMPQ  CX, $8
	JB    ag4

ag8:
	MOVOU  (SI), X0
	MOVOU  16(SI), X1
	MOVO   X0, X4
	MOVO   X1, X5
	PSHUFB X6, X4
	PSHUFB X6, X5
	PSHUFB X7, X0
	PSHUFB X7, X1
	PADDB  X4, X0
	PADDB  X5, X1
	MOVOU  X0, (DI)
	MOVOU  X1, 16(DI)
	ADDQ   $32, SI
	ADDQ   $32, DI
	SUBQ   $8, CX
	CMPQ   CX, $8
	JAE    ag8

ag4:
	CMPQ   CX, $4
	JB     ag1
	MOVOU  (SI), X0
	MOVO   X0, X4
	PSHUFB X6, X4
	PSHUFB X7, X0
	PADDB  X4, X0
	MOVOU  X0, (DI)
	ADDQ   $16, SI
	ADDQ   $16, DI
	SUBQ   $4, CX

ag1:
	TESTQ  CX, CX
	JZ     agdone
	MOVL   (SI), X0
	MOVO   X0, X4
	PSHUFB X6, X4
	PSHUFB X7, X0
	PADDB  X4, X0
	MOVL   X0, (DI)
	ADDQ   $4, SI
	ADDQ   $4, DI
	DECQ   CX
	JMP    ag1

agdone:
	RET

// func emitAddGreenAVX2(src *uint32, dst *byte, n int)
//
// emitAddGreenSSE41 with 8 pixels per vector.
//
// SI src, DI dst, CX pixels left; Y7 swzRGBA, Y6 greenBR; Y0-Y3 pixels,
// Y4, Y5 greens.
TEXT ·emitAddGreenAVX2(SB), NOSPLIT, $0-24
	MOVQ           src+0(FP), SI
	MOVQ           dst+8(FP), DI
	MOVQ           n+16(FP), CX
	VBROADCASTI128 swzRGBA<>(SB), Y7
	VBROADCASTI128 greenBR<>(SB), Y6
	CMPQ           CX, $16
	JB             ag8

ag16:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VPSHUFB Y6, Y0, Y4
	VPSHUFB Y6, Y1, Y5
	VPSHUFB Y7, Y0, Y0
	VPSHUFB Y7, Y1, Y1
	VPADDB  Y4, Y0, Y0
	VPADDB  Y5, Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	SUBQ    $16, CX
	CMPQ    CX, $16
	JAE     ag16

ag8:
	CMPQ    CX, $8
	JB      ag4
	VMOVDQU (SI), Y0
	VPSHUFB Y6, Y0, Y4
	VPSHUFB Y7, Y0, Y0
	VPADDB  Y4, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX

ag4:
	CMPQ    CX, $4
	JB      ag1
	VMOVDQU (SI), X0
	VPSHUFB X6, X0, X4
	VPSHUFB X7, X0, X0
	VPADDB  X4, X0, X0
	VMOVDQU X0, (DI)
	ADDQ    $16, SI
	ADDQ    $16, DI
	SUBQ    $4, CX

ag1:
	TESTQ   CX, CX
	JZ      agdone
	VMOVD   (SI), X0
	VPSHUFB X6, X0, X4
	VPSHUFB X7, X0, X0
	VPADDB  X4, X0, X0
	VMOVD   X0, (DI)
	ADDQ    $4, SI
	ADDQ    $4, DI
	DECQ    CX
	JMP     ag1

agdone:
	VZEROUPPER
	RET

// func addGreenSSE41(in, out *uint32, n int)
//
// out[i] = in[i] with green added to red and blue (libwebp
// AddGreenToBlueAndRed_SSE2, with PSHUFB). in may alias out: each vector
// is loaded before it is stored.
//
// SI in, DI out, CX pixels left; X6 greenBR; X0-X3 pixels, X4, X5 greens.
TEXT ·addGreenSSE41(SB), NOSPLIT, $0-24
	MOVQ  in+0(FP), SI
	MOVQ  out+8(FP), DI
	MOVQ  n+16(FP), CX
	MOVOU greenBR<>(SB), X6
	CMPQ  CX, $8
	JB    ga4

ga8:
	MOVOU  (SI), X0
	MOVOU  16(SI), X1
	MOVO   X0, X4
	MOVO   X1, X5
	PSHUFB X6, X4
	PSHUFB X6, X5
	PADDB  X4, X0
	PADDB  X5, X1
	MOVOU  X0, (DI)
	MOVOU  X1, 16(DI)
	ADDQ   $32, SI
	ADDQ   $32, DI
	SUBQ   $8, CX
	CMPQ   CX, $8
	JAE    ga8

ga4:
	CMPQ   CX, $4
	JB     ga1
	MOVOU  (SI), X0
	MOVO   X0, X4
	PSHUFB X6, X4
	PADDB  X4, X0
	MOVOU  X0, (DI)
	ADDQ   $16, SI
	ADDQ   $16, DI
	SUBQ   $4, CX

ga1:
	TESTQ  CX, CX
	JZ     gadone
	MOVL   (SI), X0
	MOVO   X0, X4
	PSHUFB X6, X4
	PADDB  X4, X0
	MOVL   X0, (DI)
	ADDQ   $4, SI
	ADDQ   $4, DI
	DECQ   CX
	JMP    ga1

gadone:
	RET

// func addGreenAVX2(in, out *uint32, n int)
//
// addGreenSSE41 with 8 pixels per vector (libwebp
// AddGreenToBlueAndRed_AVX2).
//
// SI in, DI out, CX pixels left; Y6 greenBR; Y0-Y3 pixels, Y4, Y5 greens.
TEXT ·addGreenAVX2(SB), NOSPLIT, $0-24
	MOVQ           in+0(FP), SI
	MOVQ           out+8(FP), DI
	MOVQ           n+16(FP), CX
	VBROADCASTI128 greenBR<>(SB), Y6
	CMPQ           CX, $16
	JB             ga8

ga16:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VPSHUFB Y6, Y0, Y4
	VPSHUFB Y6, Y1, Y5
	VPADDB  Y4, Y0, Y0
	VPADDB  Y5, Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	SUBQ    $16, CX
	CMPQ    CX, $16
	JAE     ga16

ga8:
	CMPQ    CX, $8
	JB      ga4
	VMOVDQU (SI), Y0
	VPSHUFB Y6, Y0, Y4
	VPADDB  Y4, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX

ga4:
	CMPQ    CX, $4
	JB      ga1
	VMOVDQU (SI), X0
	VPSHUFB X6, X0, X4
	VPADDB  X4, X0, X0
	VMOVDQU X0, (DI)
	ADDQ    $16, SI
	ADDQ    $16, DI
	SUBQ    $4, CX

ga1:
	TESTQ   CX, CX
	JZ      gadone
	VMOVD   (SI), X0
	VPSHUFB X6, X0, X4
	VPADDB  X4, X0, X0
	VMOVD   X0, (DI)
	ADDQ    $4, SI
	ADDQ    $4, DI
	DECQ    CX
	JMP     ga1

gadone:
	VZEROUPPER
	RET

// CCMULTS loads the multipliers of the next tile from (DX) into
// R9 = [g2b*8, g2r*8] and AX = [r2b*8, 0] (16-bit lanes, low first): the
// sign-extended multipliers pre-scaled so that PMULHW of a channel byte
// in the high half of a 16-bit lane, (c<<8) * (m*8) >> 16, gives
// (c * m) >> 5, colorTransformDelta. g2r is byte 0, g2b byte 1, r2b byte 2
// of the multiplier word. Clobbers R10.
#define CCMULTS \
	MOVL    (DX), AX     \
	ADDQ    $4, DX       \
	MOVBLSX AL, R9       \
	SHLL    $19, R9      \
	MOVL    AX, R10      \
	SHRL    $8, R10      \
	MOVBLSX R10B, R10    \
	SHLL    $3, R10      \
	ANDL    $0xffff, R10 \
	ORL     R10, R9      \
	SHRL    $16, AX      \
	MOVBLSX AL, AX       \
	SHLL    $3, AX       \
	ANDL    $0xffff, AX

// CCOLOR undoes the cross-color transform on the pixels in X1 (4, or one
// in the low word) into X2 (libwebp TransformColorInverse_SSE41). Per
// pixel, in 16-bit lanes [low half, high half] = [G<<8|B, A<<8|R]:
//   X2 = [G<<8, G<<8] * [g2b, g2r] >> 5, bytewise + X1: B' = B+dB at
//        byte 0, R' = R+dR at byte 2;
//   X3 = [R'<<8, 0] * [r2b, 0] >> 5 = [dB2, 0]; X2 += X3 bytewise;
//   G and A restored from X1 (PBLENDVB, mask X0).
// Only the low byte of each delta matters, as in colorTransformDelta.
// Uses X0 ccMaskAG, X6 ccGreen, X7 ccRed, X5 [g2b, g2r], X4 [r2b, 0].
#define CCOLOR \
	MOVO     X1, X2 \
	PSHUFB   X6, X2 \
	PMULHW   X5, X2 \
	PADDB    X1, X2 \
	MOVO     X2, X3 \
	PSHUFB   X7, X3 \
	PMULHW   X4, X3 \
	PADDB    X3, X2 \
	PBLENDVB X0, X1, X2

// func crossColorRowSSE41(mults, src, dst *uint32, width, bits int)
//
// Undoes the cross-color transform on one row of width pixels, the tile
// of 1<<bits pixels at x using the multipliers mults[x>>bits]. src may
// alias dst.
//
// DX mults, SI src, DI dst, BX pixels left in the row, R8 tile width,
// R11 pixels left in the tile; AX, R9, R10 temporaries.
TEXT ·crossColorRowSSE41(SB), NOSPLIT, $0-40
	MOVQ  mults+0(FP), DX
	MOVQ  src+8(FP), SI
	MOVQ  dst+16(FP), DI
	MOVQ  width+24(FP), BX
	MOVQ  bits+32(FP), CX
	MOVL  $1, R8
	SHLQ  CX, R8
	MOVOU ccMaskAG<>(SB), X0
	MOVOU ccGreen<>(SB), X6
	MOVOU ccRed<>(SB), X7

cctile:
	TESTQ   BX, BX
	JZ      ccdone
	CCMULTS
	MOVL    R9, X5
	PSHUFD  $0, X5, X5
	MOVL    AX, X4
	PSHUFD  $0, X4, X4
	MOVQ    R8, R11
	CMPQ    BX, R11
	CMOVQLT BX, R11
	SUBQ    R11, BX

cc4:
	CMPQ  R11, $4
	JB    cc1
	MOVOU (SI), X1
	CCOLOR
	MOVOU X2, (DI)
	ADDQ  $16, SI
	ADDQ  $16, DI
	SUBQ  $4, R11
	JMP   cc4

cc1:
	TESTQ R11, R11
	JZ    cctile
	MOVL  (SI), X1
	CCOLOR
	MOVL  X2, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  R11
	JMP   cc1

ccdone:
	RET

// CCOLORV is CCOLOR in VEX form for register width R (X or Y): v1 in,
// v2 out, v3 temporary; v0 mask, v6 ccGreen, v7 ccRed, v5 [g2b, g2r],
// v4 [r2b, 0].
#define CCOLORV(v0, v1, v2, v3, v4, v5, v6, v7) \
	VPSHUFB   v6, v1, v2     \
	VPMULHW   v5, v2, v2     \
	VPADDB    v1, v2, v2     \
	VPSHUFB   v7, v2, v3     \
	VPMULHW   v4, v3, v3     \
	VPADDB    v3, v2, v2     \
	VPBLENDVB v0, v1, v2, v2

// func crossColorRowAVX2(mults, src, dst *uint32, width, bits int)
//
// crossColorRowSSE41 with 8 pixels per vector (libwebp
// TransformColorInverse_AVX2), then 4 and 1.
//
// Registers as in crossColorRowSSE41, Y vectors.
TEXT ·crossColorRowAVX2(SB), NOSPLIT, $0-40
	MOVQ           mults+0(FP), DX
	MOVQ           src+8(FP), SI
	MOVQ           dst+16(FP), DI
	MOVQ           width+24(FP), BX
	MOVQ           bits+32(FP), CX
	MOVL           $1, R8
	SHLQ           CX, R8
	VBROADCASTI128 ccMaskAG<>(SB), Y0
	VBROADCASTI128 ccGreen<>(SB), Y6
	VBROADCASTI128 ccRed<>(SB), Y7

cctile:
	TESTQ        BX, BX
	JZ           ccdone
	CCMULTS
	VMOVD        R9, X5
	VPBROADCASTD X5, Y5
	VMOVD        AX, X4
	VPBROADCASTD X4, Y4
	MOVQ         R8, R11
	CMPQ         BX, R11
	CMOVQLT      BX, R11
	SUBQ         R11, BX

cc8:
	CMPQ    R11, $8
	JB      cc4
	VMOVDQU (SI), Y1
	CCOLORV(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
	VMOVDQU Y2, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, R11
	JMP     cc8

cc4:
	CMPQ    R11, $4
	JB      cc1
	VMOVDQU (SI), X1
	CCOLORV(X0, X1, X2, X3, X4, X5, X6, X7)
	VMOVDQU X2, (DI)
	ADDQ    $16, SI
	ADDQ    $16, DI
	SUBQ    $4, R11

cc1:
	TESTQ R11, R11
	JZ    cctile
	VMOVD (SI), X1
	CCOLORV(X0, X1, X2, X3, X4, X5, X6, X7)
	VMOVD X2, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  R11
	JMP   cc1

ccdone:
	VZEROUPPER
	RET
