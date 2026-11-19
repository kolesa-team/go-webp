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

//go:build amd64 && !purego

// SSE4.1 and AVX2 kernels of the color-indexing transform with bundled
// indices (at most 16 colors): each packed byte holds 2, 4 or 8 indices of
// 4, 2 or 1 bits (bits 1, 2, 3), the first in the low bits. libwebp has
// no x86 SIMD version (VP8LColorIndexInverseTransformAlpha,
// VP8LColorIndexInverseTransform); the Go references expandBundledRowGo
// (transform.go) and expandBundledRGBAGo (idx8.go) are the specification.
// The functions process whole blocks of packed bytes, the wrappers in
// dsp_amd64.go the rest of each row in Go; the AVX2 ones use VEX
// encodings only and end with VZEROUPPER.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/lossless.c
// (VP8LColorIndexInverseTransform, VP8LColorIndexInverseTransformAlpha).

#include "textflag.h"

// e8Nib: 0x0f in each byte; e8Two: 0x03 in each byte.
DATA e8Nib<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA e8Nib<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL e8Nib<>(SB), RODATA|NOPTR, $16
DATA e8Two<>+0(SB)/8, $0x0303030303030303
DATA e8Two<>+8(SB)/8, $0x0303030303030303
GLOBL e8Two<>(SB), RODATA|NOPTR, $16

// e8Rep4: PSHUFB indices repeating bytes 0 and 1 eight times each, and in
// the high lane bytes 2 and 3.
DATA e8Rep4<>+0(SB)/8, $0
DATA e8Rep4<>+8(SB)/8, $0x0101010101010101
DATA e8Rep4<>+16(SB)/8, $0x0202020202020202
DATA e8Rep4<>+24(SB)/8, $0x0303030303030303
GLOBL e8Rep4<>(SB), RODATA|NOPTR, $32

// e8Bit: bit k in bytes k and k+8.
DATA e8Bit<>+0(SB)/8, $0x8040201008040201
DATA e8Bit<>+8(SB)/8, $0x8040201008040201
GLOBL e8Bit<>(SB), RODATA|NOPTR, $16

// e8Step2, e8Step4: 2 and 4 in each byte.
DATA e8Step2<>+0(SB)/8, $0x0202020202020202
DATA e8Step2<>+8(SB)/8, $0x0202020202020202
GLOBL e8Step2<>(SB), RODATA|NOPTR, $16
DATA e8Step4<>+0(SB)/8, $0x0404040404040404
DATA e8Step4<>+8(SB)/8, $0x0404040404040404
GLOBL e8Step4<>(SB), RODATA|NOPTR, $16

// e8Groups: VPERMD indices 0, 2, 4, 6, 1, 3, 5, 7 (groups of 4 packed
// bytes, so that the in-lane interleaves give the indices in order).
DATA e8Groups<>+0(SB)/8, $0x0000000200000000
DATA e8Groups<>+8(SB)/8, $0x0000000600000004
DATA e8Groups<>+16(SB)/8, $0x0000000300000001
DATA e8Groups<>+24(SB)/8, $0x0000000700000005
GLOBL e8Groups<>(SB), RODATA|NOPTR, $32

// UNPACK4 sets X2, X1 to the 32 4-bit indices of the 16 packed bytes in
// X0 (X6 e8Nib); clobbers X0.
#define UNPACK4 \
	MOVO      X0, X1 \
	PAND      X6, X1 \
	PSRLW     $4, X0 \
	PAND      X6, X0 \
	MOVO      X1, X2 \
	PUNPCKLBW X0, X2 \
	PUNPCKHBW X0, X1

// UNPACK2 sets X0, X4, X2, X1 to the 64 2-bit indices of the 16 packed
// bytes in X0 (X6 e8Two): the four fields of each byte, interleaved by
// bytes and then by 16-bit pairs. Clobbers X3, X5.
#define UNPACK2 \
	MOVO      X0, X1 \
	PAND      X6, X1 \
	MOVO      X0, X2 \
	PSRLW     $2, X2 \
	PAND      X6, X2 \
	MOVO      X0, X3 \
	PSRLW     $4, X3 \
	PAND      X6, X3 \
	PSRLW     $6, X0 \
	PAND      X6, X0 \
	MOVO      X1, X4 \
	PUNPCKLBW X2, X4 \
	PUNPCKHBW X2, X1 \
	MOVO      X3, X5 \
	PUNPCKLBW X0, X5 \
	PUNPCKHBW X0, X3 \
	MOVO      X4, X0 \
	PUNPCKLWL X5, X0 \
	PUNPCKHWL X5, X4 \
	MOVO      X1, X2 \
	PUNPCKLWL X3, X2 \
	PUNPCKHWL X3, X1

// LOOK8(x, off) stores at off(DI) the palette bytes (X7) of the 16
// indices in x; clobbers X8.
#define LOOK8(x, off) \
	MOVO   X7, X8 \
	PSHUFB x, X8  \
	MOVOU  X8, off(DI)

// BIT8(off) stores at off(DI) the colors of the next 16 1-bit indices of
// the packed bytes in X0 (two bytes, PSHUFB indices X14, advanced by 2):
// c0 ^ (mask & (c0 ^ c1)), the mask from the bits (X9 e8Bit). X12 c0,
// X13 c0 ^ c1, X10 e8Step2; clobbers X1.
#define BIT8(off) \
	MOVO    X0, X1      \
	PSHUFB  X14, X1     \
	PAND    X9, X1      \
	PCMPEQB X9, X1      \
	PAND    X13, X1     \
	PXOR    X12, X1     \
	MOVOU   X1, off(DI) \
	PADDB   X10, X14

// func expandBundled8SSE41(pal *[256]byte, src, dst *byte, n, bits int)
//
// dst[k<<bits+j] = pal[index j of src[k]] for k < n (a multiple of 16),
// bits 1 to 3 (the 8-bit alpha path, libwebp MapAlpha after
// VP8LColorIndexInverseTransformAlpha's unpacking): one PSHUFB per 16
// output bytes from the palette in X7, or for 1-bit indices a select
// between the two colors.
//
// AX pal, SI src, DI dst, CX blocks of 16 packed bytes left, DX bits.
TEXT ·expandBundled8SSE41(SB), NOSPLIT, $0-40
	MOVQ  pal+0(FP), AX
	MOVQ  src+8(FP), SI
	MOVQ  dst+16(FP), DI
	MOVQ  n+24(FP), CX
	MOVQ  bits+32(FP), DX
	MOVOU (AX), X7
	SHRQ  $4, CX
	JZ    e8done
	CMPQ  DX, $2
	JEQ   e8bits2
	JA    e8bits3
	MOVOU e8Nib<>(SB), X6

e8bits1:
	MOVOU (SI), X0
	UNPACK4
	LOOK8(X2, 0)
	LOOK8(X1, 16)
	ADDQ  $16, SI
	ADDQ  $32, DI
	DECQ  CX
	JNZ   e8bits1
	RET

e8bits2:
	MOVOU e8Two<>(SB), X6

e8bits2l:
	MOVOU (SI), X0
	UNPACK2
	LOOK8(X0, 0)
	LOOK8(X4, 16)
	LOOK8(X2, 32)
	LOOK8(X1, 48)
	ADDQ  $16, SI
	ADDQ  $64, DI
	DECQ  CX
	JNZ   e8bits2l
	RET

e8bits3:
	MOVOU  e8Bit<>(SB), X9
	MOVOU  e8Step2<>(SB), X10
	PXOR   X11, X11
	MOVO   X7, X12
	PSHUFB X11, X12          // c0 in every byte
	MOVO   X7, X13
	PSRLDQ $1, X13
	PSHUFB X11, X13          // c1
	PXOR   X12, X13          // c0 ^ c1

e8bits3l:
	MOVOU (SI), X0
	MOVOU e8Rep4<>(SB), X14
	BIT8(0)
	BIT8(16)
	BIT8(32)
	BIT8(48)
	BIT8(64)
	BIT8(80)
	BIT8(96)
	BIT8(112)
	ADDQ  $16, SI
	ADDQ  $128, DI
	DECQ  CX
	JNZ   e8bits3l

e8done:
	RET

// func expandBundled8AVX2(pal *[256]byte, src, dst *byte, n, bits int)
//
// expandBundled8SSE41 with 32-byte vectors: blocks of 32 packed bytes,
// then one of 16 (n a multiple of 16). For bits 1 and 2 the packed bytes
// are first permuted across the lanes (VPERMQ, VPERMD) so that the
// in-lane interleaves give the indices in order.
//
// AX pal, SI src, DI dst, CX packed bytes left, DX bits; Y7 the palette
// in both lanes.
TEXT ·expandBundled8AVX2(SB), NOSPLIT, $0-40
	MOVQ           pal+0(FP), AX
	MOVQ           src+8(FP), SI
	MOVQ           dst+16(FP), DI
	MOVQ           n+24(FP), CX
	MOVQ           bits+32(FP), DX
	VBROADCASTI128 (AX), Y7
	CMPQ           DX, $2
	JEQ            v8bits2
	JA             v8bits3
	VBROADCASTI128 e8Nib<>(SB), Y6

v8bits1:
	CMPQ       CX, $32
	JB         v8bits1x
	VPERMQ     $0xd8, (SI), Y0
	VPAND      Y6, Y0, Y1
	VPSRLW     $4, Y0, Y0
	VPAND      Y6, Y0, Y0
	VPUNPCKLBW Y0, Y1, Y2
	VPUNPCKHBW Y0, Y1, Y3
	VPSHUFB    Y2, Y7, Y2
	VPSHUFB    Y3, Y7, Y3
	VMOVDQU    Y2, (DI)
	VMOVDQU    Y3, 32(DI)
	ADDQ       $32, SI
	ADDQ       $64, DI
	SUBQ       $32, CX
	JMP        v8bits1

v8bits1x:
	TESTQ      CX, CX
	JZ         v8done
	VMOVDQU    (SI), X0
	VPAND      X6, X0, X1
	VPSRLW     $4, X0, X0
	VPAND      X6, X0, X0
	VPUNPCKLBW X0, X1, X2
	VPUNPCKHBW X0, X1, X3
	VPSHUFB    X2, X7, X2
	VPSHUFB    X3, X7, X3
	VMOVDQU    X2, (DI)
	VMOVDQU    X3, 16(DI)
	JMP        v8done

v8bits2:
	VBROADCASTI128 e8Two<>(SB), Y6
	VMOVDQU        e8Groups<>(SB), Y8

v8bits2l:
	CMPQ       CX, $32
	JB         v8bits2x
	VPERMD     (SI), Y8, Y0
	VPAND      Y6, Y0, Y1
	VPSRLW     $2, Y0, Y2
	VPAND      Y6, Y2, Y2
	VPSRLW     $4, Y0, Y3
	VPAND      Y6, Y3, Y3
	VPSRLW     $6, Y0, Y0
	VPAND      Y6, Y0, Y0
	VPUNPCKLBW Y2, Y1, Y4
	VPUNPCKHBW Y2, Y1, Y1
	VPUNPCKLBW Y0, Y3, Y5
	VPUNPCKHBW Y0, Y3, Y3
	VPUNPCKLWD Y5, Y4, Y0
	VPUNPCKHWD Y5, Y4, Y4
	VPUNPCKLWD Y3, Y1, Y2
	VPUNPCKHWD Y3, Y1, Y1
	VPSHUFB    Y0, Y7, Y0
	VPSHUFB    Y4, Y7, Y4
	VPSHUFB    Y2, Y7, Y2
	VPSHUFB    Y1, Y7, Y1
	VMOVDQU    Y0, (DI)
	VMOVDQU    Y4, 32(DI)
	VMOVDQU    Y2, 64(DI)
	VMOVDQU    Y1, 96(DI)
	ADDQ       $32, SI
	ADDQ       $128, DI
	SUBQ       $32, CX
	JMP        v8bits2l

v8bits2x:
	TESTQ      CX, CX
	JZ         v8done
	VMOVDQU    (SI), X0
	VPAND      X6, X0, X1
	VPSRLW     $2, X0, X2
	VPAND      X6, X2, X2
	VPSRLW     $4, X0, X3
	VPAND      X6, X3, X3
	VPSRLW     $6, X0, X0
	VPAND      X6, X0, X0
	VPUNPCKLBW X2, X1, X4
	VPUNPCKHBW X2, X1, X1
	VPUNPCKLBW X0, X3, X5
	VPUNPCKHBW X0, X3, X3
	VPUNPCKLWD X5, X4, X0
	VPUNPCKHWD X5, X4, X4
	VPUNPCKLWD X3, X1, X2
	VPUNPCKHWD X3, X1, X1
	VPSHUFB    X0, X7, X0
	VPSHUFB    X4, X7, X4
	VPSHUFB    X2, X7, X2
	VPSHUFB    X1, X7, X1
	VMOVDQU    X0, (DI)
	VMOVDQU    X4, 16(DI)
	VMOVDQU    X2, 32(DI)
	VMOVDQU    X1, 48(DI)
	JMP        v8done

	// 1-bit indices, 16 packed bytes (in both lanes of Y0) per block:
	// each of 4 steps repeats 4 bytes 8 times (Y14, advanced by 4),
	// isolates bit k of byte k%8 and selects c1 or c0.
v8bits3:
	VPBROADCASTB   (AX), Y12
	VPBROADCASTB   1(AX), Y13
	VBROADCASTI128 e8Bit<>(SB), Y9
	VBROADCASTI128 e8Step4<>(SB), Y10
	SHRQ           $4, CX
	JZ             v8done

v8bits3l:
	VBROADCASTI128 (SI), Y0
	VMOVDQU        e8Rep4<>(SB), Y14
	VPSHUFB        Y14, Y0, Y1
	VPADDB         Y10, Y14, Y14
	VPSHUFB        Y14, Y0, Y2
	VPADDB         Y10, Y14, Y14
	VPSHUFB        Y14, Y0, Y3
	VPADDB         Y10, Y14, Y14
	VPSHUFB        Y14, Y0, Y4
	VPAND          Y9, Y1, Y1
	VPAND          Y9, Y2, Y2
	VPAND          Y9, Y3, Y3
	VPAND          Y9, Y4, Y4
	VPCMPEQB       Y9, Y1, Y1
	VPCMPEQB       Y9, Y2, Y2
	VPCMPEQB       Y9, Y3, Y3
	VPCMPEQB       Y9, Y4, Y4
	VPBLENDVB      Y1, Y13, Y12, Y1
	VPBLENDVB      Y2, Y13, Y12, Y2
	VPBLENDVB      Y3, Y13, Y12, Y3
	VPBLENDVB      Y4, Y13, Y12, Y4
	VMOVDQU        Y1, (DI)
	VMOVDQU        Y2, 32(DI)
	VMOVDQU        Y3, 64(DI)
	VMOVDQU        Y4, 96(DI)
	ADDQ           $16, SI
	ADDQ           $128, DI
	DECQ           CX
	JNZ            v8bits3l

v8done:
	VZEROUPPER
	RET

// erShift1, erShift2, erShift3: VPSRLVD counts moving index j of the
// packed bytes broadcast to every dword to the low bits of dword j.
DATA erShift1<>+0(SB)/8, $0x0000000400000000
DATA erShift1<>+8(SB)/8, $0x0000000c00000008
DATA erShift1<>+16(SB)/8, $0x0000001400000010
DATA erShift1<>+24(SB)/8, $0x0000001c00000018
GLOBL erShift1<>(SB), RODATA|NOPTR, $32
DATA erShift2<>+0(SB)/8, $0x0000000200000000
DATA erShift2<>+8(SB)/8, $0x0000000600000004
DATA erShift2<>+16(SB)/8, $0x0000000a00000008
DATA erShift2<>+24(SB)/8, $0x0000000e0000000c
GLOBL erShift2<>(SB), RODATA|NOPTR, $32
DATA erShift3<>+0(SB)/8, $0x0000000100000000
DATA erShift3<>+8(SB)/8, $0x0000000300000002
DATA erShift3<>+16(SB)/8, $0x0000000500000004
DATA erShift3<>+24(SB)/8, $0x0000000700000006
GLOBL erShift3<>(SB), RODATA|NOPTR, $32

// func expandBundledRGBAAVX2(pal *[256]uint32, src, dst *byte, n, bits int)
//
// dst[4(k<<bits+j):][:4] = the RGBA word pal[index j of src[k]] for k < n,
// n a multiple of the packed bytes of 8 pixels (4, 2, 1 for bits 1, 2,
// 3): per 8 pixels, the packed bytes broadcast to every dword
// (VPBROADCASTD/W/B), shifted per lane (VPSRLVD) and masked give the 8
// indices, VPERMD looks them up in the palette entries 0-7 (and for 4-bit
// indices 8-15, picked by index bit 3 moved to the sign bit). The
// emitPalette of the byte path (libwebp VP8LColorIndexInverseTransform and
// VP8LConvertBGRAToRGBA).
//
// AX pal, SI src, DI dst, CX packed bytes left, DX bits; Y8, Y9 palette
// entries 0-7 and 8-15, Y10 shift counts, Y11 index mask.
TEXT ·expandBundledRGBAAVX2(SB), NOSPLIT, $0-40
	MOVQ    pal+0(FP), AX
	MOVQ    src+8(FP), SI
	MOVQ    dst+16(FP), DI
	MOVQ    n+24(FP), CX
	MOVQ    bits+32(FP), DX
	VMOVDQU (AX), Y8
	TESTQ   CX, CX
	JZ      erdone
	CMPQ    DX, $2
	JEQ     erbits2
	JA      erbits3
	VMOVDQU 32(AX), Y9
	VMOVDQU erShift1<>(SB), Y10
	MOVL    $15, DX
	VMOVD   DX, X11
	VPBROADCASTD X11, Y11

erbits1:
	VPBROADCASTD (SI), Y0
	VPSRLVD      Y10, Y0, Y0
	VPAND        Y11, Y0, Y0
	VPERMD       Y8, Y0, Y1
	VPERMD       Y9, Y0, Y2
	VPSLLD       $28, Y0, Y0
	VBLENDVPS    Y0, Y2, Y1, Y1
	VMOVDQU      Y1, (DI)
	ADDQ         $4, SI
	ADDQ         $32, DI
	SUBQ         $4, CX
	JNZ          erbits1
	JMP          erdone

erbits2:
	VMOVDQU      erShift2<>(SB), Y10
	MOVL         $3, DX
	VMOVD        DX, X11
	VPBROADCASTD X11, Y11

erbits2l:
	VPBROADCASTW (SI), Y0
	VPSRLVD      Y10, Y0, Y0
	VPAND        Y11, Y0, Y0
	VPERMD       Y8, Y0, Y1
	VMOVDQU      Y1, (DI)
	ADDQ         $2, SI
	ADDQ         $32, DI
	SUBQ         $2, CX
	JNZ          erbits2l
	JMP          erdone

erbits3:
	VMOVDQU      erShift3<>(SB), Y10
	MOVL         $1, DX
	VMOVD        DX, X11
	VPBROADCASTD X11, Y11

erbits3l:
	VPBROADCASTB (SI), Y0
	VPSRLVD      Y10, Y0, Y0
	VPAND        Y11, Y0, Y0
	VPERMD       Y8, Y0, Y1
	VMOVDQU      Y1, (DI)
	ADDQ         $1, SI
	ADDQ         $32, DI
	DECQ         CX
	JNZ          erbits3l

erdone:
	VZEROUPPER
	RET
