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

// AVX-512 VBMI kernels of the color-indexing transform with 256-entry
// palettes: lookups of index bytes into RGBA words (lookupRGBA) and alpha
// bytes (lookupBytes), and an AVX2 gather of RGBA words. libwebp has no
// x86 SIMD version of these (MapARGB_C, MapAlpha_C); the Go references
// lookupRGBAGo (idx8.go) and lookupBytesGo (transform.go) are the
// specification. The VBMI functions process exact counts, the tails with
// masked loads and stores (which do not fault on the masked-off bytes);
// they need AVX512-F/BW (x86-64-v4) and AVX512-VBMI (VPERMB, VPERMI2B).
// All end with VZEROUPPER.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/lossless.c
// (MapARGB_C, MapAlpha_C).

#include "textflag.h"

// lrPlanar: VPERMB indices gathering byte c of the 16 words of a vector
// into its 128-bit lane c (lane c, byte q <- byte 4q+c).
DATA lrPlanar<>+0(SB)/8, $0x1c1814100c080400
DATA lrPlanar<>+8(SB)/8, $0x3c3834302c282420
DATA lrPlanar<>+16(SB)/8, $0x1d1915110d090501
DATA lrPlanar<>+24(SB)/8, $0x3d3935312d292521
DATA lrPlanar<>+32(SB)/8, $0x1e1a16120e0a0602
DATA lrPlanar<>+40(SB)/8, $0x3e3a36322e2a2622
DATA lrPlanar<>+48(SB)/8, $0x1f1b17130f0b0703
DATA lrPlanar<>+56(SB)/8, $0x3f3b37332f2b2723
GLOBL lrPlanar<>(SB), RODATA|NOPTR, $64

// lrOrder: VPERMB indices of the 64 index bytes of a block such that the
// in-lane byte and word interleaves of the four channel planes give the
// RGBA words in pixel order: position 16j+4k+i holds pixel 16k+4j+i
// (a transpose of 4-byte groups).
DATA lrOrder<>+0(SB)/8, $0x1312111003020100
DATA lrOrder<>+8(SB)/8, $0x3332313023222120
DATA lrOrder<>+16(SB)/8, $0x1716151407060504
DATA lrOrder<>+24(SB)/8, $0x3736353427262524
DATA lrOrder<>+32(SB)/8, $0x1b1a19180b0a0908
DATA lrOrder<>+40(SB)/8, $0x3b3a39382b2a2928
DATA lrOrder<>+48(SB)/8, $0x1f1e1d1c0f0e0d0c
DATA lrOrder<>+56(SB)/8, $0x3f3e3d3c2f2e2d2c
GLOBL lrOrder<>(SB), RODATA|NOPTR, $64

// LRPLANES transposes the palette words off(AX)..off+255(AX) (64 colors)
// into the channel planes r, g, b, a (64 bytes each). Z14 lrPlanar;
// clobbers Z0-Z7.
#define LRPLANES(off, r, g, b, a) \
	VMOVDQU32  off(AX), Z0        \
	VMOVDQU32  off+64(AX), Z1     \
	VMOVDQU32  off+128(AX), Z2    \
	VMOVDQU32  off+192(AX), Z3    \
	VPERMB     Z0, Z14, Z0        \
	VPERMB     Z1, Z14, Z1        \
	VPERMB     Z2, Z14, Z2        \
	VPERMB     Z3, Z14, Z3        \
	VSHUFI64X2 $0x44, Z1, Z0, Z4  \
	VSHUFI64X2 $0xee, Z1, Z0, Z5  \
	VSHUFI64X2 $0x44, Z3, Z2, Z6  \
	VSHUFI64X2 $0xee, Z3, Z2, Z7  \
	VSHUFI64X2 $0x88, Z6, Z4, r   \
	VSHUFI64X2 $0xdd, Z6, Z4, g   \
	VSHUFI64X2 $0x88, Z7, Z5, b   \
	VSHUFI64X2 $0xdd, Z7, Z5, a

// LRPLANE sets dst to the bytes plane[idx] of the 64 indices in Z1, the
// plane in t0-t3 (256 bytes): VPERMI2B looks up the 128 entries of t0:t1
// and of t2:t3 (index bit 7 ignored), and K1 (bit 7) picks one. Clobbers
// x, y.
#define LRPLANE(t0, t1, t2, t3, x, y, dst) \
	VMOVDQA64 Z1, x            \
	VMOVDQA64 Z1, y            \
	VPERMI2B  t1, t0, x        \
	VPERMI2B  t3, t2, y        \
	VPBLENDMB y, x, K1, dst

// LRBLOCK looks up the 64 indices in Z0 into the RGBA words Z0-Z3 (16
// pixels each, in order). Z15 lrOrder, Z16-Z31 the planes.
#define LRBLOCK \
	VPERMB     Z0, Z15, Z1                             \
	VPMOVB2M   Z1, K1                                  \
	LRPLANE(Z16, Z17, Z18, Z19, Z2, Z3, Z4)            \
	LRPLANE(Z20, Z21, Z22, Z23, Z6, Z7, Z5)            \
	LRPLANE(Z24, Z25, Z26, Z27, Z2, Z3, Z8)            \
	LRPLANE(Z28, Z29, Z30, Z31, Z6, Z7, Z9)            \
	VPUNPCKLBW Z5, Z4, Z10                             \
	VPUNPCKHBW Z5, Z4, Z11                             \
	VPUNPCKLBW Z9, Z8, Z12                             \
	VPUNPCKHBW Z9, Z8, Z13                             \
	VPUNPCKLWD Z12, Z10, Z0                            \
	VPUNPCKHWD Z12, Z10, Z1                            \
	VPUNPCKLWD Z13, Z11, Z2                            \
	VPUNPCKHWD Z13, Z11, Z3

// func lookupRGBAVBMI(pal *[256]uint32, src, dst *byte, n int)
//
// dst[4i:4i+4] = pal[src[i]] for i < n, 64 pixels per block, the last
// n%64 with masked loads and stores. The palette is first transposed into
// four channel planes of 256 bytes (Z16-Z31); per block each plane is
// looked up with two VPERMI2B and a blend, and the planes are interleaved
// back into words. (VPGATHERDD measured only 1.2x faster than the Go
// loop on Zen 4, and is much slower on Intel CPUs with the GDS microcode
// fix.)
//
// AX pal, SI src, DI dst, CX pixels left, DX tail mask; Z14 lrPlanar,
// Z15 lrOrder.
TEXT ·lookupRGBAVBMI(SB), NOSPLIT, $0-32
	MOVQ      pal+0(FP), AX
	MOVQ      src+8(FP), SI
	MOVQ      dst+16(FP), DI
	MOVQ      n+24(FP), CX
	VMOVDQU64 lrPlanar<>(SB), Z14
	VMOVDQU64 lrOrder<>(SB), Z15
	LRPLANES(0, Z16, Z20, Z24, Z28)
	LRPLANES(256, Z17, Z21, Z25, Z29)
	LRPLANES(512, Z18, Z22, Z26, Z30)
	LRPLANES(768, Z19, Z23, Z27, Z31)

lr64:
	CMPQ      CX, $64
	JB        lrtail
	VMOVDQU8  (SI), Z0
	LRBLOCK
	VMOVDQU32 Z0, (DI)
	VMOVDQU32 Z1, 64(DI)
	VMOVDQU32 Z2, 128(DI)
	VMOVDQU32 Z3, 192(DI)
	ADDQ      $64, SI
	ADDQ      $256, DI
	SUBQ      $64, CX
	JMP       lr64

lrtail:
	TESTQ      CX, CX
	JZ         lrdone
	MOVQ       $1, DX
	SHLQ       CX, DX
	DECQ       DX
	KMOVQ      DX, K2
	VMOVDQU8.Z (SI), K2, Z0
	LRBLOCK
	KMOVW      DX, K2
	VMOVDQU32  Z0, K2, (DI)
	SHRQ       $16, DX
	KMOVW      DX, K2
	VMOVDQU32  Z1, K2, 64(DI)
	SHRQ       $16, DX
	KMOVW      DX, K2
	VMOVDQU32  Z2, K2, 128(DI)
	SHRQ       $16, DX
	KMOVW      DX, K2
	VMOVDQU32  Z3, K2, 192(DI)

lrdone:
	VZEROUPPER
	RET

// func lookupBytesVBMI(pal *[256]byte, in, out *byte, n int)
//
// out[i] = pal[in[i]] for i < n (the 8-bit alpha palette, libwebp
// MapAlpha), 64 bytes per block, the last n%64 with masked loads and
// stores: the table lives in Z16-Z19, VPERMI2B looks up the indices in
// its halves 0-127 and 128-255, and bit 7 of each index (K1) picks one.
//
// AX pal, SI in, DI out, CX bytes left, DX tail mask.
TEXT ·lookupBytesVBMI(SB), NOSPLIT, $0-32
	MOVQ      pal+0(FP), AX
	MOVQ      in+8(FP), SI
	MOVQ      out+16(FP), DI
	MOVQ      n+24(FP), CX
	VMOVDQU64 (AX), Z16
	VMOVDQU64 64(AX), Z17
	VMOVDQU64 128(AX), Z18
	VMOVDQU64 192(AX), Z19
	CMPQ      CX, $128
	JB        lb64

lb128:
	VMOVDQU8  (SI), Z0
	VMOVDQU8  64(SI), Z1
	VPMOVB2M  Z0, K1
	VPMOVB2M  Z1, K2
	VMOVDQA64 Z0, Z2
	VMOVDQA64 Z1, Z3
	VPERMI2B  Z17, Z16, Z0
	VPERMI2B  Z17, Z16, Z1
	VPERMI2B  Z19, Z18, Z2
	VPERMI2B  Z19, Z18, Z3
	VPBLENDMB Z2, Z0, K1, Z0
	VPBLENDMB Z3, Z1, K2, Z1
	VMOVDQU8  Z0, (DI)
	VMOVDQU8  Z1, 64(DI)
	ADDQ      $128, SI
	ADDQ      $128, DI
	SUBQ      $128, CX
	CMPQ      CX, $128
	JAE       lb128

lb64:
	CMPQ      CX, $64
	JB        lbtail
	VMOVDQU8  (SI), Z0
	VPMOVB2M  Z0, K1
	VMOVDQA64 Z0, Z2
	VPERMI2B  Z17, Z16, Z0
	VPERMI2B  Z19, Z18, Z2
	VPBLENDMB Z2, Z0, K1, Z0
	VMOVDQU8  Z0, (DI)
	ADDQ      $64, SI
	ADDQ      $64, DI
	SUBQ      $64, CX

lbtail:
	TESTQ      CX, CX
	JZ         lbdone
	MOVQ       $1, DX
	SHLQ       CX, DX
	DECQ       DX
	KMOVQ      DX, K2
	VMOVDQU8.Z (SI), K2, Z0
	VPMOVB2M   Z0, K1
	VMOVDQA64  Z0, Z2
	VPERMI2B   Z17, Z16, Z0
	VPERMI2B   Z19, Z18, Z2
	VPBLENDMB  Z2, Z0, K1, Z0
	VMOVDQU8   Z0, K2, (DI)

lbdone:
	VZEROUPPER
	RET

// func lookupRGBAAVX2(pal *[256]uint32, src, dst *byte, n int)
//
// Writes the RGBA words pal[src[i]] to dst for i < n, n a multiple of 8
// (the wrapper does the rest in Go): VPMOVZXBD widens 8 index bytes to
// dwords and VPGATHERDD loads their palette words (every index is below
// 256, inside the palette). The gather clears its mask register, which is
// set again from Y15 (all ones) before each one.
//
// AX pal, SI src, DI dst, CX pixels left; Y0, Y1 indices, Y2, Y3 masks,
// Y4, Y5 RGBA words.
TEXT ·lookupRGBAAVX2(SB), NOSPLIT, $0-32
	MOVQ     pal+0(FP), AX
	MOVQ     src+8(FP), SI
	MOVQ     dst+16(FP), DI
	MOVQ     n+24(FP), CX
	VPCMPEQD Y15, Y15, Y15

lg16:
	CMPQ       CX, $16
	JB         lg8
	VPMOVZXBD  (SI), Y0
	VPMOVZXBD  8(SI), Y1
	VMOVDQA    Y15, Y2
	VMOVDQA    Y15, Y3
	VPGATHERDD Y2, (AX)(Y0*4), Y4
	VPGATHERDD Y3, (AX)(Y1*4), Y5
	VMOVDQU    Y4, (DI)
	VMOVDQU    Y5, 32(DI)
	ADDQ       $16, SI
	ADDQ       $64, DI
	SUBQ       $16, CX
	JMP        lg16

lg8:
	CMPQ       CX, $8
	JB         lgdone
	VPMOVZXBD  (SI), Y0
	VMOVDQA    Y15, Y2
	VPGATHERDD Y2, (AX)(Y0*4), Y4
	VMOVDQU    Y4, (DI)

lgdone:
	VZEROUPPER
	RET
