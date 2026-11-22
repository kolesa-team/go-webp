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

// SSE4.1 and AVX2 kernels of the ALPH unfilters, adapted from libwebp
// (BSD-3-Clause) src/dsp/filters_sse2.c (HorizontalUnfilter_SSE2,
// VerticalUnfilter_SSE2). The Go references in filters.go are the
// specification. The functions process n bytes, n a multiple of 16; the
// AVX2 ones use VEX encodings only and end with VZEROUPPER.

#include "textflag.h"

// ufLast: PSHUFB indices broadcasting byte 15 (of each 128-bit lane).
DATA ufLast<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA ufLast<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL ufLast<>(SB), RODATA|NOPTR, $16

// PREFIX(x, t) replaces the 16 bytes of x by their prefix sums: four
// shifted adds (by 1, 2, 4 and 8 bytes); clobbers t.
#define PREFIX(x, t) \
	MOVO   x, t  \
	PSLLDQ $1, t \
	PADDB  t, x  \
	MOVO   x, t  \
	PSLLDQ $2, t \
	PADDB  t, x  \
	MOVO   x, t  \
	PSLLDQ $4, t \
	PADDB  t, x  \
	MOVO   x, t  \
	PSLLDQ $8, t \
	PADDB  t, x

// PREFIXV is PREFIX in VEX form, per 128-bit lane for Y registers.
#define PREFIXV(x, t) \
	VPSLLDQ $1, x, t \
	VPADDB  t, x, x  \
	VPSLLDQ $2, x, t \
	VPADDB  t, x, x  \
	VPSLLDQ $4, x, t \
	VPADDB  t, x, x  \
	VPSLLDQ $8, x, t \
	VPADDB  t, x, x

// PREFIX32(x, t) is PREFIXV on the 32 bytes of x: the total of the low
// lane (its byte 15, Y7 ufLast) is then added to the high lane.
#define PREFIX32(x, t) \
	PREFIXV(x, t)                \
	VPSHUFB    Y7, x, t          \
	VPERM2I128 $0x08, t, t, t    \
	VPADDB     t, x, x

// func unfilterHorizontalSSE41(in, out *byte, n, pred int)
//
// out[i] = pred + in[0] + ... + in[i] (mod 256) for i < n, n a multiple of
// 16; in may alias out (libwebp HorizontalUnfilter_SSE2, 16 bytes per
// vector). Each block gets its prefix sums off the serial path; the carry
// from the previous block (all bytes of X0) costs an add and a broadcast
// per 32 bytes: the second block of an iteration adds the total of the
// first before the carry.
//
// SI in, DI out, CX bytes left; X0 carry, X7 ufLast, X1/X2 blocks,
// X3/X4 temporaries.
TEXT ·unfilterHorizontalSSE41(SB), NOSPLIT, $0-32
	MOVQ   in+0(FP), SI
	MOVQ   out+8(FP), DI
	MOVQ   n+16(FP), CX
	MOVQ   pred+24(FP), AX
	MOVOU  ufLast<>(SB), X7
	MOVL   AX, X0
	PXOR   X6, X6
	PSHUFB X6, X0
	CMPQ   CX, $32
	JB     h16

h32:
	MOVOU  (SI), X1
	MOVOU  16(SI), X2
	PREFIX(X1, X3)
	PREFIX(X2, X4)
	MOVO   X1, X3
	PSHUFB X7, X3
	PADDB  X3, X2
	PADDB  X0, X1
	PADDB  X0, X2
	MOVO   X2, X0
	PSHUFB X7, X0
	MOVOU  X1, (DI)
	MOVOU  X2, 16(DI)
	ADDQ   $32, SI
	ADDQ   $32, DI
	SUBQ   $32, CX
	CMPQ   CX, $32
	JAE    h32

h16:
	TESTQ CX, CX
	JZ    hdone
	MOVOU (SI), X1
	PREFIX(X1, X3)
	PADDB X0, X1
	MOVOU X1, (DI)

hdone:
	RET

// func unfilterHorizontalAVX2(in, out *byte, n, pred int)
//
// unfilterHorizontalSSE41 with 32-byte vectors, two per iteration, then
// one of 32 and one of 16 bytes. The carry is broadcast from byte 31
// (VPSHUFB in-lane, then VPERMQ of the high qword).
//
// SI in, DI out, CX bytes left; Y0 carry, Y7 ufLast, Y1/Y2 blocks,
// Y3/Y4 temporaries.
TEXT ·unfilterHorizontalAVX2(SB), NOSPLIT, $0-32
	MOVQ           in+0(FP), SI
	MOVQ           out+8(FP), DI
	MOVQ           n+16(FP), CX
	MOVQ           pred+24(FP), AX
	VBROADCASTI128 ufLast<>(SB), Y7
	VMOVD          AX, X0
	VPBROADCASTB   X0, Y0
	CMPQ           CX, $64
	JB             h32

h64:
	VMOVDQU (SI), Y1
	VMOVDQU 32(SI), Y2
	PREFIX32(Y1, Y3)
	PREFIX32(Y2, Y4)
	VPSHUFB Y7, Y1, Y3
	VPERMQ  $0xff, Y3, Y3
	VPADDB  Y3, Y2, Y2
	VPADDB  Y0, Y1, Y1
	VPADDB  Y0, Y2, Y2
	VPSHUFB Y7, Y2, Y0
	VPERMQ  $0xff, Y0, Y0
	VMOVDQU Y1, (DI)
	VMOVDQU Y2, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	SUBQ    $64, CX
	CMPQ    CX, $64
	JAE     h64

h32:
	CMPQ    CX, $32
	JB      h16
	VMOVDQU (SI), Y1
	PREFIX32(Y1, Y3)
	VPADDB  Y0, Y1, Y1
	VPSHUFB Y7, Y1, Y0
	VPERMQ  $0xff, Y0, Y0
	VMOVDQU Y1, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $32, CX

h16:
	TESTQ   CX, CX
	JZ      hdone
	VMOVDQU (SI), X1
	PREFIXV(X1, X3)
	VPADDB  X0, X1, X1
	VMOVDQU X1, (DI)

hdone:
	VZEROUPPER
	RET

// func unfilterVerticalSSE41(prev, in, out *byte, n int)
//
// out[i] = prev[i] + in[i] (mod 256) for i < n, n a multiple of 16; in
// may alias out (libwebp VerticalUnfilter_SSE2).
//
// AX prev, SI in, DI out, CX bytes left; X0-X3 prev, X4-X7 in.
TEXT ·unfilterVerticalSSE41(SB), NOSPLIT, $0-32
	MOVQ prev+0(FP), AX
	MOVQ in+8(FP), SI
	MOVQ out+16(FP), DI
	MOVQ n+24(FP), CX
	CMPQ CX, $64
	JB   v16

v64:
	MOVOU (AX), X0
	MOVOU 16(AX), X1
	MOVOU 32(AX), X2
	MOVOU 48(AX), X3
	MOVOU (SI), X4
	MOVOU 16(SI), X5
	MOVOU 32(SI), X6
	MOVOU 48(SI), X7
	PADDB X4, X0
	PADDB X5, X1
	PADDB X6, X2
	PADDB X7, X3
	MOVOU X0, (DI)
	MOVOU X1, 16(DI)
	MOVOU X2, 32(DI)
	MOVOU X3, 48(DI)
	ADDQ  $64, AX
	ADDQ  $64, SI
	ADDQ  $64, DI
	SUBQ  $64, CX
	CMPQ  CX, $64
	JAE   v64

v16:
	TESTQ CX, CX
	JZ    vdone
	MOVOU (AX), X0
	MOVOU (SI), X4
	PADDB X4, X0
	MOVOU X0, (DI)
	ADDQ  $16, AX
	ADDQ  $16, SI
	ADDQ  $16, DI
	SUBQ  $16, CX
	JMP   v16

vdone:
	RET

// func unfilterVerticalAVX2(prev, in, out *byte, n int)
//
// unfilterVerticalSSE41 with 32-byte vectors, then one of 16 bytes.
//
// AX prev, SI in, DI out, CX bytes left; Y0-Y3 prev, Y4-Y7 in.
TEXT ·unfilterVerticalAVX2(SB), NOSPLIT, $0-32
	MOVQ prev+0(FP), AX
	MOVQ in+8(FP), SI
	MOVQ out+16(FP), DI
	MOVQ n+24(FP), CX
	CMPQ CX, $128
	JB   v32

v128:
	VMOVDQU (AX), Y0
	VMOVDQU 32(AX), Y1
	VMOVDQU 64(AX), Y2
	VMOVDQU 96(AX), Y3
	VPADDB  (SI), Y0, Y0
	VPADDB  32(SI), Y1, Y1
	VPADDB  64(SI), Y2, Y2
	VPADDB  96(SI), Y3, Y3
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ    $128, AX
	ADDQ    $128, SI
	ADDQ    $128, DI
	SUBQ    $128, CX
	CMPQ    CX, $128
	JAE     v128

v32:
	CMPQ    CX, $32
	JB      v16
	VMOVDQU (AX), Y0
	VPADDB  (SI), Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ    $32, AX
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $32, CX
	JMP     v32

v16:
	TESTQ   CX, CX
	JZ      vdone
	VMOVDQU (AX), X0
	VPADDB  (SI), X0, X0
	VMOVDQU X0, (DI)

vdone:
	VZEROUPPER
	RET
