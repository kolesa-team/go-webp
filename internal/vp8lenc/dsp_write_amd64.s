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

#include "go_asm.h"
#include "textflag.h"

// Kernels of the input conversion, the histograms of a parse and the
// entropy-coded pixel writer (see dsp_write_amd64.go for the wrappers and
// their bound checks). Each one is bit-exact with the Go reference it
// replaces. libwebp has no SIMD version of the last two (its C code does
// the same as the Go one); the conversion is the swizzle of libwebp's
// VP8LConvertBGRAToRGBA_SSE2 run the other way.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/lossless_sse2.c
// (ConvertBGRAToRGBA_SSE2), src/enc/histogram_enc.c and src/enc/vp8l_enc.c.

// ---------------------------------------------------------------------------
// func nrgbaRowSSE41(dst *uint32, src *byte, n int, exact bool) (flags int)
//
// nrgbaRowGo for n >= 4 pixels, 4 per step: PSHUFB swaps the R and B bytes
// of each pixel (NRGBA bytes R, G, B, A give the little-endian uint32
// A<<24 | R<<16 | G<<8 | B), a pixel whose alpha is 0 is zeroed unless
// exact (the pixel shifted right by 24 compared with 0, PANDN), and PMINUB
// keeps the byte minima, whose alpha bytes give the flags at the end. Both
// rows have 4 bytes per pixel, so AX is the offset in both; a partial last
// block is converted again as the block ending at n (BX). Returns 1 if a
// pixel is not opaque, plus 2 if one is fully transparent.
// X7 = shuffle, X6 = minima, X5 = 0, X4 = all ones unless exact.
TEXT ·nrgbaRowSSE41(SB), NOSPLIT, $0-40
	MOVQ    dst+0(FP), DI
	MOVQ    src+8(FP), SI
	MOVQ    n+16(FP), CX
	SHLQ    $2, CX
	LEAQ    -16(CX), BX
	MOVQ    $0x0704050603000102, AX
	MOVQ    AX, X7
	MOVQ    $0x0f0c0d0e0b08090a, AX
	PINSRQ  $1, AX, X7
	PCMPEQB X6, X6
	PXOR    X5, X5
	PXOR    X4, X4
	MOVBLZX exact+24(FP), AX
	TESTL   AX, AX
	JNZ     nrs_start
	PCMPEQB X4, X4

nrs_start:
	XORQ AX, AX

nrs_loop:
	MOVOU   (SI)(AX*1), X0
	PMINUB  X0, X6
	PSHUFB  X7, X0
	MOVO    X0, X1
	PSRLL   $24, X1
	PCMPEQL X5, X1
	PAND    X4, X1
	PANDN   X0, X1
	MOVOU   X1, (DI)(AX*1)
	ADDQ    $16, AX
	CMPQ    AX, BX
	JLE     nrs_loop
	CMPQ    AX, CX
	JEQ     nrs_flags
	MOVQ    BX, AX
	JMP     nrs_loop

nrs_flags:
	PSRLL    $24, X6 // the lowest alpha of each lane
	PCMPEQB  X3, X3
	PSRLL    $24, X3
	PCMPEQL  X6, X3  // lane opaque
	PCMPEQL  X5, X6  // lane has a transparent pixel
	MOVMSKPS X3, AX
	MOVMSKPS X6, DX
	XORL     $15, AX
	NEGL     AX
	SBBL     AX, AX
	ANDL     $1, AX
	NEGL     DX
	SBBL     DX, DX
	ANDL     $2, DX
	ORL      DX, AX
	MOVQ     AX, flags+32(FP)
	RET

// func nrgbaRowAVX2(dst *uint32, src *byte, n int, exact bool) (flags int)
//
// nrgbaRowSSE41 with 8 pixels per step (n >= 8), the same registers in
// their ymm forms.
TEXT ·nrgbaRowAVX2(SB), NOSPLIT, $0-40
	MOVQ        dst+0(FP), DI
	MOVQ        src+8(FP), SI
	MOVQ        n+16(FP), CX
	SHLQ        $2, CX
	LEAQ        -32(CX), BX
	MOVQ        $0x0704050603000102, AX
	VMOVQ       AX, X7
	MOVQ        $0x0f0c0d0e0b08090a, AX
	VPINSRQ     $1, AX, X7, X7
	VINSERTI128 $1, X7, Y7, Y7
	VPCMPEQB    Y6, Y6, Y6
	VPXOR       Y5, Y5, Y5
	VPXOR       Y4, Y4, Y4
	MOVBLZX     exact+24(FP), AX
	TESTL       AX, AX
	JNZ         nrv_start
	VPCMPEQB    Y4, Y4, Y4

nrv_start:
	XORQ AX, AX

nrv_loop:
	VMOVDQU  (SI)(AX*1), Y0
	VPMINUB  Y0, Y6, Y6
	VPSHUFB  Y7, Y0, Y0
	VPSRLD   $24, Y0, Y1
	VPCMPEQD Y5, Y1, Y1
	VPAND    Y4, Y1, Y1
	VPANDN   Y0, Y1, Y1
	VMOVDQU  Y1, (DI)(AX*1)
	ADDQ     $32, AX
	CMPQ     AX, BX
	JLE      nrv_loop
	CMPQ     AX, CX
	JEQ      nrv_flags
	MOVQ     BX, AX
	JMP      nrv_loop

nrv_flags:
	VPSRLD    $24, Y6, Y6
	VPCMPEQB  Y3, Y3, Y3
	VPSRLD    $24, Y3, Y3
	VPCMPEQD  Y6, Y3, Y3
	VPCMPEQD  Y5, Y6, Y6
	VMOVMSKPS Y3, AX
	VMOVMSKPS Y6, DX
	VZEROUPPER
	XORL      $255, AX
	NEGL      AX
	SBBL      AX, AX
	ANDL      $1, AX
	NEGL      DX
	SBBL      DX, DX
	ANDL      $2, DX
	ORL       DX, AX
	MOVQ      AX, flags+32(FP)
	RET

// ---------------------------------------------------------------------------
// func subtractGreenSSE41(p *uint32, n int)
//
// subtractGreenGo for n pixels, n a multiple of 4 (libwebp
// SubtractGreenFromBlueAndRed_SSE2, with PSHUFB): the shuffle copies the
// green byte of each pixel to its blue and red bytes and zeroes the
// others, and PSUBB subtracts it (mod 256) in place.
// DI = p, CX = end offset, AX = offset, X7 = shuffle.
TEXT ·subtractGreenSSE41(SB), NOSPLIT, $0-16
	MOVQ   p+0(FP), DI
	MOVQ   n+8(FP), CX
	SHLQ   $2, CX
	MOVQ   $0x8005800580018001, AX
	MOVQ   AX, X7
	MOVQ   $0x800d800d80098009, AX
	PINSRQ $1, AX, X7
	XORQ   AX, AX
	TESTQ  CX, CX
	JZ     sgs_done

sgs_loop:
	MOVOU  (DI)(AX*1), X0
	MOVO   X0, X1
	PSHUFB X7, X1
	PSUBB  X1, X0
	MOVOU  X0, (DI)(AX*1)
	ADDQ   $16, AX
	CMPQ   AX, CX
	JB     sgs_loop

sgs_done:
	RET

// func subtractGreenAVX2(p *uint32, n int)
//
// subtractGreenSSE41 with 8 pixels per ymm (n a multiple of 8), two ymm
// per step while 16 pixels remain.
TEXT ·subtractGreenAVX2(SB), NOSPLIT, $0-16
	MOVQ        p+0(FP), DI
	MOVQ        n+8(FP), CX
	SHLQ        $2, CX
	MOVQ        $0x8005800580018001, AX
	VMOVQ       AX, X7
	MOVQ        $0x800d800d80098009, AX
	VPINSRQ     $1, AX, X7, X7
	VINSERTI128 $1, X7, Y7, Y7
	XORQ        AX, AX
	LEAQ        -64(CX), BX

sgv_loop2:
	CMPQ    AX, BX
	JGT     sgv_one
	VMOVDQU (DI)(AX*1), Y0
	VMOVDQU 32(DI)(AX*1), Y2
	VPSHUFB Y7, Y0, Y1
	VPSHUFB Y7, Y2, Y3
	VPSUBB  Y1, Y0, Y0
	VPSUBB  Y3, Y2, Y2
	VMOVDQU Y0, (DI)(AX*1)
	VMOVDQU Y2, 32(DI)(AX*1)
	ADDQ    $64, AX
	JMP     sgv_loop2

sgv_one:
	CMPQ    AX, CX
	JAE     sgv_done
	VMOVDQU (DI)(AX*1), Y0
	VPSHUFB Y7, Y0, Y1
	VPSUBB  Y1, Y0, Y0
	VMOVDQU Y0, (DI)(AX*1)

sgv_done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// PREFIXCODE sets r (32 bits) to prefixCode(v) for v in r: d = uint32(v-1)
// is its own code below 4, else the code is 2*hb + bit hb-1 of d, hb being
// the index of its highest set bit (BTL sets the carry to that bit). t is
// clobbered.
#define PREFIXCODE(r, t, done) \
	DECL r;            \
	CMPL r, $4;        \
	JCS  done;         \
	BSRL r, t;         \
	DECL t;            \
	BTL  t, r;         \
	LEAL 2(t)(t*1), r; \
	ADCL $0, r;        \
done:

// PLANECODE sets AX to distanceToPlaneCode(xsize, dist) (64 bits) for
// dist = the uint32 at (SI) and xsize = xs (a register or a memory
// operand, 1 <= xsize < 2^31): DIVL gives yoff (AX) and xoff (DX); the two
// neighborhoods of the distance in the plane are looked up in planeToCode,
// every other distance is dist+120. Clobbers DX and t.
#define PLANECODE(xs, t, near2, far, done) \
	MOVL    (SI), AX;                 \
	XORL    DX, DX;                   \
	DIVL    xs;                       \
	CMPL    DX, $8;                   \
	JHI     near2;                    \
	CMPL    AX, $8;                   \
	JCC     near2;                    \
	SHLL    $4, AX;                   \
	ADDL    $8, AX;                   \
	SUBL    DX, AX;                   \
	LEAQ    ·planeToCode(SB), t;      \
	MOVBLZX (t)(AX*1), AX;            \
	INCL    AX;                       \
	JMP     done;                     \
near2:                                \
	MOVL    xs, t;                    \
	SUBL    DX, t;                    \
	CMPL    t, $8;                    \
	JCC     far;                      \
	CMPL    AX, $7;                   \
	JCC     far;                      \
	INCL    AX;                       \
	SHLL    $4, AX;                   \
	ADDL    $8, AX;                   \
	ADDL    t, AX;                    \
	LEAQ    ·planeToCode(SB), t;      \
	MOVBLZX (t)(AX*1), AX;            \
	INCL    AX;                       \
	JMP     done;                     \
far:                                  \
	MOVL    (SI), AX;                 \
	ADDQ    $120, AX;                 \
done:

// func addRefsSSE41(q *uint32, stride, litLen int, refs *pixOrCopy, n, xsize int, raw bool) (done int)
//
// The loop of addRefsGo (addRefsRawDistGo when raw) over the n references,
// each one being added to the partial histogram (address/8)&3 of q (4
// histograms of stride bins: literal[litLen] | red | blue | alpha |
// dist[40]), so that consecutive references go to different histograms
// and consecutive increments of a bin do not wait for each other. A cache index past the bins or a distance code from 40 stops the
// loop (the Go code panics on it): returns the references added. The
// literal and length bins are in range by construction (the wrapper checks
// litLen >= 280 and stride = litLen+808).
// SI = reference, R9 = end, DI = q, R12 = 4*stride, R8 = the partial
// histogram of the reference, BX = litLen, R10 = stride, R11 = raw,
// R13 = xsize.
TEXT ·addRefsSSE41(SB), NOSPLIT, $0-64
	MOVQ    q+0(FP), DI
	MOVQ    stride+8(FP), R10
	MOVQ    litLen+16(FP), BX
	MOVQ    refs+24(FP), SI
	MOVQ    n+32(FP), R9
	MOVQ    xsize+40(FP), R13
	MOVBQZX raw+48(FP), R11
	LEAQ    (SI)(R9*8), R9
	LEAQ    (R10*4), R12

ar_loop:
	CMPQ    SI, R9
	JEQ     ar_done
	MOVQ    SI, R8
	SHRQ    $3, R8
	ANDQ    $3, R8
	IMULQ   R12, R8
	ADDQ    DI, R8
	MOVL    4(SI), CX
	TESTL   $3, CX
	JNZ     ar_nonlit
	MOVBLZX 1(SI), AX      // green
	INCL    (R8)(AX*4)
	MOVBLZX 2(SI), AX      // red
	ADDQ    BX, AX
	INCL    (R8)(AX*4)
	MOVBLZX (SI), AX       // blue
	ADDQ    BX, AX
	INCL    1024(R8)(AX*4)
	MOVBLZX 3(SI), AX      // alpha
	ADDQ    BX, AX
	INCL    2048(R8)(AX*4)
	ADDQ    $8, SI
	JMP     ar_loop

ar_nonlit:
	MOVL  CX, AX
	ANDL  $3, AX
	CMPL  AX, $1
	JNE   ar_copy
	MOVL  (SI), AX         // cache index
	ADDQ  $280, AX
	CMPQ  AX, R10
	JCC   ar_bad
	INCL  (R8)(AX*4)
	ADDQ  $8, SI
	JMP   ar_loop

ar_copy:
	SHRL       $2, CX      // length
	PREFIXCODE(CX, AX, ar_lc)
	INCL       1024(R8)(CX*4) // 256+code
	TESTQ      R11, R11
	JZ         ar_plane
	MOVL       (SI), AX
	JMP        ar_dist

ar_plane:
	PLANECODE(R13, CX, ar_near2, ar_far, ar_dist)
	PREFIXCODE(AX, CX, ar_dc)
	CMPL       AX, $40
	JCC        ar_bad
	ADDQ       BX, AX
	INCL       3072(R8)(AX*4) // litLen+768+code
	ADDQ       $8, SI
	JMP        ar_loop

ar_bad:
	MOVQ SI, R9

ar_done:
	SUBQ refs+24(FP), R9
	SHRQ $3, R9
	MOVQ R9, done+56(FP)
	RET

// ---------------------------------------------------------------------------
// Entropy-coded pixels (writeImageDataGo).
//
// The bit writer: AX = accumulator, CX = its bit count (< 32 between
// putBits), DI = buffer, R8 = length. PUTCODE adds the code of symbol s of
// code k of the tile (g = R9) at the bit count and moves it, as a putBits
// of the code; two of them then FLUSH are the putBits of their
// concatenation. FLUSH is the end of putBits without a branch: the low 32
// bits of the accumulator are stored at the end of the buffer (there is
// room for them), then kept (length += 4, accumulator >>= 32, count -= 32)
// if the count reached 32. The bits are the Go ones, and so are the flushes
// (the state of the Go writer is the same after each putBits).

// PUTCODE: s in DX; clobbers DX, R12, R13.
#define PUTCODE(k) \
	MOVQ    k*huffCode__size+huffCode_lengths(R9), R12; \
	MOVBQZX (R12)(DX*1), R13;                           \
	MOVQ    k*huffCode__size+huffCode_codes(R9), R12;   \
	MOVWQZX (R12)(DX*2), DX;                            \
	SHLQ    CL, DX;                                     \
	ORQ     DX, AX;                                     \
	ADDQ    R13, CX

// FLUSH clobbers DX, R12, R13.
#define FLUSH \
	MOVL    AX, (DI)(R8*1); \
	MOVQ    AX, DX;         \
	SHRQ    $32, DX;        \
	LEAQ    4(R8), R12;     \
	LEAQ    -32(CX), R13;   \
	CMPQ    CX, $32;        \
	CMOVQCC DX, AX;         \
	CMOVQCC R12, R8;        \
	CMOVQCC R13, CX

// PREFIXENC sets R12, R13 and DX to the symbol, the extra bit count and
// the extra bits of prefixEncode(v) for v in DX: d = uint32(v-1) is its own
// symbol below 4 (no extra bits), else hb is the index of its highest set
// bit, the symbol 2*hb + bit hb-1 of d (the carry of the BTRL that clears
// it) and the extra bits are the hb-1 bits of d below.
#define PREFIXENC(small, done) \
	DECL DX;                \
	MOVL DX, R12;           \
	XORL R13, R13;          \
	CMPL DX, $4;            \
	JCS  small;             \
	BSRL DX, R12;           \
	LEAL -1(R12), R13;      \
	BTRL R12, DX;           \
	BTRL R13, DX;           \
	LEAL (R12)(R12*1), R12; \
	ADCL $0, R12;           \
	JMP  done;              \
small:                      \
	XORL DX, DX;            \
done:

// func writeImageDataSSE41(st *imgWriter, refs *pixOrCopy, n int) (done int)
//
// The loop of writeImageDataGo over the n references, with the state of
// the writer, the position and the tile in st: before each reference, the
// tile's group is looked up again if x or y left the tile (never when
// there is no entropy image: the wrapper then sets tileMask, tileX and
// tileY to 0); the literals are written as 2 putBits, the cache indices as
// 1 and the copies as 4, as in the Go code. Stops before a reference when
// the buffer length passed st.limit or when a symbol, a tile or a group is
// out of range (st.err = 1; the wrapper then runs the Go code, which
// panics on it). Returns the references written.
// BX = st, SI = reference, R9 = g, R10 = x, R11 = y, the bit writer as
// above; locals: the end of refs, the length of a copy, the extra bits of
// a prefix code and their count, the accumulator during a division.
TEXT ·writeImageDataSSE41(SB), NOSPLIT, $40-32
	MOVQ st+0(FP), BX
	MOVQ refs+8(FP), SI
	MOVQ n+16(FP), DX
	LEAQ (SI)(DX*8), DX
	MOVQ DX, end-8(SP)
	MOVQ imgWriter_buf(BX), DI
	MOVQ imgWriter_pos(BX), R8
	MOVQ imgWriter_acc(BX), AX
	MOVQ imgWriter_nbits(BX), CX
	MOVQ imgWriter_g(BX), R9
	MOVQ imgWriter_x(BX), R10
	MOVQ imgWriter_y(BX), R11

wr_loop:
	CMPQ SI, end-8(SP)
	JEQ  wr_done
	CMPQ R8, imgWriter_limit(BX)
	JGT  wr_done
	MOVQ R10, DX
	ANDQ imgWriter_tileMask(BX), DX
	CMPQ DX, imgWriter_tileX(BX)
	JNE  wr_tile
	MOVQ R11, DX
	ANDQ imgWriter_tileMask(BX), DX
	CMPQ DX, imgWriter_tileY(BX)
	JNE  wr_tile

wr_ref:
	MOVL  4(SI), DX
	TESTL $3, DX
	JNZ   wr_nonlit
	MOVBLZX 1(SI), DX // green
	PUTCODE(0)
	MOVBLZX 2(SI), DX // red
	PUTCODE(1)
	FLUSH
	MOVBLZX (SI), DX  // blue
	PUTCODE(2)
	MOVBLZX 3(SI), DX // alpha
	PUTCODE(3)
	FLUSH
	ADDQ    $8, SI
	INCQ    R10
	CMPQ    R10, imgWriter_w(BX)
	JLT     wr_loop
	XORL    R10, R10
	INCQ    R11
	JMP     wr_loop

wr_nonlit:
	ANDL $3, DX
	CMPL DX, $1
	JNE  wr_copy
	MOVL (SI), DX     // cache index
	ADDQ $280, DX
	CMPQ DX, imgWriter_len0(BX)
	JCC  wr_bad
	PUTCODE(0)
	FLUSH
	ADDQ $8, SI
	INCQ R10
	CMPQ R10, imgWriter_w(BX)
	JLT  wr_loop
	XORL R10, R10
	INCQ R11
	JMP  wr_loop

wr_copy:
	MOVL      4(SI), DX
	SHRL      $2, DX
	MOVQ      DX, len-16(SP)
	PREFIXENC(wr_ls, wr_le)
	ADDQ      $256, R12
	CMPQ      R12, imgWriter_len0(BX)
	JCC       wr_bad
	MOVQ      DX, ex-24(SP)
	MOVQ      R13, nb-32(SP)
	MOVQ      R12, DX
	PUTCODE(0)
	FLUSH
	MOVQ      ex-24(SP), DX
	SHLQ      CL, DX
	ORQ       DX, AX
	ADDQ      nb-32(SP), CX
	FLUSH
	MOVQ      AX, acc-40(SP)
	PLANECODE(imgWriter_w(BX), R12, wr_near2, wr_far, wr_pc)
	MOVQ      AX, DX
	MOVQ      acc-40(SP), AX
	PREFIXENC(wr_ds, wr_de)
	CMPQ      R12, imgWriter_len4(BX)
	JCC       wr_bad
	MOVQ      DX, ex-24(SP)
	MOVQ      R13, nb-32(SP)
	MOVQ      R12, DX
	PUTCODE(4)
	FLUSH
	MOVQ      ex-24(SP), DX
	SHLQ      CL, DX
	ORQ       DX, AX
	ADDQ      nb-32(SP), CX
	FLUSH
	ADDQ      $8, SI
	ADDQ      len-16(SP), R10

wr_wrap:
	CMPQ R10, imgWriter_w(BX)
	JLT  wr_loop
	SUBQ imgWriter_w(BX), R10
	INCQ R11
	JMP  wr_wrap

wr_tile: // the group of the tile at (x, y): codes[5*symbols[(y>>hb)*hx + x>>hb]:]
	MOVQ    R10, DX
	ANDQ    imgWriter_tileMask(BX), DX
	MOVQ    DX, imgWriter_tileX(BX)
	MOVQ    R11, DX
	ANDQ    imgWriter_tileMask(BX), DX
	MOVQ    DX, imgWriter_tileY(BX)
	MOVQ    CX, R13
	MOVQ    imgWriter_hb(BX), CX
	MOVQ    R11, DX
	SHRQ    CL, DX
	IMULQ   imgWriter_hx(BX), DX
	MOVQ    R10, R12
	SHRQ    CL, R12
	ADDQ    R12, DX
	MOVQ    R13, CX
	CMPQ    DX, imgWriter_nsym(BX)
	JCC     wr_bad
	MOVQ    imgWriter_symbols(BX), R12
	MOVWQZX (R12)(DX*2), DX
	CMPQ    DX, imgWriter_ngroups(BX)
	JCC     wr_bad
	IMUL3Q  $(5*huffCode__size), DX, DX
	MOVQ    imgWriter_codes(BX), R9
	ADDQ    DX, R9
	JMP     wr_ref

wr_bad:
	MOVQ $1, imgWriter_err(BX)

wr_done:
	MOVQ R8, imgWriter_pos(BX)
	MOVQ AX, imgWriter_acc(BX)
	MOVQ CX, imgWriter_nbits(BX)
	MOVQ R9, imgWriter_g(BX)
	MOVQ R10, imgWriter_x(BX)
	MOVQ R11, imgWriter_y(BX)
	SUBQ refs+8(FP), SI
	SHRQ $3, SI
	MOVQ SI, done+24(FP)
	RET

// ---------------------------------------------------------------------------
// func addTileRefsSSE41(data *uint32, stride, litLen int, refs *pixOrCopy, n, xsize, hb, hx, ntiles, x0, y0 int) (done, x, y int)
//
// The loop of addTileRefsFrom over the n references, the first one at
// (x0, y0) (0 <= x0 < xsize): each one is added to
// the histogram of its tile (data + ix*stride, ix = (y>>hb)*hx + x>>hb),
// as in addRefsSSE41, then (x, y) moves past it. The alpha increments of
// consecutive literals usually hit the same bin (opaque images): they are
// counted in a register (DX for the bin at R10) and added to the bin when
// another one comes, so that they do not wait for each other. A tile, a
// cache index or a distance code out of range stops the loop (the Go code
// panics on it): returns the references added and the position of the
// next one.
// SI = reference, DI = data, R8 = tile histogram, BX = litLen, R9 = x,
// R11 = y; locals: the end of refs, the pending count during a division,
// xsize, the length of a copy, 4*stride.
TEXT ·addTileRefsSSE41(SB), NOSPLIT, $40-112
	MOVQ data+0(FP), DI
	MOVQ stride+8(FP), AX
	SHLQ $2, AX
	MOVQ AX, s4-40(SP)
	MOVQ litLen+16(FP), BX
	MOVQ refs+24(FP), SI
	MOVQ n+32(FP), AX
	LEAQ (SI)(AX*8), AX
	MOVQ AX, end-8(SP)
	MOVQ xsize+40(FP), AX
	MOVQ AX, xs-24(SP)
	MOVQ x0+72(FP), R9
	MOVQ y0+80(FP), R11
	MOVQ DI, R10
	XORL DX, DX

at_loop:
	CMPQ  SI, end-8(SP)
	JEQ   at_done
	MOVQ  hb+48(FP), CX
	MOVQ  R11, AX
	SHRQ  CL, AX
	IMULQ hx+56(FP), AX
	MOVQ  R9, R12
	SHRQ  CL, R12
	ADDQ  R12, AX
	CMPQ  AX, ntiles+64(FP)
	JCC   at_done
	IMULQ s4-40(SP), AX
	LEAQ  (DI)(AX*1), R8
	MOVL  4(SI), AX
	TESTL $3, AX
	JNZ   at_nonlit
	MOVBLZX 1(SI), AX      // green
	INCL    (R8)(AX*4)
	MOVBLZX 2(SI), AX      // red
	ADDQ    BX, AX
	INCL    (R8)(AX*4)
	MOVBLZX (SI), AX       // blue
	ADDQ    BX, AX
	INCL    1024(R8)(AX*4)
	MOVBLZX 3(SI), AX      // alpha: pending in DX while the bin repeats
	ADDQ    BX, AX
	LEAQ    2048(R8)(AX*4), AX
	CMPQ    AX, R10
	JNE     at_alpha
	INCL    DX
	JMP     at_lnext

at_alpha:
	ADDL DX, (R10)
	MOVQ AX, R10
	MOVL $1, DX

at_lnext:
	ADDQ $8, SI
	INCQ R9
	CMPQ R9, xs-24(SP)
	JLT  at_loop
	XORL R9, R9
	INCQ R11
	JMP  at_loop

at_nonlit:
	ANDL $3, AX
	CMPL AX, $1
	JNE  at_copy
	MOVL (SI), AX          // cache index
	ADDQ $280, AX
	CMPQ AX, stride+8(FP)
	JCC  at_done
	INCL (R8)(AX*4)
	JMP  at_lnext

at_copy:
	MOVL       4(SI), AX
	SHRL       $2, AX      // length
	MOVQ       AX, len-32(SP)
	PREFIXCODE(AX, R12, at_lc)
	INCL       1024(R8)(AX*4) // 256+code
	MOVQ       DX, cnt-16(SP)
	PLANECODE(xs-24(SP), R12, at_near2, at_far, at_pc)
	MOVQ       cnt-16(SP), DX
	PREFIXCODE(AX, R12, at_dc)
	CMPL       AX, $40
	JCC        at_done
	ADDQ       BX, AX
	INCL       3072(R8)(AX*4) // litLen+768+code
	ADDQ       $8, SI
	ADDQ       len-32(SP), R9

at_wrap:
	CMPQ R9, xs-24(SP)
	JLT  at_loop
	SUBQ xs-24(SP), R9
	INCQ R11
	JMP  at_wrap

at_done:
	ADDL DX, (R10)
	SUBQ refs+24(FP), SI
	SHRQ $3, SI
	MOVQ SI, done+88(FP)
	MOVQ R9, x+96(FP)
	MOVQ R11, y+104(FP)
	RET
