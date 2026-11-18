//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 and AVX2 kernels of the lossless encoder (see dsp_amd64.go for
// the wrappers and their bound checks). Algorithms adapted from libwebp
// (src/dsp/lossless_enc_sse2.c, lossless_enc_sse41.c, lossless_enc_avx2.c,
// BSD-3-Clause); every kernel is bit-exact with the Go reference it
// replaces.

// ---------------------------------------------------------------------------
// predictRow: residuals of the 14 spatial predictors (libwebp PredictorSub*
// _SSE2 and _AVX2), out[i] = cur[i] - pred(i) per byte, for i in [0, n).
//
// Arguments: R8 = mode, SI = &cur[x0], DX = &up[x0], DI = &out[0], CX = 4*n.
// The neighbours of pixel i are L = cur[i-1], T = up[i], TL = up[i-1] and
// TR = up[i+1] (the wrapper checks that cur[x0-1:x1] and up[x0-1:x1+1] are
// in bounds). The residuals of a block only depend on the source pixels,
// so the blocks are independent: AX walks the blocks (a byte offset), BX
// is the offset of the last full block, and a partial last block is
// computed again as the full block ending at n (overlapping the previous
// one, with the same values), so nothing outside the rows is touched. The
// SSE4.1 kernel needs n >= 4, the AVX2 one n >= 8.
//
// average2 (floor of the per-byte mean) is PAVGB (which rounds up) minus
// ((a ^ b) & 1). Predictor 11 (select) compares the sums of the absolute
// byte differences |L-TL| and |T-TL| of each pixel (PMADDUBSW with 1s,
// then PMADDWD with 1s); 12 and 13 work on 16-bit lanes and clamp with
// PACKUSWB, as libwebp does.
//
// Constants: X14/Y14 = 0x01 bytes, X13/Y13 = 0x0001 words, X15/Y15 = 0,
// X12/Y12 = 0xff000000 (ARGB black).

// SSE_NEXT stores the residuals of the block at AX (in X0) and moves to the
// next block, or to the overlapping last block, or returns.
#define SSE_NEXT(loop) \
	MOVOU X0, (DI)(AX*1); \
	ADDQ  $16, AX;        \
	CMPQ  AX, BX;         \
	JLE   loop;           \
	CMPQ  AX, CX;         \
	JEQ   sse_done;       \
	MOVQ  BX, AX;         \
	JMP   loop

// AVGS sets b to average2(a, b); a is kept, t is clobbered.
#define AVGS(a, b, t) \
	MOVO  a, t;   \
	PXOR  b, t;   \
	PAVGB a, b;   \
	PAND  X14, t; \
	PSUBB t, b

#define SRC_S MOVOU (SI)(AX*1), X0

// func predictRowSSE41(mode int, cur, up, out *uint32, n int)
TEXT ·predictRowSSE41(SB), NOSPLIT, $0-40
	MOVQ mode+0(FP), R8
	MOVQ cur+8(FP), SI
	MOVQ up+16(FP), DX
	MOVQ out+24(FP), DI
	MOVQ n+32(FP), CX
	SHLQ $2, CX
	LEAQ -16(CX), BX
	XORQ AX, AX
	PCMPEQB X14, X14
	MOVO    X14, X12
	PSLLL   $24, X12
	PABSB   X14, X14
	MOVO    X14, X13
	PSRLW   $8, X13
	PXOR    X15, X15
	CMPQ    R8, $7
	JAE     sse_hi
	CMPQ    R8, $0
	JEQ     sse_m0
	CMPQ    R8, $1
	JEQ     sse_m1
	CMPQ    R8, $2
	JEQ     sse_m2
	CMPQ    R8, $3
	JEQ     sse_m3
	CMPQ    R8, $4
	JEQ     sse_m4
	CMPQ    R8, $5
	JEQ     sse_m5
	JMP     sse_m6

sse_hi:
	CMPQ R8, $7
	JEQ  sse_m7
	CMPQ R8, $8
	JEQ  sse_m8
	CMPQ R8, $9
	JEQ  sse_m9
	CMPQ R8, $10
	JEQ  sse_m10
	CMPQ R8, $11
	JEQ  sse_m11
	CMPQ R8, $12
	JEQ  sse_m12
	JMP  sse_m13

sse_m0: // black
	SRC_S
	PSUBB X12, X0
	SSE_NEXT(sse_m0)

sse_m1: // L
	SRC_S
	MOVOU -4(SI)(AX*1), X1
	PSUBB X1, X0
	SSE_NEXT(sse_m1)

sse_m2: // T
	SRC_S
	MOVOU (DX)(AX*1), X1
	PSUBB X1, X0
	SSE_NEXT(sse_m2)

sse_m3: // TR
	SRC_S
	MOVOU 4(DX)(AX*1), X1
	PSUBB X1, X0
	SSE_NEXT(sse_m3)

sse_m4: // TL
	SRC_S
	MOVOU -4(DX)(AX*1), X1
	PSUBB X1, X0
	SSE_NEXT(sse_m4)

sse_m5: // avg(avg(L, TR), T)
	MOVOU -4(SI)(AX*1), X1
	MOVOU 4(DX)(AX*1), X2
	AVGS(X2, X1, X3)
	MOVOU (DX)(AX*1), X2
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m5)

sse_m6: // avg(L, TL)
	MOVOU -4(SI)(AX*1), X1
	MOVOU -4(DX)(AX*1), X2
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m6)

sse_m7: // avg(L, T)
	MOVOU -4(SI)(AX*1), X1
	MOVOU (DX)(AX*1), X2
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m7)

sse_m8: // avg(TL, T)
	MOVOU -4(DX)(AX*1), X1
	MOVOU (DX)(AX*1), X2
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m8)

sse_m9: // avg(T, TR)
	MOVOU (DX)(AX*1), X1
	MOVOU 4(DX)(AX*1), X2
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m9)

sse_m10: // avg(avg(L, TL), avg(T, TR))
	MOVOU -4(SI)(AX*1), X1
	MOVOU -4(DX)(AX*1), X2
	AVGS(X2, X1, X3)
	MOVOU (DX)(AX*1), X2
	MOVOU 4(DX)(AX*1), X4
	AVGS(X4, X2, X3)
	AVGS(X2, X1, X3)
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m10)

sse_m11: // select: L if sum|L-TL| > sum|T-TL|, else T
	MOVOU -4(SI)(AX*1), X1 // L
	MOVOU (DX)(AX*1), X2   // T
	MOVOU -4(DX)(AX*1), X3 // TL
	MOVO  X2, X4
	PMAXUB X3, X4
	MOVO  X2, X5
	PMINUB X3, X5
	PSUBB X5, X4           // |T-TL|
	MOVO  X1, X5
	PMAXUB X3, X5
	MOVO  X1, X6
	PMINUB X3, X6
	PSUBB X6, X5           // |L-TL|
	PMADDUBSW X14, X4
	PMADDUBSW X14, X5
	PMADDWL X13, X4        // pa = sum|T-TL|
	PMADDWL X13, X5        // pb = sum|L-TL|
	PCMPGTL X4, X5         // pb > pa
	PAND  X5, X1
	PANDN X2, X5
	POR   X5, X1
	SRC_S
	PSUBB X1, X0
	SSE_NEXT(sse_m11)

sse_m12: // clamp(L + T - TL)
	MOVOU -4(SI)(AX*1), X1
	MOVOU (DX)(AX*1), X2
	MOVOU -4(DX)(AX*1), X3
	MOVO  X1, X4
	PUNPCKLBW X15, X4
	MOVO  X2, X5
	PUNPCKLBW X15, X5
	MOVO  X3, X6
	PUNPCKLBW X15, X6
	PADDW X5, X4
	PSUBW X6, X4
	PUNPCKHBW X15, X1
	PUNPCKHBW X15, X2
	PUNPCKHBW X15, X3
	PADDW X2, X1
	PSUBW X3, X1
	PACKUSWB X1, X4
	SRC_S
	PSUBB X4, X0
	SSE_NEXT(sse_m12)

sse_m13: // clamp(a + (a - TL) / 2), a = avg(L, T)
	MOVOU -4(SI)(AX*1), X1
	MOVOU (DX)(AX*1), X2
	MOVOU -4(DX)(AX*1), X3
	MOVO  X1, X4
	PUNPCKLBW X15, X4
	MOVO  X2, X5
	PUNPCKLBW X15, X5
	PADDW X5, X4
	PSRLW $1, X4           // a (low half)
	MOVO  X3, X6
	PUNPCKLBW X15, X6
	MOVO  X4, X7
	PSUBW X6, X7           // a - TL
	PCMPGTW X4, X6         // TL > a: -1
	PSUBW X6, X7           // rounds the halving towards 0
	PSRAW $1, X7
	PADDW X7, X4
	PUNPCKHBW X15, X1
	PUNPCKHBW X15, X2
	PADDW X2, X1
	PSRLW $1, X1           // a (high half)
	PUNPCKHBW X15, X3
	MOVO  X1, X7
	PSUBW X3, X7
	PCMPGTW X1, X3
	PSUBW X3, X7
	PSRAW $1, X7
	PADDW X7, X1
	PACKUSWB X1, X4
	SRC_S
	PSUBB X4, X0
	SSE_NEXT(sse_m13)

sse_done:
	RET

// AVX_NEXT is SSE_NEXT for the 8-pixel blocks of the AVX2 kernel.
#define AVX_NEXT(loop) \
	VMOVDQU Y0, (DI)(AX*1); \
	ADDQ    $32, AX;        \
	CMPQ    AX, BX;         \
	JLE     loop;           \
	CMPQ    AX, CX;         \
	JEQ     avx_done;       \
	MOVQ    BX, AX;         \
	JMP     loop

// AVGV sets dst to average2(a, b); t is clobbered (dst may be a or b).
#define AVGV(a, b, dst, t) \
	VPXOR  a, b, t;     \
	VPAVGB a, b, dst;   \
	VPAND  Y14, t, t;   \
	VPSUBB t, dst, dst

#define SRC_V VMOVDQU (SI)(AX*1), Y0

// func predictRowAVX2(mode int, cur, up, out *uint32, n int)
TEXT ·predictRowAVX2(SB), NOSPLIT, $0-40
	MOVQ mode+0(FP), R8
	MOVQ cur+8(FP), SI
	MOVQ up+16(FP), DX
	MOVQ out+24(FP), DI
	MOVQ n+32(FP), CX
	SHLQ $2, CX
	LEAQ -32(CX), BX
	XORQ AX, AX
	VPCMPEQB Y14, Y14, Y14
	VPSLLD   $24, Y14, Y12
	VPABSB   Y14, Y14
	VPSRLW   $8, Y14, Y13
	VPXOR    Y15, Y15, Y15
	CMPQ     R8, $7
	JAE      avx_hi
	CMPQ     R8, $0
	JEQ      avx_m0
	CMPQ     R8, $1
	JEQ      avx_m1
	CMPQ     R8, $2
	JEQ      avx_m2
	CMPQ     R8, $3
	JEQ      avx_m3
	CMPQ     R8, $4
	JEQ      avx_m4
	CMPQ     R8, $5
	JEQ      avx_m5
	JMP      avx_m6

avx_hi:
	CMPQ R8, $7
	JEQ  avx_m7
	CMPQ R8, $8
	JEQ  avx_m8
	CMPQ R8, $9
	JEQ  avx_m9
	CMPQ R8, $10
	JEQ  avx_m10
	CMPQ R8, $11
	JEQ  avx_m11
	CMPQ R8, $12
	JEQ  avx_m12
	JMP  avx_m13

avx_m0: // black
	SRC_V
	VPSUBB Y12, Y0, Y0
	AVX_NEXT(avx_m0)

avx_m1: // L
	SRC_V
	VPSUBB -4(SI)(AX*1), Y0, Y0
	AVX_NEXT(avx_m1)

avx_m2: // T
	SRC_V
	VPSUBB (DX)(AX*1), Y0, Y0
	AVX_NEXT(avx_m2)

avx_m3: // TR
	SRC_V
	VPSUBB 4(DX)(AX*1), Y0, Y0
	AVX_NEXT(avx_m3)

avx_m4: // TL
	SRC_V
	VPSUBB -4(DX)(AX*1), Y0, Y0
	AVX_NEXT(avx_m4)

avx_m5: // avg(avg(L, TR), T)
	VMOVDQU -4(SI)(AX*1), Y1
	VMOVDQU 4(DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	VMOVDQU (DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m5)

avx_m6: // avg(L, TL)
	VMOVDQU -4(SI)(AX*1), Y1
	VMOVDQU -4(DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m6)

avx_m7: // avg(L, T)
	VMOVDQU -4(SI)(AX*1), Y1
	VMOVDQU (DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m7)

avx_m8: // avg(TL, T)
	VMOVDQU -4(DX)(AX*1), Y1
	VMOVDQU (DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m8)

avx_m9: // avg(T, TR)
	VMOVDQU (DX)(AX*1), Y1
	VMOVDQU 4(DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m9)

avx_m10: // avg(avg(L, TL), avg(T, TR))
	VMOVDQU -4(SI)(AX*1), Y1
	VMOVDQU -4(DX)(AX*1), Y2
	AVGV(Y2, Y1, Y1, Y3)
	VMOVDQU (DX)(AX*1), Y2
	VMOVDQU 4(DX)(AX*1), Y4
	AVGV(Y4, Y2, Y2, Y3)
	AVGV(Y2, Y1, Y1, Y3)
	SRC_V
	VPSUBB Y1, Y0, Y0
	AVX_NEXT(avx_m10)

avx_m11: // select: L if sum|L-TL| > sum|T-TL|, else T
	VMOVDQU    -4(SI)(AX*1), Y1 // L
	VMOVDQU    (DX)(AX*1), Y2   // T
	VMOVDQU    -4(DX)(AX*1), Y3 // TL
	VPMAXUB    Y3, Y2, Y4
	VPMINUB    Y3, Y2, Y5
	VPSUBB     Y5, Y4, Y4       // |T-TL|
	VPMAXUB    Y3, Y1, Y5
	VPMINUB    Y3, Y1, Y6
	VPSUBB     Y6, Y5, Y5       // |L-TL|
	VPMADDUBSW Y14, Y4, Y4
	VPMADDUBSW Y14, Y5, Y5
	VPMADDWD   Y13, Y4, Y4      // pa = sum|T-TL|
	VPMADDWD   Y13, Y5, Y5      // pb = sum|L-TL|
	VPCMPGTD   Y4, Y5, Y5       // pb > pa
	VPBLENDVB  Y5, Y1, Y2, Y1   // L where pb > pa, else T
	SRC_V
	VPSUBB     Y1, Y0, Y0
	AVX_NEXT(avx_m11)

avx_m12: // clamp(L + T - TL)
	VMOVDQU    -4(SI)(AX*1), Y1
	VMOVDQU    (DX)(AX*1), Y2
	VMOVDQU    -4(DX)(AX*1), Y3
	VPUNPCKLBW Y15, Y1, Y4
	VPUNPCKLBW Y15, Y2, Y5
	VPUNPCKLBW Y15, Y3, Y6
	VPADDW     Y5, Y4, Y4
	VPSUBW     Y6, Y4, Y4
	VPUNPCKHBW Y15, Y1, Y1
	VPUNPCKHBW Y15, Y2, Y2
	VPUNPCKHBW Y15, Y3, Y3
	VPADDW     Y2, Y1, Y1
	VPSUBW     Y3, Y1, Y1
	VPACKUSWB  Y1, Y4, Y4
	SRC_V
	VPSUBB     Y4, Y0, Y0
	AVX_NEXT(avx_m12)

avx_m13: // clamp(a + (a - TL) / 2), a = avg(L, T)
	VMOVDQU    -4(SI)(AX*1), Y1
	VMOVDQU    (DX)(AX*1), Y2
	VMOVDQU    -4(DX)(AX*1), Y3
	VPUNPCKLBW Y15, Y1, Y4
	VPUNPCKLBW Y15, Y2, Y5
	VPADDW     Y5, Y4, Y4
	VPSRLW     $1, Y4, Y4       // a (low halves)
	VPUNPCKLBW Y15, Y3, Y6
	VPSUBW     Y6, Y4, Y7       // a - TL
	VPCMPGTW   Y4, Y6, Y6       // TL > a: -1
	VPSUBW     Y6, Y7, Y7       // rounds the halving towards 0
	VPSRAW     $1, Y7, Y7
	VPADDW     Y7, Y4, Y4
	VPUNPCKHBW Y15, Y1, Y1
	VPUNPCKHBW Y15, Y2, Y2
	VPADDW     Y2, Y1, Y1
	VPSRLW     $1, Y1, Y1       // a (high halves)
	VPUNPCKHBW Y15, Y3, Y3
	VPSUBW     Y3, Y1, Y7
	VPCMPGTW   Y1, Y3, Y3
	VPSUBW     Y3, Y7, Y7
	VPSRAW     $1, Y7, Y7
	VPADDW     Y7, Y1, Y1
	VPACKUSWB  Y1, Y4, Y4
	SRC_V
	VPSUBB     Y4, Y0, Y0
	AVX_NEXT(avx_m13)

avx_done:
	VZEROUPPER
	RET
