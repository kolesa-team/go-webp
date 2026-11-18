//go:build amd64 && !purego

#include "go_asm.h"
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

// ---------------------------------------------------------------------------
// Entropy helpers. They keep the floating-point operations of the Go code,
// in its order, so that the results are bit-identical: only the integer
// bookkeeping (which bins to visit) changes.

// SLOG2_SLOW sets X2 to slog2Slow(AX) (AX >= 4096, zero-extended):
// float64(v) * (log2Tab[m] + float64(shift) + float64(rem) /
// (float64(m) * Ln2 * float64(1<<shift))) with shift = Len32(v) - 12,
// m = v >> shift, rem = v & (1<<shift - 1), as entropy.go evaluates it.
// Clobbers CX, R9, R13, X3, X4, X5; X14 holds Ln2.
#define SLOG2_SLOW \
	BSRL     AX, CX;             \
	SUBL     $11, CX;            \
	MOVL     AX, R13;            \
	SHRL     CX, R13;            \
	MOVL     $1, R9;             \
	SHLL     CX, R9;             \
	DECL     R9;                 \
	ANDL     AX, R9;             \
	CVTSQ2SD R9, X3;             \
	CVTSQ2SD R13, X4;            \
	MULSD    X14, X4;            \
	MOVL     $1, R9;             \
	SHLQ     CX, R9;             \
	CVTSQ2SD R9, X5;             \
	MULSD    X5, X4;             \
	DIVSD    X4, X3;             \
	LEAQ     ·log2Tab(SB), R9;   \
	MOVSD    (R9)(R13*8), X2;    \
	CVTSQ2SD CX, X5;             \
	ADDSD    X5, X2;             \
	ADDSD    X3, X2;             \
	CVTSQ2SD AX, X5;             \
	MULSD    X5, X2

// SLOG2 sets X2 to slog2(AX) (R8 = &slog2Tab), through slow (which jumps
// back to back) for AX >= 4096.
#define SLOG2(slow, back) \
	CMPL  AX, $4096;      \
	JAE   slow;           \
	MOVSD (R8)(AX*8), X2; \
back:

// ZEROMASK16 ORs into R12, at bit sh, the mask of the zero bins among the
// 16 int32 at off(AX): the bins are packed with signed saturation to bytes
// (a bin is zero iff its byte is), compared with 0 (X15) and PMOVMSKB'd.
#define ZEROMASK16(off, sh) \
	MOVOU    off(AX), X1;      \
	MOVOU    off+16(AX), X2;   \
	MOVOU    off+32(AX), X3;   \
	MOVOU    off+48(AX), X4;   \
	PACKSSLW X2, X1;           \
	PACKSSLW X4, X3;           \
	PACKSSWB X3, X1;           \
	PCMPEQB  X15, X1;          \
	PMOVMSKB X1, R13;          \
	SHLQ     $sh, R13;         \
	ORQ      R13, R12

// func combinedShannonEntropySSE41(x, y *[256]int32, acc *accEntropy) float64
//
// combinedShannonEntropyGo (libwebp CombinedShannonEntropy_SSE2/_AVX2 use
// the same bit-mask walk): for each chunk of 64 bins, a mask of the
// non-zero bins of x, then for each set bit i (BSF, clear lowest):
// sumX += x[i]; r += (acc.s[i] - slog2(x[i])) - slog2(x[i]+y[i]).
// SI = x, DI = y, DX = acc, R8 = &slog2Tab, R10 = sumX, R11 = chunk base,
// R12 = mask, BX = bin, X0 = r, X14 = Ln2, X15 = 0.
TEXT ·combinedShannonEntropySSE41(SB), NOSPLIT, $0-32
	MOVQ  x+0(FP), SI
	MOVQ  y+8(FP), DI
	MOVQ  acc+16(FP), DX
	LEAQ  ·slog2Tab(SB), R8
	MOVQ  $0x3fe62e42fefa39ef, AX // math.Ln2
	MOVQ  AX, X14
	PXOR  X15, X15
	MOVQ  $0x8000000000000000, AX
	MOVQ  AX, X1
	MOVSD accEntropy_total(DX), X0
	XORPD X1, X0                 // r = -acc.total
	XORL  R10, R10
	XORL  R11, R11

cse_chunk:
	LEAQ (SI)(R11*4), AX
	XORL R12, R12
	ZEROMASK16(0, 0)
	ZEROMASK16(64, 16)
	ZEROMASK16(128, 32)
	ZEROMASK16(192, 48)
	NOTQ R12

cse_bin:
	TESTQ R12, R12
	JZ    cse_next
	BSFQ  R12, BX
	LEAQ  -1(R12), AX
	ANDQ  AX, R12
	ADDQ  R11, BX
	MOVL  (SI)(BX*4), AX
	ADDL  AX, R10
	MOVSD accEntropy_s(DX)(BX*8), X1
	SLOG2(cse_slow1, cse_back1)
	SUBSD X2, X1
	ADDL  (DI)(BX*4), AX
	SLOG2(cse_slow2, cse_back2)
	SUBSD X2, X1
	ADDSD X1, X0
	JMP   cse_bin

cse_next:
	ADDQ $64, R11
	CMPQ R11, $256
	JB   cse_chunk
	MOVL R10, AX
	SLOG2(cse_slow3, cse_back3)
	ADDSD X2, X0
	MOVL  R10, AX
	ADDL  accEntropy_sum(DX), AX
	SLOG2(cse_slow4, cse_back4)
	ADDSD X2, X0
	MOVSD X0, ret+24(FP)
	RET

cse_slow1:
	SLOG2_SLOW
	JMP cse_back1

cse_slow2:
	SLOG2_SLOW
	JMP cse_back2

cse_slow3:
	SLOG2_SLOW
	JMP cse_back3

cse_slow4:
	SLOG2_SLOW
	JMP cse_back4

// SLOG2T is SLOG2 with the table address loaded into R9 (which the slow
// path clobbers anyway), for kernels short of registers.
#define SLOG2T(slow, back) \
	CMPL  AX, $4096;          \
	JAE   slow;               \
	LEAQ  ·slog2Tab(SB), R9;  \
	MOVSD (R9)(AX*8), X2;     \
back:


// CHANGEMASK4 ORs into R12, at bit sh, the mask of the positions p of
// [BX+off/4, BX+off/4+4) where a[p]+b[p] == a[p-1]+b[p-1] (SI = a,
// DI = b), and folds the sums a[p]+b[p] (X0) into the maximum X9.
#define CHANGEMASK4(off, sh) \
	MOVOU    off(SI)(BX*4), X0;   \
	MOVOU    off(DI)(BX*4), X1;   \
	PADDL    X1, X0;              \
	MOVOU    off-4(SI)(BX*4), X1; \
	MOVOU    off-4(DI)(BX*4), X2; \
	PADDL    X2, X1;              \
	PMAXUD   X0, X9;              \
	PCMPEQL  X0, X1;              \
	MOVMSKPS X1, AX;              \
	SHLL     $sh, AX;             \
	ORL      AX, R12

// CHANGEMASK4S is CHANGEMASK4 that also adds the sums into X8 and counts
// the zero sums into X10 (X15 = 0).
#define CHANGEMASK4S(off, sh) \
	CHANGEMASK4(off, sh); \
	PADDL   X0, X8;       \
	PCMPEQL X15, X0;      \
	PSUBL   X0, X10

// CHANGEMASK16 sets R12 to the mask of the positions of [BX, BX+16) where
// a streak ends (a[p]+b[p] != a[p-1]+b[p-1]); CHANGEMASK16S also adds
// their sums and zero counts.
#define CHANGEMASK16 \
	XORL R12, R12;        \
	CHANGEMASK4(0, 0);    \
	CHANGEMASK4(16, 4);   \
	CHANGEMASK4(32, 8);   \
	CHANGEMASK4(48, 12);  \
	XORL $0xffff, R12

#define CHANGEMASK16S \
	XORL R12, R12;        \
	CHANGEMASK4S(0, 0);   \
	CHANGEMASK4S(16, 4);  \
	CHANGEMASK4S(32, 8);  \
	CHANGEMASK4S(48, 12); \
	XORL $0xffff, R12

// STREAK adds a streak of R13 bins of value R10, as combinedPopulationCostGo
// does, without branches: with nz = (R10 != 0) and long = (R13 > 3), the
// long streaks are counted and their lengths summed per kind in R8 (four
// 16-bit fields: count of zero streaks, count of non-zero streaks, length
// of zero streaks, length of non-zero streaks), and X7 -= slog2(R10) *
// float64(R13). For a zero value X7 - (+0 * R13) == X7 exactly, as when
// the Go code skips it. The other statistics follow from the bins
// themselves (popStatsSSE41). Clobbers AX, CX, R9, R13, X2-X6.
#define STREAK(slow, back) \
	XORL     CX, CX;      \
	TESTL    R10, R10;    \
	SETNE    CX;          \
	XORL     R9, R9;      \
	CMPQ     R13, $3;     \
	SETGT    R9;          \
	NEGQ     R9;          \
	MOVQ     R13, AX;     \
	SHLQ     $32, AX;     \
	INCQ     AX;          \
	ANDQ     R9, AX;      \
	SHLL     $4, CX;      \
	SHLQ     CX, AX;      \
	ADDQ     AX, R8;      \
	CVTSQ2SD R13, X6;     \
	MOVL     R10, AX;     \
	SLOG2T(slow, back);   \
	MULSD    X6, X2;      \
	SUBSD    X2, X7

// func popStatsSSE41(a, b *uint32, n int, st *popStats)
//
// The streak loop of combinedPopulationCostGo (16 < n < 65536, st zeroed):
// a streak ends at each position p in [1, n) where a[p]+b[p] differs from
// the previous sum, and at n. The positions are scanned 16 at a time
// (CHANGEMASK16S); the last, partial group is the 16 positions ending at n
// with the positions already scanned masked out (and their sums added one
// by one). Each streak end (BSF, clear lowest) is added with STREAK.
// The statistics that do not depend on the streaks are computed from the
// sums: e.sum is their total (mod 2^32: the Go code adds value*length per
// non-zero streak), e.maxVal their maximum, e.nonzeros the count of the
// non-zero ones, which is also the total length of the non-zero streaks
// (the short ones are the rest).
// SI = a, DI = b, DX = st, BX = first position of the group, R10 = value
// of the current streak, R11 = its start, R12 = mask, R8 = long streaks,
// X7 = entropy, X8 = sums, X9 = maximum, X10 = zero counts (4 lanes),
// X14 = Ln2, X15 = 0.
TEXT ·popStatsSSE41(SB), NOSPLIT, $0-32
	MOVQ  a+0(FP), SI
	MOVQ  b+8(FP), DI
	MOVQ  st+24(FP), DX
	MOVQ  $0x3fe62e42fefa39ef, AX // math.Ln2
	MOVQ  AX, X14
	PXOR  X15, X15
	XORPS X7, X7
	MOVL  (SI), R10
	ADDL  (DI), R10
	MOVQ  R10, X8
	MOVQ  R10, X9
	XORL  AX, AX
	TESTL R10, R10
	SETEQ AX
	MOVQ  AX, X10
	XORL  R8, R8
	XORL  R11, R11
	MOVL  $1, BX

pop_group:
	LEAQ 16(BX), AX
	CMPQ AX, n+16(FP)
	JA   pop_tail
	CHANGEMASK16S

pop_walk:
	TESTL R12, R12
	JZ    pop_walked
	BSFL  R12, AX
	LEAL  -1(R12), CX
	ANDL  CX, R12
	ADDQ  BX, AX
	MOVQ  AX, R13
	SUBQ  R11, R13
	MOVQ  AX, R11
	STREAK(pop_slow1, pop_back1)
	MOVL  (SI)(R11*4), R10
	ADDL  (DI)(R11*4), R10
	JMP   pop_walk

pop_walked:
	ADDQ $16, BX
	JMP  pop_group

pop_tail:
	MOVQ BX, CX
	CMPQ CX, n+16(FP)
	JAE  pop_final

pop_tailsum: // the sums of the positions not scanned yet, one by one
	MOVL  (SI)(CX*4), AX
	ADDL  (DI)(CX*4), AX
	MOVQ  AX, X0
	PADDL X0, X8
	XORL  R9, R9
	TESTL AX, AX
	SETEQ R9
	MOVQ  R9, X0
	PADDL X0, X10
	INCQ  CX
	CMPQ  CX, n+16(FP)
	JB    pop_tailsum
	MOVQ  BX, CX
	MOVQ  n+16(FP), BX
	SUBQ  $16, BX        // the last group
	SUBQ  BX, CX         // its positions already scanned (1..15)
	CHANGEMASK16
	SHRL  CX, R12
	SHLL  CX, R12
	JMP   pop_walk

pop_final:
	MOVQ   n+16(FP), R13
	SUBQ   R11, R13
	STREAK(pop_slow2, pop_back2)
	MOVSD  X7, popStats_e+bitEntropy_entropy(DX)
	PSHUFD $0x4e, X8, X0
	PADDL  X0, X8
	PSHUFD $0xb1, X8, X0
	PADDL  X0, X8
	MOVQ   X8, AX
	MOVL   AX, popStats_e+bitEntropy_sum(DX)
	PSHUFD $0x4e, X9, X0
	PMAXUD X0, X9
	PSHUFD $0xb1, X9, X0
	PMAXUD X0, X9
	MOVQ   X9, AX
	MOVL   AX, popStats_e+bitEntropy_maxVal(DX)
	PSHUFD $0x4e, X10, X0
	PADDL  X0, X10
	PSHUFD $0xb1, X10, X0
	PADDL  X0, X10
	MOVQ   X10, CX
	MOVL   CX, CX                 // zero bins
	MOVQ   n+16(FP), AX
	SUBQ   CX, AX                 // non-zero bins
	MOVQ   AX, popStats_e+bitEntropy_nonzeros(DX)
	MOVQ   R8, R9
	SHRQ   $48, R9                // length of the long non-zero streaks
	MOVQ   R9, popStats_s+streaks_streaks+24(DX)
	SUBQ   R9, AX
	MOVQ   AX, popStats_s+streaks_streaks+16(DX)
	MOVQ   R8, R9
	SHRQ   $32, R9
	ANDQ   $0xffff, R9            // length of the long zero streaks
	MOVQ   R9, popStats_s+streaks_streaks+8(DX)
	SUBQ   R9, CX
	MOVQ   CX, popStats_s+streaks_streaks+0(DX)
	MOVQ   R8, R9
	ANDQ   $0xffff, R9
	MOVQ   R9, popStats_s+streaks_counts+0(DX)
	SHRQ   $16, R8
	ANDQ   $0xffff, R8
	MOVQ   R8, popStats_s+streaks_counts+8(DX)
	RET

pop_slow1:
	SLOG2_SLOW
	JMP pop_back1

pop_slow2:
	SLOG2_SLOW
	JMP pop_back2
