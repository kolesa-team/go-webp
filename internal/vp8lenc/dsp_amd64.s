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

// NZMASK16 ORs into R12, at bit sh, the mask of the zero bins among the 16
// int32 at off(SI)(R11*4) (as ZEROMASK16), after adding them into X8-X11.
#define NZMASK16(off, sh) \
	MOVOU    off(SI)(R11*4), X1;    \
	MOVOU    off+16(SI)(R11*4), X2; \
	MOVOU    off+32(SI)(R11*4), X3; \
	MOVOU    off+48(SI)(R11*4), X4; \
	PADDL    X1, X8;                \
	PADDL    X2, X9;                \
	PADDL    X3, X10;               \
	PADDL    X4, X11;               \
	PACKSSLW X2, X1;                \
	PACKSSLW X4, X3;                \
	PACKSSWB X3, X1;                \
	PCMPEQB  X15, X1;               \
	PMOVMSKB X1, R13;               \
	SHLQ     $sh, R13;              \
	ORQ      R13, R12

// ZERO8 stores 8 zero float64 at off(AX).
#define ZERO8(off) \
	MOVOU X15, off(AX);    \
	MOVOU X15, off+16(AX); \
	MOVOU X15, off+32(AX); \
	MOVOU X15, off+48(AX)

// func accEntropyInitSSE41(h *[256]int32, a *accEntropy)
//
// accEntropy.initGo (libwebp has no counterpart: it recomputes the
// entropy of the accumulated histogram in every CombinedShannonEntropy).
// For each chunk of 64 bins: the bins are added into 4 lanes of sums, a
// mask of the non-zero bins is built, a.s is zeroed for the chunk, then for
// each set bit i (BSF, clear lowest) a.s[i] = slog2(h[i]) and
// total += a.s[i], in the order of the Go loop.
// SI = h, DX = a, R8 = &slog2Tab, R11 = chunk base, R12 = mask, BX = bin,
// X0 = total, X8-X11 = sums, X14 = Ln2, X15 = 0.
TEXT ·accEntropyInitSSE41(SB), NOSPLIT, $0-16
	MOVQ  h+0(FP), SI
	MOVQ  a+8(FP), DX
	LEAQ  ·slog2Tab(SB), R8
	MOVQ  $0x3fe62e42fefa39ef, AX // math.Ln2
	MOVQ  AX, X14
	PXOR  X15, X15
	XORPS X0, X0
	PXOR  X8, X8
	PXOR  X9, X9
	PXOR  X10, X10
	PXOR  X11, X11
	XORL  R11, R11

ae_chunk:
	XORL R12, R12
	NZMASK16(0, 0)
	NZMASK16(64, 16)
	NZMASK16(128, 32)
	NZMASK16(192, 48)
	NOTQ R12
	LEAQ accEntropy_s(DX)(R11*8), AX
	ZERO8(0)
	ZERO8(64)
	ZERO8(128)
	ZERO8(192)
	ZERO8(256)
	ZERO8(320)
	ZERO8(384)
	ZERO8(448)

ae_bin:
	TESTQ R12, R12
	JZ    ae_next
	BSFQ  R12, BX
	LEAQ  -1(R12), AX
	ANDQ  AX, R12
	ADDQ  R11, BX
	MOVL  (SI)(BX*4), AX
	SLOG2(ae_slow, ae_back)
	MOVSD X2, accEntropy_s(DX)(BX*8)
	ADDSD X2, X0
	JMP   ae_bin

ae_next:
	ADDQ   $64, R11
	CMPQ   R11, $256
	JB     ae_chunk
	MOVSD  X0, accEntropy_total(DX)
	PADDL  X9, X8
	PADDL  X11, X10
	PADDL  X10, X8
	PSHUFD $0x4e, X8, X1
	PADDL  X1, X8
	PSHUFD $0xb1, X8, X1
	PADDL  X1, X8
	MOVQ   X8, AX
	MOVL   AX, accEntropy_sum(DX)
	RET

ae_slow:
	SLOG2_SLOW
	JMP ae_back

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

// ---------------------------------------------------------------------------
// Hash chain search (libwebp VP8LHashChainFill with VectorMismatch_SSE2).

// MISMATCH sets AX to the length of the common prefix of the uint32 at R12
// and at R13, at most DX: 4 at a time (PCMPEQL, MOVMSKPS, BSF on the first
// difference) while 4 more fit, then one at a time. Clobbers CX, X0, X1.
#define MISMATCH(loop, found, tail, done) \
	XORL     AX, AX;          \
loop:                         \
	LEAQ     4(AX), CX;       \
	CMPQ     CX, DX;          \
	JHI      tail;            \
	MOVOU    (R12)(AX*4), X0; \
	MOVOU    (R13)(AX*4), X1; \
	PCMPEQL  X1, X0;          \
	MOVMSKPS X0, CX;          \
	XORL     $15, CX;         \
	JNZ      found;           \
	ADDQ     $4, AX;          \
	JMP      loop;            \
found:                        \
	BSFL     CX, CX;          \
	ADDQ     CX, AX;          \
	JMP      done;            \
tail:                         \
	CMPQ     AX, DX;          \
	JCC      done;            \
	MOVL     (R12)(AX*4), CX; \
	CMPL     CX, (R13)(AX*4); \
	JNE      done;            \
	INCQ     AX;              \
	JMP      tail;            \
done:

// func hcRunSSE41(s *hcSearcher, hi, lo, stop int, marks *uint64, pf int) (last, next int)
//
// hcSearcher.runGo (see its comment): from base = hi down to stop, search
// the best match at base (the distances xsize and 1 first unless
// lowEffort, then the hash chain from chain[base], at most iterMax-1
// links within the window, a candidate being compared in full only when
// its pixel at the current best length matches), mark base, and write the
// match into ol, extended to the left. Before each search, the first node
// of the chain of the search pf positions below is prefetched (pixels and
// link).
// SI = argb, DI = chain, BX = base, R8 = bestLen, R9 = &argb[bestLen],
// R10 = argb[base+bestLen], R11 = p, R12 = iter, R13 = minPos, DX = maxLen;
// locals: the best distance, the length that ends the search, and R12 and
// R13 while MISMATCH uses them.
TEXT ·hcRunSSE41(SB), NOSPLIT, $32-64
	MOVQ s+0(FP), AX
	MOVQ hcSearcher_argb(AX), SI
	MOVQ hcSearcher_chain(AX), DI
	MOVQ hi+8(FP), BX
	MOVQ $0, last+48(FP)

hc_search:
	CMPQ BX, stop+24(FP)
	JLT  hc_finish
	MOVQ BX, AX
	SUBQ pf+40(FP), AX
	JLT  hc_nopf
	MOVLQSX    (DI)(AX*4), CX
	TESTQ      CX, CX
	JLT        hc_nopf
	PREFETCHT0 (SI)(CX*4)
	PREFETCHT0 (DI)(CX*4)

hc_nopf:
	MOVQ s+0(FP), AX
	MOVQ hcSearcher_size(AX), DX
	SUBQ BX, DX
	DECQ DX                      // maxLen = min(size-1-base, maxLength)
	CMPQ DX, $4095
	JLE  hc_maxlen
	MOVQ $4095, DX

hc_maxlen:
	MOVQ DX, CX
	CMPQ CX, $256
	JLE  hc_lmax
	MOVQ $256, CX

hc_lmax:
	MOVQ    CX, lmax-32(SP)      // lengthMax = min(maxLen, 256)
	MOVQ    hcSearcher_iterMax(AX), R12
	MOVQ    BX, R13
	SUBQ    hcSearcher_window(AX), R13
	JGT     hc_minpos
	XORL    R13, R13             // minPos = max(base-window, 0)

hc_minpos:
	MOVLQSX (DI)(BX*4), R11
	XORL    R8, R8
	MOVQ    $0, dist-24(SP)
	MOVBLZX hcSearcher_lowEffort(AX), CX
	TESTL   CX, CX
	JNZ     hc_walkinit
	MOVQ    hcSearcher_xsize(AX), CX
	CMPQ    BX, CX
	JLT     hc_d1
	MOVQ    BX, AX
	SUBQ    CX, AX
	MOVL    (SI)(AX*4), AX
	CMPL    AX, (SI)(BX*4)
	JNE     hc_xsdone
	MOVQ    R12, iter-8(SP)
	MOVQ    R13, minpos-16(SP)
	MOVQ    BX, R12
	SUBQ    CX, R12
	LEAQ    (SI)(R12*4), R12
	LEAQ    (SI)(BX*4), R13
	MISMATCH(hc_m1loop, hc_m1found, hc_m1tail, hc_m1done)
	MOVQ    iter-8(SP), R12
	MOVQ    minpos-16(SP), R13
	TESTQ   AX, AX
	JZ      hc_xsdone
	MOVQ    AX, R8
	MOVQ    s+0(FP), CX
	MOVQ    hcSearcher_xsize(CX), CX
	MOVQ    CX, dist-24(SP)

hc_xsdone:
	DECQ R12

hc_d1:
	LEAQ  (BX)(R8*1), AX
	MOVL  -4(SI)(AX*4), CX
	CMPL  CX, (SI)(AX*4)
	JNE   hc_d1done
	MOVQ  R12, iter-8(SP)
	MOVQ  R13, minpos-16(SP)
	LEAQ  -4(SI)(BX*4), R12
	LEAQ  (SI)(BX*4), R13
	MISMATCH(hc_m2loop, hc_m2found, hc_m2tail, hc_m2done)
	MOVQ  iter-8(SP), R12
	MOVQ  minpos-16(SP), R13
	CMPQ  AX, R8
	JLE   hc_d1done
	MOVQ  AX, R8
	MOVQ  $1, dist-24(SP)

hc_d1done:
	DECQ R12
	CMPQ R8, $4095
	JNE  hc_walkinit
	LEAQ -1(R13), R11

hc_walkinit:
	LEAQ (BX)(R8*1), AX
	MOVL (SI)(AX*4), R10
	LEAQ (SI)(R8*4), R9

hc_walk:
	CMPQ R11, R13
	JLT  hc_found
	DECQ R12
	JLE  hc_found
	MOVL (R9)(R11*4), AX
	CMPL AX, R10
	JNE  hc_next
	MOVQ R12, iter-8(SP)
	MOVQ R13, minpos-16(SP)
	LEAQ (SI)(R11*4), R12
	LEAQ (SI)(BX*4), R13
	MISMATCH(hc_m3loop, hc_m3found, hc_m3tail, hc_m3done)
	MOVQ iter-8(SP), R12
	MOVQ minpos-16(SP), R13
	CMPQ AX, R8
	JLE  hc_next
	MOVQ AX, R8
	MOVQ BX, CX
	SUBQ R11, CX
	MOVQ CX, dist-24(SP)
	LEAQ (BX)(R8*1), CX
	MOVL (SI)(CX*4), R10
	LEAQ (SI)(R8*4), R9
	CMPQ R8, lmax-32(SP)
	JGE  hc_found

hc_next:
	MOVLQSX (DI)(R11*4), R11
	JMP     hc_walk

hc_found:
	MOVQ  marks+32(FP), AX
	TESTQ AX, AX
	JZ    hc_nomark
	MOVQ  BX, CX
	ANDL  $63, CX
	MOVL  $1, DX
	SHLQ  CX, DX
	MOVQ  BX, CX
	SHRQ  $6, CX
	ORQ   DX, (AX)(CX*8)

hc_nomark:
	MOVQ BX, last+48(FP)
	MOVQ s+0(FP), AX
	MOVQ hcSearcher_ol(AX), AX
	MOVQ dist-24(SP), CX
	MOVQ BX, DX                  // maxBase

hc_ext: // ol[base] = dist<<12 | bestLen, then extend to the left
	MOVQ  CX, R11
	SHLQ  $12, R11
	ORQ   R8, R11
	MOVL  R11, (AX)(BX*4)
	DECQ  BX
	TESTQ CX, CX
	JZ    hc_search
	TESTQ BX, BX
	JZ    hc_search
	CMPQ  BX, lo+16(FP)
	JLT   hc_search
	CMPQ  BX, CX
	JLT   hc_search
	MOVQ  BX, R11
	SUBQ  CX, R11
	MOVL  (SI)(R11*4), R12
	CMPL  R12, (SI)(BX*4)
	JNE   hc_search
	CMPQ  R8, $4095
	JNE   hc_extinc
	CMPQ  CX, $1
	JEQ   hc_ext
	LEAQ  4095(BX), R11
	CMPQ  R11, DX
	JLT   hc_search
	JMP   hc_ext

hc_extinc:
	INCQ R8
	MOVQ BX, DX
	JMP  hc_ext

hc_finish:
	MOVQ BX, next+56(FP)
	RET

// ---------------------------------------------------------------------------
// func cacheCountsSSE41(argb *uint32, npix int, refs *pixOrCopy, nrefs, maxBits int, h *cacheHistos) int
//
// The counting loop of bestCacheBitsGo (1 <= maxBits <= 10): for each
// literal, its channels are counted for the cache size 0, then for each
// cache size i from maxBits down to 1 (the key of size i is the key of
// maxBits shifted right) either the cache hit (lit[i][280+key]) or, on a
// miss, the channels, the cache taking the pixel; for each copy, its
// length prefix is counted for every size and each pixel that differs from
// the previous one goes into every cache. Returns the number of pixels
// covered, or -1 when a reference does not fit argb (or has a length the Go
// code would reject: 0 or above 4096) — the caller then runs the Go code.
// SI = argb, BX = pos, DI = ref, R8 = end of refs, DX = h, R9 = maxBits,
// CX = 32-maxBits, AX = pixel, R10 = key, R11 = cache size. (A branchless
// count of the cache sizes, every channel counter getting 1-hit, was 11%
// slower on the photo.)
TEXT ·cacheCountsSSE41(SB), NOSPLIT, $0-56
	MOVQ argb+0(FP), SI
	MOVQ refs+16(FP), DI
	MOVQ nrefs+24(FP), R8
	SHLQ $3, R8
	ADDQ DI, R8
	MOVQ maxBits+32(FP), R9
	MOVQ h+40(FP), DX
	MOVL $32, CX
	SUBL R9, CX
	XORL BX, BX

cc_ref:
	CMPQ  DI, R8
	JEQ   cc_ok
	MOVL  4(DI), R12
	TESTL $3, R12
	JNZ   cc_copy
	CMPQ  BX, npix+8(FP)
	JCC   cc_bad
	MOVL  (SI)(BX*4), AX
	INCQ  BX
	ADDQ  $8, DI
	MOVL  AX, R12
	SHRL  $24, R12
	INCL  cacheHistos_alpha(DX)(R12*4)
	MOVL  AX, R12
	SHRL  $16, R12
	ANDL  $0xff, R12
	INCL  cacheHistos_red(DX)(R12*4)
	MOVL  AX, R12
	ANDL  $0xff, R12
	INCL  cacheHistos_blue(DX)(R12*4)
	MOVL  AX, R12
	SHRL  $8, R12
	ANDL  $0xff, R12
	MOVQ  cacheHistos_lit(DX), R13
	INCL  (R13)(R12*4)
	MOVL  AX, R10
	IMUL3L $0x1e35a7bd, R10, R10
	SHRL  CX, R10
	MOVQ  R9, R11

cc_lvl:
	MOVQ cacheHistos_cache(DX)(R11*8), R13
	CMPL AX, (R13)(R10*4)
	JNE  cc_miss
	MOVQ cacheHistos_lit(DX)(R11*8), R13
	INCL 1120(R13)(R10*4) // lit[280+key]
	JMP  cc_lvlnext

cc_miss:
	MOVL AX, (R13)(R10*4)
	MOVQ R11, R13
	SHLQ $10, R13
	ADDQ DX, R13 // + 1024*i: the histograms of size i
	MOVL AX, R12
	SHRL $24, R12
	INCL cacheHistos_alpha(R13)(R12*4)
	MOVL AX, R12
	SHRL $16, R12
	ANDL $0xff, R12
	INCL cacheHistos_red(R13)(R12*4)
	MOVL AX, R12
	ANDL $0xff, R12
	INCL cacheHistos_blue(R13)(R12*4)
	MOVL AX, R12
	SHRL $8, R12
	ANDL $0xff, R12
	MOVQ cacheHistos_lit(DX)(R11*8), R13
	INCL (R13)(R12*4)

cc_lvlnext:
	SHRL $1, R10
	DECQ R11
	JNZ  cc_lvl
	JMP  cc_ref

cc_copy:
	SHRL $2, R12 // length
	JZ   cc_bad
	CMPL R12, $4096
	JHI  cc_bad
	MOVQ BX, R13
	ADDQ R12, R13
	CMPQ R13, npix+8(FP)
	JHI  cc_bad
	ADDQ $8, DI
	LEAL -1(R12), R13 // prefix code of the length
	CMPL R13, $4
	JCS  cc_code
	BSRL R13, R10
	LEAL -1(R10), AX
	BTL  AX, R13
	MOVL $0, AX
	ADCL R10, AX
	ADDL R10, AX
	MOVL AX, R13

cc_code:
	MOVQ R9, R11

cc_codeloop:
	MOVQ cacheHistos_lit(DX)(R11*8), R10
	INCL 1024(R10)(R13*4) // lit[256+code]
	DECQ R11
	JGE  cc_codeloop
	ADDQ BX, R12          // end of the copy
	MOVL (SI)(BX*4), AX
	NOTL AX               // prev

cc_px:
	MOVL (SI)(BX*4), R13
	CMPL R13, AX
	JEQ  cc_pxnext
	MOVL R13, AX
	MOVL R13, R10
	IMUL3L $0x1e35a7bd, R10, R10
	SHRL CX, R10
	MOVQ R9, R11

cc_ins:
	MOVQ cacheHistos_cache(DX)(R11*8), R13
	MOVL AX, (R13)(R10*4)
	SHRL $1, R10
	DECQ R11
	JNZ  cc_ins

cc_pxnext:
	INCQ BX
	CMPQ BX, R12
	JLT  cc_px
	JMP  cc_ref

cc_ok:
	MOVQ BX, ret+48(FP)
	RET

cc_bad:
	MOVQ $-1, ret+48(FP)
	RET

// ---------------------------------------------------------------------------
// func rleSSE41(argb *uint32, n, xsize, i int, refs *pixOrCopy, room int) (count, next int)
//
// The parse loop of backwardRefsRleGo from position i (1 <= i): at most
// room references are written to refs. A pixel equal to neither its left
// nor its top neighbor is a literal; otherwise the lengths of the matches
// at distances 1 and xsize (at most min(n-i, maxLength)) are measured with
// MISMATCH and the longer one of at least minLength (distance 1 on a tie)
// is taken, else a literal. Returns the references written and the next
// position.
// SI = argb, BX = i, R8 = n, R9 = xsize, DI = next reference, R10 = end of
// refs, R11 = length at distance 1.
TEXT ·rleSSE41(SB), NOSPLIT, $0-64
	MOVQ argb+0(FP), SI
	MOVQ n+8(FP), R8
	MOVQ xsize+16(FP), R9
	MOVQ i+24(FP), BX
	MOVQ refs+32(FP), DI
	MOVQ room+40(FP), R10
	SHLQ $3, R10
	ADDQ DI, R10

rle_loop:
	CMPQ BX, R8
	JGE  rle_done
	CMPQ DI, R10
	JEQ  rle_done
	MOVL (SI)(BX*4), AX
	CMPL AX, -4(SI)(BX*4)
	JEQ  rle_match
	CMPQ BX, R9
	JLT  rle_lit
	MOVQ BX, CX
	SUBQ R9, CX
	CMPL AX, (SI)(CX*4)
	JEQ  rle_match

rle_lit:
	MOVL AX, (DI)
	MOVL $4, 4(DI) // 1<<2 | modeLiteral
	ADDQ $8, DI
	INCQ BX
	JMP  rle_loop

rle_match:
	MOVQ R8, DX
	SUBQ BX, DX
	CMPQ DX, $4095
	JLE  rle_maxlen
	MOVQ $4095, DX

rle_maxlen:
	LEAQ (SI)(BX*4), R12
	LEAQ -4(SI)(BX*4), R13
	MISMATCH(rle_m1loop, rle_m1found, rle_m1tail, rle_m1done)
	MOVQ AX, R11
	XORL AX, AX
	CMPQ BX, R9
	JLT  rle_choose
	LEAQ (SI)(BX*4), R12
	MOVQ BX, CX
	SUBQ R9, CX
	LEAQ (SI)(CX*4), R13
	MISMATCH(rle_m2loop, rle_m2found, rle_m2tail, rle_m2done)

rle_choose:
	CMPQ R11, AX
	JLT  rle_prow
	CMPQ R11, $4
	JLT  rle_prow
	MOVL $1, (DI)
	MOVL R11, CX
	SHLL $2, CX
	ORL  $2, CX // len<<2 | modeCopy
	MOVL CX, 4(DI)
	ADDQ $8, DI
	ADDQ R11, BX
	JMP  rle_loop

rle_prow:
	CMPQ AX, $4
	JLT  rle_lit2
	MOVL R9, (DI)
	MOVL AX, CX
	SHLL $2, CX
	ORL  $2, CX
	MOVL CX, 4(DI)
	ADDQ $8, DI
	ADDQ AX, BX
	JMP  rle_loop

rle_lit2:
	MOVL (SI)(BX*4), AX
	JMP  rle_lit

rle_done:
	MOVQ DI, AX
	SUBQ refs+32(FP), AX
	SHRQ $3, AX
	MOVQ AX, count+48(FP)
	MOVQ BX, next+56(FP)
	RET

// ---------------------------------------------------------------------------
// Cross-color statistics (libwebp CollectColorRedTransforms_SSE2 and
// CollectColorBlueTransforms_SSE2 compute the transformed values the same
// way, then add them to the histogram one by one).

// CCBIN adds the count k of the block to the bin in the low byte of AX,
// then moves to the next byte. Clobbers DX, R12.
#define CCBIN(k) \
	MOVBLZX AL, DX;               \
	MOVL    k*4(R10)(BX*4), R12;  \
	ADDL    R12, (DI)(DX*4);      \
	SHRQ    $8, AX

// func ccHistoSSE41(histo *[256]int32, d, s1, s2 *int8, c *int32, n int, m1, m2 int)
//
// ccHistoBlueGo with b = d, g = s1, r = s2, mg = m1, mr = m2 (ccHistoRedGo
// when m2 = 0): histo[uint8(d[i] - (m1*s1[i])>>5 - (m2*s2[i])>>5)] += c[i]
// for i < n (n >= 1). 8 entries per step: PMOVSXBW widens the int8 to
// 16-bit lanes, where m*s (in [-16256, 16384]) and the difference (in
// [-1152, 1143]) are exact, PMULLW/PSRAW/PSUBW compute it, PSHUFB packs the
// low bytes (the bins) into AX, and CCBIN adds the 8 counts. The last
// n%8 entries are done one by one with the same integer operations.
// DI = histo, SI = d, R8 = s1, R9 = s2, R10 = c, CX = n, BX = i,
// R11 = m1, R13 = m2, X6/X7 = m1/m2 in every 16-bit lane, X5 = shuffle.
TEXT ·ccHistoSSE41(SB), NOSPLIT, $0-64
	MOVQ    histo+0(FP), DI
	MOVQ    d+8(FP), SI
	MOVQ    s1+16(FP), R8
	MOVQ    s2+24(FP), R9
	MOVQ    c+32(FP), R10
	MOVQ    n+40(FP), CX
	MOVQ    m1+48(FP), R11
	MOVQ    m2+56(FP), R13
	MOVQ    R11, X6
	PSHUFLW $0, X6, X6
	PSHUFD  $0, X6, X6
	MOVQ    R13, X7
	PSHUFLW $0, X7, X7
	PSHUFD  $0, X7, X7
	MOVQ    $0x0e0c0a0806040200, AX
	MOVQ    AX, X5
	XORQ    BX, BX

cc_block:
	LEAQ     8(BX), DX
	CMPQ     DX, CX
	JHI      cc_tail
	PMOVSXBW (SI)(BX*1), X0
	PMOVSXBW (R8)(BX*1), X1
	PMOVSXBW (R9)(BX*1), X2
	PMULLW   X6, X1
	PMULLW   X7, X2
	PSRAW    $5, X1
	PSRAW    $5, X2
	PSUBW    X1, X0
	PSUBW    X2, X0
	PSHUFB   X5, X0
	MOVQ     X0, AX
	CCBIN(0)
	CCBIN(1)
	CCBIN(2)
	CCBIN(3)
	CCBIN(4)
	CCBIN(5)
	CCBIN(6)
	CCBIN(7)
	ADDQ     $8, BX
	JMP      cc_block

cc_tail:
	CMPQ    BX, CX
	JAE     cc_done
	MOVBQSX (SI)(BX*1), AX
	MOVBQSX (R8)(BX*1), DX
	IMULQ   R11, DX
	SARQ    $5, DX
	SUBQ    DX, AX
	MOVBQSX (R9)(BX*1), DX
	IMULQ   R13, DX
	SARQ    $5, DX
	SUBQ    DX, AX
	MOVBLZX AL, AX
	MOVL    (R10)(BX*4), DX
	ADDL    DX, (DI)(AX*4)
	INCQ    BX
	JMP     cc_tail

cc_done:
	RET

// ---------------------------------------------------------------------------
// func lz77ReachSSE41(ol *uint32, i, jMax, n int) (l int)
//
// lz77ReachGo for jMax-i >= 4, 0 < n < 2^30, jMax < 2^30 (libwebp
// BackwardReferencesLz77 runs the same loop in C): the result is j-i for
// the first j in (i, jMax] with the largest min(reach(j), n), reach(j)
// being j+lj for lj = ol[j]&0xfff >= 4, else j+1 (the Go loop stops at
// the first j reaching n, the first one with the largest capped reach;
// otherwise it keeps the first strict maximum). 4 positions per block:
// the capped reaches (PCMPGTL/PAND/PANDN/POR select lj or 1, PADDL adds j,
// PMINUD caps), their maximum (PSHUFD/PMAXUD) compared with the best so
// far, and only when it is larger the first lane equal to it (PCMPEQL,
// MOVMSKPS, BSF). The last partial block is the block ending at jMax: its
// lanes already seen are at most the best, so they cannot match.
// SI = ol, R8 = i, R9 = jMax, R10 = n, BX = block start, R11 = best capped
// reach, DX = its position, R12 = jMax-3, R13 = 1 in the last block,
// X3 = the positions of the block, X2 = 4s, X4 = n, X5 = 1s, X6 = 3s,
// X7 = 0xfff, X9 = (0, 1, 2, 3).
TEXT ·lz77ReachSSE41(SB), NOSPLIT, $0-40
	MOVQ   ol+0(FP), SI
	MOVQ   i+8(FP), R8
	MOVQ   jMax+16(FP), R9
	MOVQ   n+24(FP), R10
	MOVL   $0xfff, AX
	MOVQ   AX, X7
	PSHUFD $0, X7, X7
	MOVL   $3, AX
	MOVQ   AX, X6
	PSHUFD $0, X6, X6
	MOVL   $1, AX
	MOVQ   AX, X5
	PSHUFD $0, X5, X5
	MOVQ   R10, X4
	PSHUFD $0, X4, X4
	MOVL   $4, AX
	MOVQ   AX, X2
	PSHUFD $0, X2, X2
	MOVQ   $0x0000000100000000, AX
	MOVQ   AX, X9
	MOVQ   $0x0000000300000002, AX
	PINSRQ $1, AX, X9
	LEAQ   1(R8), BX
	MOVQ   BX, X3
	PSHUFD $0, X3, X3
	PADDL  X9, X3
	XORL   R11, R11
	MOVQ   BX, DX
	LEAQ   -3(R9), R12
	XORL   R13, R13

lzr_loop:
	CMPQ BX, R12
	JGT  lzr_last

lzr_block:
	MOVOU    (SI)(BX*4), X0
	PAND     X7, X0
	MOVO     X0, X1
	PCMPGTL  X6, X1
	PAND     X1, X0
	PANDN    X5, X1
	POR      X1, X0
	PADDL    X3, X0
	PMINUD   X4, X0
	PSHUFD   $0x4e, X0, X1
	PMAXUD   X0, X1
	PSHUFD   $0xb1, X1, X8
	PMAXUD   X8, X1
	MOVQ     X1, AX
	MOVL     AX, AX
	CMPQ     AX, R11
	JLE      lzr_next
	PCMPEQL  X1, X0
	MOVMSKPS X0, CX
	BSFL     CX, CX
	LEAQ     (BX)(CX*1), DX
	MOVQ     AX, R11
	CMPQ     R11, R10
	JGE      lzr_done

lzr_next:
	TESTQ R13, R13
	JNZ   lzr_done
	ADDQ  $4, BX
	PADDL X2, X3
	JMP   lzr_loop

lzr_last:
	CMPQ   BX, R9
	JGT    lzr_done
	MOVQ   R12, BX
	MOVQ   BX, X3
	PSHUFD $0, X3, X3
	PADDL  X9, X3
	MOVL   $1, R13
	JMP    lzr_block

lzr_done:
	SUBQ R8, DX
	MOVQ DX, l+32(FP)
	RET
