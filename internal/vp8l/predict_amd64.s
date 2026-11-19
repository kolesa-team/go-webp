//go:build amd64 && !purego

// SSE4.1 and AVX2 kernels of the inverse predictor transform. Algorithms
// adapted from libwebp (BSD-3-Clause), src/dsp/lossless_sse2.c
// (PredictorAdd0_SSE2 to PredictorAdd12_SSE2) and lossless_avx2.c; the Go
// reference predictRowGo (transform.go) is the specification.
//
// Pixels are ARGB words, added per byte modulo 256 (PADDB). Every loop
// processes exact pixel counts: 8 (AVX2), 4, then one pixel at a time in
// the low 32 bits of a vector register, so nothing is read or written
// outside in[x:width], out[x-1:width] and up[x-1:width+1].
//
// The modes whose prediction depends on the left pixel through averages
// (5, 6, 7, 10) run on the complements of the pixels: with avg(a, b) the
// floor average of predictRun and PAVGB the rounding-up one,
// ^avg(a, b) = PAVGB(^a, ^b) per byte, so that the serial chain of a pixel
// is one PAVGB per average and one PSUBB: with nL = ^L,
// ^(res + avg(...)) = PAVGB(...) - res. The complements of the top pixels
// are computed four at a time, off the serial path.

#include "textflag.h"

// prBlack: 0xff000000 in each pixel (mode 0).
DATA prBlack<>+0(SB)/8, $0xff000000ff000000
DATA prBlack<>+8(SB)/8, $0xff000000ff000000
GLOBL prBlack<>(SB), RODATA|NOPTR, $16

// prOne: 1 in each byte (the low bit of floor averages).
DATA prOne<>+0(SB)/8, $0x0101010101010101
DATA prOne<>+8(SB)/8, $0x0101010101010101
GLOBL prOne<>(SB), RODATA|NOPTR, $16

// SERIAL4 runs the serial step PSTEP on four pixels, the operands of the
// next pixel moved to the low lane by PSHIFT; X1 is ^L (lane 0), and the
// four results, complemented back, are stored at (DI).
#define SERIAL4 \
	PSTEP              \
	MOVO       X1, X4  \
	PSHIFT             \
	PSTEP              \
	PUNPCKLLQ  X1, X4  \
	PSHIFT             \
	PSTEP              \
	MOVO       X1, X5  \
	PSHIFT             \
	PSTEP              \
	PUNPCKLLQ  X1, X5  \
	PUNPCKLQDQ X5, X4  \
	PXOR       X13, X4 \
	MOVOU      X4, (DI)

// SERIAL1 stores the complement of ^L (X1) at (DI) and advances one pixel.
#define SERIAL1 \
	MOVO X1, X4  \
	PXOR X13, X4 \
	MOVL X4, (DI) \
	ADDQ $4, SI  \
	ADDQ $4, DI  \
	ADDQ $4, R11

// func predictRowSSE41(in, out, up, modes *uint32, x, width, bits int) int
//
// Undoes the predictor transform on the pixels x to width-1 of a row
// (x >= 1; out[0:x] done), the tile of 1<<bits pixels at x using the mode
// in bits 8-11 of modes[x>>bits]. up is the previous output row extended
// by out[0] (width+1 pixels). in may alias out. It returns width, or the
// first pixel of a tile in mode 13, which the caller does in Go.
// libwebp: VP8LPredictorsAdd_SSE2, modes 0-12.
//
//   0, 14, 15  in + 0xff000000                vector
//   1          L: prefix sums of 4 pixels      vector, serial carry
//   2, 3, 4    in + T, TR, TL                  vector
//   8, 9       in + avg(TL, T), avg(T, TR)     vector
//   5, 6, 7    in + avg(avg(L, TR), T), avg(L, TL), avg(L, T)
//   10         in + avg(avg(L, TL), avg(T, TR))
//              one pixel at a time on complements (see above)
//   11         Select: one pixel at a time (see pr11)
//   12         ClampedAddSubtractFull: one pixel at a time, with
//              saturating byte operations (see pr12)
//
// AX in, BX out, R10 up, DX modes, R8 x (start of the next tile), R9 width,
// CX bits; per tile SI &in[x], DI &out[x], R11 &up[x], R12 a second top
// pointer, R13 pixels left in the tile. X14 prBlack, X13 all ones,
// X12 prOne.
TEXT ·predictRowSSE41(SB), NOSPLIT, $0-64
	MOVQ    in+0(FP), AX
	MOVQ    out+8(FP), BX
	MOVQ    up+16(FP), R10
	MOVQ    modes+24(FP), DX
	MOVQ    x+32(FP), R8
	MOVQ    width+40(FP), R9
	MOVQ    bits+48(FP), CX
	MOVOU   prBlack<>(SB), X14
	PCMPEQB X13, X13
	MOVOU   prOne<>(SB), X12

prtile:
	CMPQ    R8, R9
	JGE     prdone
	MOVQ    R8, R13
	SHRQ    CX, R13
	MOVL    (DX)(R13*4), R12
	SHRL    $8, R12
	ANDL    $15, R12             // mode
	INCQ    R13
	SHLQ    CX, R13
	CMPQ    R13, R9
	CMOVQGT R9, R13              // end of the tile
	CMPQ    R12, $13
	JEQ     prgo
	LEAQ    (AX)(R8*4), SI
	LEAQ    (BX)(R8*4), DI
	LEAQ    (R10)(R8*4), R11
	SUBQ    R8, R13              // pixels in the tile
	ADDQ    R13, R8
	CMPQ    R12, $1
	JEQ     pr1
	CMPQ    R12, $11
	JEQ     pr11
	CMPQ    R12, $12
	JEQ     pr12
	CMPQ    R12, $2
	JEQ     pr2
	CMPQ    R12, $7
	JEQ     pr7
	CMPQ    R12, $4
	JEQ     pr4
	CMPQ    R12, $5
	JEQ     pr5
	CMPQ    R12, $3
	JEQ     pr3
	CMPQ    R12, $6
	JEQ     pr6
	CMPQ    R12, $8
	JEQ     pr8
	CMPQ    R12, $9
	JEQ     pr9
	CMPQ    R12, $10
	JEQ     pr10
	JMP     pr0                  // 0, 14, 15

prgo:
	MOVQ R8, ret+56(FP)
	RET

prdone:
	MOVQ R9, ret+56(FP)
	RET

	// Mode 0 (and 14, 15): in + 0xff000000.
pr0:
	CMPQ  R13, $4
	JB    pr0s
	MOVOU (SI), X0
	PADDB X14, X0
	MOVOU X0, (DI)
	ADDQ  $16, SI
	ADDQ  $16, DI
	SUBQ  $4, R13
	JMP   pr0

pr0s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  (SI), X0
	PADDB X14, X0
	MOVL  X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  R13
	JMP   pr0s

	// Mode 1, L: per 4 pixels a b c d, the prefix sums a, a+b, a+b+c,
	// a+b+c+d from two shifted adds, plus L in all lanes; the new L is
	// lane 3. X1 = L (all lanes). With 8 pixels, the second group adds the
	// total of the first off the serial path: one add and one broadcast
	// per 8 pixels on the serial path.
pr1:
	MOVL   -4(DI), X1
	PSHUFD $0, X1, X1

pr1v8:
	CMPQ   R13, $8
	JB     pr1v
	MOVOU  (SI), X0
	MOVOU  16(SI), X3
	MOVO   X0, X2
	MOVO   X3, X4
	PSLLDQ $4, X2
	PSLLDQ $4, X4
	PADDB  X2, X0
	PADDB  X4, X3
	MOVO   X0, X2
	MOVO   X3, X4
	PSLLDQ $8, X2
	PSLLDQ $8, X4
	PADDB  X2, X0
	PADDB  X4, X3
	PSHUFD $0xff, X0, X2
	PADDB  X2, X3
	PADDB  X1, X0
	PADDB  X1, X3
	PSHUFD $0xff, X3, X1
	MOVOU  X0, (DI)
	MOVOU  X3, 16(DI)
	ADDQ   $32, SI
	ADDQ   $32, DI
	SUBQ   $8, R13
	JMP    pr1v8

pr1v:
	CMPQ   R13, $4
	JB     pr1s
	MOVOU  (SI), X0
	MOVO   X0, X2
	PSLLDQ $4, X2
	PADDB  X2, X0
	MOVO   X0, X2
	PSLLDQ $8, X2
	PADDB  X2, X0
	PADDB  X1, X0
	PSHUFD $0xff, X0, X1
	MOVOU  X0, (DI)
	ADDQ   $16, SI
	ADDQ   $16, DI
	SUBQ   $4, R13

pr1s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  (SI), X0
	PADDB X0, X1
	MOVL  X1, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  R13
	JMP   pr1s

	// Modes 2, 3, 4: in + the top pixel at R12.
pr2:
	MOVQ R11, R12
	JMP  prtop

pr3:
	LEAQ 4(R11), R12
	JMP  prtop

pr4:
	LEAQ -4(R11), R12

prtop:
	CMPQ  R13, $4
	JB    prtops
	MOVOU (SI), X0
	MOVOU (R12), X1
	PADDB X1, X0
	MOVOU X0, (DI)
	ADDQ  $16, SI
	ADDQ  $16, R12
	ADDQ  $16, DI
	SUBQ  $4, R13
	JMP   prtop

prtops:
	TESTQ R13, R13
	JZ    prtile
	MOVL  (SI), X0
	MOVL  (R12), X1
	PADDB X1, X0
	MOVL  X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R12
	ADDQ  $4, DI
	DECQ  R13
	JMP   prtops

	// Modes 8, 9: in + avg(the top pixels at R12 and R11), the floor
	// average PAVGB(a, b) - ((a ^ b) & 1) (libwebp Average2_m128i).
pr8:
	LEAQ -4(R11), R12
	JMP  pravg

pr9:
	MOVQ R11, R12
	ADDQ $4, R11

pravg:
	CMPQ  R13, $4
	JB    pravgs
	MOVOU (R12), X1
	MOVOU (R11), X2
	MOVO  X1, X3
	PXOR  X2, X3
	PAND  X12, X3
	PAVGB X2, X1
	PSUBB X3, X1
	MOVOU (SI), X0
	PADDB X1, X0
	MOVOU X0, (DI)
	ADDQ  $16, SI
	ADDQ  $16, R12
	ADDQ  $16, R11
	ADDQ  $16, DI
	SUBQ  $4, R13
	JMP   pravg

pravgs:
	TESTQ R13, R13
	JZ    prtile
	MOVL  (R12), X1
	MOVL  (R11), X2
	MOVO  X1, X3
	PXOR  X2, X3
	PAND  X12, X3
	PAVGB X2, X1
	PSUBB X3, X1
	MOVL  (SI), X0
	PADDB X1, X0
	MOVL  X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R12
	ADDQ  $4, R11
	ADDQ  $4, DI
	DECQ  R13
	JMP   pravgs

	// Mode 7: ^L' = PAVGB(^L, ^T) - res. X2 ^T, X3 residuals.
#define PSTEP PAVGB X2, X1; PSUBB X3, X1
#define PSHIFT PSRLDQ $4, X2; PSRLDQ $4, X3
pr7:
	MOVL -4(DI), X1
	PXOR X13, X1

pr7v:
	CMPQ  R13, $4
	JB    pr7s
	MOVOU (R11), X2
	PXOR  X13, X2
	MOVOU (SI), X3
	SERIAL4
	ADDQ  $16, SI
	ADDQ  $16, DI
	ADDQ  $16, R11
	SUBQ  $4, R13
	JMP   pr7v

pr7s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  (R11), X2
	PXOR  X13, X2
	MOVL  (SI), X3
	PSTEP
	SERIAL1
	DECQ  R13
	JMP   pr7s
#undef PSTEP
#undef PSHIFT

	// Mode 6: ^L' = PAVGB(^L, ^TL) - res. X2 ^TL, X3 residuals.
#define PSTEP PAVGB X2, X1; PSUBB X3, X1
#define PSHIFT PSRLDQ $4, X2; PSRLDQ $4, X3
pr6:
	MOVL -4(DI), X1
	PXOR X13, X1

pr6v:
	CMPQ  R13, $4
	JB    pr6s
	MOVOU -4(R11), X2
	PXOR  X13, X2
	MOVOU (SI), X3
	SERIAL4
	ADDQ  $16, SI
	ADDQ  $16, DI
	ADDQ  $16, R11
	SUBQ  $4, R13
	JMP   pr6v

pr6s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  -4(R11), X2
	PXOR  X13, X2
	MOVL  (SI), X3
	PSTEP
	SERIAL1
	DECQ  R13
	JMP   pr6s
#undef PSTEP
#undef PSHIFT

	// Mode 5: ^L' = PAVGB(PAVGB(^L, ^TR), ^T) - res. X2 ^TR, X6 ^T,
	// X3 residuals.
#define PSTEP PAVGB X2, X1; PAVGB X6, X1; PSUBB X3, X1
#define PSHIFT PSRLDQ $4, X2; PSRLDQ $4, X6; PSRLDQ $4, X3
pr5:
	MOVL -4(DI), X1
	PXOR X13, X1

pr5v:
	CMPQ  R13, $4
	JB    pr5s
	MOVOU 4(R11), X2
	PXOR  X13, X2
	MOVOU (R11), X6
	PXOR  X13, X6
	MOVOU (SI), X3
	SERIAL4
	ADDQ  $16, SI
	ADDQ  $16, DI
	ADDQ  $16, R11
	SUBQ  $4, R13
	JMP   pr5v

pr5s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  4(R11), X2
	PXOR  X13, X2
	MOVL  (R11), X6
	PXOR  X13, X6
	MOVL  (SI), X3
	PSTEP
	SERIAL1
	DECQ  R13
	JMP   pr5s
#undef PSTEP
#undef PSHIFT

	// Mode 10: ^L' = PAVGB(PAVGB(^L, ^TL), PAVGB(^T, ^TR)) - res, the
	// second average off the serial path. X2 ^TL, X6 PAVGB(^T, ^TR),
	// X3 residuals.
#define PSTEP PAVGB X2, X1; PAVGB X6, X1; PSUBB X3, X1
#define PSHIFT PSRLDQ $4, X2; PSRLDQ $4, X6; PSRLDQ $4, X3
pr10:
	MOVL -4(DI), X1
	PXOR X13, X1

pr10v:
	CMPQ  R13, $4
	JB    pr10s
	MOVOU -4(R11), X2
	PXOR  X13, X2
	MOVOU (R11), X6
	MOVOU 4(R11), X7
	PXOR  X13, X6
	PXOR  X13, X7
	PAVGB X7, X6
	MOVOU (SI), X3
	SERIAL4
	ADDQ  $16, SI
	ADDQ  $16, DI
	ADDQ  $16, R11
	SUBQ  $4, R13
	JMP   pr10v

pr10s:
	TESTQ R13, R13
	JZ    prtile
	MOVL  -4(R11), X2
	PXOR  X13, X2
	MOVL  (R11), X6
	MOVL  4(R11), X7
	PXOR  X13, X6
	PXOR  X13, X7
	PAVGB X7, X6
	MOVL  (SI), X3
	PSTEP
	SERIAL1
	DECQ  R13
	JMP   pr10s
#undef PSTEP
#undef PSHIFT

	// Mode 11, Select: out = withT = res + T if pb = sum over the channels
	// of |L - TL| is at most pa = sum |T - TL|, else withL = res + L
	// (libwebp Select, PSADBW sums of pixels in the low lane, zero above).
	// The pb of the next pixel (whose TL is this T) is computed for both
	// outcomes before the decision: sum |withT - T| and sum |withL - T|,
	// picked by the same mask as the output, so that the serial chain
	// from one decision to the next is a compare and a blend.
	//
	// X1 L, X2 pb, X3 T, X4 TL, X5 res then withL, X6 pa, X7 withT then
	// the output, X8 the next pb, X9 sum |withL - T|, X0 the mask.
pr11:
	MOVL   -4(DI), X1
	MOVL   -4(R11), X4
	MOVO   X1, X2
	PSADBW X4, X2

pr11l:
	MOVL     (R11), X3
	MOVL     -4(R11), X4
	MOVL     (SI), X5
	MOVO     X3, X6
	PSADBW   X4, X6
	MOVO     X5, X7
	PADDB    X3, X7
	MOVO     X7, X8
	PSADBW   X3, X8
	PADDB    X1, X5
	MOVO     X5, X9
	PSADBW   X3, X9
	MOVO     X2, X0
	PCMPGTL  X6, X0
	PBLENDVB X0, X5, X7
	PBLENDVB X0, X9, X8
	MOVL     X7, (DI)
	MOVO     X7, X1
	MOVO     X8, X2
	ADDQ     $4, SI
	ADDQ     $4, DI
	ADDQ     $4, R11
	DECQ     R13
	JNZ      pr11l
	JMP      prtile

	// Mode 12, ClampedAddSubtractFull: per channel
	// L' = clamp(L + T - TL) + res (libwebp PredictorAdd12_SSE2, which
	// widens L to 16 bits). Here in bytes: with dpos = T -sat TL and
	// dneg = TL -sat T (one of them 0), clamp(L + T - TL) =
	// (L +sat dpos) -sat dneg, so the serial chain of a pixel is PADDUSB,
	// PSUBUSB and PADDB; T, TL and the residual are loaded per pixel
	// (MOVL), off that chain. Only the low lane of X1 matters.
	//
	// X1 L, X2 T then dpos, X4 TL then dneg, X3 residual.
#define P12(off) \
	MOVL    off(R11), X2   \
	MOVL    off-4(R11), X4 \
	MOVO    X2, X5         \
	PSUBUSB X4, X5         \
	PSUBUSB X2, X4         \
	MOVL    off(SI), X3    \
	PADDUSB X5, X1         \
	PSUBUSB X4, X1         \
	PADDB   X3, X1         \
	MOVL    X1, off(DI)

pr12:
	MOVL -4(DI), X1

pr12v:
	CMPQ R13, $4
	JB   pr12s
	P12(0)
	P12(4)
	P12(8)
	P12(12)
	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, R11
	SUBQ $4, R13
	JMP  pr12v

pr12s:
	TESTQ R13, R13
	JZ    prtile
	P12(0)
	ADDQ  $4, SI
	ADDQ  $4, DI
	ADDQ  $4, R11
	DECQ  R13
	JMP   pr12s

#undef P12
