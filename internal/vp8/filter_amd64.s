//go:build amd64 && !purego

#include "textflag.h"

// SSE4.1 in-loop filters, adapted from libwebp (BSD-3-Clause)
// src/dsp/dec_sse2.c (NeedsFilter_SSE2, GetNotHEV_SSE2, DoFilter2/4/6_SSE2,
// ComplexMask_SSE2, Load16x4_SSE2/Store16x4_SSE2 and the edge functions)
// and bit-exact with the Go code in filter.go. Legacy SSE encodings only.
//
// One call filters all edges of one macroblock, in libwebp's DoFilter order
// per plane: left edge, inner vertical edges, top edge, inner horizontal
// edges. Luma and chroma are independent, so luma is done first; U and V
// share registers, U in the low and V in the high 8 bytes (libwebp's *8
// variants).
//
// Register conventions of the filter macros:
//   X0-X7    p3, p2, p1, p0, q0, q1, q2, q3 (the 8 rows or columns across
//            the edge, one byte per line along the edge)
//   X8, X9   filter mask and not-hev mask (COMPLEX_MASK)
//   X10-X15  temporaries
// The broadcast thresholds live in the frame: macroblock edge threshold
// (limit+4) at 0(SP), inner edge threshold (limit) at 16(SP), interior
// limit at 32(SP), hev threshold at 48(SP). The constants below are
// 32-byte aligned (the linker aligns data symbols to their size, up to
// 32), as legacy SSE memory operands require; the frame is not, so it is
// read with MOVOU.

DATA fK80<>+0(SB)/8, $0x8080808080808080
DATA fK80<>+8(SB)/8, $0x8080808080808080
DATA fK80<>+16(SB)/8, $0x8080808080808080
DATA fK80<>+24(SB)/8, $0x8080808080808080
GLOBL fK80<>(SB), RODATA|NOPTR, $32

DATA fK3<>+0(SB)/8, $0x0303030303030303
DATA fK3<>+8(SB)/8, $0x0303030303030303
DATA fK3<>+16(SB)/8, $0x0303030303030303
DATA fK3<>+24(SB)/8, $0x0303030303030303
GLOBL fK3<>(SB), RODATA|NOPTR, $32

DATA fK4<>+0(SB)/8, $0x0404040404040404
DATA fK4<>+8(SB)/8, $0x0404040404040404
DATA fK4<>+16(SB)/8, $0x0404040404040404
DATA fK4<>+24(SB)/8, $0x0404040404040404
GLOBL fK4<>(SB), RODATA|NOPTR, $32

DATA fKFE<>+0(SB)/8, $0xFEFEFEFEFEFEFEFE
DATA fKFE<>+8(SB)/8, $0xFEFEFEFEFEFEFEFE
DATA fKFE<>+16(SB)/8, $0xFEFEFEFEFEFEFEFE
DATA fKFE<>+24(SB)/8, $0xFEFEFEFEFEFEFEFE
GLOBL fKFE<>(SB), RODATA|NOPTR, $32

DATA fK1F<>+0(SB)/8, $0x1F1F1F1F1F1F1F1F
DATA fK1F<>+8(SB)/8, $0x1F1F1F1F1F1F1F1F
DATA fK1F<>+16(SB)/8, $0x1F1F1F1F1F1F1F1F
DATA fK1F<>+24(SB)/8, $0x1F1F1F1F1F1F1F1F
GLOBL fK1F<>(SB), RODATA|NOPTR, $32

DATA fK10<>+0(SB)/8, $0x1010101010101010
DATA fK10<>+8(SB)/8, $0x1010101010101010
DATA fK10<>+16(SB)/8, $0x1010101010101010
DATA fK10<>+24(SB)/8, $0x1010101010101010
GLOBL fK10<>(SB), RODATA|NOPTR, $32

DATA fK40<>+0(SB)/8, $0x4040404040404040
DATA fK40<>+8(SB)/8, $0x4040404040404040
DATA fK40<>+16(SB)/8, $0x4040404040404040
DATA fK40<>+24(SB)/8, $0x4040404040404040
GLOBL fK40<>(SB), RODATA|NOPTR, $32

DATA fK0900<>+0(SB)/8, $0x0900090009000900
DATA fK0900<>+8(SB)/8, $0x0900090009000900
DATA fK0900<>+16(SB)/8, $0x0900090009000900
DATA fK0900<>+24(SB)/8, $0x0900090009000900
GLOBL fK0900<>(SB), RODATA|NOPTR, $32

DATA fK63<>+0(SB)/8, $0x003F003F003F003F
DATA fK63<>+8(SB)/8, $0x003F003F003F003F
DATA fK63<>+16(SB)/8, $0x003F003F003F003F
DATA fK63<>+24(SB)/8, $0x003F003F003F003F
GLOBL fK63<>(SB), RODATA|NOPTR, $32

// PSHUFB control transposing the 4x4 bytes of each 128-bit lane (row
// major to column major): out[4c+r] = in[4r+c].
DATA fT4x4<>+0(SB)/8, $0x0D0905010C080400
DATA fT4x4<>+8(SB)/8, $0x0F0B07030E0A0602
DATA fT4x4<>+16(SB)/8, $0x0D0905010C080400
DATA fT4x4<>+24(SB)/8, $0x0F0B07030E0A0602
GLOBL fT4x4<>(SB), RODATA|NOPTR, $32

// ABSDIFF sets D = |A - B| (bytes, unsigned), libwebp's MM_ABS; T is a
// temporary.
#define ABSDIFF(A, B, D, T) \
	MOVO A, D \
	PSUBUSB B, D \
	MOVO B, T \
	PSUBUSB A, T \
	POR T, D

// SSHR3 is the signed byte shift X >>= 3 (libwebp's SignedShift8b_SSE2)
// with X15 = 0x80 bytes: ((x ^ 0x80) >> 3) - 16 on unsigned bytes is
// floor(x / 8).
#define SSHR3(X) \
	PXOR X15, X \
	PSRLW $3, X \
	PAND fK1F<>(SB), X \
	PSUBB fK10<>(SB), X

// COMPLEX_MASK is ComplexMask_SSE2 and GetNotHEV_SSE2 on X0-X7 (unsigned):
// leaves in X8 the filter mask
//   max(|p3-p2|, |p2-p1|, |p1-p0|, |q1-q0|, |q2-q1|, |q3-q2|) <= ilevel &&
//   2*|p0-q0| + |p1-q1|/2 <= thresh  (saturating; == 4a+b <= 2*thresh+1)
// and in X9 the not-hev mask max(|p1-p0|, |q1-q0|) <= hev_thresh, where
// thresh is the frame threshold at THROFF(SP). The two "<= limit" tests
// are folded into one compare of the OR of the saturated differences with
// zero. Clobbers X10-X15.
#define COMPLEX_MASK(THROFF) \
	ABSDIFF(X2, X3, X10, X15) \
	ABSDIFF(X5, X4, X11, X15) \
	MOVO X10, X9 \
	PMAXUB X11, X9 \
	ABSDIFF(X0, X1, X12, X15) \
	PMAXUB X12, X10 \
	ABSDIFF(X1, X2, X12, X15) \
	PMAXUB X12, X10 \
	ABSDIFF(X7, X6, X12, X15) \
	PMAXUB X12, X11 \
	ABSDIFF(X6, X5, X12, X15) \
	PMAXUB X12, X11 \
	PMAXUB X11, X10 \
	MOVOU 32(SP), X15 \
	PSUBUSB X15, X10 \
	ABSDIFF(X3, X4, X11, X15) \
	PADDUSB X11, X11 \
	ABSDIFF(X2, X5, X12, X15) \
	PAND fKFE<>(SB), X12 \
	PSRLW $1, X12 \
	PADDUSB X12, X11 \
	MOVOU THROFF(SP), X15 \
	PSUBUSB X15, X11 \
	POR X11, X10 \
	PXOR X8, X8 \
	PCMPEQB X10, X8 \
	MOVOU 48(SP), X15 \
	PSUBUSB X15, X9 \
	PXOR X10, X10 \
	PCMPEQB X10, X9

// UPDATE2 is Update2Pixels_SSE2 without the sign flips: with the words
// X12 (low 8 lines) and X13 (high 8 lines) holding k*f + 63, P += and
// Q -= sat8((k*f + 63) >> 7). Clobbers X8, X14.
#define UPDATE2(P, Q) \
	MOVO X12, X14 \
	PSRAW $7, X14 \
	MOVO X13, X8 \
	PSRAW $7, X8 \
	PACKSSWB X8, X14 \
	PADDSB X14, P \
	PSUBSB X14, Q

// FILTER6 is DoFilter6_SSE2 (macroblock edges) on p2..q2 in X1-X6 with the
// masks of COMPLEX_MASK: with a = sat8((p1-q1) + 3*(q0-p0)) (signed),
// hev lanes get p0 += sat8(a+3)>>3, q0 -= sat8(a+4)>>3 (DoFilter2), the
// other filtered lanes p0/q0 +-= (27a+63)>>7, p1/q1 +-= (18a+63)>>7,
// p2/q2 +-= (9a+63)>>7 (saturating, i.e. clipped to [0, 255]). Each update
// leaves the other kind of lane unchanged (its filter value is 0).
// Clobbers X8-X15.
#define FILTER6 \
	MOVOU fK80<>(SB), X15 \
	PXOR X15, X1 \
	PXOR X15, X2 \
	PXOR X15, X3 \
	PXOR X15, X4 \
	PXOR X15, X5 \
	PXOR X15, X6 \
	MOVO X2, X10 \
	PSUBSB X5, X10 \
	MOVO X4, X11 \
	PSUBSB X3, X11 \
	PADDSB X11, X10 \
	PADDSB X11, X10 \
	PADDSB X11, X10 \
	MOVO X9, X11 \
	PANDN X8, X11 \
	PAND X10, X11 \
	PAND X8, X9 \
	PAND X10, X9 \
	MOVO X11, X12 \
	PADDSB fK3<>(SB), X12 \
	SSHR3(X12) \
	PADDSB X12, X3 \
	PADDSB fK4<>(SB), X11 \
	SSHR3(X11) \
	PSUBSB X11, X4 \
	PXOR X10, X10 \
	PUNPCKLBW X9, X10 \
	PMULHW fK0900<>(SB), X10 \
	PXOR X11, X11 \
	PUNPCKHBW X9, X11 \
	PMULHW fK0900<>(SB), X11 \
	MOVO X10, X12 \
	PADDW fK63<>(SB), X12 \
	MOVO X11, X13 \
	PADDW fK63<>(SB), X13 \
	UPDATE2(X1, X6) \
	PADDW X10, X12 \
	PADDW X11, X13 \
	UPDATE2(X2, X5) \
	PADDW X10, X12 \
	PADDW X11, X13 \
	UPDATE2(X3, X4) \
	PXOR X15, X1 \
	PXOR X15, X2 \
	PXOR X15, X3 \
	PXOR X15, X4 \
	PXOR X15, X5 \
	PXOR X15, X6

// FILTER4 is DoFilter4_SSE2 (inner edges) on p1..q1 in X2-X5 with the
// masks of COMPLEX_MASK: a = sat8(hev ? (p1-q1) + 3*(q0-p0) : 3*(q0-p0))
// on filtered lanes, p0 += sat8(a+3)>>3, q0 -= a1 = sat8(a+4)>>3, and on
// the non-hev lanes p1 += a3, q1 -= a3 with a3 = (a1+1)>>1 (computed as
// avg(a1 ^ 0x80, 0) - 64). Clobbers X10-X12, X15.
#define FILTER4 \
	MOVOU fK80<>(SB), X15 \
	PXOR X15, X2 \
	PXOR X15, X3 \
	PXOR X15, X4 \
	PXOR X15, X5 \
	MOVO X2, X10 \
	PSUBSB X5, X10 \
	MOVO X9, X11 \
	PANDN X10, X11 \
	MOVO X4, X12 \
	PSUBSB X3, X12 \
	PADDSB X12, X11 \
	PADDSB X12, X11 \
	PADDSB X12, X11 \
	PAND X8, X11 \
	MOVO X11, X12 \
	PADDSB fK3<>(SB), X12 \
	SSHR3(X12) \
	PADDSB fK4<>(SB), X11 \
	SSHR3(X11) \
	PADDSB X12, X3 \
	PSUBSB X11, X4 \
	PXOR X15, X11 \
	PXOR X12, X12 \
	PAVGB X12, X11 \
	PSUBB fK40<>(SB), X11 \
	PAND X9, X11 \
	PADDSB X11, X2 \
	PSUBSB X11, X5 \
	PXOR X15, X2 \
	PXOR X15, X3 \
	PXOR X15, X4 \
	PXOR X15, X5

// SIMPLE is NeedsFilter_SSE2 + DoFilter2_SSE2 (simple filter) on p1..q1 in
// X2-X5 (unsigned in and out): where 2*|p0-q0| + |p1-q1|/2 <= thresh
// (frame threshold at THROFF(SP)), a = sat8((p1-q1) + 3*(q0-p0)),
// p0 += sat8(a+3)>>3, q0 -= sat8(a+4)>>3. Clobbers X8, X10-X12, X15.
#define SIMPLE(THROFF) \
	ABSDIFF(X3, X4, X10, X15) \
	PADDUSB X10, X10 \
	ABSDIFF(X2, X5, X11, X15) \
	PAND fKFE<>(SB), X11 \
	PSRLW $1, X11 \
	PADDUSB X11, X10 \
	MOVOU THROFF(SP), X11 \
	PSUBUSB X11, X10 \
	PXOR X8, X8 \
	PCMPEQB X10, X8 \
	MOVOU fK80<>(SB), X15 \
	MOVO X2, X10 \
	PXOR X15, X10 \
	MOVO X5, X11 \
	PXOR X15, X11 \
	PSUBSB X11, X10 \
	PXOR X15, X3 \
	PXOR X15, X4 \
	MOVO X4, X11 \
	PSUBSB X3, X11 \
	PADDSB X11, X10 \
	PADDSB X11, X10 \
	PADDSB X11, X10 \
	PAND X8, X10 \
	MOVO X10, X11 \
	PADDSB fK3<>(SB), X11 \
	SSHR3(X11) \
	PADDSB X11, X3 \
	PADDSB fK4<>(SB), X10 \
	SSHR3(X10) \
	PSUBSB X10, X4 \
	PXOR X15, X3 \
	PXOR X15, X4

// TRANSPOSE8 transposes the two 8x8 byte blocks held in the low and high
// halves of X0-X7 (row i of each block in Xi) in place (column j in Xj),
// with byte, word, dword and qword unpacks. Clobbers X8-X13.
#define TRANSPOSE8 \
	MOVO X0, X8 \
	PUNPCKLBW X1, X0 \
	PUNPCKHBW X1, X8 \
	MOVO X2, X9 \
	PUNPCKLBW X3, X2 \
	PUNPCKHBW X3, X9 \
	MOVO X4, X10 \
	PUNPCKLBW X5, X4 \
	PUNPCKHBW X5, X10 \
	MOVO X6, X11 \
	PUNPCKLBW X7, X6 \
	PUNPCKHBW X7, X11 \
	MOVO X0, X1 \
	PUNPCKLWL X2, X0 \
	PUNPCKHWL X2, X1 \
	MOVO X8, X3 \
	PUNPCKLWL X9, X8 \
	PUNPCKHWL X9, X3 \
	MOVO X4, X5 \
	PUNPCKLWL X6, X4 \
	PUNPCKHWL X6, X5 \
	MOVO X10, X7 \
	PUNPCKLWL X11, X10 \
	PUNPCKHWL X11, X7 \
	MOVO X0, X2 \
	PUNPCKLLQ X4, X0 \
	PUNPCKHLQ X4, X2 \
	MOVO X1, X6 \
	PUNPCKLLQ X5, X1 \
	PUNPCKHLQ X5, X6 \
	MOVO X8, X9 \
	PUNPCKLLQ X10, X8 \
	PUNPCKHLQ X10, X9 \
	MOVO X3, X11 \
	PUNPCKLLQ X7, X3 \
	PUNPCKHLQ X7, X11 \
	MOVO X0, X12 \
	PUNPCKLQDQ X8, X0 \
	PUNPCKHQDQ X8, X12 \
	MOVO X2, X13 \
	PUNPCKLQDQ X9, X2 \
	PUNPCKHQDQ X9, X13 \
	MOVO X1, X4 \
	PUNPCKLQDQ X3, X4 \
	PUNPCKHQDQ X3, X1 \
	MOVO X1, X5 \
	MOVO X6, X7 \
	PUNPCKLQDQ X11, X6 \
	PUNPCKHQDQ X11, X7 \
	MOVO X12, X1 \
	MOVO X13, X3

// LOAD8T loads 8 bytes from each of the 8 rows at PA (low halves) and the
// 8 rows at PB (high halves), stride R8 (R11 = 3*R8), and transposes them
// into the columns X0-X7. Clobbers R9, R12, X8-X13.
#define LOAD8T(PA, PB) \
	MOVQ (PA), X0 \
	MOVHPS (PB), X0 \
	MOVQ (PA)(R8*1), X1 \
	MOVHPS (PB)(R8*1), X1 \
	MOVQ (PA)(R8*2), X2 \
	MOVHPS (PB)(R8*2), X2 \
	MOVQ (PA)(R11*1), X3 \
	MOVHPS (PB)(R11*1), X3 \
	LEAQ (PA)(R8*4), R9 \
	LEAQ (PB)(R8*4), R12 \
	MOVQ (R9), X4 \
	MOVHPS (R12), X4 \
	MOVQ (R9)(R8*1), X5 \
	MOVHPS (R12)(R8*1), X5 \
	MOVQ (R9)(R8*2), X6 \
	MOVHPS (R12)(R8*2), X6 \
	MOVQ (R9)(R11*1), X7 \
	MOVHPS (R12)(R11*1), X7 \
	TRANSPOSE8

// STORE8T is the reverse of LOAD8T (the transpose is its own inverse).
#define STORE8T(PA, PB) \
	TRANSPOSE8 \
	MOVQ X0, (PA) \
	MOVHPS X0, (PB) \
	MOVQ X1, (PA)(R8*1) \
	MOVHPS X1, (PB)(R8*1) \
	MOVQ X2, (PA)(R8*2) \
	MOVHPS X2, (PB)(R8*2) \
	MOVQ X3, (PA)(R11*1) \
	MOVHPS X3, (PB)(R11*1) \
	LEAQ (PA)(R8*4), R9 \
	LEAQ (PB)(R8*4), R12 \
	MOVQ X4, (R9) \
	MOVHPS X4, (R12) \
	MOVQ X5, (R9)(R8*1) \
	MOVHPS X5, (R12)(R8*1) \
	MOVQ X6, (R9)(R8*2) \
	MOVHPS X6, (R12)(R8*2) \
	MOVQ X7, (R9)(R11*1) \
	MOVHPS X7, (R12)(R11*1)

// LOAD4T(PA, PB, D0, D1, D2, D3, T0, T1) is Load16x4_SSE2: loads 4 bytes
// from each of the 8 rows at PA and the 8 rows at PB (stride R8, R11 =
// 3*R8) and transposes them into the columns D0-D3 (16 lines each, rows
// of PA first). Each 4x4 block (4 rows of 4 bytes in one register) is
// transposed with PSHUFB, then dword and qword unpacks gather the
// columns. Clobbers R9, T0, T1.
#define LOAD4T(PA, PB, D0, D1, D2, D3, T0, T1) \
	MOVL (PA), D0 \
	PINSRD $1, (PA)(R8*1), D0 \
	PINSRD $2, (PA)(R8*2), D0 \
	PINSRD $3, (PA)(R11*1), D0 \
	LEAQ (PA)(R8*4), R9 \
	MOVL (R9), D1 \
	PINSRD $1, (R9)(R8*1), D1 \
	PINSRD $2, (R9)(R8*2), D1 \
	PINSRD $3, (R9)(R11*1), D1 \
	MOVL (PB), D2 \
	PINSRD $1, (PB)(R8*1), D2 \
	PINSRD $2, (PB)(R8*2), D2 \
	PINSRD $3, (PB)(R11*1), D2 \
	LEAQ (PB)(R8*4), R9 \
	MOVL (R9), D3 \
	PINSRD $1, (R9)(R8*1), D3 \
	PINSRD $2, (R9)(R8*2), D3 \
	PINSRD $3, (R9)(R11*1), D3 \
	PSHUFB fT4x4<>(SB), D0 \
	PSHUFB fT4x4<>(SB), D1 \
	PSHUFB fT4x4<>(SB), D2 \
	PSHUFB fT4x4<>(SB), D3 \
	MOVO D0, T0 \
	PUNPCKLLQ D1, D0 \
	PUNPCKHLQ D1, T0 \
	MOVO D2, T1 \
	PUNPCKLLQ D3, D2 \
	PUNPCKHLQ D3, T1 \
	MOVO D0, D1 \
	PUNPCKLQDQ D2, D0 \
	PUNPCKHQDQ D2, D1 \
	MOVO T0, D2 \
	PUNPCKLQDQ T1, D2 \
	MOVO T0, D3 \
	PUNPCKHQDQ T1, D3

// STORE4T(PA, PB, S0, S1, S2, S3, T0, T1) is Store16x4_SSE2, the reverse
// of LOAD4T: the qword unpacks are undone with SHUFPS (even/odd dwords),
// then each 4x4 block is transposed back with PSHUFB and stored row by
// row. Clobbers R9, S0-S3, T0, T1.
#define STORE4T(PA, PB, S0, S1, S2, S3, T0, T1) \
	MOVO S0, T0 \
	PUNPCKLQDQ S1, S0 \
	PUNPCKHQDQ S1, T0 \
	MOVO S2, T1 \
	PUNPCKLQDQ S3, S2 \
	PUNPCKHQDQ S3, T1 \
	MOVO S0, S1 \
	SHUFPS $0x88, S2, S0 \
	SHUFPS $0xDD, S2, S1 \
	MOVO T0, S2 \
	SHUFPS $0x88, T1, S2 \
	SHUFPS $0xDD, T1, T0 \
	PSHUFB fT4x4<>(SB), S0 \
	PSHUFB fT4x4<>(SB), S1 \
	PSHUFB fT4x4<>(SB), S2 \
	PSHUFB fT4x4<>(SB), T0 \
	MOVL S0, (PA) \
	PEXTRD $1, S0, (PA)(R8*1) \
	PEXTRD $2, S0, (PA)(R8*2) \
	PEXTRD $3, S0, (PA)(R11*1) \
	LEAQ (PA)(R8*4), R9 \
	MOVL S1, (R9) \
	PEXTRD $1, S1, (R9)(R8*1) \
	PEXTRD $2, S1, (R9)(R8*2) \
	PEXTRD $3, S1, (R9)(R11*1) \
	MOVL S2, (PB) \
	PEXTRD $1, S2, (PB)(R8*1) \
	PEXTRD $2, S2, (PB)(R8*2) \
	PEXTRD $3, S2, (PB)(R11*1) \
	LEAQ (PB)(R8*4), R9 \
	MOVL T0, (R9) \
	PEXTRD $1, T0, (R9)(R8*1) \
	PEXTRD $2, T0, (R9)(R8*2) \
	PEXTRD $3, T0, (R9)(R11*1)

// BCAST(SRC, OFF) broadcasts the low byte of the argument SRC to the 16
// bytes at OFF(SP) (X15 must be zero).
#define BCAST(SRC, OFF) \
	MOVQ SRC, X0 \
	PSHUFB X15, X0 \
	MOVOU X0, OFF(SP)

// func filterMBComplexSSE41(y, u, v *byte, yStride, uvStride, thrMB, thrIn, ilevel, hevThresh, flags int)
//
// Mirrors libwebp's DoFilter for the complex filter: HFilter16/8 (left
// edge, flags&1), HFilter16i/8i (inner vertical edges, flags&4),
// VFilter16/8 (top edge, flags&2), VFilter16i/8i (inner horizontal edges,
// flags&4). y, u and v point at the top-left pixel of the macroblock;
// thrMB = limit+4, thrIn = limit (both <= 254), ilevel and hevThresh as in
// fInfo. The caller guarantees that rows -4..15 (luma) / -4..7 (chroma)
// and columns -4..15 / -4..7 are inside the planes whenever the
// corresponding edge is filtered.
//
// Vertical macroblock edges load 8 columns across the edge with LOAD8T
// (rows 0-7 and 8-15 of luma, or the 8 rows of U and of V, in the two
// halves); the inner luma edges roll 4-column loads (LOAD4T) along the
// block as HFilter16i_SSE2 does, keeping p3..p0 of the next edge in
// registers; the chroma inner edge loads its 8 columns with LOAD8T.
// Horizontal edges load whole rows (U|V halves for chroma).
//
// GPRs: AX y / u, CX v, R8 stride of the plane, R11 = 3*stride, R10
// flags, DI/DX/SI/R13 row and column pointers, R9/R12 scratch, BX edge
// counter.
TEXT ·filterMBComplexSSE41(SB), NOSPLIT, $64-80
	PXOR X15, X15
	BCAST(thrMB+40(FP), 0)
	BCAST(thrIn+48(FP), 16)
	BCAST(ilevel+56(FP), 32)
	BCAST(hevThresh+64(FP), 48)
	MOVQ flags+72(FP), R10
	MOVQ y+0(FP), AX
	MOVQ yStride+24(FP), R8
	LEAQ (R8)(R8*2), R11

	// ---- luma
	TESTQ $1, R10
	JZ ylnoleft
	LEAQ -4(AX), DI
	LEAQ (DI)(R8*8), DX
	LOAD8T(DI, DX)
	COMPLEX_MASK(0)
	FILTER6
	STORE8T(DI, DX)

ylnoleft:
	TESTQ $4, R10
	JZ ylnoinner
	// inner vertical edges at columns 4, 8, 12: p3..p0 = columns k-4..k-1
	// in X0-X3, q0..q3 = columns k..k+3 in X4-X7.
	MOVQ AX, DI
	LEAQ (AX)(R8*8), DX
	LOAD4T(DI, DX, X0, X1, X2, X3, X8, X9)
	MOVQ $3, BX

ylhinner:
	ADDQ $4, DI
	ADDQ $4, DX
	LOAD4T(DI, DX, X4, X5, X6, X7, X8, X9)
	COMPLEX_MASK(16)
	FILTER4
	MOVO X4, X10
	MOVO X5, X11
	LEAQ -2(DI), SI
	LEAQ -2(DX), R13
	STORE4T(SI, R13, X2, X3, X10, X11, X8, X9)
	MOVO X4, X0
	MOVO X5, X1
	MOVO X6, X2
	MOVO X7, X3
	DECQ BX
	JNZ ylhinner

ylnoinner:
	TESTQ $2, R10
	JZ ylnotop
	// top edge: rows -4..3
	MOVQ AX, DI
	SUBQ R8, DI
	SUBQ R11, DI
	MOVOU (DI), X0
	MOVOU (DI)(R8*1), X1
	MOVOU (DI)(R8*2), X2
	MOVOU (DI)(R11*1), X3
	MOVOU (AX), X4
	MOVOU (AX)(R8*1), X5
	MOVOU (AX)(R8*2), X6
	MOVOU (AX)(R11*1), X7
	COMPLEX_MASK(0)
	FILTER6
	MOVOU X1, (DI)(R8*1)
	MOVOU X2, (DI)(R8*2)
	MOVOU X3, (DI)(R11*1)
	MOVOU X4, (AX)
	MOVOU X5, (AX)(R8*1)
	MOVOU X6, (AX)(R8*2)

ylnotop:
	TESTQ $4, R10
	JZ chroma
	// inner horizontal edges at rows 4, 8, 12, rolling like the vertical
	// ones (VFilter16i_SSE2).
	MOVOU (AX), X0
	MOVOU (AX)(R8*1), X1
	MOVOU (AX)(R8*2), X2
	MOVOU (AX)(R11*1), X3
	MOVQ AX, DI
	MOVQ $3, BX

ylvinner:
	LEAQ (DI)(R8*4), DI
	MOVOU (DI), X4
	MOVOU (DI)(R8*1), X5
	MOVOU (DI)(R8*2), X6
	MOVOU (DI)(R11*1), X7
	COMPLEX_MASK(16)
	FILTER4
	MOVQ DI, SI
	SUBQ R8, SI
	MOVOU X3, (SI)
	SUBQ R8, SI
	MOVOU X2, (SI)
	MOVOU X4, (DI)
	MOVOU X5, (DI)(R8*1)
	MOVO X4, X0
	MOVO X5, X1
	MOVO X6, X2
	MOVO X7, X3
	DECQ BX
	JNZ ylvinner

chroma:
	// ---- chroma: U in the low, V in the high 8 bytes
	MOVQ u+8(FP), AX
	MOVQ v+16(FP), CX
	MOVQ uvStride+32(FP), R8
	LEAQ (R8)(R8*2), R11
	TESTQ $1, R10
	JZ uvnoleft
	LEAQ -4(AX), DI
	LEAQ -4(CX), DX
	LOAD8T(DI, DX)
	COMPLEX_MASK(0)
	FILTER6
	STORE8T(DI, DX)

uvnoleft:
	TESTQ $4, R10
	JZ uvnoinner
	// inner vertical edge at column 4: columns 0..7
	LOAD8T(AX, CX)
	COMPLEX_MASK(16)
	FILTER4
	STORE8T(AX, CX)

uvnoinner:
	TESTQ $2, R10
	JZ uvnotop
	// top edge: rows -4..3
	MOVQ AX, DI
	SUBQ R8, DI
	SUBQ R11, DI
	MOVQ CX, DX
	SUBQ R8, DX
	SUBQ R11, DX
	MOVQ (DI), X0
	MOVHPS (DX), X0
	MOVQ (DI)(R8*1), X1
	MOVHPS (DX)(R8*1), X1
	MOVQ (DI)(R8*2), X2
	MOVHPS (DX)(R8*2), X2
	MOVQ (DI)(R11*1), X3
	MOVHPS (DX)(R11*1), X3
	MOVQ (AX), X4
	MOVHPS (CX), X4
	MOVQ (AX)(R8*1), X5
	MOVHPS (CX)(R8*1), X5
	MOVQ (AX)(R8*2), X6
	MOVHPS (CX)(R8*2), X6
	MOVQ (AX)(R11*1), X7
	MOVHPS (CX)(R11*1), X7
	COMPLEX_MASK(0)
	FILTER6
	MOVQ X1, (DI)(R8*1)
	MOVHPS X1, (DX)(R8*1)
	MOVQ X2, (DI)(R8*2)
	MOVHPS X2, (DX)(R8*2)
	MOVQ X3, (DI)(R11*1)
	MOVHPS X3, (DX)(R11*1)
	MOVQ X4, (AX)
	MOVHPS X4, (CX)
	MOVQ X5, (AX)(R8*1)
	MOVHPS X5, (CX)(R8*1)
	MOVQ X6, (AX)(R8*2)
	MOVHPS X6, (CX)(R8*2)

uvnotop:
	TESTQ $4, R10
	JZ done
	// inner horizontal edge at row 4: rows 0..7
	MOVQ (AX), X0
	MOVHPS (CX), X0
	MOVQ (AX)(R8*1), X1
	MOVHPS (CX)(R8*1), X1
	MOVQ (AX)(R8*2), X2
	MOVHPS (CX)(R8*2), X2
	MOVQ (AX)(R11*1), X3
	MOVHPS (CX)(R11*1), X3
	LEAQ (AX)(R8*4), DI
	LEAQ (CX)(R8*4), DX
	MOVQ (DI), X4
	MOVHPS (DX), X4
	MOVQ (DI)(R8*1), X5
	MOVHPS (DX)(R8*1), X5
	MOVQ (DI)(R8*2), X6
	MOVHPS (DX)(R8*2), X6
	MOVQ (DI)(R11*1), X7
	MOVHPS (DX)(R11*1), X7
	COMPLEX_MASK(16)
	FILTER4
	MOVQ X2, (AX)(R8*2)
	MOVHPS X2, (CX)(R8*2)
	MOVQ X3, (AX)(R11*1)
	MOVHPS X3, (CX)(R11*1)
	MOVQ X4, (DI)
	MOVHPS X4, (DX)
	MOVQ X5, (DI)(R8*1)
	MOVHPS X5, (DX)(R8*1)

done:
	RET

// func filterMBSimpleSSE41(y *byte, yStride, thrMB, thrIn, flags int)
//
// Mirrors libwebp's DoFilter for the simple filter (luma only):
// SimpleHFilter16 (left edge, flags&1), SimpleHFilter16i (flags&4),
// SimpleVFilter16 (top edge, flags&2), SimpleVFilter16i (flags&4), one
// edge at a time. Vertical edges load the 4 columns p1..q1 with LOAD4T,
// horizontal edges 4 rows. The caller guarantees that rows -2..15 and
// columns -2..15 are inside the plane whenever the edges reaching them are
// filtered.
//
// GPRs: AX y, R8 stride, R11 = 3*stride, R10 flags, BX edge offset k, CX
// last edge offset, DI/DX pointers, R9 scratch. The frame holds the
// macroblock edge threshold at 0 and the inner one at 16; the threshold
// of the current edge is copied (through X14) to 32(SP) for SIMPLE.
TEXT ·filterMBSimpleSSE41(SB), NOSPLIT, $48-40
	PXOR X15, X15
	BCAST(thrMB+16(FP), 0)
	BCAST(thrIn+24(FP), 16)
	MOVQ y+0(FP), AX
	MOVQ yStride+8(FP), R8
	LEAQ (R8)(R8*2), R11
	MOVQ flags+32(FP), R10

	// Edges k in [first, last], step 4: first = 0 if the left/top edge is
	// filtered, else 4; last = 12 with inner edges, else 0.
	XORQ CX, CX
	TESTQ $4, R10
	JZ sfirstv
	MOVQ $12, CX

sfirstv:
	XORQ BX, BX
	TESTQ $1, R10
	JNZ svloop
	MOVQ $4, BX

svloop:
	CMPQ BX, CX
	JGT shoriz
	MOVOU 16(SP), X14
	TESTQ BX, BX
	JNZ 2(PC)
	MOVOU 0(SP), X14
	MOVOU X14, 32(SP)
	LEAQ -2(AX)(BX*1), DI
	LEAQ (DI)(R8*8), DX
	LOAD4T(DI, DX, X2, X3, X4, X5, X8, X9)
	SIMPLE(32)
	STORE4T(DI, DX, X2, X3, X4, X5, X8, X9)
	ADDQ $4, BX
	JMP svloop

shoriz:
	XORQ BX, BX
	TESTQ $2, R10
	JNZ shloop
	MOVQ $4, BX

shloop:
	CMPQ BX, CX
	JGT sdone
	MOVOU 16(SP), X14
	TESTQ BX, BX
	JNZ 2(PC)
	MOVOU 0(SP), X14
	MOVOU X14, 32(SP)
	// rows k-2..k+1
	MOVQ BX, DI
	IMULQ R8, DI
	ADDQ AX, DI
	SUBQ R8, DI
	SUBQ R8, DI
	MOVOU (DI), X2
	MOVOU (DI)(R8*1), X3
	MOVOU (DI)(R8*2), X4
	MOVOU (DI)(R11*1), X5
	SIMPLE(32)
	MOVOU X3, (DI)(R8*1)
	MOVOU X4, (DI)(R8*2)
	ADDQ $4, BX
	JMP shloop

sdone:
	RET
