//go:build amd64 && !purego

// SSE4.1 and AVX2 kernels of the 8-bit alpha palette lookup (lookupBytes,
// libwebp MapAlpha, which has no x86 SIMD version) for CPUs without
// AVX512-VBMI (lookupBytesVBMI in palette_amd64.s); the Go reference
// lookupBytesGo (transform.go) is the specification. The functions process
// whole blocks of 16 bytes, the wrapper in dsp_amd64.go the rest in Go;
// the AVX2 one uses VEX encodings only and ends with VZEROUPPER.

#include "textflag.h"

// lbSixteen: 16 in each byte; lbHigh: 0x80 in each byte.
DATA lbSixteen<>+0(SB)/8, $0x1010101010101010
DATA lbSixteen<>+8(SB)/8, $0x1010101010101010
GLOBL lbSixteen<>(SB), RODATA|NOPTR, $16
DATA lbHigh<>+0(SB)/8, $0x8080808080808080
DATA lbHigh<>+8(SB)/8, $0x8080808080808080
GLOBL lbHigh<>(SB), RODATA|NOPTR, $16

// LBTABLES stores at 0(SP)..255(SP) the PSHUFB tables of lookupBytesSSE41
// and lookupBytesAVX2: for each half h of the palette (AX) and k < 8,
// row 16(8h+k) of the palette XOR row 16(8h+k-1) (row 16(8h) alone).
// Clobbers X0-X2.
#define LBROW(k) \
	MOVOU k*16(AX), X1  \
	MOVO  X1, X2        \
	PXOR  X0, X1        \
	MOVOU X1, k*16(SP)  \
	MOVO  X2, X0

#define LBTABLES \
	MOVOU (AX), X0    \
	MOVOU X0, (SP)    \
	LBROW(1)          \
	LBROW(2)          \
	LBROW(3)          \
	LBROW(4)          \
	LBROW(5)          \
	LBROW(6)          \
	LBROW(7)          \
	MOVOU 128(AX), X0 \
	MOVOU X0, 128(SP) \
	LBROW(9)          \
	LBROW(10)         \
	LBROW(11)         \
	LBROW(12)         \
	LBROW(13)         \
	LBROW(14)         \
	LBROW(15)

// LBTABLESV is LBTABLES in VEX encoding (for lookupBytesAVX2).
#define LBROWV(k) \
	VMOVDQU k*16(AX), X1     \
	VPXOR   X0, X1, X2       \
	VMOVDQU X2, k*16(SP)     \
	VMOVDQA X1, X0

#define LBTABLESV \
	VMOVDQU (AX), X0    \
	VMOVDQU X0, (SP)    \
	LBROWV(1)           \
	LBROWV(2)           \
	LBROWV(3)           \
	LBROWV(4)           \
	LBROWV(5)           \
	LBROWV(6)           \
	LBROWV(7)           \
	VMOVDQU 128(AX), X0 \
	VMOVDQU X0, 128(SP) \
	LBROWV(9)           \
	LBROWV(10)          \
	LBROWV(11)          \
	LBROWV(12)          \
	LBROWV(13)          \
	LBROWV(14)          \
	LBROWV(15)

// LBSTEP looks up the indices X4 in the table at off(SP), XORs the
// result into acc and moves the indices to the next row (X8 lbSixteen).
#define LBSTEP(acc, off) \
	MOVOU  off(SP), X6 \
	PSHUFB X4, X6      \
	PXOR   X6, acc     \
	PSUBB  X8, X4

// LBSTEP2 is LBSTEP for two vectors of indices (Y4, Y5) into the
// accumulators a0, a1; Y8 lbSixteen.
#define LBSTEP2(a0, a1, off) \
	VBROADCASTI128 off(SP), Y6 \
	VPSHUFB        Y4, Y6, Y7  \
	VPXOR          Y7, a0, a0  \
	VPSHUFB        Y5, Y6, Y7  \
	VPXOR          Y7, a1, a1  \
	VPSUBB         Y8, Y4, Y4  \
	VPSUBB         Y8, Y5, Y5

// func lookupBytesSSE41(pal *[256]byte, in, out *byte, n int)
//
// out[i] = pal[in[i]] for i < n, n a multiple of 16, with PSHUFB, which
// looks up 16-byte tables and yields 0 for indices with bit 7 set. For
// the indices j < 128 of a half of the palette, j - 16k is a valid index
// (low nibble that of j) for the rows k <= j>>4 and has bit 7 set for the
// others; so the XOR of the PSHUFB lookups of j - 16k, k < 8, in the
// tables row k XOR row k-1 telescopes to row j>>4 at j&15. The lower half
// is looked up with in[i], the upper with in[i] ^ 0x80, and bit 7 of
// in[i] picks one (PBLENDVB, mask X0).
//
// AX pal, SI in, DI out, CX bytes left; tables at 0(SP)-255(SP); X0 the
// indices, X4 shifted indices, X2, X3 the two halves' results, X8
// lbSixteen, X9 lbHigh.
TEXT ·lookupBytesSSE41(SB), NOSPLIT, $256-32
	MOVQ  pal+0(FP), AX
	MOVQ  in+8(FP), SI
	MOVQ  out+16(FP), DI
	MOVQ  n+24(FP), CX
	LBTABLES
	MOVOU lbSixteen<>(SB), X8
	MOVOU lbHigh<>(SB), X9
	SHRQ  $4, CX
	JZ    lsdone

ls16:
	MOVOU    (SI), X0
	MOVO     X0, X4
	PXOR     X2, X2
	LBSTEP(X2, 0)
	LBSTEP(X2, 16)
	LBSTEP(X2, 32)
	LBSTEP(X2, 48)
	LBSTEP(X2, 64)
	LBSTEP(X2, 80)
	LBSTEP(X2, 96)
	LBSTEP(X2, 112)
	MOVO     X0, X4
	PXOR     X9, X4
	PXOR     X3, X3
	LBSTEP(X3, 128)
	LBSTEP(X3, 144)
	LBSTEP(X3, 160)
	LBSTEP(X3, 176)
	LBSTEP(X3, 192)
	LBSTEP(X3, 208)
	LBSTEP(X3, 224)
	LBSTEP(X3, 240)
	PBLENDVB X0, X3, X2
	MOVOU    X2, (DI)
	ADDQ     $16, SI
	ADDQ     $16, DI
	DECQ     CX
	JNZ      ls16

lsdone:
	RET

// func lookupBytesAVX2(pal *[256]byte, in, out *byte, n int)
//
// lookupBytesSSE41 with 32-byte vectors, two per iteration, then one
// vector of 32 and one of 16 bytes (n a multiple of 16).
//
// AX pal, SI in, DI out, CX bytes left; tables at 0(SP)-255(SP); Y0, Y1
// indices, Y4, Y5 shifted indices, Y2, Y3 and Y10, Y11 the halves'
// results, Y8 lbSixteen, Y9 lbHigh.
TEXT ·lookupBytesAVX2(SB), NOSPLIT, $256-32
	MOVQ           pal+0(FP), AX
	MOVQ           in+8(FP), SI
	MOVQ           out+16(FP), DI
	MOVQ           n+24(FP), CX
	LBTABLESV
	VBROADCASTI128 lbSixteen<>(SB), Y8
	VBROADCASTI128 lbHigh<>(SB), Y9

la64:
	CMPQ      CX, $64
	JB        la32
	VMOVDQU   (SI), Y0
	VMOVDQU   32(SI), Y1
	VPXOR     Y2, Y2, Y2
	VPXOR     Y3, Y3, Y3
	VMOVDQA   Y0, Y4
	VMOVDQA   Y1, Y5
	LBSTEP2(Y2, Y3, 0)
	LBSTEP2(Y2, Y3, 16)
	LBSTEP2(Y2, Y3, 32)
	LBSTEP2(Y2, Y3, 48)
	LBSTEP2(Y2, Y3, 64)
	LBSTEP2(Y2, Y3, 80)
	LBSTEP2(Y2, Y3, 96)
	LBSTEP2(Y2, Y3, 112)
	VPXOR     Y9, Y0, Y4
	VPXOR     Y9, Y1, Y5
	VPXOR     Y10, Y10, Y10
	VPXOR     Y11, Y11, Y11
	LBSTEP2(Y10, Y11, 128)
	LBSTEP2(Y10, Y11, 144)
	LBSTEP2(Y10, Y11, 160)
	LBSTEP2(Y10, Y11, 176)
	LBSTEP2(Y10, Y11, 192)
	LBSTEP2(Y10, Y11, 208)
	LBSTEP2(Y10, Y11, 224)
	LBSTEP2(Y10, Y11, 240)
	VPBLENDVB Y0, Y10, Y2, Y2
	VPBLENDVB Y1, Y11, Y3, Y3
	VMOVDQU   Y2, (DI)
	VMOVDQU   Y3, 32(DI)
	ADDQ      $64, SI
	ADDQ      $64, DI
	SUBQ      $64, CX
	JMP       la64

	// One or two vectors left (32 and/or 16 bytes): the 64-byte step on
	// a 32-byte vector, its second lane unused for 16 bytes.
la32:
	TESTQ     CX, CX
	JZ        ladone
	CMPQ      CX, $32
	JB        la16
	VMOVDQU   (SI), Y0
	JMP       la1

la16:
	VMOVDQU   (SI), X0

la1:
	VPXOR     Y2, Y2, Y2
	VPXOR     Y3, Y3, Y3
	VMOVDQA   Y0, Y4
	VMOVDQA   Y0, Y5
	LBSTEP2(Y2, Y3, 0)
	LBSTEP2(Y2, Y3, 16)
	LBSTEP2(Y2, Y3, 32)
	LBSTEP2(Y2, Y3, 48)
	LBSTEP2(Y2, Y3, 64)
	LBSTEP2(Y2, Y3, 80)
	LBSTEP2(Y2, Y3, 96)
	LBSTEP2(Y2, Y3, 112)
	VPXOR     Y9, Y0, Y4
	VPXOR     Y9, Y0, Y5
	VPXOR     Y10, Y10, Y10
	VPXOR     Y11, Y11, Y11
	LBSTEP2(Y10, Y11, 128)
	LBSTEP2(Y10, Y11, 144)
	LBSTEP2(Y10, Y11, 160)
	LBSTEP2(Y10, Y11, 176)
	LBSTEP2(Y10, Y11, 192)
	LBSTEP2(Y10, Y11, 208)
	LBSTEP2(Y10, Y11, 224)
	LBSTEP2(Y10, Y11, 240)
	VPBLENDVB Y0, Y10, Y2, Y2
	CMPQ      CX, $32
	JB        la1x
	VMOVDQU   Y2, (DI)
	ADDQ      $32, SI
	ADDQ      $32, DI
	SUBQ      $32, CX
	JMP       la32

la1x:
	VMOVDQU   X2, (DI)

ladone:
	VZEROUPPER
	RET
