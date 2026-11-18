//go:build amd64 && !purego

#include "textflag.h"

// Constants of residualCostSSE41 (word lanes, low to high): encBands[n] *
// 408 for the positions n = 0..15 (408 bytes per band of a coefficient
// type's levelCost: 3 contexts of 68 uint16), the lane numbers 0..7, 67,
// 2 and 136 (the bytes of one context's table).
DATA ·rcBand+0(SB)/8, $0x04c8033001980000
DATA ·rcBand+8(SB)/8, $0x099007f806600990
DATA ·rcBand+16(SB)/8, $0x0990099009900990
DATA ·rcBand+24(SB)/8, $0x0b28099009900990
GLOBL ·rcBand(SB), RODATA|NOPTR, $32
DATA ·rcLane+0(SB)/8, $0x0003000200010000
DATA ·rcLane+8(SB)/8, $0x0007000600050004
GLOBL ·rcLane(SB), RODATA|NOPTR, $16
DATA ·rck67+0(SB)/8, $0x0043004300430043
DATA ·rck67+8(SB)/8, $0x0043004300430043
GLOBL ·rck67(SB), RODATA|NOPTR, $16
DATA ·rck2+0(SB)/8, $0x0002000200020002
DATA ·rck2+8(SB)/8, $0x0002000200020002
GLOBL ·rck2(SB), RODATA|NOPTR, $16
DATA ·rck136+0(SB)/8, $0x0088008800880088
DATA ·rck136+8(SB)/8, $0x0088008800880088
GLOBL ·rck136(SB), RODATA|NOPTR, $16

// ---------------------------------------------------------------------------
// func residualCostSSE41(costs *[numBands][numCtx]levelCostTable, prob *[numBands][numCtx][numProbas]uint8,
//	coeffs *[16]int16, first, last, ctx0 int) int
//
// residualCostGo (cost.go) for first in {0, 1}, last <= 15, ctx0 <= 2,
// without data-dependent branches; libwebp's GetResidualCost_SSE2. The
// Go loop picks the cost table of position n+1 from the context
// min(|level n|, 2), so the address of every looked-up entry is known up
// front: for the 16 positions,
//   fixed[n] = 2 * (|c[n]| & 2047)            (levelFixedCosts offset)
//   var[n]   = encBands[n]*408 + ctx(n-1)*136 + 2*min(|c[n]|, 67)
// with ctx(n-1) = min(|c[n-1]|, 2) and ctx0 at position first (PABSW,
// PMINUW, PALIGNR/PSLLDQ for the shift by one position, a lane mask for
// ctx0, PMULLW). Both offset vectors go to the stack, and the loop over
// first..last only adds the two looked-up costs. The head (p0 terms) and
// the tail (the end-of-block bit with the last level's context) are as in
// the Go code.
//
// Registers: DI costs, R9 prob, SI coeffs, CX first, R8 last, BX ctx0,
// R10 cost, R11 levelFixedCosts, R12 entropyCost; the stack holds var[]
// at 0(SP) and fixed[] at 32(SP).
TEXT ·residualCostSSE41(SB), NOSPLIT, $64-56
	MOVQ	costs+0(FP), DI
	MOVQ	prob+8(FP), R9
	MOVQ	coeffs+16(FP), SI
	MOVQ	first+24(FP), CX
	MOVQ	last+32(FP), R8
	MOVQ	ctx0+40(FP), BX
	LEAQ	·entropyCost(SB), R12
	IMUL3Q	$33, CX, AX
	IMUL3Q	$11, BX, DX
	ADDQ	DX, AX
	MOVBLZX	(R9)(AX*1), AX // p0 = prob[first][ctx0][0]
	TESTQ	R8, R8
	JS	rcempty
	XORL	DX, DX
	XORL	$255, AX
	MOVWLZX	(R12)(AX*2), AX // bitCost(1, p0)
	TESTQ	BX, BX
	CMOVLNE	DX, AX
	MOVL	AX, R10

	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	PABSW	X0, X0
	PABSW	X1, X1
	MOVOU	·k2047w(SB), X2
	MOVO	X0, X3
	PAND	X2, X3
	PAND	X1, X2
	PSLLW	$1, X3
	PSLLW	$1, X2
	MOVOU	X3, 32(SP)
	MOVOU	X2, 48(SP)
	MOVO	X0, X4
	PMINUW	·rck67(SB), X4
	MOVO	X1, X5
	PMINUW	·rck67(SB), X5
	PMINUW	·rck2(SB), X0
	PMINUW	·rck2(SB), X1
	MOVO	X1, X6
	PALIGNR	$14, X0, X6 // contexts of positions 7..14
	PSLLO	$2, X0      // contexts of positions -..6
	MOVL	CX, X7
	PSHUFB	·wSplat0(SB), X7
	PCMPEQW	·rcLane(SB), X7 // lane first
	MOVL	BX, X8
	PSHUFB	·wSplat0(SB), X8
	PAND	X7, X8
	PANDN	X0, X7
	POR	X8, X7
	PMULLW	·rck136(SB), X7
	PMULLW	·rck136(SB), X6
	PADDW	X4, X4
	PADDW	X5, X5
	PADDW	X4, X7
	PADDW	X5, X6
	PADDW	·rcBand(SB), X7
	PADDW	·rcBand+16(SB), X6
	MOVOU	X7, 0(SP)
	MOVOU	X6, 16(SP)

	LEAQ	·levelFixedCosts(SB), R11
	MOVQ	CX, DX

rcloop:
	MOVWLZX	0(SP)(DX*2), AX
	MOVWLZX	(DI)(AX*1), AX
	ADDL	AX, R10
	MOVWLZX	32(SP)(DX*2), AX
	MOVWLZX	(R11)(AX*1), AX
	ADDL	AX, R10
	INCQ	DX
	CMPQ	DX, R8
	JLE	rcloop

	// DX = n + 1, n the last position coded: max(first, last) as in the
	// Go loop
	CMPQ	DX, $16
	JGE	rcdone
	MOVWLSX	-2(SI)(DX*2), AX
	MOVL	AX, R8
	NEGL	R8
	CMOVLLT	AX, R8 // |c[n]|
	XORL	AX, AX
	CMPL	R8, $1
	SETNE	AL
	INCL	AX // context: 1 if |c[n]| == 1, else 2
	LEAQ	·encBands(SB), R8
	MOVBLZX	(R8)(DX*1), DX
	IMUL3Q	$33, DX, DX
	IMUL3Q	$11, AX, AX
	ADDQ	AX, DX
	MOVBLZX	(R9)(DX*1), AX
	MOVWLZX	(R12)(AX*2), AX // bitCost(0, prob[band][ctx][0])
	ADDL	AX, R10

rcdone:
	MOVQ	R10, ret+48(FP)
	RET

rcempty:
	MOVWLZX	(R12)(AX*2), AX // bitCost(0, p0)
	MOVQ	AX, ret+48(FP)
	RET
