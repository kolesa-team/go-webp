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

//go:build arm64 && !purego

// NEON kernels of the ALPH unfilters, adapted from libwebp (BSD-3-Clause)
// src/dsp/filters_neon.c (HorizontalUnfilter_NEON, VerticalUnfilter_NEON).
// The Go references in filters.go are the specification.

#include "textflag.h"

// func unfilterHorizontalAsm(in, out *byte, n, pred int)
//
// out[i] = pred + in[0] + ... + in[i] (mod 256) for i < n, n a multiple of
// 16; in may alias out. Each block of 16 bytes gets its prefix sums from
// four shifted adds (by 1, 2, 4 and 8 bytes, EXT with zero), off the serial
// path; the carry from the previous block (all lanes of V0) costs one add
// and one lane broadcast per 32 bytes: the second block of an iteration
// adds the total of the first before the carry.
//
// R0 in, R1 out, R2 bytes left, R3 pred; V0 carry, V31 zero, V1/V2 blocks,
// V3/V4 shifted copies.
TEXT ·unfilterHorizontalAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD n+16(FP), R2
	MOVD pred+24(FP), R3
	VDUP R3, V0.B16
	VEOR V31.B16, V31.B16, V31.B16
	CMP  $32, R2
	BLT  h16

h32:
	VLD1.P 32(R0), [V1.B16, V2.B16]
	VEXT   $15, V1.B16, V31.B16, V3.B16
	VEXT   $15, V2.B16, V31.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VEXT   $14, V1.B16, V31.B16, V3.B16
	VEXT   $14, V2.B16, V31.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VEXT   $12, V1.B16, V31.B16, V3.B16
	VEXT   $12, V2.B16, V31.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VEXT   $8, V1.B16, V31.B16, V3.B16
	VEXT   $8, V2.B16, V31.B16, V4.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V4.B16, V2.B16, V2.B16
	VDUP   V1.B[15], V3.B16
	VADD   V3.B16, V2.B16, V2.B16
	VADD   V0.B16, V1.B16, V1.B16
	VADD   V0.B16, V2.B16, V2.B16
	VDUP   V2.B[15], V0.B16
	VST1.P [V1.B16, V2.B16], 32(R1)
	SUB    $32, R2, R2
	CMP    $32, R2
	BGE    h32

h16:
	CBZ    R2, hdone
	VLD1.P 16(R0), [V1.B16]
	VEXT   $15, V1.B16, V31.B16, V3.B16
	VADD   V3.B16, V1.B16, V1.B16
	VEXT   $14, V1.B16, V31.B16, V3.B16
	VADD   V3.B16, V1.B16, V1.B16
	VEXT   $12, V1.B16, V31.B16, V3.B16
	VADD   V3.B16, V1.B16, V1.B16
	VEXT   $8, V1.B16, V31.B16, V3.B16
	VADD   V3.B16, V1.B16, V1.B16
	VADD   V0.B16, V1.B16, V1.B16
	VDUP   V1.B[15], V0.B16
	VST1.P [V1.B16], 16(R1)
	SUB    $16, R2, R2
	B      h16

hdone:
	RET

// func unfilterVerticalAsm(prev, in, out *byte, n int)
//
// out[i] = prev[i] + in[i] (mod 256) for i < n, n a multiple of 16; in may
// alias out.
//
// R0 prev, R1 in, R2 out, R3 bytes left; V0-V3 prev, V4-V7 in.
TEXT ·unfilterVerticalAsm(SB), NOSPLIT, $0-32
	MOVD prev+0(FP), R0
	MOVD in+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3
	CMP  $64, R3
	BLT  v16

v64:
	VLD1.P 64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	VADD   V4.B16, V0.B16, V0.B16
	VADD   V5.B16, V1.B16, V1.B16
	VADD   V6.B16, V2.B16, V2.B16
	VADD   V7.B16, V3.B16, V3.B16
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R2)
	SUB    $64, R3, R3
	CMP    $64, R3
	BGE    v64

v16:
	CBZ    R3, vdone
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V4.B16]
	VADD   V4.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R2)
	SUB    $16, R3, R3
	B      v16

vdone:
	RET
