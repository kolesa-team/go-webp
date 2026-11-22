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

#include "textflag.h"

// NEON in-loop filters, adapted from libwebp (BSD-3-Clause) src/dsp/dec_neon.c
// (NeedsFilter2_NEON, NeedsHev_NEON, DoFilter2/4/6_NEON and the transposing
// loads and stores) and bit-exact with the Go code in filter.go.
//
// One call filters all edges of one macroblock, in libwebp's DoFilter order
// per plane: left edge, inner vertical edges, top edge, inner horizontal
// edges. Luma and chroma are independent, so luma is done first; U and V
// are packed in the low and high halves of the same registers (libwebp's
// *8 variants).
//
// The pixels are kept in signed form (x ^ 0x80) between the load and the
// store, so the filters need no per-edge sign flips; the absolute
// differences are then taken with SABD, which gives the same |a-b| as UABD
// on the unsigned values.
//
// Instructions the Go assembler does not know are emitted with WORD; each
// macro names the instruction it encodes. The encodings were checked
// against clang's assembler.

// SABD Vd.16B, Vn.16B, Vm.16B
#define SABD(d, n, m) WORD $(0x4E207400 | ((m)<<16) | ((n)<<5) | (d))
// SQRSHRN Vd.8B, Vn.8H, #sh (lower half; clears the upper half)
#define SQRSHRN_8B(d, n, sh) WORD $(0x0F009C00 | ((16-(sh))<<16) | ((n)<<5) | (d))
// SQRSHRN2 Vd.16B, Vn.8H, #sh (upper half)
#define SQRSHRN2_16B(d, n, sh) WORD $(0x4F009C00 | ((16-(sh))<<16) | ((n)<<5) | (d))

// Register conventions of filterMBComplexNEON:
//   V0-V19   pixels (rows, columns, or the 4 rows/columns across an edge)
//   V20      macroblock edge threshold, limit+4 (NeedsFilter's thresh)
//   V21      inner edge threshold, limit
//   V22      interior limit (ilevel)
//   V23      high edge variance threshold (hev_thresh)
//   V24, V25 the int8 constants 3 and 4
//   V26-V31  temporaries T0-T5 of the macros below
// The 16x16 transposes need all 32 registers, so V20-V25 are reloaded
// from R5-R8 (CONSTS) after each of them.

#define CONSTS \
	VDUP R5, V20.B16 \
	VDUP R6, V21.B16 \
	VDUP R7, V22.B16 \
	VDUP R8, V23.B16 \
	VMOVI $3, V24.B16 \
	VMOVI $4, V25.B16

// SIGN0x80 loads the sign flip constant into T5 (V31); FLIP(R) toggles the
// sign bits of R with it.
#define SIGN0x80 VMOVI $0x80, V31.B16
#define FLIP(R) VEOR V31.B16, R.B16, R.B16

// NEEDS(p3, p2, p1, p0, q0, q1, q2, q3, THR) is NeedsFilter2_NEON and
// NeedsHev_NEON on register numbers p3..q3: leaves the filter mask
//   max(|p3-p2|, |p2-p1|, |p1-p0|, |q3-q2|, |q2-q1|, |q1-q0|) <= ilevel &&
//   2*|p0-q0| + |p1-q1|/2 <= THR      (saturating, == 4a+b <= 2*THR+1)
// in T0 (V26) and the hev mask max(|p1-p0|, |q1-q0|) > hev_thresh in T1
// (V27). The maximum is a tree to keep the dependency chain short.
// Clobbers T2-T5.
#define NEEDS(p3, p2, p1, p0, q0, q1, q2, q3, THR) \
	SABD(26, p3, p2) \
	SABD(27, p2, p1) \
	SABD(28, p1, p0) \
	SABD(29, q1, q0) \
	SABD(30, q3, q2) \
	SABD(31, q2, q1) \
	VUMAX V27.B16, V26.B16, V26.B16 \
	VUMAX V31.B16, V30.B16, V30.B16 \
	VUMAX V29.B16, V28.B16, V27.B16 \
	VUMAX V30.B16, V26.B16, V26.B16 \
	VUMAX V27.B16, V26.B16, V26.B16 \
	VCMHS V26.B16, V22.B16, V26.B16 \
	VCMHI V23.B16, V27.B16, V27.B16 \
	SABD(28, p0, q0) \
	SABD(29, p1, q1) \
	VUQADD V28.B16, V28.B16, V28.B16 \
	VUSHR $1, V29.B16, V29.B16 \
	VUQADD V29.B16, V28.B16, V28.B16 \
	VCMHS V28.B16, THR.B16, V28.B16 \
	VAND V28.B16, V26.B16, V26.B16

// FILTER2(P1, P0, Q0, Q1) is the hev part of DoFilter6_NEON, with the
// masks of NEEDS in T0/T1: on lanes with mask && hev (DoFilter2_C)
//   a = sat8((p1-q1) + 3*(q0-p0)); p0 += sat8(a+3)>>3; q0 -= sat8(a+4)>>3.
// Leaves T0 = mask &^ hev and T3 = a (unmasked). Clobbers T1, T2.
#define FILTER2(P1, P0, Q0, Q1) \
	VAND V27.B16, V26.B16, V27.B16 \
	VEOR V27.B16, V26.B16, V26.B16 \
	VSQSUB P0.B16, Q0.B16, V28.B16 \
	VSQSUB Q1.B16, P1.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VAND V29.B16, V27.B16, V27.B16 \
	VSQADD V24.B16, V27.B16, V28.B16 \
	VSQADD V25.B16, V27.B16, V27.B16 \
	VSSHR $3, V28.B16, V28.B16 \
	VSSHR $3, V27.B16, V27.B16 \
	VSQADD V28.B16, P0.B16, P0.B16 \
	VSQSUB V27.B16, Q0.B16, Q0.B16

// FILTER6(P2, P1, P0, Q0, Q1, Q2) is DoFilter6_NEON (macroblock edges) on
// the masks of NEEDS: FILTER2 on hev lanes, and on the other filtered
// lanes, with a = sat8((p1-q1) + 3*(q0-p0)):
//   a1 = (27a+63)>>7 = sqrshrn(27a - 1, 7), a2 = (18a+63)>>7 =
//   sqrshrn(9a - 1, 6), a3 = (9a+63)>>7 = sqrshrn(9a - 1, 7);
//   p0 += a1, q0 -= a1, p1 += a2, q1 -= a2, p2 += a3, q2 -= a3
// (saturating, i.e. clip to [0, 255] in unsigned form). libwebp forms
// 27a - 1 as (9a - 1) + 18a; computing it directly gives the same value
// with a shorter dependency chain. Lanes outside the complex mask get
// a = 0, hence a1 = a2 = a3 = 0. Clobbers T0-T5.
#define FILTER6(P2, P1, P0, Q0, Q1, Q2) \
	FILTER2(P1, P0, Q0, Q1) \
	VAND V29.B16, V26.B16, V26.B16 \
	VMOVI $9, V27.B16 \
	VMOVI $0xff, V28.B16 \
	VMOVI $0xff, V29.B16 \
	VSMLAL V27.B8, V26.B8, V28.H8 \
	VSMLAL2 V27.B16, V26.B16, V29.H8 \
	SQRSHRN_8B(30, 28, 7) \
	SQRSHRN2_16B(30, 29, 7) \
	SQRSHRN_8B(31, 28, 6) \
	SQRSHRN2_16B(31, 29, 6) \
	VMOVI $27, V27.B16 \
	VMOVI $0xff, V28.B16 \
	VMOVI $0xff, V29.B16 \
	VSMLAL V27.B8, V26.B8, V28.H8 \
	VSMLAL2 V27.B16, V26.B16, V29.H8 \
	SQRSHRN_8B(26, 28, 7) \
	SQRSHRN2_16B(26, 29, 7) \
	VSQADD V26.B16, P0.B16, P0.B16 \
	VSQSUB V26.B16, Q0.B16, Q0.B16 \
	VSQADD V31.B16, P1.B16, P1.B16 \
	VSQSUB V31.B16, Q1.B16, Q1.B16 \
	VSQADD V30.B16, P2.B16, P2.B16 \
	VSQSUB V30.B16, Q2.B16, Q2.B16

// FILTER4(P1, P0, Q0, Q1) is DoFilter4_NEON (inner edges) on the masks of
// NEEDS. On hev lanes it is DoFilter2 with a = sat8((p1-q1) + 3*(q0-p0)),
// on the other filtered lanes it uses a = sat8(3*(q0-p0)); both then do
//   p0 += sat8(a+3)>>3, q0 -= a1 = sat8(a+4)>>3,
// and the non-hev lanes also p1 += a3, q1 -= a3 with a3 = (a1+1)>>1. The
// per-lane delta is selected with BIT (hev ? simple : complex) and applied
// once; libwebp applies the two deltas one after the other, which gives
// the same result because each step leaves the other step's lanes
// unchanged. This keeps the dependency chain short. Clobbers T0-T4.
#define FILTER4(P1, P0, Q0, Q1) \
	VSQSUB P0.B16, Q0.B16, V28.B16 \
	VSQSUB Q1.B16, P1.B16, V29.B16 \
	VSQADD V28.B16, V28.B16, V30.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V30.B16, V30.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VBIT V27.B16, V29.B16, V30.B16 \
	VAND V26.B16, V30.B16, V30.B16 \
	VBIC V27.B16, V26.B16, V26.B16 \
	VSQADD V24.B16, V30.B16, V28.B16 \
	VSQADD V25.B16, V30.B16, V29.B16 \
	VSSHR $3, V28.B16, V28.B16 \
	VSSHR $3, V29.B16, V29.B16 \
	VSRSHR $1, V29.B16, V30.B16 \
	VSQADD V28.B16, P0.B16, P0.B16 \
	VSQSUB V29.B16, Q0.B16, Q0.B16 \
	VAND V26.B16, V30.B16, V30.B16 \
	VSQADD V30.B16, P1.B16, P1.B16 \
	VSQSUB V30.B16, Q1.B16, Q1.B16

// TRANSPOSE16 transposes the 16x16 bytes in V0-V15 (row i in Vi) in place
// (column j in Vj), with TRN1/TRN2 on bytes, halfwords, words and
// doublewords, ping-ponging through V16-V31.
#define TRANSPOSE16 \
	VTRN1 V1.B16, V0.B16, V16.B16 \
	VTRN2 V1.B16, V0.B16, V17.B16 \
	VTRN1 V3.B16, V2.B16, V18.B16 \
	VTRN2 V3.B16, V2.B16, V19.B16 \
	VTRN1 V5.B16, V4.B16, V20.B16 \
	VTRN2 V5.B16, V4.B16, V21.B16 \
	VTRN1 V7.B16, V6.B16, V22.B16 \
	VTRN2 V7.B16, V6.B16, V23.B16 \
	VTRN1 V9.B16, V8.B16, V24.B16 \
	VTRN2 V9.B16, V8.B16, V25.B16 \
	VTRN1 V11.B16, V10.B16, V26.B16 \
	VTRN2 V11.B16, V10.B16, V27.B16 \
	VTRN1 V13.B16, V12.B16, V28.B16 \
	VTRN2 V13.B16, V12.B16, V29.B16 \
	VTRN1 V15.B16, V14.B16, V30.B16 \
	VTRN2 V15.B16, V14.B16, V31.B16 \
	VTRN1 V18.H8, V16.H8, V0.H8 \
	VTRN2 V18.H8, V16.H8, V2.H8 \
	VTRN1 V19.H8, V17.H8, V1.H8 \
	VTRN2 V19.H8, V17.H8, V3.H8 \
	VTRN1 V22.H8, V20.H8, V4.H8 \
	VTRN2 V22.H8, V20.H8, V6.H8 \
	VTRN1 V23.H8, V21.H8, V5.H8 \
	VTRN2 V23.H8, V21.H8, V7.H8 \
	VTRN1 V26.H8, V24.H8, V8.H8 \
	VTRN2 V26.H8, V24.H8, V10.H8 \
	VTRN1 V27.H8, V25.H8, V9.H8 \
	VTRN2 V27.H8, V25.H8, V11.H8 \
	VTRN1 V30.H8, V28.H8, V12.H8 \
	VTRN2 V30.H8, V28.H8, V14.H8 \
	VTRN1 V31.H8, V29.H8, V13.H8 \
	VTRN2 V31.H8, V29.H8, V15.H8 \
	VTRN1 V4.S4, V0.S4, V16.S4 \
	VTRN2 V4.S4, V0.S4, V20.S4 \
	VTRN1 V5.S4, V1.S4, V17.S4 \
	VTRN2 V5.S4, V1.S4, V21.S4 \
	VTRN1 V6.S4, V2.S4, V18.S4 \
	VTRN2 V6.S4, V2.S4, V22.S4 \
	VTRN1 V7.S4, V3.S4, V19.S4 \
	VTRN2 V7.S4, V3.S4, V23.S4 \
	VTRN1 V12.S4, V8.S4, V24.S4 \
	VTRN2 V12.S4, V8.S4, V28.S4 \
	VTRN1 V13.S4, V9.S4, V25.S4 \
	VTRN2 V13.S4, V9.S4, V29.S4 \
	VTRN1 V14.S4, V10.S4, V26.S4 \
	VTRN2 V14.S4, V10.S4, V30.S4 \
	VTRN1 V15.S4, V11.S4, V27.S4 \
	VTRN2 V15.S4, V11.S4, V31.S4 \
	VTRN1 V24.D2, V16.D2, V0.D2 \
	VTRN2 V24.D2, V16.D2, V8.D2 \
	VTRN1 V25.D2, V17.D2, V1.D2 \
	VTRN2 V25.D2, V17.D2, V9.D2 \
	VTRN1 V26.D2, V18.D2, V2.D2 \
	VTRN2 V26.D2, V18.D2, V10.D2 \
	VTRN1 V27.D2, V19.D2, V3.D2 \
	VTRN2 V27.D2, V19.D2, V11.D2 \
	VTRN1 V28.D2, V20.D2, V4.D2 \
	VTRN2 V28.D2, V20.D2, V12.D2 \
	VTRN1 V29.D2, V21.D2, V5.D2 \
	VTRN2 V29.D2, V21.D2, V13.D2 \
	VTRN1 V30.D2, V22.D2, V6.D2 \
	VTRN2 V30.D2, V22.D2, V14.D2 \
	VTRN1 V31.D2, V23.D2, V7.D2 \
	VTRN2 V31.D2, V23.D2, V15.D2

// T8X2_FWD transposes the two side by side 8x8 blocks of V8-V15 (row i in
// V8+i) into V0-V7 (column j in Vj, each half its own block), as libwebp's
// Load8x8x2T_NEON.
#define T8X2_FWD \
	VTRN1 V9.B16, V8.B16, V0.B16 \
	VTRN2 V9.B16, V8.B16, V1.B16 \
	VTRN1 V11.B16, V10.B16, V2.B16 \
	VTRN2 V11.B16, V10.B16, V3.B16 \
	VTRN1 V13.B16, V12.B16, V4.B16 \
	VTRN2 V13.B16, V12.B16, V5.B16 \
	VTRN1 V15.B16, V14.B16, V6.B16 \
	VTRN2 V15.B16, V14.B16, V7.B16 \
	VTRN1 V2.H8, V0.H8, V8.H8 \
	VTRN2 V2.H8, V0.H8, V10.H8 \
	VTRN1 V3.H8, V1.H8, V9.H8 \
	VTRN2 V3.H8, V1.H8, V11.H8 \
	VTRN1 V6.H8, V4.H8, V12.H8 \
	VTRN2 V6.H8, V4.H8, V14.H8 \
	VTRN1 V7.H8, V5.H8, V13.H8 \
	VTRN2 V7.H8, V5.H8, V15.H8 \
	VTRN1 V12.S4, V8.S4, V0.S4 \
	VTRN2 V12.S4, V8.S4, V4.S4 \
	VTRN1 V13.S4, V9.S4, V1.S4 \
	VTRN2 V13.S4, V9.S4, V5.S4 \
	VTRN1 V14.S4, V10.S4, V2.S4 \
	VTRN2 V14.S4, V10.S4, V6.S4 \
	VTRN1 V15.S4, V11.S4, V3.S4 \
	VTRN2 V15.S4, V11.S4, V7.S4

// T8X2_BWD is the same transpose from V0-V7 back into V8-V15.
#define T8X2_BWD \
	VTRN1 V1.B16, V0.B16, V8.B16 \
	VTRN2 V1.B16, V0.B16, V9.B16 \
	VTRN1 V3.B16, V2.B16, V10.B16 \
	VTRN2 V3.B16, V2.B16, V11.B16 \
	VTRN1 V5.B16, V4.B16, V12.B16 \
	VTRN2 V5.B16, V4.B16, V13.B16 \
	VTRN1 V7.B16, V6.B16, V14.B16 \
	VTRN2 V7.B16, V6.B16, V15.B16 \
	VTRN1 V10.H8, V8.H8, V0.H8 \
	VTRN2 V10.H8, V8.H8, V2.H8 \
	VTRN1 V11.H8, V9.H8, V1.H8 \
	VTRN2 V11.H8, V9.H8, V3.H8 \
	VTRN1 V14.H8, V12.H8, V4.H8 \
	VTRN2 V14.H8, V12.H8, V6.H8 \
	VTRN1 V15.H8, V13.H8, V5.H8 \
	VTRN2 V15.H8, V13.H8, V7.H8 \
	VTRN1 V4.S4, V0.S4, V8.S4 \
	VTRN2 V4.S4, V0.S4, V12.S4 \
	VTRN1 V5.S4, V1.S4, V9.S4 \
	VTRN2 V5.S4, V1.S4, V13.S4 \
	VTRN1 V6.S4, V2.S4, V10.S4 \
	VTRN2 V6.S4, V2.S4, V14.S4 \
	VTRN1 V7.S4, V3.S4, V11.S4 \
	VTRN2 V7.S4, V3.S4, V15.S4

// LOAD16X8(PA, PB, STRD) loads 8 rows of 8 bytes from PA (stride STRD) into the
// low halves and 8 rows from PB into the high halves of V8-V15 (libwebp's
// Load8x8x2_NEON); PA and PB are advanced by 8*STRD. STORE16X8 is the
// reverse.
#define LOAD16X8(PA, PB, STRD) \
	VLD1.P (PA)(STRD), V8.D[0] \
	VLD1.P (PA)(STRD), V9.D[0] \
	VLD1.P (PA)(STRD), V10.D[0] \
	VLD1.P (PA)(STRD), V11.D[0] \
	VLD1.P (PA)(STRD), V12.D[0] \
	VLD1.P (PA)(STRD), V13.D[0] \
	VLD1.P (PA)(STRD), V14.D[0] \
	VLD1.P (PA)(STRD), V15.D[0] \
	VLD1.P (PB)(STRD), V8.D[1] \
	VLD1.P (PB)(STRD), V9.D[1] \
	VLD1.P (PB)(STRD), V10.D[1] \
	VLD1.P (PB)(STRD), V11.D[1] \
	VLD1.P (PB)(STRD), V12.D[1] \
	VLD1.P (PB)(STRD), V13.D[1] \
	VLD1.P (PB)(STRD), V14.D[1] \
	VLD1.P (PB)(STRD), V15.D[1]

#define STORE16X8(PA, PB, STRD) \
	VST1.P V8.D[0], (PA)(STRD) \
	VST1.P V9.D[0], (PA)(STRD) \
	VST1.P V10.D[0], (PA)(STRD) \
	VST1.P V11.D[0], (PA)(STRD) \
	VST1.P V12.D[0], (PA)(STRD) \
	VST1.P V13.D[0], (PA)(STRD) \
	VST1.P V14.D[0], (PA)(STRD) \
	VST1.P V15.D[0], (PA)(STRD) \
	VST1.P V8.D[1], (PB)(STRD) \
	VST1.P V9.D[1], (PB)(STRD) \
	VST1.P V10.D[1], (PB)(STRD) \
	VST1.P V11.D[1], (PB)(STRD) \
	VST1.P V12.D[1], (PB)(STRD) \
	VST1.P V13.D[1], (PB)(STRD) \
	VST1.P V14.D[1], (PB)(STRD) \
	VST1.P V15.D[1], (PB)(STRD)

// LOAD4X16(PA, PB, STRD) loads 4 bytes from each of 8 rows at PA and 8
// rows at PB (stride STRD) and transposes them into the 4 columns V16-V19, rows of
// PA in lanes 0-7 and rows of PB in lanes 8-15 (libwebp's Load4x16_NEON,
// WORK_AROUND_GCC variant): row r goes to word lane r/4 of V16+r%4, then
// TRN on bytes and halfwords. Clobbers T0-T3; advances PA and PB by 8*STRD.
#define LOAD4X16(PA, PB, STRD) \
	VLD1.P (PA)(STRD), V16.S[0] \
	VLD1.P (PA)(STRD), V17.S[0] \
	VLD1.P (PA)(STRD), V18.S[0] \
	VLD1.P (PA)(STRD), V19.S[0] \
	VLD1.P (PA)(STRD), V16.S[1] \
	VLD1.P (PA)(STRD), V17.S[1] \
	VLD1.P (PA)(STRD), V18.S[1] \
	VLD1.P (PA)(STRD), V19.S[1] \
	VLD1.P (PB)(STRD), V16.S[2] \
	VLD1.P (PB)(STRD), V17.S[2] \
	VLD1.P (PB)(STRD), V18.S[2] \
	VLD1.P (PB)(STRD), V19.S[2] \
	VLD1.P (PB)(STRD), V16.S[3] \
	VLD1.P (PB)(STRD), V17.S[3] \
	VLD1.P (PB)(STRD), V18.S[3] \
	VLD1.P (PB)(STRD), V19.S[3] \
	VTRN1 V17.B16, V16.B16, V26.B16 \
	VTRN2 V17.B16, V16.B16, V27.B16 \
	VTRN1 V19.B16, V18.B16, V28.B16 \
	VTRN2 V19.B16, V18.B16, V29.B16 \
	VTRN1 V28.H8, V26.H8, V16.H8 \
	VTRN2 V28.H8, V26.H8, V18.H8 \
	VTRN1 V29.H8, V27.H8, V17.H8 \
	VTRN2 V29.H8, V27.H8, V19.H8

// STORE4X16(PA, PB, STRD) is the reverse of LOAD4X16 (V16-V19 are clobbered).
#define STORE4X16(PA, PB, STRD) \
	VTRN1 V18.H8, V16.H8, V26.H8 \
	VTRN2 V18.H8, V16.H8, V28.H8 \
	VTRN1 V19.H8, V17.H8, V27.H8 \
	VTRN2 V19.H8, V17.H8, V29.H8 \
	VTRN1 V27.B16, V26.B16, V16.B16 \
	VTRN2 V27.B16, V26.B16, V17.B16 \
	VTRN1 V29.B16, V28.B16, V18.B16 \
	VTRN2 V29.B16, V28.B16, V19.B16 \
	VST1.P V16.S[0], (PA)(STRD) \
	VST1.P V17.S[0], (PA)(STRD) \
	VST1.P V18.S[0], (PA)(STRD) \
	VST1.P V19.S[0], (PA)(STRD) \
	VST1.P V16.S[1], (PA)(STRD) \
	VST1.P V17.S[1], (PA)(STRD) \
	VST1.P V18.S[1], (PA)(STRD) \
	VST1.P V19.S[1], (PA)(STRD) \
	VST1.P V16.S[2], (PB)(STRD) \
	VST1.P V17.S[2], (PB)(STRD) \
	VST1.P V18.S[2], (PB)(STRD) \
	VST1.P V19.S[2], (PB)(STRD) \
	VST1.P V16.S[3], (PB)(STRD) \
	VST1.P V17.S[3], (PB)(STRD) \
	VST1.P V18.S[3], (PB)(STRD) \
	VST1.P V19.S[3], (PB)(STRD)

#define FLIP16_19 \
	FLIP(V16) \
	FLIP(V17) \
	FLIP(V18) \
	FLIP(V19)

#define FLIP0_7 \
	FLIP(V0) \
	FLIP(V1) \
	FLIP(V2) \
	FLIP(V3) \
	FLIP(V4) \
	FLIP(V5) \
	FLIP(V6) \
	FLIP(V7)

#define FLIP8_15 \
	FLIP(V8) \
	FLIP(V9) \
	FLIP(V10) \
	FLIP(V11) \
	FLIP(V12) \
	FLIP(V13) \
	FLIP(V14) \
	FLIP(V15)

// The three inner edges of a 16x16 luma block held as columns or rows in
// V0-V15: edge k (4, 8, 12) has p3..q3 = V(k-4)..V(k+3).
#define INNER16 \
	NEEDS(0, 1, 2, 3, 4, 5, 6, 7, V21) \
	FILTER4(V2, V3, V4, V5) \
	NEEDS(4, 5, 6, 7, 8, 9, 10, 11, V21) \
	FILTER4(V6, V7, V8, V9) \
	NEEDS(8, 9, 10, 11, 12, 13, 14, 15, V21) \
	FILTER4(V10, V11, V12, V13)

// func filterMBComplexNEON(y, u, v *byte, yStride, uvStride, thrMB, thrIn, ilevel, hevThresh, flags int)
//
// Mirrors libwebp's DoFilter for the complex filter: VP8HFilter16/8 (left
// edge, flags&1), VP8HFilter16i/8i (inner vertical edges, flags&4),
// VP8VFilter16/8 (top edge, flags&2), VP8VFilter16i/8i (inner horizontal
// edges, flags&4). y, u and v point at the top-left pixel of the
// macroblock; thrMB = limit+4, thrIn = limit (both <= 254), ilevel and
// hevThresh as in fInfo. The caller guarantees that rows -4..15 (luma) /
// -4..7 (chroma) and columns -4..15 / -4..7 are inside the planes whenever
// the corresponding edge is filtered.
//
// With inner edges, the whole luma block is loaded and transposed once
// (columns in V0-V15), the left edge uses the four columns to its left
// (V16-V19), the inner vertical edges are filtered in registers, the
// block is transposed back, then the top edge (rows -4..-1 in V16-V19) and
// the inner horizontal edges are filtered and everything is stored once.
// Chroma does the same with U|V packed rows. Without inner edges only the
// pixels next to the edges are loaded: an 8x16 strip transposed through
// V8-V15/V0-V7 for the left edge, 8 rows for the top edge.
//
// GPRs: R0-R2 plane pointers, R3/R4 strides, R5-R8 thresholds, R9 flags,
// R10-R14 scratch pointers, R15 plane counter.
TEXT ·filterMBComplexNEON(SB), NOSPLIT, $0-80
	MOVD y+0(FP), R0
	MOVD u+8(FP), R1
	MOVD v+16(FP), R2
	MOVD yStride+24(FP), R3
	MOVD uvStride+32(FP), R4
	MOVD thrMB+40(FP), R5
	MOVD thrIn+48(FP), R6
	MOVD ilevel+56(FP), R7
	MOVD hevThresh+64(FP), R8
	MOVD flags+72(FP), R9
	TBZ $2, R9, edges

	// ---- luma, with inner edges
	MOVD R0, R10
	VLD1.P (R10)(R3), [V0.B16]
	VLD1.P (R10)(R3), [V1.B16]
	VLD1.P (R10)(R3), [V2.B16]
	VLD1.P (R10)(R3), [V3.B16]
	VLD1.P (R10)(R3), [V4.B16]
	VLD1.P (R10)(R3), [V5.B16]
	VLD1.P (R10)(R3), [V6.B16]
	VLD1.P (R10)(R3), [V7.B16]
	VLD1.P (R10)(R3), [V8.B16]
	VLD1.P (R10)(R3), [V9.B16]
	VLD1.P (R10)(R3), [V10.B16]
	VLD1.P (R10)(R3), [V11.B16]
	VLD1.P (R10)(R3), [V12.B16]
	VLD1.P (R10)(R3), [V13.B16]
	VLD1.P (R10)(R3), [V14.B16]
	VLD1.P (R10)(R3), [V15.B16]
	SIGN0x80
	FLIP0_7
	FLIP8_15
	TRANSPOSE16
	CONSTS
	TBZ $0, R9, ylnoleft

	// left edge: columns -4..-1 of rows 0-7 and 8-15
	SUB $4, R0, R10
	ADD R3<<3, R10, R11
	LOAD4X16(R10, R11, R3)
	SIGN0x80
	FLIP16_19
	NEEDS(16, 17, 18, 19, 0, 1, 2, 3, V20)
	FILTER6(V17, V18, V19, V0, V1, V2)
	SIGN0x80
	FLIP16_19
	SUB $4, R0, R10
	ADD R3<<3, R10, R11
	STORE4X16(R10, R11, R3)

ylnoleft:
	INNER16
	TRANSPOSE16
	CONSTS
	TBZ $1, R9, ylnotop

	// top edge: rows -4..-1
	SUB R3<<2, R0, R10
	VLD1.P (R10)(R3), [V16.B16]
	VLD1.P (R10)(R3), [V17.B16]
	VLD1.P (R10)(R3), [V18.B16]
	VLD1.P (R10)(R3), [V19.B16]
	SIGN0x80
	FLIP16_19
	NEEDS(16, 17, 18, 19, 0, 1, 2, 3, V20)
	FILTER6(V17, V18, V19, V0, V1, V2)
	SIGN0x80
	FLIP(V17)
	FLIP(V18)
	FLIP(V19)
	SUB R3<<2, R0, R10
	ADD R3, R10, R10
	VST1.P [V17.B16], (R10)(R3)
	VST1.P [V18.B16], (R10)(R3)
	VST1.P [V19.B16], (R10)(R3)

ylnotop:
	INNER16
	SIGN0x80
	FLIP0_7
	FLIP8_15
	MOVD R0, R10
	VST1.P [V0.B16], (R10)(R3)
	VST1.P [V1.B16], (R10)(R3)
	VST1.P [V2.B16], (R10)(R3)
	VST1.P [V3.B16], (R10)(R3)
	VST1.P [V4.B16], (R10)(R3)
	VST1.P [V5.B16], (R10)(R3)
	VST1.P [V6.B16], (R10)(R3)
	VST1.P [V7.B16], (R10)(R3)
	VST1.P [V8.B16], (R10)(R3)
	VST1.P [V9.B16], (R10)(R3)
	VST1.P [V10.B16], (R10)(R3)
	VST1.P [V11.B16], (R10)(R3)
	VST1.P [V12.B16], (R10)(R3)
	VST1.P [V13.B16], (R10)(R3)
	VST1.P [V14.B16], (R10)(R3)
	VST1.P [V15.B16], (R10)(R3)

	// ---- chroma, with inner edges: rows of U in the low halves, V in
	// the high halves
	MOVD R1, R10
	MOVD R2, R11
	LOAD16X8(R10, R11, R4)
	SIGN0x80
	FLIP8_15
	T8X2_FWD
	TBZ $0, R9, uvlnoleft

	SUB $4, R1, R10
	SUB $4, R2, R11
	LOAD4X16(R10, R11, R4)
	SIGN0x80
	FLIP16_19
	NEEDS(16, 17, 18, 19, 0, 1, 2, 3, V20)
	FILTER6(V17, V18, V19, V0, V1, V2)
	SIGN0x80
	FLIP16_19
	SUB $4, R1, R10
	SUB $4, R2, R11
	STORE4X16(R10, R11, R4)

uvlnoleft:
	NEEDS(0, 1, 2, 3, 4, 5, 6, 7, V21)
	FILTER4(V2, V3, V4, V5)
	T8X2_BWD
	TBZ $1, R9, uvlnotop

	SUB R4<<2, R1, R10
	SUB R4<<2, R2, R11
	VLD1.P (R10)(R4), V16.D[0]
	VLD1.P (R10)(R4), V17.D[0]
	VLD1.P (R10)(R4), V18.D[0]
	VLD1.P (R10)(R4), V19.D[0]
	VLD1.P (R11)(R4), V16.D[1]
	VLD1.P (R11)(R4), V17.D[1]
	VLD1.P (R11)(R4), V18.D[1]
	VLD1.P (R11)(R4), V19.D[1]
	SIGN0x80
	FLIP16_19
	NEEDS(16, 17, 18, 19, 8, 9, 10, 11, V20)
	FILTER6(V17, V18, V19, V8, V9, V10)
	SIGN0x80
	FLIP(V17)
	FLIP(V18)
	FLIP(V19)
	SUB R4<<2, R1, R10
	SUB R4<<2, R2, R11
	ADD R4, R10, R10
	ADD R4, R11, R11
	VST1.P V17.D[0], (R10)(R4)
	VST1.P V18.D[0], (R10)(R4)
	VST1.P V19.D[0], (R10)(R4)
	VST1.P V17.D[1], (R11)(R4)
	VST1.P V18.D[1], (R11)(R4)
	VST1.P V19.D[1], (R11)(R4)

uvlnotop:
	NEEDS(8, 9, 10, 11, 12, 13, 14, 15, V21)
	FILTER4(V10, V11, V12, V13)
	SIGN0x80
	FLIP8_15
	MOVD R1, R10
	MOVD R2, R11
	STORE16X8(R10, R11, R4)
	RET

	// ---- no inner edges: the same code for luma (plane 0) and chroma
	// (plane 1). R12 = stride, R10/R11 = left strip rows 0-7 / 8-15,
	// R13/R14 = top rows -4..3, left / right half.
edges:
	CONSTS
	MOVD $0, R15

eplane:
	CBNZ R15, euv
	MOVD R3, R12
	SUB $4, R0, R10
	ADD R12<<3, R10, R11
	SUB R12<<2, R0, R13
	ADD $8, R13, R14
	B eleft

euv:
	MOVD R4, R12
	SUB $4, R1, R10
	SUB $4, R2, R11
	SUB R12<<2, R1, R13
	SUB R12<<2, R2, R14

eleft:
	TBZ $0, R9, etop
	LOAD16X8(R10, R11, R12)
	SIGN0x80
	FLIP8_15
	T8X2_FWD
	NEEDS(0, 1, 2, 3, 4, 5, 6, 7, V20)
	FILTER6(V1, V2, V3, V4, V5, V6)
	T8X2_BWD
	SIGN0x80
	FLIP8_15
	SUB R12<<3, R10, R10
	SUB R12<<3, R11, R11
	STORE16X8(R10, R11, R12)

etop:
	TBZ $1, R9, enext
	MOVD R13, R10
	MOVD R14, R11
	LOAD16X8(R10, R11, R12)
	SIGN0x80
	FLIP8_15
	NEEDS(8, 9, 10, 11, 12, 13, 14, 15, V20)
	FILTER6(V9, V10, V11, V12, V13, V14)
	SIGN0x80
	FLIP8_15
	MOVD R13, R10
	MOVD R14, R11
	STORE16X8(R10, R11, R12)

enext:
	ADD $1, R15, R15
	CMP $2, R15
	BNE eplane
	RET

// SIMPLE(P1, P0, Q0, Q1, p1, p0, q0, q1, THR) is libwebp's
// NeedsFilter_NEON + DoFilter2_NEON (simple filter) on signed pixels:
// where 2*|p0-q0| + |p1-q1|/2 <= THR,
//   a = sat8((p1-q1) + 3*(q0-p0)); p0 += sat8(a+3)>>3; q0 -= sat8(a+4)>>3.
// Clobbers T0-T3.
#define SIMPLE(P1, P0, Q0, Q1, p1, p0, q0, q1, THR) \
	SABD(26, p0, q0) \
	VUQADD V26.B16, V26.B16, V26.B16 \
	SABD(27, p1, q1) \
	VUSHR $1, V27.B16, V27.B16 \
	VUQADD V27.B16, V26.B16, V26.B16 \
	VCMHS V26.B16, THR.B16, V26.B16 \
	VSQSUB P0.B16, Q0.B16, V28.B16 \
	VSQSUB Q1.B16, P1.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VSQADD V28.B16, V29.B16, V29.B16 \
	VAND V29.B16, V26.B16, V26.B16 \
	VSQADD V24.B16, V26.B16, V28.B16 \
	VSQADD V25.B16, V26.B16, V26.B16 \
	VSSHR $3, V28.B16, V28.B16 \
	VSSHR $3, V26.B16, V26.B16 \
	VSQADD V28.B16, P0.B16, P0.B16 \
	VSQSUB V26.B16, Q0.B16, Q0.B16

// func filterMBSimpleNEON(y *byte, yStride, thrMB, thrIn, flags int)
//
// Mirrors libwebp's DoFilter for the simple filter (luma only):
// SimpleHFilter16 (left edge, flags&1), SimpleHFilter16i (flags&4),
// SimpleVFilter16 (top edge, flags&2), SimpleVFilter16i (flags&4), one
// edge at a time. Vertical edges load 4 columns of 16 rows with LOAD4X16,
// horizontal edges 4 rows. The caller guarantees that rows -2..15 and
// columns -2..15 are inside the plane whenever the edges reaching them
// are filtered.
//
// GPRs: R0 y, R1 stride, R2/R3 thresholds, R4 flags, R5 edge offset, R6
// last edge offset, R7 threshold of the current edge, R10/R11 pointers.
// V16-V19 hold p1, p0, q0, q1; V20 the current threshold.
TEXT ·filterMBSimpleNEON(SB), NOSPLIT, $0-40
	MOVD y+0(FP), R0
	MOVD yStride+8(FP), R1
	MOVD thrMB+16(FP), R2
	MOVD thrIn+24(FP), R3
	MOVD flags+32(FP), R4
	VMOVI $3, V24.B16
	VMOVI $4, V25.B16
	VMOVI $0x80, V31.B16

	// Edges k in [first, last], step 4: first = 0 if the left/top edge is
	// filtered, else 4; last = 12 with inner edges, else 0.
	MOVD $0, R6
	TBZ $2, R4, sfirstv
	MOVD $12, R6

sfirstv:
	MOVD $0, R5
	TBNZ $0, R4, svloop
	MOVD $4, R5

svloop:
	CMP R6, R5
	BGT shoriz
	MOVD R3, R7
	CBNZ R5, svthr
	MOVD R2, R7

svthr:
	VDUP R7, V20.B16
	ADD R5, R0, R10
	SUB $2, R10, R10
	ADD R1<<3, R10, R11
	LOAD4X16(R10, R11, R1)
	FLIP16_19
	SIMPLE(V16, V17, V18, V19, 16, 17, 18, 19, V20)
	FLIP16_19
	ADD R5, R0, R10
	SUB $2, R10, R10
	ADD R1<<3, R10, R11
	STORE4X16(R10, R11, R1)
	ADD $4, R5, R5
	B svloop

shoriz:
	MOVD $0, R5
	TBNZ $1, R4, shloop
	MOVD $4, R5

shloop:
	CMP R6, R5
	BGT sdone
	MOVD R3, R7
	CBNZ R5, shthr
	MOVD R2, R7

shthr:
	VDUP R7, V20.B16
	MUL R5, R1, R10
	ADD R0, R10, R10
	SUB R1<<1, R10, R10
	MOVD R10, R11
	VLD1.P (R10)(R1), [V16.B16]
	VLD1.P (R10)(R1), [V17.B16]
	VLD1.P (R10)(R1), [V18.B16]
	VLD1.P (R10)(R1), [V19.B16]
	FLIP16_19
	SIMPLE(V16, V17, V18, V19, 16, 17, 18, 19, V20)
	FLIP(V17)
	FLIP(V18)
	ADD R1, R11, R11
	VST1.P [V17.B16], (R11)(R1)
	VST1.P [V18.B16], (R11)(R1)
	ADD $4, R5, R5
	B shloop

sdone:
	RET
