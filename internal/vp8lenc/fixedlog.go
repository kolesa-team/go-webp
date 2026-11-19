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

package vp8lenc

import (
	"math"
	"math/bits"
)

// Fixed-point entropy helpers matching libwebp 1.6 (src/dsp/lossless_enc.c,
// BSD-3-Clause), so that cost comparisons resolve ties like libwebp does.

const (
	log2PrecisionBits = 23
	log2Reciprocal    = 12102203 // round((1 << log2PrecisionBits) / ln(2))
	// log2ReciprocalDouble is (1 << log2PrecisionBits) / ln(2).
	log2ReciprocalDouble = 12102203.161561485379934310913085937500
	approxLogMax         = 4096
	approxLogCorrMax     = 65536
)

// fastLog2 returns log2(v) in fixed point (libwebp VP8LFastLog2).
func fastLog2(v uint32) uint32 {
	if v < 256 {
		return kLog2Table[v]
	}
	if v < approxLogCorrMax {
		logCnt := uint32(bits.Len32(v)) - 1 - 7
		y := uint32(1) << logCnt
		r := kLog2Table[v>>logCnt] + logCnt<<log2PrecisionBits
		if v >= approxLogMax {
			corr := uint64(log2Reciprocal) * uint64(v&(y-1))
			r += uint32(divRound(int64(corr), int64(v)))
		}
		return r
	}
	return uint32(log2ReciprocalDouble*math.Log(float64(v)) + .5)
}

// fastSLog2 returns v*log2(v) in fixed point (libwebp VP8LFastSLog2).
func fastSLog2(v uint32) uint64 {
	if v < 256 {
		return kSLog2Table[v]
	}
	if v < approxLogCorrMax {
		logCnt := uint32(bits.Len32(v)) - 1 - 7
		y := uint32(1) << logCnt
		corr := uint64(log2Reciprocal) * uint64(v&(y-1))
		return uint64(v)*(uint64(kLog2Table[v>>logCnt])+uint64(logCnt)<<log2PrecisionBits) + corr
	}
	return uint64(log2ReciprocalDouble*float64(v)*math.Log(float64(v)) + .5)
}

// divRound is libwebp's DivRound: a/b rounded half away from zero.
func divRound(a, b int64) int64 {
	if (a < 0) == (b < 0) {
		return (a + b/2) / b
	}
	return (a - b/2) / b
}
