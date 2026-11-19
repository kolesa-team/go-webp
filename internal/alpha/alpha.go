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

// Package alpha implements the helpers around the WebP ALPH chunk: header
// byte, spatial prediction filters, level quantization (encoder) and level
// dequantization smoothing (decoder "alpha dithering"), with libwebp's exact
// semantics.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/filters.c,
// src/utils/filters_utils.c, src/utils/quant_levels_utils.c,
// src/utils/quant_levels_dec_utils.c, src/enc/alpha_enc.c and
// src/dec/alpha_dec.c.
package alpha

import (
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Prediction filters of the ALPH chunk.
const (
	FilterNone       = 0
	FilterHorizontal = 1
	FilterVertical   = 2
	FilterGradient   = 3
)

// Compression methods of the ALPH chunk.
const (
	CompressionNone     = 0 // raw alpha bytes
	CompressionLossless = 1 // VP8L-compressed green channel
)

// PreprocessingLevels marks an alpha plane quantized by QuantizeLevels; a
// decoder may then smooth it with Dequantize.
const PreprocessingLevels = 1

// Header is the decoded first byte of an ALPH chunk.
type Header struct {
	Compression   int // 0: none, 1: lossless
	Filter        int // FilterNone .. FilterGradient
	PreProcessing int // 0: none, 1: quantized levels
}

// ParseHeader decodes the ALPH header byte
// (compression | filter<<2 | preprocessing<<4, bits 6-7 reserved), with the
// same validation as libwebp's ALPHInit.
func ParseHeader(b byte) (Header, error) {
	h := Header{
		Compression:   int(b & 3),
		Filter:        int(b>>2) & 3,
		PreProcessing: int(b>>4) & 3,
	}
	if h.Compression > CompressionLossless || h.PreProcessing > PreprocessingLevels || b>>6 != 0 {
		return Header{}, status.Bitstream("invalid ALPH header")
	}
	return h, nil
}

// Byte encodes the header.
func (h Header) Byte() byte {
	return byte(h.Compression&3) | byte(h.Filter&3)<<2 | byte(h.PreProcessing&3)<<4
}

// AlphaLevels returns the number of quantization levels libwebp's encoder
// uses for the given alpha_quality (0..100); 256 or more means "no
// quantization" (alpha_quality == 100).
func AlphaLevels(quality int) int {
	if quality <= 70 {
		return 2 + quality/5
	}
	return 16 + (quality-70)*8
}
