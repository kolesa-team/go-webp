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

// Package vp8lenc implements a pure-Go WebP lossless (VP8L) encoder.
//
// The encoder implements the full VP8L toolbox: subtract-green, spatial
// prediction with per-tile mode selection, cross-color decorrelation,
// color indexing (palette) with pixel bundling, a color cache, LZ77 backward
// references (hash chain, RLE and cost-based optimal parsing), 2D distance
// codes, meta-Huffman (entropy image) clustering and length-limited Huffman
// codes with run-length compressed code-length codes.
//
// Algorithm adapted from libwebp (BSD-3-Clause).
package vp8lenc

import "math/bits"

const (
	numLiteralCodes   = 256
	numLengthCodes    = 24
	numDistanceCodes  = 40
	maxColorCacheBits = 10
	codeLengthCodes   = 19

	maxLength     = 4095 // maximum LZ77 copy length
	maxLengthBits = 12
	minLength     = 4
	windowSize    = (1 << 20) - 120

	maxDim = 16383

	maxHuffImageSize = 2600
	minHuffmanBits   = 2
	maxHuffmanBits   = 9
	minTransformBits = 2
	maxTransformBits = 9 // MIN_TRANSFORM_BITS + (1 << NUM_TRANSFORM_BITS) - 1

	// maxPredictorImageSize bounds the size of the predictor image (in
	// tiles) when choosing its sampling.
	maxPredictorImageSize = 1 << 14

	argbBlack = 0xff000000

	// Transform types.
	predictorTransform     = 0
	crossColorTransform    = 1
	subtractGreenTransform = 2
	colorIndexingTransform = 3

	vp8lSignature = 0x2f
)

// codeToPlane maps a 2D distance code (1-based, minus 1) to (yoffset<<4)|(8-xoffset).
var codeToPlane = [120]uint8{
	0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
	0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
	0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
	0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
	0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
	0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
	0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
	0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
	0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
	0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
	0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
	0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
}

// planeToCode is the inverse of codeToPlane.
var planeToCode [128]uint8

func init() {
	for i, p := range codeToPlane {
		planeToCode[p] = uint8(i)
	}
}

// distanceToPlaneCode converts a linear backward distance into the VP8L
// distance code (1..120 for short 2D distances, dist+120 otherwise).
func distanceToPlaneCode(xsize int, dist int) int {
	yoff := dist / xsize
	xoff := dist - yoff*xsize
	if xoff <= 8 && yoff < 8 {
		return int(planeToCode[yoff*16+8-xoff]) + 1
	} else if xoff > xsize-8 && yoff < 7 {
		return int(planeToCode[(yoff+1)*16+8+(xsize-xoff)]) + 1
	}
	return dist + 120
}

// prefixEncode splits a length or distance code value (>= 1) into its prefix
// symbol, number of extra bits and extra bits value.
func prefixEncode(v int) (code, nbits, extra int) {
	d := uint32(v - 1)
	if d < 4 {
		return int(d), 0, 0
	}
	hb := bits.Len32(d) - 1
	shb := int(d>>(uint(hb)-1)) & 1
	nbits = hb - 1
	extra = int(d) & (1<<uint(nbits) - 1)
	return 2*hb + shb, nbits, extra
}

// prefixCode returns only the prefix symbol of v.
func prefixCode(v int) int {
	d := uint32(v - 1)
	if d < 4 {
		return int(d)
	}
	hb := bits.Len32(d) - 1
	return 2*hb + int(d>>(uint(hb)-1))&1
}

func subSampleSize(size, bits int) int {
	return (size + (1 << uint(bits)) - 1) >> uint(bits)
}

func log2Floor(v uint32) int { return bits.Len32(v) - 1 }

// subPixels returns the per-channel difference a-b (mod 256).
func subPixels(a, b uint32) uint32 {
	ag := 0x00ff00ff + (a & 0xff00ff00) - (b & 0xff00ff00)
	rb := 0xff00ff00 + (a & 0x00ff00ff) - (b & 0x00ff00ff)
	return (ag & 0xff00ff00) | (rb & 0x00ff00ff)
}

// subtractGreenGo applies the subtract-green transform to p in place: green
// is subtracted (mod 256) from red and blue. It is the Go reference of
// subtractGreen (dsp_amd64.go, dsp_generic.go).
func subtractGreenGo(p []uint32) {
	for i, px := range p {
		g := (px >> 8) & 0xff
		p[i] = px&0xff00ff00 | (((px >> 16) - g) & 0xff << 16) | ((px - g) & 0xff)
	}
}

// colorCacheHash returns the color cache key of argb for the given bits.
func colorCacheHash(argb uint32, shift uint) uint32 {
	return (argb * 0x1e35a7bd) >> shift
}

// ---- pooled scratch buffers ----

// getU32 returns a []uint32 of length n (contents unspecified), from the
// pool (see freelist.go) when it holds one of about that size.
func getU32(n int) []uint32 { return u32Pool.Get(n) }

// putU32 returns b to the pool, if the budget allows it.
func putU32(b []uint32) { u32Pool.Put(b) }
