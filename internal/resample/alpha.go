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

package resample

// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/alpha_processing.c (WebPMultARGBRow and
// the GetScale tables).

const (
	mfix    = 24
	half    = uint32(1) << (mfix - 1)
	kInv255 = uint32((1 << mfix) / 255)
)

// premulScale[a] = a * KINV_255 and unpremulScale[a] = (255 << 24) / a, as
// in libwebp's GetScale (USE_TABLES_FOR_ALPHA_MULT == 0).
var premulScale, unpremulScale [256]uint32

func init() {
	for a := uint32(1); a < 256; a++ {
		premulScale[a] = a * kInv255
		unpremulScale[a] = (255 << mfix) / a
	}
}

// mult is libwebp's Mult (32-bit arithmetic, wrapping like the C code).
func mult(x byte, scale uint32) byte { return byte((uint32(x)*scale + half) >> mfix) }

// PremultiplyRGBA multiplies the R, G and B samples of an RGBA row by alpha,
// exactly like libwebp's WebPMultARGBRow(..., inverse=0); fully transparent
// pixels become 0,0,0,0.
func PremultiplyRGBA(row []byte) { premultiplyRow(row) }

// UnpremultiplyRGBA reverses PremultiplyRGBA, exactly like libwebp's
// WebPMultARGBRow(..., inverse=1).
func UnpremultiplyRGBA(row []byte) { unpremultiplyRow(row) }

func multRGBA(row []byte, table *[256]uint32) {
	for i := 0; i+4 <= len(row); i += 4 {
		p := row[i : i+4 : i+4]
		a := p[3]
		if a == 255 {
			continue
		}
		if a == 0 {
			p[0], p[1], p[2] = 0, 0, 0
			continue
		}
		s := table[a]
		p[0] = mult(p[0], s)
		p[1] = mult(p[1], s)
		p[2] = mult(p[2], s)
	}
}
