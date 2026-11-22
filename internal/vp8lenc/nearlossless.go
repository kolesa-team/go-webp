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

// Near-lossless preprocessing.
// Algorithm adapted from libwebp (BSD-3-Clause).

const minDimForNearLossless = 64

// nearLosslessBits maps a near-lossless quality (0..100) to quantization bits.
func nearLosslessBits(q int) int { return 5 - q/20 }

func findClosestDiscretized(a uint32, bits uint) uint32 {
	mask := uint32(1)<<bits - 1
	biased := a + (mask >> 1) + ((a >> bits) & 1)
	if biased > 0xff {
		return 0xff
	}
	return biased &^ mask
}

func closestDiscretizedArgb(a uint32, bits uint) uint32 {
	return findClosestDiscretized(a>>24, bits)<<24 |
		findClosestDiscretized((a>>16)&0xff, bits)<<16 |
		findClosestDiscretized((a>>8)&0xff, bits)<<8 |
		findClosestDiscretized(a&0xff, bits)
}

func isNear(a, b uint32, limit int32) bool {
	for k := uint(0); k < 32; k += 8 {
		d := int32((a>>k)&0xff) - int32((b>>k)&0xff)
		if d >= limit || d <= -limit {
			return false
		}
	}
	return true
}

func nearLosslessPass(xsize, ysize int, argb []uint32, limitBits int, buf []uint32) {
	limit := int32(1) << uint(limitBits)
	prev := buf[:xsize]
	cur := buf[xsize : 2*xsize]
	next := buf[2*xsize : 3*xsize]
	copy(cur, argb[:xsize])
	copy(next, argb[xsize:2*xsize])
	for y := 0; y < ysize; y++ {
		dst := argb[y*xsize : (y+1)*xsize]
		if y != 0 && y != ysize-1 {
			copy(next, argb[(y+1)*xsize:(y+2)*xsize])
			for x := 1; x < xsize-1; x++ {
				c := cur[x]
				if isNear(c, cur[x-1], limit) && isNear(c, cur[x+1], limit) &&
					isNear(c, prev[x], limit) && isNear(c, next[x], limit) {
					dst[x] = c
				} else {
					dst[x] = closestDiscretizedArgb(c, uint(limitBits))
				}
			}
		}
		prev, cur, next = cur, next, prev
	}
}

// applyNearLossless quantizes non-smooth pixels of argb in place.
func applyNearLossless(xsize, ysize int, quality int, argb []uint32) {
	bits := nearLosslessBits(quality)
	if bits <= 0 {
		return
	}
	if (xsize < minDimForNearLossless && ysize < minDimForNearLossless) || ysize < 3 {
		return
	}
	buf := getU32(3 * xsize)
	for b := bits; b > 0; b-- {
		nearLosslessPass(xsize, ysize, argb, b, buf)
	}
	putU32(buf)
}

// ---- near-lossless quantization inside the predictor ----

func addPixels(a, b uint32) uint32 {
	ag := (a & 0xff00ff00) + (b & 0xff00ff00)
	rb := (a & 0x00ff00ff) + (b & 0x00ff00ff)
	return (ag & 0xff00ff00) | (rb & 0x00ff00ff)
}

func addGreenToBlueAndRed(p uint32) uint32 {
	g := (p >> 8) & 0xff
	rb := (p & 0x00ff00ff) + (g<<16 | g)
	return p&0xff00ff00 | rb&0x00ff00ff
}

func maxDiffBetweenPixels(p1, p2 uint32) int {
	m := 0
	for k := uint(0); k < 32; k += 8 {
		d := int((p1>>k)&0xff) - int((p2>>k)&0xff)
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

// computeMaxDiffs fills md with, for every interior pixel, the maximum
// channel difference to its 4 neighbours (libwebp MaxDiffsForRow).
func computeMaxDiffs(width, height int, argb []uint32, usedSubGreen bool, md []uint8) {
	conv := func(p uint32) uint32 {
		if usedSubGreen {
			return addGreenToBlueAndRed(p)
		}
		return p
	}
	for y := 1; y+1 < height; y++ {
		row := argb[y*width : (y+1)*width]
		up := argb[(y-1)*width : y*width]
		down := argb[(y+1)*width : (y+2)*width]
		out := md[y*width : (y+1)*width]
		if width <= 2 {
			continue
		}
		cur := conv(row[0])
		right := conv(row[1])
		for x := 1; x < width-1; x++ {
			left := cur
			cur = right
			right = conv(row[x+1])
			u, d := conv(up[x]), conv(down[x])
			m := max(maxDiffBetweenPixels(cur, u), maxDiffBetweenPixels(cur, d),
				maxDiffBetweenPixels(cur, left), maxDiffBetweenPixels(cur, right))
			if m > 255 {
				m = 255
			}
			out[x] = uint8(m)
		}
	}
}

func nearLosslessComponent(value, predict, boundary, quant int) int {
	residual := (value - predict) & 0xff
	boundaryResidual := (boundary - predict) & 0xff
	lower := residual &^ (quant - 1)
	upper := lower + quant
	bias := 0
	if (boundary-value)&0xff < boundaryResidual {
		bias = 1
	}
	if residual-lower < upper-residual+bias {
		if residual > boundaryResidual && lower <= boundaryResidual {
			return lower + quant>>1
		}
		return lower
	}
	if residual <= boundaryResidual && upper > boundaryResidual {
		return lower + quant>>1
	}
	return upper & 0xff
}

// nearLosslessResidual quantizes the residual value-predict.
func nearLosslessResidual(value, predict uint32, maxQuant, maxDiff int, usedSubGreen bool) uint32 {
	if maxDiff <= 2 {
		return subPixels(value, predict)
	}
	q := maxQuant
	for q >= maxDiff {
		q >>= 1
	}
	var a int
	va := int(value >> 24)
	if va == 0 || va == 0xff {
		a = (va - int(predict>>24)) & 0xff
	} else {
		a = nearLosslessComponent(va, int(predict>>24), 0xff, q)
	}
	g := nearLosslessComponent(int((value>>8)&0xff), int((predict>>8)&0xff), 0xff, q)
	newGreen, greenDiff := 0, 0
	if usedSubGreen {
		newGreen = (int((predict>>8)&0xff) + g) & 0xff
		greenDiff = (newGreen - int((value>>8)&0xff)) & 0xff
	}
	r := nearLosslessComponent((int((value>>16)&0xff)-greenDiff)&0xff, int((predict>>16)&0xff), 0xff-newGreen, q)
	b := nearLosslessComponent((int(value&0xff)-greenDiff)&0xff, int(predict&0xff), 0xff-newGreen, q)
	return uint32(a)<<24 | uint32(r)<<16 | uint32(g)<<8 | uint32(b)
}
