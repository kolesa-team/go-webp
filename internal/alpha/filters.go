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

package alpha

// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/filters.c and src/utils/filters_utils.c
// (WebPFilters, WebPUnfilters, GradientPredictor, WebPEstimateBestFilter).

import "encoding/binary"

const (
	hiBits = 0x8080808080808080
)

// addBytes sets out[i] = a[i] + b[i] (mod 256) for i < n, eight bytes at a
// time.
func addBytes(out, a, b []byte, n int) {
	out, a, b = out[:n], a[:n], b[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		x := binary.LittleEndian.Uint64(a[i:])
		y := binary.LittleEndian.Uint64(b[i:])
		binary.LittleEndian.PutUint64(out[i:], ((x&^hiBits)+(y&^hiBits))^((x^y)&hiBits))
	}
	for ; i < n; i++ {
		out[i] = a[i] + b[i]
	}
}

// subBytes sets out[i] = a[i] - b[i] (mod 256) for i < n, eight bytes at a
// time.
func subBytes(out, a, b []byte, n int) {
	out, a, b = out[:n], a[:n], b[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		x := binary.LittleEndian.Uint64(a[i:])
		y := binary.LittleEndian.Uint64(b[i:])
		binary.LittleEndian.PutUint64(out[i:], ((x|hiBits)-(y&^hiBits))^((x^^y)&hiBits))
	}
	for ; i < n; i++ {
		out[i] = a[i] - b[i]
	}
}

// gradientPredictor is libwebp's GradientPredictor: clip(a + b - c).
func gradientPredictor(a, b, c byte) byte {
	g := int(a) + int(b) - int(c)
	if g&^0xff == 0 {
		return byte(g)
	}
	if g < 0 {
		return 0
	}
	return 255
}

// UnfilterRow reverses filter on one row: out = unfilter(in) using prev as
// the previous (already reconstructed) row, or nil for the first row. in and
// out may be the same slice; prev must not alias out. This is the streaming
// form used by alpha decoders (libwebp's WebPUnfilters).
func UnfilterRow(filter int, prev, in, out []byte) {
	width := len(in)
	if width == 0 {
		return
	}
	out = out[:width]
	switch filter {
	case FilterHorizontal:
		var pred byte
		if prev != nil {
			pred = prev[0]
		}
		unfilterHorizontal(pred, in, out)
	case FilterVertical:
		if prev == nil {
			unfilterHorizontal(0, in, out)
		} else {
			unfilterVertical(prev, in, out)
		}
	case FilterGradient:
		if prev == nil {
			unfilterHorizontal(0, in, out)
		} else {
			gradientUnfilter(prev[:width], in, out)
		}
	default:
		if &out[0] != &in[0] {
			copy(out, in)
		}
	}
}

// unfilterHorizontalGo is the reference of unfilterHorizontal
// (unfilter_*.go): out[i] = pred + in[0] + ... + in[i] (mod 256), libwebp's
// HorizontalUnfilter_C with pred the left prediction of the first pixel.
// in may alias out.
func unfilterHorizontalGo(pred byte, in, out []byte) {
	out = out[:len(in)]
	for i, v := range in {
		pred += v
		out[i] = pred
	}
}

// unfilterVerticalGo is the reference of unfilterVertical (unfilter_*.go):
// out[i] = prev[i] + in[i] (mod 256) for i < len(in). in may alias out.
func unfilterVerticalGo(prev, in, out []byte) {
	addBytes(out, prev, in, len(in))
}

func gradientUnfilter(prev, in, out []byte) {
	prev = prev[:len(in)]
	out = out[:len(in)]
	top := prev[0]
	topLeft, left := top, top
	for i, v := range in {
		top = prev[i]
		left = v + gradientPredictor(left, top, topLeft)
		topLeft = top
		out[i] = left
	}
}

// Unfilter reverses the prediction filter in place on a width x height
// plane, row by row, with libwebp's semantics (first row is
// left-predicted from 0, first column of the other rows predicted from
// above).
func Unfilter(filter int, plane []byte, width, height, stride int) {
	if filter == FilterNone || width <= 0 || height <= 0 {
		return
	}
	var prev []byte
	for y := 0; y < height; y++ {
		row := plane[y*stride : y*stride+width]
		UnfilterRow(filter, prev, row, row)
		prev = row
	}
}

// Filter applies the forward prediction filter to the width x height plane
// src and writes the residuals to dst (libwebp's WebPFilters). With
// FilterNone the plane is copied.
func Filter(filter int, src []byte, width, height, srcStride int, dst []byte, dstStride int) {
	if width <= 0 || height <= 0 {
		return
	}
	if filter == FilterNone {
		for y := 0; y < height; y++ {
			copy(dst[y*dstStride:y*dstStride+width], src[y*srcStride:y*srcStride+width])
		}
		return
	}
	// Top row: first pixel copied, the rest left-predicted, for all filters.
	in := src[:width]
	out := dst[:width]
	out[0] = in[0]
	subBytes(out[1:], in[1:], in, width-1)
	for y := 1; y < height; y++ {
		in := src[y*srcStride : y*srcStride+width]
		up := src[(y-1)*srcStride : (y-1)*srcStride+width]
		out := dst[y*dstStride : y*dstStride+width]
		switch filter {
		case FilterHorizontal:
			out[0] = in[0] - up[0]
			subBytes(out[1:], in[1:], in, width-1)
		case FilterVertical:
			subBytes(out, in, up, width)
		default: // FilterGradient
			out[0] = in[0] - up[0]
			gradientFilterRow(in, up, out)
		}
	}
}

// gradientFilterRow computes out[x] = in[x] - clip(in[x-1] + up[x] - up[x-1])
// for x >= 1. There is no dependency between iterations, so the clipping is
// done branch-free.
func gradientFilterRow(in, up, out []byte) {
	n := len(in)
	up = up[:n]
	out = out[:n]
	for x := 1; x < n; x++ {
		g := min(max(int(in[x-1])+int(up[x])-int(up[x-1]), 0), 255)
		out[x] = in[x] - byte(g)
	}
}

// EstimateBestFilter mirrors libwebp's WebPEstimateBestFilter: it samples
// every other pixel and picks the filter with the most compact residual
// histogram. (libwebp addresses the row above with the width; it is only
// ever called with stride == width, where both agree.)
func EstimateBestFilter(plane []byte, width, height, stride int) int {
	const smax = 16
	var bins [4][smax]bool
	for j := 2; j < height-1; j += 2 {
		p := plane[j*stride : j*stride+width]
		up := plane[(j-1)*stride : (j-1)*stride+width]
		mean := int(p[0])
		for i := 2; i < width-1; i += 2 {
			v := int(p[i])
			bins[FilterNone][sdiff(v, mean)] = true
			bins[FilterHorizontal][sdiff(v, int(p[i-1]))] = true
			bins[FilterVertical][sdiff(v, int(up[i]))] = true
			bins[FilterGradient][sdiff(v, int(gradientPredictor(p[i-1], up[i], up[i-1])))] = true
			mean = (3*mean + v + 2) >> 2
		}
	}
	best, bestScore := FilterNone, int(^uint(0)>>1)
	for f := FilterNone; f <= FilterGradient; f++ {
		score := 0
		for i, used := range bins[f] {
			if used {
				score += i
			}
		}
		if score < bestScore {
			best, bestScore = f, score
		}
	}
	return best
}

func sdiff(a, b int) int {
	d := a - b
	if d < 0 {
		d = -d
	}
	return (d >> 4) & 15
}
