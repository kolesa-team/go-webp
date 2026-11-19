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

//go:build amd64 && !purego

package vp8

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// SSE4.1 and AVX2 loop filters (filter_amd64.s), one call per macroblock,
// selected by filterLevel. The simple filter has an SSE4.1 kernel only
// (the AVX2 level uses it).

// filterMBComplexSSE41 filters the edges of one macroblock selected by
// flags (filterEdgeLeft, filterEdgeTop, filterEdgeInner) with the complex
// filter, luma then U and V. y, u and v point at the macroblock's top-left
// samples; rows -4..15 / -4..7 and columns -4..15 / -4..7 around them must
// be accessible for the filtered edges (see blockInBounds). thrMB =
// limit+4 and thrIn = limit must be at most 254.
//
//go:noescape
func filterMBComplexSSE41(y, u, v *byte, yStride, uvStride, thrMB, thrIn, ilevel, hevThresh, flags int)

// filterMBComplexAVX2 is filterMBComplexSSE41 filtering luma and chroma
// together in the two lanes of ymm registers.
//
//go:noescape
func filterMBComplexAVX2(y, u, v *byte, yStride, uvStride, thrMB, thrIn, ilevel, hevThresh, flags int)

// filterMBSimpleSSE41 filters the luma edges selected by flags with the
// simple filter; rows -2..15 and columns -2..15 must be accessible for the
// filtered edges.
//
//go:noescape
func filterMBSimpleSSE41(y *byte, yStride, thrMB, thrIn, flags int)

// filterMB applies the in-loop filter to the macroblock at column mbX of
// the current cache row (libwebp's DoFilter); it matches filterMBGo bit for
// bit.
func (d *decoder) filterMB(mbX, mbY int, f fInfo) {
	limit := int(f.limit)
	if limit == 0 {
		return
	}
	flags := 0
	if mbX > 0 {
		flags |= filterEdgeLeft
	}
	if mbY > 0 {
		flags |= filterEdgeTop
	}
	if f.inner != 0 {
		flags |= filterEdgeInner
	}
	if flags == 0 {
		return
	}
	// The kernels compare thresholds as bytes; VP8 limits are at most
	// 2*63+63 = 189, so the fallback is never taken by valid streams.
	if limit > 250 || filterLevel < cpuinfo.SSE41 {
		d.filterMBGo(mbX, mbY, f)
		return
	}
	ys := d.cacheYStride
	yo := d.cacheYOff + mbX*16
	cy := d.cacheY
	if d.filterType == 1 {
		if !blockInBounds(len(cy), yo, ys, 16, 2, flags) {
			d.filterMBGo(mbX, mbY, f)
			return
		}
		filterMBSimpleSSE41(&cy[yo], ys, limit+4, limit, flags)
		return
	}
	uvs := d.cacheUVStride
	uvo := d.cacheUVOff + mbX*8
	cu, cv := d.cacheU, d.cacheV
	if !blockInBounds(len(cy), yo, ys, 16, 4, flags) || !blockInBounds(min(len(cu), len(cv)), uvo, uvs, 8, 4, flags) {
		d.filterMBGo(mbX, mbY, f)
		return
	}
	if filterLevel >= cpuinfo.AVX2 {
		filterMBComplexAVX2(&cy[yo], &cu[uvo], &cv[uvo], ys, uvs, limit+4, limit, int(f.ilevel), int(f.hevThresh), flags)
	} else {
		filterMBComplexSSE41(&cy[yo], &cu[uvo], &cv[uvo], ys, uvs, limit+4, limit, int(f.ilevel), int(f.hevThresh), flags)
	}
}
