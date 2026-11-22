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

// Row kernels in pure Go. They are the reference for the assembly ones and
// the implementation on other architectures (see rescaler_generic.go).
// Algorithm adapted from libwebp (BSD-3-Clause): src/dsp/rescaler.c.
//
// All of them work on whole rows. The import kernels walk the column table
// built by initCols; acc, when not nil, receives the new row added to it
// (the irow accumulation of a vertical shrink).

// shrinkRowRef imports a row of a horizontal shrink with the Go kernels.
func (r *Rescaler) shrinkRowRef(src []byte, acc []uint32) {
	xSub := uint32(r.xSub)
	switch r.numChannels {
	case 1:
		shrinkRow1Go(r.frow, acc, src, r.cols, xSub, r.fxScale, 0)
	case 4:
		shrinkRow4Go(r.frow, acc, src, r.cols, xSub, r.fxScale, [4]uint32{})
	default:
		shrinkRowGo(r.frow, acc, src, r.cols, r.numChannels, xSub, r.fxScale)
	}
}

// expandRowRef imports a row of a horizontal expansion with the Go kernels.
func (r *Rescaler) expandRowRef(src []byte, acc []uint32) {
	xAdd := uint32(r.xAdd)
	if r.srcWidth > 1 {
		switch r.numChannels {
		case 1:
			expandRow1Go(r.frow, acc, src, r.cols, xAdd)
			return
		case 4:
			expandRow4Go(r.frow, acc, src, r.cols, xAdd)
			return
		}
	}
	right := r.numChannels
	if r.srcWidth == 1 {
		right = 0
	}
	expandRowGo(r.frow, acc, src, r.cols, r.numChannels, right, xAdd)
}

// shrinkRowGo is WebPRescalerImportRowShrink_C for ch interleaved channels,
// starting at destination pixel 0 with zero carries.
func shrinkRowGo(frow, acc []uint32, src []byte, cols []uint32, ch int, xSub, fx uint32) {
	nOut := len(cols) / 2
	for c := 0; c < ch; c++ {
		xIn := c
		var sum uint32
		for x := 0; x < nOut; x++ {
			n, na := int(cols[2*x]), cols[2*x+1]
			var base uint32
			for ; n > 0; n-- {
				base = uint32(src[xIn])
				sum += base
				xIn += ch
			}
			fr := base * na
			v := sum*xSub - fr
			frow[x*ch+c] = v
			if acc != nil {
				acc[x*ch+c] += v
			}
			sum = multFix(fr, fx)
		}
	}
}

// shrinkRow1Go is shrinkRowGo for one channel. It resumes at the
// destination pixel the slices start at: src starts at that pixel's first
// source pixel and carry is the rounded fractional part left over by the
// previous destination pixel (0 at the start of a row).
func shrinkRow1Go(frow, acc []uint32, src []byte, cols []uint32, xSub, fx, carry uint32) {
	nOut := len(cols) / 2
	cols = cols[:2*nOut]
	frow = frow[:nOut]
	if acc != nil {
		acc = acc[:nOut]
	}
	sum := carry
	for x := range frow {
		n, na := int(cols[2*x]), cols[2*x+1]
		p := src[:n]
		src = src[n:]
		for _, v := range p[:n-1] {
			sum += uint32(v)
		}
		base := uint32(p[n-1])
		sum += base
		fr := base * na
		v := sum*xSub - fr
		frow[x] = v
		if acc != nil {
			acc[x] += v
		}
		sum = multFix(fr, fx)
	}
}

// shrinkRow4Go is shrinkRowGo for four channels, resuming like
// shrinkRow1Go (carry holds the four channels' carries).
func shrinkRow4Go(frow, acc []uint32, src []byte, cols []uint32, xSub, fx uint32, carry [4]uint32) {
	nOut := len(cols) / 2
	cols = cols[:2*nOut]
	frow = frow[:4*nOut]
	if acc != nil {
		acc = acc[:4*nOut]
	}
	s0, s1, s2, s3 := carry[0], carry[1], carry[2], carry[3]
	for x := 0; x < nOut; x++ {
		n, na := int(cols[2*x]), cols[2*x+1]
		p := src[:4*n]
		src = src[4*n:]
		for ; len(p) > 4; p = p[4:] {
			s0 += uint32(p[0])
			s1 += uint32(p[1])
			s2 += uint32(p[2])
			s3 += uint32(p[3])
		}
		p = p[:4]
		b0, b1, b2, b3 := uint32(p[0]), uint32(p[1]), uint32(p[2]), uint32(p[3])
		s0 += b0
		s1 += b1
		s2 += b2
		s3 += b3
		fr0, fr1, fr2, fr3 := b0*na, b1*na, b2*na, b3*na
		f := frow[4*x : 4*x+4 : 4*x+4]
		f[0] = s0*xSub - fr0
		f[1] = s1*xSub - fr1
		f[2] = s2*xSub - fr2
		f[3] = s3*xSub - fr3
		if acc != nil {
			a := acc[4*x : 4*x+4 : 4*x+4]
			a[0] += f[0]
			a[1] += f[1]
			a[2] += f[2]
			a[3] += f[3]
		}
		s0, s1, s2, s3 = multFix(fr0, fx), multFix(fr1, fx), multFix(fr2, fx), multFix(fr3, fx)
	}
}

// expandRowGo is WebPRescalerImportRowExpand_C for ch interleaved channels.
// right is the offset of the right source pixel from the left one: ch, or
// 0 for a one pixel wide source.
func expandRowGo(frow, acc []uint32, src []byte, cols []uint32, ch, right int, xAdd uint32) {
	nOut := len(cols) / 2
	for x := 0; x < nOut; x++ {
		l, w := int(cols[2*x])*ch, cols[2*x+1]
		for c := 0; c < ch; c++ {
			left, rt := uint32(src[l+c]), uint32(src[l+c+right])
			v := rt*xAdd + (left-rt)*w
			frow[x*ch+c] = v
			if acc != nil {
				acc[x*ch+c] += v
			}
		}
	}
}

// expandRow4Go is expandRowGo for four channels and a source at least two
// pixels wide.
func expandRow4Go(frow, acc []uint32, src []byte, cols []uint32, xAdd uint32) {
	nOut := len(cols) / 2
	cols = cols[:2*nOut]
	frow = frow[:4*nOut]
	if acc != nil {
		acc = acc[:4*nOut]
	}
	for x := 0; x < nOut; x++ {
		l, w := int(cols[2*x])*4, cols[2*x+1]
		p := src[l : l+8 : l+8]
		f := frow[4*x : 4*x+4 : 4*x+4]
		r0, r1, r2, r3 := uint32(p[4]), uint32(p[5]), uint32(p[6]), uint32(p[7])
		f[0] = r0*xAdd + (uint32(p[0])-r0)*w
		f[1] = r1*xAdd + (uint32(p[1])-r1)*w
		f[2] = r2*xAdd + (uint32(p[2])-r2)*w
		f[3] = r3*xAdd + (uint32(p[3])-r3)*w
		if acc != nil {
			a := acc[4*x : 4*x+4 : 4*x+4]
			a[0] += f[0]
			a[1] += f[1]
			a[2] += f[2]
			a[3] += f[3]
		}
	}
}

// expandRow1Go is expandRowGo for one channel and a source at least two
// pixels wide.
func expandRow1Go(frow, acc []uint32, src []byte, cols []uint32, xAdd uint32) {
	nOut := len(cols) / 2
	cols = cols[:2*nOut]
	frow = frow[:nOut]
	if acc != nil {
		acc = acc[:nOut]
	}
	for x := range frow {
		l, w := int(cols[2*x]), cols[2*x+1]
		p := src[l : l+2 : l+2]
		rt := uint32(p[1])
		v := rt*xAdd + (uint32(p[0])-rt)*w
		frow[x] = v
		if acc != nil {
			acc[x] += v
		}
	}
}

// exportExpandGo is the y_accum == 0 branch of
// WebPRescalerExportRowExpand_C: dst = clip8(frow * fy).
func exportExpandGo(dst []byte, frow []uint32, fy uint32) {
	frow = frow[:len(dst)]
	for i, v := range frow {
		dst[i] = clip8(multFix(v, fy))
	}
}

// exportExpandMixGo is the interpolating branch of
// WebPRescalerExportRowExpand_C: dst = clip8(((a*frow + b*irow) >> 32) *
// fy), both products rounded. a + b == 2^32.
func exportExpandMixGo(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	n := len(dst)
	frow, irow = frow[:n], irow[:n]
	a64, b64, fy64 := uint64(a), uint64(b), uint64(fy)
	i := 0
	for ; i+4 <= n; i += 4 {
		f := frow[i : i+4 : i+4]
		g := irow[i : i+4 : i+4]
		d := dst[i : i+4 : i+4]
		j0 := (a64*uint64(f[0]) + b64*uint64(g[0]) + rounder) >> rfix
		j1 := (a64*uint64(f[1]) + b64*uint64(g[1]) + rounder) >> rfix
		j2 := (a64*uint64(f[2]) + b64*uint64(g[2]) + rounder) >> rfix
		j3 := (a64*uint64(f[3]) + b64*uint64(g[3]) + rounder) >> rfix
		d[0] = clip8(uint32((uint64(uint32(j0))*fy64 + rounder) >> rfix))
		d[1] = clip8(uint32((uint64(uint32(j1))*fy64 + rounder) >> rfix))
		d[2] = clip8(uint32((uint64(uint32(j2))*fy64 + rounder) >> rfix))
		d[3] = clip8(uint32((uint64(uint32(j3))*fy64 + rounder) >> rfix))
	}
	for ; i < n; i++ {
		j := uint32((a64*uint64(frow[i]) + b64*uint64(irow[i]) + rounder) >> rfix)
		dst[i] = clip8(uint32((uint64(j)*fy64 + rounder) >> rfix))
	}
}

// exportShrinkGo is WebPRescalerExportRowShrink_C with a non-zero yscale:
// the fraction of the last row that belongs to the next output row moves
// from irow to the next accumulation.
func exportShrinkGo(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	n := len(dst)
	irow, frow = irow[:n], frow[:n]
	for i := range dst {
		fr := uint32((uint64(frow[i]) * uint64(yscale)) >> rfix)
		dst[i] = clip8(multFix(irow[i]-fr, fxy))
		irow[i] = fr
	}
}

// exportShrinkZeroGo is WebPRescalerExportRowShrink_C with yscale == 0.
func exportShrinkZeroGo(dst []byte, irow []uint32, fxy uint32) {
	irow = irow[:len(dst)]
	for i, v := range irow {
		dst[i] = clip8(multFix(v, fxy))
		irow[i] = 0
	}
}
