// Package resample implements libwebp's fixed-point image rescaler
// (WebPRescaler): area-averaging when shrinking, bilinear interpolation when
// expanding, processed one source row at a time with O(width) state.
//
// Results are bit-exact with libwebp's C implementation
// (src/utils/rescaler_utils.c, src/dsp/rescaler.c).
//
// Algorithm adapted from libwebp (BSD-3-Clause).
package resample

import (
	"errors"
	"math"
)

const (
	rfix    = 32
	one     = uint64(1) << rfix
	rounder = one >> 1
)

// frac is WEBP_RESCALER_FRAC(x, y).
func frac(x, y uint64) uint32 { return uint32((x << rfix) / y) }

// multFix is MULT_FIX: rounded 32.32 fixed-point product.
func multFix(x, y uint32) uint32 { return uint32((uint64(x)*uint64(y) + rounder) >> rfix) }

// Rescaler scales an image with numChannels interleaved 8-bit channels from
// srcWidth x srcHeight to dstWidth x dstHeight. Rows are fed with
// ImportRow and produced with ExportRow; the zero value is not usable, see
// New and Init.
type Rescaler struct {
	xExpand, yExpand bool
	numChannels      int
	fxScale          uint32
	fyScale          uint32
	fxyScale         uint32
	yAccum           int
	yAdd, ySub       int
	xAdd, xSub       int
	srcWidth         int
	srcHeight        int
	dstWidth         int
	dstHeight        int
	srcY, dstY       int
	irow, frow       []uint32
	work             []uint32
}

// ErrInvalid is returned for invalid rescaler dimensions.
var ErrInvalid = errors.New("resample: invalid dimensions")

// WorkSize returns the number of uint32 work cells a rescaler with the given
// destination width and channel count needs.
func WorkSize(dstWidth, numChannels int) int { return 2 * dstWidth * numChannels }

// New returns a rescaler with a freshly allocated work buffer.
func New(srcWidth, srcHeight, dstWidth, dstHeight, numChannels int) (*Rescaler, error) {
	r := new(Rescaler)
	if err := r.Init(srcWidth, srcHeight, dstWidth, dstHeight, numChannels, nil); err != nil {
		return nil, err
	}
	return r, nil
}

// Init (re)initializes r, like WebPRescalerInit. work is reused when its
// capacity is at least WorkSize(dstWidth, numChannels); otherwise a new
// buffer is allocated.
func (r *Rescaler) Init(srcWidth, srcHeight, dstWidth, dstHeight, numChannels int, work []uint32) error {
	if srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0 ||
		numChannels <= 0 || numChannels > 4 ||
		srcWidth > math.MaxInt32/2 || srcHeight > math.MaxInt32/2 ||
		dstWidth > math.MaxInt32/2 || dstHeight > math.MaxInt32/2 {
		return ErrInvalid
	}
	n := WorkSize(dstWidth, numChannels)
	if cap(work) < n {
		work = make([]uint32, n)
	} else {
		work = work[:n]
		clear(work)
	}
	*r = Rescaler{
		xExpand:     srcWidth < dstWidth,
		yExpand:     srcHeight < dstHeight,
		numChannels: numChannels,
		srcWidth:    srcWidth,
		srcHeight:   srcHeight,
		dstWidth:    dstWidth,
		dstHeight:   dstHeight,
		work:        work,
		irow:        work[: n/2 : n/2],
		frow:        work[n/2:],
	}
	xAdd, xSub := srcWidth, dstWidth
	yAdd, ySub := srcHeight, dstHeight
	if r.xExpand {
		r.xAdd, r.xSub = xSub-1, xAdd-1
	} else {
		r.xAdd, r.xSub = xAdd, xSub
		r.fxScale = frac(1, uint64(r.xSub))
	}
	if r.yExpand {
		r.yAdd, r.ySub = yAdd-1, ySub-1
		r.yAccum = r.ySub
		r.fyScale = frac(1, uint64(r.xAdd))
	} else {
		r.yAdd, r.ySub = yAdd, ySub
		r.yAccum = r.yAdd
		num := uint64(dstHeight) * one
		den := uint64(r.xAdd) * uint64(r.yAdd)
		ratio := num / den
		if ratio != uint64(uint32(ratio)) {
			r.fxyScale = 0 // special-cased in ExportRow
		} else {
			r.fxyScale = uint32(ratio)
		}
		r.fyScale = frac(1, uint64(r.ySub))
	}
	return nil
}

// SrcY returns the number of source rows imported so far.
func (r *Rescaler) SrcY() int { return r.srcY }

// DstY returns the number of destination rows exported so far.
func (r *Rescaler) DstY() int { return r.dstY }

// DstWidth returns the destination width in pixels.
func (r *Rescaler) DstWidth() int { return r.dstWidth }

// DstHeight returns the destination height in pixels.
func (r *Rescaler) DstHeight() int { return r.dstHeight }

// InputDone reports whether all source rows were imported.
func (r *Rescaler) InputDone() bool { return r.srcY >= r.srcHeight }

// OutputDone reports whether all destination rows were exported.
func (r *Rescaler) OutputDone() bool { return r.dstY >= r.dstHeight }

// HasPendingOutput reports whether a destination row is ready to be
// exported. While it is, no more rows may be imported.
func (r *Rescaler) HasPendingOutput() bool { return !r.OutputDone() && r.yAccum <= 0 }

// NeededLines returns how many source rows must be imported before the next
// output row is ready, capped at maxLines (WebPRescaleNeededLines).
func (r *Rescaler) NeededLines(maxLines int) int {
	n := (r.yAccum + r.ySub - 1) / r.ySub
	return min(n, maxLines)
}

// Import imports up to numLines rows of src (row stride srcStride), stopping
// early when an output row becomes pending. It returns the number of rows
// imported (WebPRescalerImport).
func (r *Rescaler) Import(numLines int, src []byte, srcStride int) int {
	total := 0
	for total < numLines && !r.HasPendingOutput() && !r.InputDone() {
		r.ImportRow(src[total*srcStride:])
		total++
	}
	return total
}

// ImportRow imports one source row of srcWidth*numChannels bytes. It must
// not be called while HasPendingOutput is true or after InputDone.
func (r *Rescaler) ImportRow(src []byte) {
	if r.InputDone() {
		return
	}
	if r.yExpand {
		r.irow, r.frow = r.frow, r.irow
	}
	src = src[:r.srcWidth*r.numChannels]
	if r.xExpand {
		switch r.numChannels {
		case 4:
			r.importExpand4(src)
		default:
			r.importExpand(src)
		}
	} else {
		switch r.numChannels {
		case 4:
			r.importShrink4(src)
		case 1:
			r.importShrink1(src)
		default:
			r.importShrink(src)
		}
	}
	if !r.yExpand {
		irow := r.irow
		frow := r.frow[:len(irow)]
		for i, v := range frow {
			irow[i] += v
		}
	}
	r.srcY++
	r.yAccum -= r.ySub
}

// importExpand is WebPRescalerImportRowExpand_C for any channel count.
func (r *Rescaler) importExpand(src []byte) {
	stride := r.numChannels
	xOutMax := r.dstWidth * stride
	frow := r.frow[:xOutMax]
	xAdd, xSub := uint32(r.xAdd), r.xSub
	for ch := 0; ch < stride; ch++ {
		xIn := ch
		accum := r.xAdd
		left := uint32(src[xIn])
		right := left
		if r.srcWidth > 1 {
			right = uint32(src[xIn+stride])
		}
		xIn += stride
		for xOut := ch; ; {
			frow[xOut] = right*xAdd + (left-right)*uint32(accum)
			xOut += stride
			if xOut >= xOutMax {
				break
			}
			accum -= xSub
			if accum < 0 {
				left = right
				xIn += stride
				right = uint32(src[xIn])
				accum += r.xAdd
			}
		}
	}
}

// importExpand4 is importExpand specialized for 4 interleaved channels.
func (r *Rescaler) importExpand4(src []byte) {
	xOutMax := r.dstWidth * 4
	frow := r.frow[:xOutMax]
	xAdd, xSub := uint32(r.xAdd), r.xSub
	accum := r.xAdd
	l0, l1, l2, l3 := uint32(src[0]), uint32(src[1]), uint32(src[2]), uint32(src[3])
	r0, r1, r2, r3 := l0, l1, l2, l3
	if r.srcWidth > 1 {
		s := src[4:8]
		r0, r1, r2, r3 = uint32(s[0]), uint32(s[1]), uint32(s[2]), uint32(s[3])
	}
	xIn := 4
	for xOut := 0; ; {
		a := uint32(accum)
		f := frow[xOut : xOut+4 : xOut+4]
		f[0] = r0*xAdd + (l0-r0)*a
		f[1] = r1*xAdd + (l1-r1)*a
		f[2] = r2*xAdd + (l2-r2)*a
		f[3] = r3*xAdd + (l3-r3)*a
		xOut += 4
		if xOut >= xOutMax {
			break
		}
		accum -= xSub
		if accum < 0 {
			l0, l1, l2, l3 = r0, r1, r2, r3
			xIn += 4
			s := src[xIn : xIn+4 : xIn+4]
			r0, r1, r2, r3 = uint32(s[0]), uint32(s[1]), uint32(s[2]), uint32(s[3])
			accum += r.xAdd
		}
	}
}

// importShrink is WebPRescalerImportRowShrink_C for any channel count.
func (r *Rescaler) importShrink(src []byte) {
	stride := r.numChannels
	xOutMax := r.dstWidth * stride
	frow := r.frow[:xOutMax]
	xSub := uint32(r.xSub)
	for ch := 0; ch < stride; ch++ {
		xIn := ch
		var sum uint32
		accum := 0
		for xOut := ch; xOut < xOutMax; xOut += stride {
			var base uint32
			accum += r.xAdd
			for accum > 0 {
				accum -= r.xSub
				base = uint32(src[xIn])
				sum += base
				xIn += stride
			}
			fr := base * uint32(-accum)
			frow[xOut] = sum*xSub - fr
			sum = multFix(fr, r.fxScale)
		}
	}
}

// importShrink1 is importShrink specialized for a single channel.
func (r *Rescaler) importShrink1(src []byte) {
	frow := r.frow[:r.dstWidth]
	xSub := uint32(r.xSub)
	xIn := 0
	var sum uint32
	accum := 0
	for xOut := range frow {
		var base uint32
		accum += r.xAdd
		for accum > 0 {
			accum -= r.xSub
			base = uint32(src[xIn])
			sum += base
			xIn++
		}
		fr := base * uint32(-accum)
		frow[xOut] = sum*xSub - fr
		sum = multFix(fr, r.fxScale)
	}
}

// importShrink4 is importShrink specialized for 4 interleaved channels.
func (r *Rescaler) importShrink4(src []byte) {
	xOutMax := r.dstWidth * 4
	frow := r.frow[:xOutMax]
	xSub := uint32(r.xSub)
	fx := r.fxScale
	xIn := 0
	var s0, s1, s2, s3 uint32
	accum := 0
	for xOut := 0; xOut < xOutMax; xOut += 4 {
		var b0, b1, b2, b3 uint32
		accum += r.xAdd
		for accum > 0 {
			accum -= r.xSub
			p := src[xIn : xIn+4 : xIn+4]
			b0, b1, b2, b3 = uint32(p[0]), uint32(p[1]), uint32(p[2]), uint32(p[3])
			s0 += b0
			s1 += b1
			s2 += b2
			s3 += b3
			xIn += 4
		}
		na := uint32(-accum)
		f := frow[xOut : xOut+4 : xOut+4]
		fr0, fr1, fr2, fr3 := b0*na, b1*na, b2*na, b3*na
		f[0] = s0*xSub - fr0
		f[1] = s1*xSub - fr1
		f[2] = s2*xSub - fr2
		f[3] = s3*xSub - fr3
		s0, s1, s2, s3 = multFix(fr0, fx), multFix(fr1, fx), multFix(fr2, fx), multFix(fr3, fx)
	}
}

// ExportRow writes the next destination row (dstWidth*numChannels bytes)
// into dst if one is pending, and reports whether it did
// (WebPRescalerExportRow).
func (r *Rescaler) ExportRow(dst []byte) bool {
	if !r.HasPendingOutput() {
		return false
	}
	n := r.dstWidth * r.numChannels
	dst = dst[:n]
	irow := r.irow[:n]
	frow := r.frow[:n]
	switch {
	case r.yExpand:
		fy := r.fyScale
		if r.yAccum == 0 {
			for i, j := range frow {
				dst[i] = clip8(multFix(j, fy))
			}
		} else {
			b := uint64(frac(uint64(-r.yAccum), uint64(r.ySub)))
			a := uint64(uint32(one - b))
			fy := uint64(fy)
			i := 0
			for ; i+4 <= n; i += 4 {
				f := frow[i : i+4 : i+4]
				g := irow[i : i+4 : i+4]
				d := dst[i : i+4 : i+4]
				j0 := (a*uint64(f[0]) + b*uint64(g[0]) + rounder) >> rfix
				j1 := (a*uint64(f[1]) + b*uint64(g[1]) + rounder) >> rfix
				j2 := (a*uint64(f[2]) + b*uint64(g[2]) + rounder) >> rfix
				j3 := (a*uint64(f[3]) + b*uint64(g[3]) + rounder) >> rfix
				d[0] = clip8(uint32((uint64(uint32(j0))*fy + rounder) >> rfix))
				d[1] = clip8(uint32((uint64(uint32(j1))*fy + rounder) >> rfix))
				d[2] = clip8(uint32((uint64(uint32(j2))*fy + rounder) >> rfix))
				d[3] = clip8(uint32((uint64(uint32(j3))*fy + rounder) >> rfix))
			}
			for ; i < n; i++ {
				j := uint32((a*uint64(frow[i]) + b*uint64(irow[i]) + rounder) >> rfix)
				dst[i] = clip8(uint32((uint64(j)*fy + rounder) >> rfix))
			}
		}
	case r.fxyScale != 0:
		yscale := r.fyScale * uint32(-r.yAccum)
		fxy := r.fxyScale
		if yscale != 0 {
			for i := range dst {
				fr := uint32((uint64(frow[i]) * uint64(yscale)) >> rfix)
				dst[i] = clip8(multFix(irow[i]-fr, fxy))
				irow[i] = fr
			}
		} else {
			for i, v := range irow {
				dst[i] = clip8(multFix(v, fxy))
				irow[i] = 0
			}
		}
	default:
		// Special case src_height == dst_height, x_add == 1.
		for i, v := range irow {
			dst[i] = byte(v)
			irow[i] = 0
		}
	}
	r.yAccum += r.yAdd
	r.dstY++
	return true
}

func clip8(v uint32) byte {
	if v > 255 {
		return 255
	}
	return byte(v)
}

// Export exports all pending rows into dst (row stride dstStride) and
// returns their number (WebPRescalerExport).
func (r *Rescaler) Export(dst []byte, dstStride int) int {
	n := 0
	for r.HasPendingOutput() {
		r.ExportRow(dst[n*dstStride:])
		n++
	}
	return n
}

// ScaledDimensions resolves the requested output size like libwebp's
// WebPRescalerGetScaledDimensions: a zero width (or height) is derived from
// the other dimension keeping the aspect ratio (rounding up). It reports
// false when the result is not a valid size.
func ScaledDimensions(srcWidth, srcHeight, width, height int) (int, int, bool) {
	const maxSize = math.MaxInt32 / 2
	if width < 0 || height < 0 || srcWidth <= 0 || srcHeight <= 0 {
		return 0, 0, false
	}
	if width == 0 {
		width = int((uint64(srcWidth)*uint64(height) + uint64(srcHeight) - 1) / uint64(srcHeight))
	}
	if height == 0 {
		height = int((uint64(srcHeight)*uint64(width) + uint64(srcWidth) - 1) / uint64(srcWidth))
	}
	if width <= 0 || height <= 0 || width > maxSize || height > maxSize {
		return 0, 0, false
	}
	return width, height, true
}
