// Package resample implements libwebp's fixed-point image rescaler
// (WebPRescaler): area-averaging when shrinking, bilinear interpolation when
// expanding, processed one source row at a time with O(width) state.
//
// Results are bit-exact with libwebp's C implementation
// (src/utils/rescaler_utils.c, src/dsp/rescaler.c).
//
// The horizontal geometry is the same for every row, so Init computes it
// once into a column table (see initCols) and the row kernels in
// kernels.go only walk that table. On arm64 the hot kernels are NEON
// assembly (rescaler_arm64.s), bit-exact with the Go ones.
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
	// cols is the column table, two cells per destination pixel x:
	//   shrinking: cols[2x] = number of source pixels whose sum makes up
	//   pixel x (at least 1), cols[2x+1] = libwebp's -accum after them, the
	//   weight of the last one that belongs to pixel x+1;
	//   expanding: cols[2x] = index of the left source pixel,
	//   cols[2x+1] = libwebp's accum, the weight of that pixel.
	cols []uint32
	arch archState // per-architecture kernel state, see initArch
}

// ErrInvalid is returned for invalid rescaler dimensions.
var ErrInvalid = errors.New("resample: invalid dimensions")

// WorkSize returns the number of uint32 row cells a rescaler with the given
// destination width and channel count needs, like libwebp's work buffer
// (2 rows of dstWidth*numChannels accumulators).
func WorkSize(dstWidth, numChannels int) int { return 2 * dstWidth * numChannels }

// BufferSize returns the number of uint32 cells Init uses from its work
// buffer: the WorkSize row cells plus the column tables. A work buffer of
// that capacity spares Init any allocation.
func BufferSize(dstWidth, numChannels int) int {
	return WorkSize(dstWidth, numChannels) + 2*dstWidth + archTableSize(dstWidth, numChannels)
}

// New returns a rescaler with a freshly allocated work buffer.
func New(srcWidth, srcHeight, dstWidth, dstHeight, numChannels int) (*Rescaler, error) {
	r := new(Rescaler)
	if err := r.Init(srcWidth, srcHeight, dstWidth, dstHeight, numChannels, nil); err != nil {
		return nil, err
	}
	return r, nil
}

// Init (re)initializes r, like WebPRescalerInit. work is reused when its
// capacity is at least BufferSize(dstWidth, numChannels); otherwise a new
// buffer is allocated.
func (r *Rescaler) Init(srcWidth, srcHeight, dstWidth, dstHeight, numChannels int, work []uint32) error {
	if srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0 ||
		numChannels <= 0 || numChannels > 4 ||
		srcWidth > math.MaxInt32/2 || srcHeight > math.MaxInt32/2 ||
		dstWidth > math.MaxInt32/2 || dstHeight > math.MaxInt32/2 {
		return ErrInvalid
	}
	n := WorkSize(dstWidth, numChannels)
	size := BufferSize(dstWidth, numChannels)
	if cap(work) < size {
		work = make([]uint32, size)
	} else {
		work = work[:size]
		clear(work[:n])
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
		frow:        work[n/2 : n : n],
		cols:        work[n : n+2*dstWidth : n+2*dstWidth],
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
	r.initCols()
	r.initArch(work[n+2*dstWidth:])
	return nil
}

// initCols fills the column table by running libwebp's horizontal loops
// (WebPRescalerImportRowExpand_C / _Shrink_C) once without the pixels.
func (r *Rescaler) initCols() {
	cols := r.cols
	if r.xExpand {
		// accum starts at x_add; whenever it drops below zero the left
		// pixel advances by one (x_sub < x_add, so by one at most).
		accum, left := r.xAdd, 0
		for x := 0; x < r.dstWidth; x++ {
			cols[2*x], cols[2*x+1] = uint32(left), uint32(accum)
			accum -= r.xSub
			if accum < 0 {
				left++
				accum += r.xAdd
			}
		}
		return
	}
	// accum gains x_add per output pixel and loses x_sub per source pixel
	// summed; it stays in (-x_sub, 0] between output pixels, and since
	// x_add >= x_sub at least one source pixel is summed per output pixel.
	accum := 0
	for x := 0; x < r.dstWidth; x++ {
		accum += r.xAdd
		k := (accum + r.xSub - 1) / r.xSub
		accum -= k * r.xSub
		cols[2*x], cols[2*x+1] = uint32(k), uint32(-accum)
	}
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
	src = src[:r.srcWidth*r.numChannels]
	// When shrinking vertically the new row is also added to the irow
	// accumulators (libwebp does that in a second pass, the sum is the
	// same); when expanding, irow and frow are the two latest rows.
	var acc []uint32
	if r.yExpand {
		r.irow, r.frow = r.frow, r.irow
	} else {
		acc = r.irow
	}
	if r.xExpand {
		r.expandRow(src, acc)
	} else {
		r.shrinkRow(src, acc)
	}
	r.srcY++
	r.yAccum -= r.ySub
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
		if r.yAccum == 0 {
			exportExpand(dst, frow, r.fyScale)
		} else {
			b := frac(uint64(-r.yAccum), uint64(r.ySub))
			// -yAccum >= 1 and ySub < 2^32, so b >= 1 and a fits 32 bits.
			a := uint32(one - uint64(b))
			exportExpandMix(dst, frow, irow, a, b, r.fyScale)
		}
	case r.fxyScale != 0:
		if yscale := r.fyScale * uint32(-r.yAccum); yscale != 0 {
			exportShrink(dst, irow, frow, yscale, r.fxyScale)
		} else {
			exportShrinkZero(dst, irow, r.fxyScale)
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
