// Package vp8 implements a pure Go VP8 (lossy WebP) key frame decoder that is
// bit-exact with libwebp 1.6 for RGBA output, including fancy upsampling,
// point sampling, filter bypass and chroma dithering.
//
// The decoder works one macroblock row at a time with a small rolling cache
// (16 luma rows plus the rows the loop filter still has to touch), so its
// memory use is O(width) and independent of the image height.
package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): dec/vp8_dec.c,
// dec/tree_dec.c, dec/frame_dec.c and dec/io_dec.c.

import (
	"image"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Options controls decoding. The zero value (or a nil *Options) selects
// libwebp's defaults.
type Options struct {
	// BypassFiltering skips the in-loop deblocking filter (dwebp -nofilter).
	BypassFiltering bool
	// NoFancyUpsampling uses point-sampled chroma instead of the bilinear
	// "fancy" upsampler (dwebp -nofancy).
	NoFancyUpsampling bool
	// DitheringStrength in [0, 100] adds libwebp's deterministic chroma
	// dithering on macroblocks quantized hard enough (dwebp -dither).
	DitheringStrength int
	// UseThreads parses the bitstream in the calling goroutine while a
	// helper goroutine reconstructs, filters and converts the previously
	// parsed batch of macroblock rows (libwebp's mt_method 2). The helper
	// comes from the process-wide budget of internal/workers; when none is
	// free, or the picture is too small to gain from it (less than 1024
	// macroblocks, about 512x512 pixels), the frame is decoded on the
	// calling goroutine. Output is identical either way.
	UseThreads bool
	// Alpha, if not nil and no alpha plane is passed to Decode or
	// DecodeRegion, is called before each batch of output rows is emitted
	// to obtain the alpha plane (same layout as the alpha argument, rows
	// counted from the top of the frame; a nil plane means opaque). rowEnd
	// is the number of frame rows whose alpha is needed: only the first
	// rowEnd*width bytes of the returned plane are read, and the plane may
	// be a different (longer) slice on each call. Like libwebp, which
	// decodes the ALPH chunk while emitting rows (VP8DecompressAlphaRows),
	// this lets the caller decode alpha only as far as the VP8 data goes.
	// An error it returns aborts decoding.
	Alpha func(rowEnd int) ([]byte, error)
}

// kFilterExtraRows is the number of rows the loop filter may still modify
// below a decoded macroblock row, per filter type (none, simple, complex).
var kFilterExtraRows = [3]int{0, 2, 8}

type quantMatrix struct {
	y1, y2, uv [2]int32
	uvQuant    int // for dithering strength evaluation
	dither     int // dithering amplitude (0 = off)
}

// fInfo holds the loop filter parameters of one macroblock.
type fInfo struct {
	limit, ilevel, inner, hevThresh uint8
}

// topSamples are the unfiltered bottom samples of the macroblock above.
type topSamples struct {
	y    [16]uint8
	u, v [8]uint8
}

// mbBlock is the parsed data of one macroblock.
type mbBlock struct {
	coeffs    [384]int16
	imodes    [16]uint8
	nonZeroY  uint32
	nonZeroUV uint32
	segment   uint8
	uvmode    uint8
	isI4x4    bool
	skip      bool
	hasCoeffs bool // at least one non-zero coefficient (inner edge filtering)
	dither    uint8
}

// decoder holds the whole decoding state. Instances are pooled; every field
// is (re)initialized per frame.
type decoder struct {
	br               bitReader
	parts            [maxNumPartitions]bitReader
	numPartsMinusOne int

	width, height, mbW, mbH int

	// segment header
	useSegment, updateMap, absoluteDelta bool
	quantizer, filterStrength            [numMBSegments]int
	segProba                             [3]uint8

	// filter header
	simple                 bool
	level, sharpness       int
	useLFDelta             bool
	refLFDelta             [numRefLFDeltas]int
	modeLFDelta            [numModeLFDeltas]int
	filterType             int
	fstrengths             [numMBSegments][2]fInfo
	proba                  [numTypes][16 + 1][numCtx][numProbas]uint8
	useSkipProba           bool
	skipP                  uint8
	dqm                    [numMBSegments]quantMatrix
	dither                 bool
	rng                    vp8Random
	ditherRow              []uint8 // per-macroblock dither amplitude of the row
	intraT                 []uint8 // 4 sub-block modes per macroblock column
	intraL                 [4]uint8
	nzT                    []uint8 // top non-zero context per column
	nzDCT                  []uint8
	nzL, nzDCL             uint8
	yuvT                   []topSamples
	blk                    mbBlock
	mtBlocks               []mbBlock // UseThreads ring of parsed row batches
	yuvB                   [yuvSize]byte
	cacheY, cacheU, cacheV []byte
	cacheYStride           int
	cacheUVStride          int
	cacheYOff, cacheUVOff  int
	extraRows              int

	// output
	sink             pipeline.Sink
	ysink            YUVSink // non-nil for DecodeYUV
	alpha            []byte
	alphaFn          func(rowEnd int) ([]byte, error)
	fancy            bool
	tmpY, tmpU, tmpV []byte
	uvT, uvB         []uint32 // upsampled chroma scratch rows
	rowsOut          int      // number of output rows committed

	// Output region (libwebp's crop_*), the macroblock area that must be
	// filtered/dithered for it (tl/br, as in VP8EnterCritical) and the
	// output width.
	cropLeft, cropTop, cropRight, cropBottom int
	tlMbX, tlMbY, brMbX, brMbY               int
	outW                                     int
}

var decoderPool = sync.Pool{New: func() any { return new(decoder) }}

// grow returns s resized to n bytes, reusing its storage when possible.
func grow(s []byte, n int) []byte {
	if cap(s) < n {
		return make([]byte, n)
	}
	return s[:n]
}

// Decode decodes the VP8 chunk payload and writes non-premultiplied RGBA rows
// into sink, top to bottom. alpha is nil (opaque, A=255) or a width*height
// plane (stride=width) whose values become the A channel.
func Decode(data []byte, alpha []byte, opts *Options, sink pipeline.Sink) error {
	return decode(data, alpha, opts, nil, sink)
}

// DecodeRegion is like Decode but only produces the crop rectangle, with
// libwebp's cropping semantics (WebPDecoderOptions.use_cropping):
// crop.Min is rounded down to even coordinates (keeping crop.Dx() and
// crop.Dy()), the cropped Y'CbCr area is converted as if it were the whole
// picture (fancy upsampling mirrors chroma at the crop borders), and sink
// receives crop.Dy() rows of crop.Dx() pixels numbered from 0. Decoding stops
// below the crop rectangle. alpha, if not nil, is still the full
// width*height plane. The rectangle must lie inside the frame.
func DecodeRegion(data []byte, alpha []byte, opts *Options, crop image.Rectangle, sink pipeline.Sink) error {
	return decode(data, alpha, opts, &crop, sink)
}

// YUVSink receives decoded Y'CbCr rows instead of RGBA (libwebp's VP8Io
// "put" callback), e.g. to implement libwebp-exact rescaling, which works on
// the Y, U and V planes. Rows arrive top to bottom in batches.
type YUVSink interface {
	// PutYUV receives luma rows [y, y+n) of the output region, n > 0. Luma
	// row j (0 <= j < n) is ys[j*yStride:][:width]. The batch carries
	// (n+1)/2 chroma rows of (width+1)/2 samples: row k is
	// us[k*uvStride:] and vs[k*uvStride:]; luma row j uses chroma row j/2
	// (y is always even). The slices are only valid during the call.
	PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error
	// NeedRows reports how many rows (from the top of the output region)
	// are still needed in total; decoding may stop once they are produced.
	NeedRows() int
}

// DecodeYUV decodes the VP8 chunk payload like DecodeRegion (crop may be nil
// for the whole frame) but delivers the filtered and dithered Y'CbCr rows to
// sink. NoFancyUpsampling has no effect.
func DecodeYUV(data []byte, opts *Options, crop *image.Rectangle, sink YUVSink) error {
	if sink == nil {
		return status.InvalidParam("nil sink")
	}
	return decode(data, nil, opts, crop, yuvAdapter{sink})
}

// yuvAdapter lets a YUVSink travel through the RGBA code path; finishRow
// recognizes it and hands it the Y'CbCr rows.
type yuvAdapter struct{ YUVSink }

func (yuvAdapter) Buffer(y, n int) []byte { return nil }
func (yuvAdapter) Stride() int            { return 0 }
func (yuvAdapter) Commit(y, n int) error  { return nil }

func decode(data, alpha []byte, opts *Options, crop *image.Rectangle, sink pipeline.Sink) error {
	if sink == nil {
		return status.InvalidParam("nil sink")
	}
	var o Options
	if opts != nil {
		o = *opts
	}
	d := decoderPool.Get().(*decoder)
	err := d.decode(data, alpha, &o, crop, sink)
	d.sink = nil
	d.ysink = nil
	d.alpha = nil
	d.alphaFn = nil
	d.br.buf = nil
	for i := range d.parts {
		d.parts[i].buf = nil
	}
	decoderPool.Put(d)
	return err
}

func (d *decoder) decode(data, alpha []byte, o *Options, crop *image.Rectangle, sink pipeline.Sink) error {
	if err := d.parseHeaders(data); err != nil {
		return err
	}
	if alpha != nil && len(alpha) < d.width*d.height {
		return status.InvalidParam("alpha plane too small")
	}
	d.alphaFn = nil
	if alpha == nil {
		d.alphaFn = o.Alpha
	}
	d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
	if crop != nil {
		// WebPIoInitFromOptions + WebPCheckCropDimensions.
		x, y := crop.Min.X&^1, crop.Min.Y&^1
		w, h := crop.Dx(), crop.Dy()
		if x < 0 || y < 0 || w <= 0 || h <= 0 || x >= d.width || w > d.width-x || y >= d.height || h > d.height-y {
			return status.InvalidParam("crop rectangle outside the frame")
		}
		d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = x, y, x+w, y+h
	}
	d.outW = d.cropRight - d.cropLeft
	if o.BypassFiltering {
		d.filterType = 0
	}
	d.enterCritical()
	d.precomputeFilterStrengths()
	d.initDithering(o.DitheringStrength)
	d.sink = sink
	d.ysink = nil
	if ya, ok := sink.(yuvAdapter); ok {
		d.ysink = ya.YUVSink
	}
	d.alpha = alpha
	d.fancy = !o.NoFancyUpsampling
	d.rowsOut = 0
	d.initFrame()
	if o.UseThreads && d.brMbY > 1 {
		return d.decodeFrameThreaded()
	}
	return d.decodeFrame()
}

// initFrame sizes and resets the per-frame state.
func (d *decoder) initFrame() {
	mbW := d.mbW
	d.intraT = grow(d.intraT, 4*mbW)
	clear(d.intraT) // B_DC_PRED
	d.nzT = grow(d.nzT, mbW)
	clear(d.nzT)
	d.nzDCT = grow(d.nzDCT, mbW)
	clear(d.nzDCT)
	if cap(d.yuvT) < mbW {
		d.yuvT = make([]topSamples, mbW)
	}
	d.yuvT = d.yuvT[:mbW]
	if d.dither {
		d.ditherRow = grow(d.ditherRow, mbW)
	}
	d.extraRows = kFilterExtraRows[d.filterType]
	d.cacheYStride = 16 * mbW
	d.cacheUVStride = 8 * mbW
	d.cacheYOff = d.extraRows * d.cacheYStride
	d.cacheUVOff = (d.extraRows / 2) * d.cacheUVStride
	d.cacheY = grow(d.cacheY, (d.extraRows+16)*d.cacheYStride)
	d.cacheU = grow(d.cacheU, (d.extraRows/2+8)*d.cacheUVStride)
	d.cacheV = grow(d.cacheV, (d.extraRows/2+8)*d.cacheUVStride)
	if d.fancy {
		w := d.outW
		d.tmpY = grow(d.tmpY, w)
		d.tmpU = grow(d.tmpU, (w+1)/2)
		d.tmpV = grow(d.tmpV, (w+1)/2)
		if cap(d.uvT) < w {
			d.uvT = make([]uint32, w)
			d.uvB = make([]uint32, w)
		}
		d.uvT, d.uvB = d.uvT[:w], d.uvB[:w]
	}
}

// enterCritical computes the macroblock area that needs filtering and the
// last macroblock row to decode for the output region (VP8EnterCritical).
func (d *decoder) enterCritical() {
	extra := kFilterExtraRows[d.filterType]
	if d.filterType == 2 {
		// The complex filter must preserve the dependency chain.
		d.tlMbX, d.tlMbY = 0, 0
	} else {
		d.tlMbX = max((d.cropLeft-extra)>>4, 0)
		d.tlMbY = max((d.cropTop-extra)>>4, 0)
	}
	d.brMbY = min((d.cropBottom+15+extra)>>4, d.mbH)
	d.brMbX = min((d.cropRight+15+extra)>>4, d.mbW)
}

// decodeFrame runs the single-goroutine row loop.
func (d *decoder) decodeFrame() error {
	for mbY := 0; mbY < d.brMbY; mbY++ {
		if d.rowsOut >= d.sink.NeedRows() {
			return nil
		}
		if err := d.decodeRow(mbY); err != nil {
			return err
		}
		if err := d.finishRow(mbY); err != nil {
			return err
		}
	}
	return nil
}

// decodeRow parses, reconstructs and filters one macroblock row into the
// cache.
func (d *decoder) decodeRow(mbY int) error {
	tokenBR := &d.parts[mbY&d.numPartsMinusOne]
	d.startParseRow()
	d.initRowEdges(mbY)
	blk := &d.blk
	for mbX := 0; mbX < d.mbW; mbX++ {
		if err := d.parseMB(mbX, blk, tokenBR); err != nil {
			return err
		}
		d.reconMB(mbX, mbY, blk)
	}
	return d.endRow()
}

// startParseRow resets the left contexts at the start of a macroblock row.
func (d *decoder) startParseRow() {
	d.nzL, d.nzDCL = 0, 0
	d.intraL = [4]uint8{}
}

// parseMB parses the modes and coefficients of the macroblock at mbX.
func (d *decoder) parseMB(mbX int, blk *mbBlock, tokenBR *bitReader) error {
	d.parseIntraMode(mbX, blk)
	skip := d.useSkipProba && blk.skip
	if !skip {
		skip = d.parseResiduals(mbX, blk, tokenBR)
	} else {
		d.nzL, d.nzT[mbX] = 0, 0
		if !blk.isI4x4 {
			d.nzDCL, d.nzDCT[mbX] = 0, 0
		}
		blk.nonZeroY = 0
		blk.nonZeroUV = 0
		blk.dither = 0
	}
	blk.hasCoeffs = !skip
	if tokenBR.eof {
		return status.NotEnoughData("premature end-of-file encountered")
	}
	return nil
}

// reconMB reconstructs and filters a parsed macroblock into the cache.
func (d *decoder) reconMB(mbX, mbY int, blk *mbBlock) {
	d.reconstructMB(mbX, mbY, blk)
	if d.filterType > 0 && mbX >= d.tlMbX && mbX < d.brMbX && mbY >= d.tlMbY {
		f := d.fstrengths[blk.segment][b2i(blk.isI4x4)]
		if blk.hasCoeffs {
			f.inner = 1
		}
		d.filterMB(mbX, mbY, f)
	}
	if d.dither {
		d.ditherRow[mbX] = blk.dither
	}
}

// endRow checks the first partition and dithers the finished row.
func (d *decoder) endRow() error {
	if d.br.eof {
		return status.NotEnoughData("premature end-of-partition0 encountered")
	}
	if d.dither {
		d.ditherCurrentRow()
	}
	return nil
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (d *decoder) ditherCurrentRow() {
	uvs := d.cacheUVStride
	for mbX := d.tlMbX; mbX < d.brMbX; mbX++ {
		if amp := d.ditherRow[mbX]; amp >= minDitherAmp {
			o := d.cacheUVOff + mbX*8
			d.rng.dither8x8(d.cacheU, o, uvs, int(amp))
			d.rng.dither8x8(d.cacheV, o, uvs, int(amp))
		}
	}
}

// finishRow emits the rows of the cache that are final after decoding
// macroblock row mbY, then rotates the cache (libwebp's FinishRow).
func (d *decoder) finishRow(mbY int) error {
	extra := d.extraRows
	isFirst := mbY == 0
	isLast := mbY >= d.brMbY-1
	yStart := 16 * mbY
	yEnd := 16 * (mbY + 1)
	var ys, us, vs []byte
	if !isFirst {
		yStart -= extra
		ys, us, vs = d.cacheY, d.cacheU, d.cacheV
	} else {
		ys = d.cacheY[d.cacheYOff:]
		us = d.cacheU[d.cacheUVOff:]
		vs = d.cacheV[d.cacheUVOff:]
	}
	if !isLast {
		yEnd -= extra
	}
	if yEnd > d.cropBottom {
		yEnd = d.cropBottom
	}
	if yStart < d.cropTop {
		delta := d.cropTop - yStart
		yStart = d.cropTop
		if yStart < yEnd {
			ys = ys[delta*d.cacheYStride:]
			us = us[(delta>>1)*d.cacheUVStride:]
			vs = vs[(delta>>1)*d.cacheUVStride:]
		}
	}
	if yStart < yEnd {
		ys, us, vs = ys[d.cropLeft:], us[d.cropLeft>>1:], vs[d.cropLeft>>1:]
		y0, y1 := yStart-d.cropTop, yEnd-d.cropTop
		var err error
		if d.ysink != nil {
			err = d.ysink.PutYUV(y0, y1-y0, ys, d.cacheYStride, us, vs, d.cacheUVStride)
			d.rowsOut = y1
		} else if d.fancy {
			err = d.emitFancy(ys, us, vs, y0, y1, yEnd >= d.cropBottom)
		} else {
			err = d.emitSampled(ys, us, vs, y0, y1)
		}
		if err != nil {
			return err
		}
	}
	if !isLast && extra > 0 {
		ysz := extra * d.cacheYStride
		uvsz := (extra / 2) * d.cacheUVStride
		copy(d.cacheY[:ysz], d.cacheY[16*d.cacheYStride:])
		copy(d.cacheU[:uvsz], d.cacheU[8*d.cacheUVStride:])
		copy(d.cacheV[:uvsz], d.cacheV[8*d.cacheUVStride:])
	}
	return nil
}

// loadAlpha obtains the deferred alpha plane (Options.Alpha) for output
// rows [0, yEnd).
func (d *decoder) loadAlpha(yEnd int) error {
	if d.alphaFn == nil {
		return nil
	}
	rowEnd := min(d.cropTop+yEnd, d.height)
	a, err := d.alphaFn(rowEnd)
	if err != nil {
		return err
	}
	if a == nil {
		d.alphaFn = nil // opaque
	} else if len(a) < rowEnd*d.width {
		return status.InvalidParam("alpha plane too small")
	}
	d.alpha = a
	return nil
}

// alphaRow returns the alpha samples of output row y, or nil when the
// output is opaque.
func (d *decoder) alphaRow(y int) []byte {
	if d.alpha == nil {
		return nil
	}
	o := (d.cropTop+y)*d.width + d.cropLeft
	return d.alpha[o : o+d.outW]
}

// emitSampled converts rows [yStart, yEnd) with point-sampled chroma.
func (d *decoder) emitSampled(ys, us, vs []byte, yStart, yEnd int) error {
	n := yEnd - yStart
	if err := d.loadAlpha(yEnd); err != nil {
		return err
	}
	dst := d.sink.Buffer(yStart, n)
	ds := d.sink.Stride()
	w := d.outW
	for j := 0; j < n; j++ {
		uo := (j >> 1) * d.cacheUVStride
		sampleRow(ys[j*d.cacheYStride:], us[uo:], vs[uo:], d.alphaRow(yStart+j), dst[j*ds:], w)
	}
	d.rowsOut = yEnd
	return d.sink.Commit(yStart, n)
}

// emitFancy converts output rows [yStart, yEnd) with libwebp's fancy
// upsampler (EmitFancyRGB). The last luma row of a non-final batch is kept
// back until the next chroma row is available.
func (d *decoder) emitFancy(ys, us, vs []byte, yStart, yEnd int, isLast bool) error {
	w := d.outW
	uvW := (w + 1) >> 1
	yS, uvS := d.cacheYStride, d.cacheUVStride
	outStart := yStart
	if yStart > 0 {
		outStart--
	}
	outEnd := yEnd
	if !isLast {
		outEnd--
	}
	n := outEnd - outStart
	if n <= 0 {
		return nil
	}
	if err := d.loadAlpha(yEnd); err != nil {
		return err
	}
	dst := d.sink.Buffer(outStart, n)
	ds := d.sink.Stride()
	row := func(y int) []byte { return dst[(y-outStart)*ds:] }

	y := yStart
	cy, cu := 0, 0
	if y == 0 {
		// First line is special cased: mirror the chroma at the boundary.
		upsamplePair(ys, nil, us, vs, us, vs, row(0), nil, d.alphaRow(0), nil, d.uvT, d.uvB, w)
	} else {
		// Finish the left-over line from the previous call.
		upsamplePair(d.tmpY, ys, d.tmpU, d.tmpV, us, vs, row(y-1), row(y), d.alphaRow(y-1), d.alphaRow(y), d.uvT, d.uvB, w)
	}
	for ; y+2 < yEnd; y += 2 {
		tu, tv := us[cu:], vs[cu:]
		cu += uvS
		cy += 2 * yS
		upsamplePair(ys[cy-yS:], ys[cy:], tu, tv, us[cu:], vs[cu:], row(y+1), row(y+2), d.alphaRow(y+1), d.alphaRow(y+2), d.uvT, d.uvB, w)
	}
	cy += yS
	if !isLast {
		copy(d.tmpY, ys[cy:cy+w])
		copy(d.tmpU, us[cu:cu+uvW])
		copy(d.tmpV, vs[cu:cu+uvW])
	} else if yEnd&1 == 0 {
		// The very last row of an even-height picture.
		upsamplePair(ys[cy:], nil, us[cu:], vs[cu:], us[cu:], vs[cu:], row(yEnd-1), nil, d.alphaRow(yEnd-1), nil, d.uvT, d.uvB, w)
	}
	d.rowsOut = outEnd
	return d.sink.Commit(outStart, n)
}
