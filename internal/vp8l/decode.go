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

// Package vp8l implements a pure Go decoder for the WebP lossless (VP8L)
// bitstream, including the ALPH chunk payload variant used for alpha planes.
//
// Algorithm adapted from libwebp (BSD-3-Clause): src/dec/vp8l_dec.c.
package vp8l

import (
	"encoding/binary"
	"errors"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

const (
	headerSize    = 5
	signature     = 0x2f
	maxDimension  = 1 << 14 // 14-bit sizes: 1..16384
	numCacheRows  = 16      // rows per inverse-transform batch
	maxCopyLength = 4096    // longest LZ77 copy
	maxPlaneDist  = 1048576 - 120

	colorCacheMult = 0x1e35a7bd
)

// metadata is the entropy coding state of one (sub-)image.
type metadata struct {
	cacheBits int
	huffBits  int
	huffMask  int // (1<<huffBits)-1, or -1 without meta Huffman codes
	huffXSize int
	huffImage []uint32
	groups    []htreeGroup
}

func (md *metadata) group(groups []htreeGroup, col, row int) *htreeGroup {
	if md.huffImage == nil {
		return &groups[0]
	}
	return &groups[md.huffImage[(row>>md.huffBits)*md.huffXSize+col>>md.huffBits]]
}

// output selects what processRows emits.
const (
	outRGBA  = iota // RGBA rows into a pipeline.Sink
	outGreen        // green channel into an alpha plane
)

// decoder holds all state of one decode. Instances are pooled.
type decoder struct {
	br bitReader

	transforms  [4]transform
	nTransforms int
	seen        uint8

	// Huffman scratch and storage, reused for each (sub-)image.
	tables      []huffCode
	packed      []packedCode
	groups      []htreeGroup
	treeOffs    [][5]int
	codeLengths [maxAlphabetSize]uint8
	sorted      [maxAlphabetSize]uint16
	lenTable    [1 << lengthsBits]huffCode
	single      [maxAlphabetSize]int32 // offset+1 of the shared table per single symbol
	singleSet   []uint16               // symbols with a non-zero single entry

	arena       []uint32 // sub-images (transform data, entropy image)
	argb        []uint32 // level-0 decoded pixels (whole image or sliding window)
	rows        []uint32 // inverse transform output: 1 top row + numCacheRows rows
	colorCache  *[1 << maxCacheBits]uint32
	pal         [256]uint32 // RGBA palette for fused color-index output
	palReady    bool
	premulPal   bool     // pal is alpha-premultiplied (pipeline.PremultipliedSink)
	bundle      []uint64 // RGBA pixels of each bundled index byte (bundleTable)
	bundleReady bool
	alphaTab    *alphaTables // 8-bit alpha path tables (built once per decode)

	// Level-0 output state.
	out      int
	sink     pipeline.Sink
	alpha    []byte
	alphaRow int // the frame row alpha[0] holds (AlphaDecoder.DecodeRowsAt)
	width    int // final image width
	height   int
	xsize    int // level-0 coded width (after color-index bundling)
	lastRow  int // rows [0, lastRow) were emitted
	needRows int
	// inPixels is set once the level-0 header (transforms, color cache,
	// Huffman codes) has been read and pixel decoding starts.
	inPixels bool

	// Resumable level-0 pixel decoding state (incremental alpha decoding).
	md0     metadata
	px      pixState
	buf     []uint32 // level-0 pixel buffer (d.argb[:bufLen])
	sliding bool     // buf (or ib) is a sliding window, not the whole image
	window  int      // LZ77 window kept when sliding
	use8b   bool     // 8-bit alpha path (decodeAlpha8Rows)
	ib      []byte   // 8-bit index buffer (whole image or sliding window)
	inTail  bool     // the 8-bit path switched to the tail bit reader
	tr      tailReader

	// Palette images decoded as bytes (idx8.go).
	idx8   bool   // the level-0 pixels are the index bytes in d.ib
	idxARB uint32 // alpha, red and blue of every level-0 pixel
	cache8 *[1 << maxCacheBits]uint16
	hash8  [256]uint16 // color cache slot of each index

	useThreads bool       // Options.UseThreads
	async      *asyncEmit // two-stage decode in progress (async.go)
	probe      asyncProbe // timing of the first rows before a two-stage decode
}

// pixState is the position of a (resumable) pixel decoding loop. base is
// the absolute pixel index of the start of the window buffer and p the
// position in it.
type pixState struct {
	base, p, col, row, lastCached int
	started                       bool
}

var decoderPool = sync.Pool{New: func() any { return new(decoder) }}

func getDecoder() *decoder {
	d := decoderPool.Get().(*decoder)
	d.nTransforms = 0
	d.seen = 0
	d.tables = d.tables[:0]
	d.packed = d.packed[:0]
	d.arena = d.arena[:0]
	d.lastRow = 0
	d.inPixels = false
	d.px = pixState{}
	d.md0 = metadata{}
	d.buf = nil
	d.use8b = false
	d.idx8 = false
	d.inTail = false
	d.palReady = false
	d.premulPal = false
	d.bundleReady = false
	d.useThreads = false
	d.async = nil
	d.probe = asyncProbe{}
	return d
}

const maxPooledPixels = 4 << 20

func putDecoder(d *decoder) {
	d.sink = nil
	d.alpha = nil
	d.br.data = nil
	for i := range d.transforms {
		d.transforms[i].data = nil
	}
	if cap(d.argb) > maxPooledPixels {
		d.argb = nil
	}
	if cap(d.arena) > maxPooledPixels {
		d.arena = nil
	}
	if cap(d.rows) > maxPooledPixels {
		d.rows = nil
	}
	if cap(d.ib) > 4*maxPooledPixels {
		d.ib = nil
	}
	d.buf = nil
	d.md0 = metadata{}
	for i := range d.groups {
		d.groups[i] = htreeGroup{}
	}
	if cap(d.groups) > 1<<12 {
		d.groups = nil
		d.treeOffs = nil
	}
	if cap(d.tables) > 1<<20 {
		d.tables = nil
		d.packed = nil
	}
	decoderPool.Put(d)
}

// DecodeHeader parses the VP8L chunk payload header (signature byte 0x2f,
// 14-bit w-1, 14-bit h-1, alpha_is_used bit, 3-bit version which must be 0).
func DecodeHeader(data []byte) (width, height int, hasAlpha bool, err error) {
	if len(data) < headerSize {
		return 0, 0, false, status.NotEnoughData("vp8l: header too short")
	}
	if data[0] != signature {
		return 0, 0, false, status.Bitstream("vp8l: invalid signature")
	}
	v := binary.LittleEndian.Uint32(data[1:])
	if v>>29 != 0 {
		return 0, 0, false, status.Bitstream("vp8l: invalid version")
	}
	width = int(v&0x3fff) + 1
	height = int(v>>14&0x3fff) + 1
	hasAlpha = v>>28&1 == 1
	return width, height, hasAlpha, nil
}

// Options configures DecodeWithOptions.
type Options struct {
	// UseThreads decodes big images (at least 256K needed pixels) in two
	// stages when the process-wide helper budget allows (see package
	// internal/workers): the calling goroutine decodes the entropy-coded
	// pixels while a helper goroutine applies the inverse transforms and
	// writes the rows into the sink. The output is the same either way.
	// libwebp ignores use_threads for lossless images.
	UseThreads bool
}

// Decode decodes a VP8L chunk payload and writes non-premultiplied RGBA rows
// into sink, top to bottom (respecting sink.NeedRows() to stop early).
//
// data may extend past the end of the chunk. libwebp hands the lossless
// decoder everything from the chunk start to the end of the input (padding
// byte and trailing chunks included), so a malformed stream that overruns
// its chunk by a few bits still decodes there; passing the same extended
// slice reproduces that behavior exactly. Bits beyond len(data) read as
// zero.
func Decode(data []byte, sink pipeline.Sink) error {
	return DecodeWithOptions(data, sink, nil)
}

// DecodeWithOptions is Decode with options (nil for the defaults). With
// UseThreads the sink may be called from a helper goroutine, never
// concurrently and always before DecodeWithOptions returns.
func DecodeWithOptions(data []byte, sink pipeline.Sink, opts *Options) error {
	w, h, _, err := DecodeHeader(data)
	if err != nil {
		return err
	}
	if sink == nil {
		return status.InvalidParam("vp8l: nil sink")
	}
	need := min(sink.NeedRows(), h)
	d := getDecoder()
	defer putDecoder(d)
	// The bit reader spans the header too: libwebp's end-of-stream limit
	// depends on the total payload size.
	d.br.init(data)
	d.br.read(8)
	d.br.read(32)
	d.out = outRGBA
	d.sink = sink
	d.needRows = max(need, 0)
	d.useThreads = opts != nil && opts.UseThreads
	err = d.decodeLevel0(w, h)
	if status.CodeOf(err) == utils.Vp8StatusNotEnoughData {
		// The whole file is available (no incremental decoding): like
		// libwebp's DecodeImageData ("if not incremental, and we are past
		// the end of buffer (eos_=1), then this is a real bitstream
		// error"), running out of data is a bitstream error.
		var se *status.Error
		msg := err.Error()
		if errors.As(err, &se) {
			msg = se.Msg
		}
		return status.Bitstream(msg)
	}
	return err
}

// DecodeAlpha decodes an ALPH chunk compressed with method 1: data is a VP8L
// image stream WITHOUT the 5-byte header (no signature/size), of the given
// width/height; the green channel of each pixel is written into dst
// (len >= width*height, stride=width).
//
// Errors in the stream header (transforms, color cache, Huffman codes) have
// the code utils.Vp8StatusOutOfMemory: libwebp's alpha decoder
// (VP8DecompressAlphaRows) reports that code when ALPHInit fails, because
// the failed lossless decoder, and its status, are already gone. Errors in
// the pixel data are bitstream errors.
func DecodeAlpha(data []byte, width, height int, dst []byte) error {
	if len(dst) < width*height {
		return status.InvalidParam("vp8l: alpha buffer too small")
	}
	var a AlphaDecoder
	if err := a.init(data, width, height); err != nil {
		return err
	}
	defer a.Close()
	return a.DecodeRows(dst, height)
}

// AlphaDecoder decodes an ALPH chunk payload (method 1) incrementally, a
// batch of rows at a time, like libwebp's VP8LDecodeAlphaImageStream: a
// caller that only needs the first rows (because the VP8 frame turns out
// to be truncated, for example) never pays for decoding the whole plane.
type AlphaDecoder struct {
	d    *decoder
	rows int
	err  error
}

// NewAlphaDecoder reads the header of the alpha stream data (transforms,
// color cache, Huffman codes; see DecodeAlpha for the format and the error
// codes) of a width x height plane. The decoder must be released with
// Close.
func NewAlphaDecoder(data []byte, width, height int) (*AlphaDecoder, error) {
	a := new(AlphaDecoder)
	if err := a.init(data, width, height); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *AlphaDecoder) init(data []byte, width, height int) error {
	if width <= 0 || height <= 0 || width > maxDimension || height > maxDimension {
		return status.InvalidParam("vp8l: invalid alpha dimensions")
	}
	d := getDecoder()
	d.br.init(data)
	d.out = outGreen
	if err := d.readLevel0Header(width, height); err != nil {
		putDecoder(d)
		var se *status.Error
		msg := err.Error()
		if errors.As(err, &se) {
			msg = se.Msg
		}
		return status.New(utils.Vp8StatusOutOfMemory, "alpha decoder initialization failed: "+msg)
	}
	d.inPixels = true
	d.use8b = d.nTransforms == 1 && d.transforms[0].kind == colorIndexingTransform && is8bOptimizable(&d.md0)
	if d.use8b {
		d.initBuffer8()
		if d.alphaTab == nil {
			d.alphaTab = new(alphaTables)
		}
		d.alphaTab.init(&d.transforms[0])
	} else {
		d.initBuffer32()
	}
	*a = AlphaDecoder{d: d}
	return nil
}

// Rows returns the number of rows decoded so far.
func (a *AlphaDecoder) Rows() int { return a.rows }

// DecodeRows decodes the plane up to row upTo (exclusive, clamped to the
// height) into dst, the width x upTo top part of the plane (stride width;
// len(dst) >= width*upTo). Rows written by earlier calls are not written
// again, so dst must hold the same top rows on every call (it may be a
// longer copy of the previous buffer). After an error every call returns
// that error.
func (a *AlphaDecoder) DecodeRows(dst []byte, upTo int) error {
	return a.DecodeRowsAt(dst, 0, upTo)
}

// DecodeRowsAt is DecodeRows into a window of the plane: dst holds the rows
// [first, upTo) (stride width), first <= Rows(), and only the rows
// [Rows(), upTo) are written. The rows before first are not needed, so a
// caller may keep a few rows of the plane at a time.
func (a *AlphaDecoder) DecodeRowsAt(dst []byte, first, upTo int) error {
	if a.err != nil {
		return a.err
	}
	d := a.d
	if d == nil {
		return status.InvalidParam("vp8l: alpha decoder closed")
	}
	upTo = min(upTo, d.height)
	if upTo <= a.rows {
		return nil
	}
	if first < 0 || first > a.rows || len(dst) < (upTo-first)*d.width {
		return status.InvalidParam("vp8l: alpha buffer too small")
	}
	d.alpha = dst
	d.alphaRow = first
	d.needRows = upTo
	var err error
	if d.use8b {
		err = d.decodeAlpha8Rows(upTo)
	} else {
		err = d.decodePixels(&d.md0, d.buf, d.xsize, d.height, true, d.sliding, d.window, &d.px)
	}
	d.alpha = nil
	if err != nil {
		a.err = err
		return err
	}
	a.rows = upTo
	return nil
}

// Close releases the decoder's buffers.
func (a *AlphaDecoder) Close() {
	if a.d != nil {
		putDecoder(a.d)
		a.d = nil
	}
}

// readLevel0Header reads the main image stream header: transforms, color
// cache and Huffman codes (into d.md0).
func (d *decoder) readLevel0Header(w, h int) error {
	br := &d.br
	d.width, d.height = w, h
	xsize := w
	for br.read(1) == 1 {
		if err := d.readTransform(&xsize, h); err != nil {
			return err
		}
	}
	if err := d.readImageCodes(&d.md0, xsize, h, true); err != nil {
		return err
	}
	d.xsize = xsize
	if br.eos() {
		return status.NotEnoughData("vp8l: truncated header")
	}
	return nil
}

// initBuffer32 allocates the inverse transform rows and the level-0 pixel
// buffer: the whole image, or a sliding window big enough for any LZ77
// distance plus the rows not yet emitted.
func (d *decoder) initBuffer32() {
	w, h, xsize := d.width, d.height, d.xsize
	if d.nTransforms > 0 {
		n := w * (numCacheRows + 1)
		if cap(d.rows) < n {
			d.rows = make([]uint32, n)
		}
		d.rows = d.rows[:n]
	}
	total := xsize * h
	d.window = max(maxPlaneDist, 7*xsize+8)
	bufLen := d.window + d.window/2 + (numCacheRows+2)*xsize + 2*maxCopyLength
	d.sliding = bufLen < total
	if !d.sliding {
		bufLen = total
	}
	if cap(d.argb) < bufLen {
		d.argb = make([]uint32, bufLen)
	}
	d.buf = d.argb[:bufLen]
}

// initBuffer8 allocates the index buffer of the 8-bit alpha path, with the
// same window logic as initBuffer32.
func (d *decoder) initBuffer8() {
	total := d.xsize * d.height
	d.window = max(maxPlaneDist, 7*d.xsize+8)
	bufLen := d.window + d.window/2 + (numCacheRows+2)*d.xsize + 2*maxCopyLength
	d.sliding = bufLen < total
	if !d.sliding {
		bufLen = total
	}
	if cap(d.ib) < bufLen {
		d.ib = make([]byte, bufLen)
	}
	d.ib = d.ib[:bufLen]
}

// decodeLevel0 decodes the main image stream: transforms, codes, pixels.
func (d *decoder) decodeLevel0(w, h int) error {
	if err := d.readLevel0Header(w, h); err != nil {
		return err
	}
	if d.needRows == 0 {
		return nil
	}
	d.inPixels = true
	// When the palette lookup is the last step, a sink that premultiplies
	// its rows (scaling) can be fed premultiplied palette colors instead:
	// premultiplying 256 palette entries once gives the same pixels as
	// premultiplying every pixel.
	if d.nTransforms > 0 && d.transforms[0].kind == colorIndexingTransform {
		if ps, ok := d.sink.(pipeline.PremultipliedSink); ok && ps.AcceptPremultiplied() {
			d.premulPal = true
		}
	}
	if arb, ok := d.idx8ARB(); ok {
		d.initIdx8(arb)
	} else {
		d.initBuffer32()
	}
	if d.asyncWanted() {
		// Stop the helper on every path; after a normal return finishAsync
		// already did, and this is a no-op.
		defer d.abortAsync()
		if asyncProbePixels > 0 {
			// The helper starts, or not, after the first rows (probeDone).
			d.probe.active = true
		} else {
			d.startAsync()
		}
		err := d.decodeLevel0Pixels(h)
		if d.async != nil {
			return d.finishAsync(err)
		}
		return err
	}
	return d.decodeLevel0Pixels(h)
}

// decodeLevel0Pixels decodes the level-0 pixels of an image of height h,
// as index bytes when d.idx8 is set (switching to 32 bits if the stream
// requires it, see idx8.go).
func (d *decoder) decodeLevel0Pixels(h int) error {
	if d.idx8 {
		err := d.decodeIdx8(&d.md0, d.ib, d.xsize, h, d.sliding, d.window, &d.px)
		if err != errWiden {
			return err
		}
		if err := d.widenIdx8(); err != nil {
			return err
		}
	}
	return d.decodePixels(&d.md0, d.buf, d.xsize, h, true, d.sliding, d.window, &d.px)
}

func is8bOptimizable(md *metadata) bool {
	if md.cacheBits > 0 {
		return false
	}
	for i := range md.groups {
		t := &md.groups[i].trees
		if t[treeRed][0].bits > 0 || t[treeBlue][0].bits > 0 || t[treeAlpha][0].bits > 0 {
			return false
		}
	}
	return true
}

// readImageCodes reads the color cache info and Huffman codes of an image.
func (d *decoder) readImageCodes(md *metadata, xsize, ysize int, level0 bool) error {
	br := &d.br
	md.cacheBits = 0
	if br.read(1) == 1 {
		bits := int(br.read(4))
		if bits < 1 || bits > maxCacheBits {
			return status.Bitstream("vp8l: invalid color cache bits")
		}
		md.cacheBits = bits
	}
	return d.readHuffmanCodes(md, xsize, ysize, md.cacheBits, level0)
}

// readTransform reads one transform; xsize is updated for color indexing.
func (d *decoder) readTransform(xsize *int, ysize int) error {
	br := &d.br
	kind := int(br.read(2))
	if d.seen&(1<<kind) != 0 {
		return status.Bitstream("vp8l: repeated transform")
	}
	d.seen |= 1 << kind
	t := &d.transforms[d.nTransforms]
	d.nTransforms++
	*t = transform{kind: kind, xsize: *xsize, ysize: ysize}
	switch kind {
	case predictorTransform, crossColorTransform:
		t.bits = int(br.read(3)) + 2
		data, err := d.decodeSubImage(subSampleSize(*xsize, t.bits), subSampleSize(ysize, t.bits))
		if err != nil {
			return err
		}
		t.data = data
	case colorIndexingTransform:
		numColors := int(br.read(8)) + 1
		bits := 0
		switch {
		case numColors > 16:
		case numColors > 4:
			bits = 1
		case numColors > 2:
			bits = 2
		default:
			bits = 3
		}
		t.bits = bits
		*xsize = subSampleSize(*xsize, bits)
		data, err := d.decodeSubImage(numColors, 1)
		if err != nil {
			return err
		}
		// Expand the delta-coded palette to its full size; unused
		// entries are transparent black.
		pal := d.allocArena(1 << (8 >> bits))
		clear(pal)
		pal[0] = data[0]
		for i := 1; i < numColors && i < len(pal); i++ {
			pal[i] = addPixels(data[i], pal[i-1])
		}
		t.data = pal
	}
	return nil
}

// allocArena returns n (not zeroed) pixels for a sub-image. Earlier
// sub-images stay valid if the arena grows.
func (d *decoder) allocArena(n int) []uint32 {
	off := len(d.arena)
	if cap(d.arena)-off < n {
		na := make([]uint32, 0, max(2*cap(d.arena), off+n, 1024))
		d.arena = na
		off = 0
	}
	d.arena = d.arena[:off+n]
	return d.arena[off : off+n : off+n]
}

// decodeSubImage decodes an entropy-coded sub-image (no transforms, no meta
// Huffman codes).
func (d *decoder) decodeSubImage(xsize, ysize int) ([]uint32, error) {
	var md metadata
	if err := d.readImageCodes(&md, xsize, ysize, false); err != nil {
		return nil, err
	}
	buf := d.allocArena(xsize * ysize)
	var st pixState
	if err := d.decodePixels(&md, buf, xsize, ysize, false, false, 0, &st); err != nil {
		return nil, err
	}
	return buf, nil
}

// planeCodeToDistance maps a distance code to a linear pixel distance.
func planeCodeToDistance(xsize, code int) int {
	if code > len(distanceMap) {
		return code - len(distanceMap)
	}
	dc := int(distanceMap[(code-1)&127])
	d := (dc>>4)*xsize + 8 - dc&0xf
	if d < 1 {
		return 1
	}
	return d
}

// distanceMap maps the 120 short distance codes to (dy<<4 | (8-dx)).
var distanceMap = [120]uint8{
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

// copyBlock32 copies length pixels from p-dist to p, with LZ77 semantics
// for overlapping ranges (the pattern of period dist repeats).
func copyBlock32(buf []uint32, p, dist, length int) {
	src := p - dist
	if dist >= length {
		copy(buf[p:p+length], buf[src:src+length])
		return
	}
	if dist == 1 {
		v := buf[src]
		s := buf[p : p+length]
		for i := range s {
			s[i] = v
		}
		return
	}
	n := copy(buf[p:p+dist], buf[src:p])
	for n < length {
		n += copy(buf[p+n:p+length], buf[p:p+n])
	}
}

func copyBlock8(buf []byte, p, dist, length int) {
	src := p - dist
	if dist >= length {
		copy(buf[p:p+length], buf[src:src+length])
		return
	}
	if dist == 1 {
		v := buf[src]
		s := buf[p : p+length]
		for i := range s {
			s[i] = v
		}
		return
	}
	n := copy(buf[p:p+dist], buf[src:p])
	for n < length {
		n += copy(buf[p+n:p+length], buf[p:p+n])
	}
}
