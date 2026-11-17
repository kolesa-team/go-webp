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

// Package decoder decodes WebP images (lossy, lossless, with alpha) with a
// pure Go implementation that produces the same pixels as libwebp.
package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"image"
	"io"
	"os"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/vp8"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// Decoder stores information to decode picture
type Decoder struct {
	data     []byte
	options  *Options
	config   *Config
	features container.Features
}

// statusError formats like v1 ("<prefix>: VP8_STATUS_...") and unwraps to
// the underlying *status.Error, which carries the libwebp status code and a
// more detailed message.
type statusError struct {
	prefix string
	err    error
}

func (e *statusError) Error() string {
	code := status.CodeOf(e.err)
	if code == utils.Vp8StatusUserAbort {
		// Cancelled through a context: say why.
		return e.prefix + ": " + code.String() + ": " + e.err.Error()
	}
	return e.prefix + ": " + code.String()
}

func (e *statusError) Unwrap() error { return e.err }

// NewDecoder return new decoder instance
func NewDecoder(r io.Reader, options *Options) (d *Decoder, err error) {
	var data []byte

	if data, err = readAll(r); err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	if options == nil {
		options = &Options{}
	}
	d = &Decoder{data: data, options: options}

	if d.config, err = d.options.GetConfig(); err != nil {
		return nil, err
	}

	// получаем WebPBitstreamFeatures
	if d.features, err = container.GetFeatures(d.data); err != nil {
		return nil, &statusError{prefix: "cannot fetch features", err: err}
	}

	return d, nil
}

// readAll reads r to EOF. When the size is known in advance (bytes.Reader,
// bytes.Buffer, strings.Reader, regular files) the buffer is allocated once.
// Otherwise (bufio.Reader as used by image.Decode, network bodies) the size
// declared in the RIFF header is used only as a growth target: the first
// allocation is at most initialReadSize and the buffer then grows by at most
// readGrowth times the bytes actually received, so a short input with a
// lying header costs at most a few times its own size.
func readAll(r io.Reader) ([]byte, error) {
	const maxHint = 1 << 30
	n := 0
	switch v := r.(type) {
	case interface{ Len() int }:
		n = v.Len()
	case *os.File:
		if fi, err := v.Stat(); err == nil && fi.Mode().IsRegular() && fi.Size() < maxHint {
			n = int(fi.Size())
		}
	}
	if n > 0 && n < maxHint {
		return readFull(r, make([]byte, 0, n+1), n)
	}
	// Unknown size: read the RIFF header first.
	var hdr [12]byte
	m, err := io.ReadFull(r, hdr[:])
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return hdr[:m:m], nil
	}
	if err != nil {
		return nil, err
	}
	hint := 0
	if string(hdr[0:4]) == "RIFF" && string(hdr[8:12]) == "WEBP" {
		hint = int(min(int64(binary.LittleEndian.Uint32(hdr[4:]))+8, maxHint))
	}
	b := make([]byte, 0, min(max(hint, 12), initialReadSize)+1)
	b = append(b, hdr[:]...)
	return readFull(r, b, hint)
}

const (
	// initialReadSize bounds the first allocation for readers of unknown size.
	initialReadSize = 64 << 10
	// readGrowth bounds each buffer growth relative to the bytes read so far.
	readGrowth = 4
)

// readFull appends the rest of r to b. hint is the expected total length
// (0 if unknown): growth aims at hint+1 so that a truthful header needs only
// a few allocations, but never exceeds readGrowth times the data received.
// One spare byte of capacity lets the final read observe io.EOF without
// growing the buffer.
func readFull(r io.Reader, b []byte, hint int) ([]byte, error) {
	for {
		if len(b) == cap(b) {
			want := 2 * cap(b)
			if hint >= len(b) {
				want = min(hint+1, readGrowth*len(b))
			}
			nb := make([]byte, len(b), max(want, len(b)+512))
			copy(nb, b)
			b = nb
		}
		m, err := r.Read(b[len(b):cap(b)])
		b = b[:len(b)+m]
		if err == io.EOF {
			return b, nil
		}
		if err != nil {
			return b, err
		}
	}
}

// Decode picture from reader
func (d *Decoder) Decode() (image.Image, error) {
	img, err := d.decode(call{})
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	return img, nil
}

// call holds the per-call arguments of the Decode variants.
type call struct {
	// dst lends its pixel memory to the output image (DecodeInto); nil
	// allocates a new image.
	dst *image.NRGBA
	// ctx cancels the decode (DecodeContext); nil if it cannot be
	// cancelled.
	ctx context.Context
}

func (d *Decoder) decode(c call) (*image.NRGBA, error) {
	if err := canceled(c.ctx); err != nil {
		return nil, err
	}
	parsed, err := container.Parse(d.data)
	if err != nil {
		return nil, err
	}
	if parsed.HasAnimation {
		return nil, status.Unsupported("animated WebP is not supported")
	}
	if parsed.VP8L != nil {
		return d.decodeLossless(parsed, c)
	}
	return d.decodeLossy(parsed, c)
}

// GetFeatures return information about picture: width, height ...
func (d *Decoder) GetFeatures() utils.BitstreamFeatures {
	return utils.BitstreamFeatures{
		Width:        d.features.Width,
		Height:       d.features.Height,
		HasAlpha:     d.features.HasAlpha,
		HasAnimation: d.features.HasAnimation,
		Format:       d.features.Format,
	}
}

func (d *Decoder) decodeLossless(parsed *container.Image, cl call) (*image.NRGBA, error) {
	c := d.config
	p := pipeline.Params{
		SrcW:        parsed.Width,
		SrcH:        parsed.Height,
		Flip:        c.Flip,
		Premultiply: true,
		Dst:         cl.dst,
	}
	if c.UseCropping {
		p.Crop = image.Rect(c.CropLeft, c.CropTop, c.CropLeft+c.CropWidth, c.CropTop+c.CropHeight)
	}
	if c.UseScaling {
		p.ScaleW, p.ScaleH = c.ScaledWidth, c.ScaledHeight
	}
	sink, result, err := pipeline.NewSink(p)
	if err != nil {
		return nil, err
	}
	if err := vp8l.DecodeWithOptions(parsed.VP8L, cl.wrap(sink), &vp8l.Options{UseThreads: c.UseThreads}); err != nil {
		return nil, err
	}
	return result(), nil
}

func (d *Decoder) decodeLossy(parsed *container.Image, cl call) (*image.NRGBA, error) {
	c := d.config
	w, h := parsed.Width, parsed.Height

	crop := image.Rect(0, 0, w, h)
	if c.UseCropping {
		crop = pipeline.SnapCropForYUV(image.Rect(c.CropLeft, c.CropTop, c.CropLeft+c.CropWidth, c.CropTop+c.CropHeight))
		if crop.Empty() || crop.Min.X < 0 || crop.Min.Y < 0 || crop.Max.X > w || crop.Max.Y > h {
			return nil, status.InvalidParam("crop rectangle outside of the frame")
		}
	}
	cw, ch := crop.Dx(), crop.Dy()

	opts := vp8.Options{
		BypassFiltering:   c.BypassFiltering,
		NoFancyUpsampling: c.NoFancyUpsampling,
		DitheringStrength: c.DitheringStrength,
		UseThreads:        c.UseThreads,
	}

	var (
		sw, sh int
		sink   pipeline.Sink
		result func() *image.NRGBA
		err    error
	)
	if c.UseScaling {
		var ok bool
		if sw, sh, ok = resample.ScaledDimensions(cw, ch, c.ScaledWidth, c.ScaledHeight); !ok {
			return nil, status.InvalidParam("invalid scaled dimensions")
		}
		// libwebp (WebPIoInitFromOptions) skips the loop filter for large
		// downscaling ratios.
		if sw < w*3/4 && sh < h*3/4 {
			opts.BypassFiltering = true
		}
	}
	if sink, result, err = pipeline.NewSink(pipeline.Params{SrcW: cw, SrcH: ch, ScaleW: sw, ScaleH: sh, Flip: c.Flip, Dst: cl.dst}); err != nil {
		return nil, err
	}

	// The alpha plane is decoded incrementally, as the VP8 decoder emits
	// rows (like libwebp's VP8DecompressAlphaRows), so that a truncated or
	// broken VP8 stream never pays for a full-frame alpha decode.
	var la *lazyAlpha
	if parsed.ALPH != nil {
		la = newLazyAlpha(parsed.ALPH, w, h, crop, c)
		defer la.release()
		opts.Alpha = la.get
	}

	switch {
	case c.UseScaling:
		ys, ok := sink.(pipeline.YUVSink)
		if !ok {
			return nil, status.InvalidParam("scaling sink does not accept YUV")
		}
		ad := &yuvAdapter{sink: ys, alpha: la, top: crop.Min.Y, left: crop.Min.X, alphaStride: w, ctx: cl.ctx}
		opts.Alpha = nil
		err = vp8.DecodeYUV(parsed.VP8, &opts, &crop, ad)
	case cw == w && ch == h:
		err = vp8.Decode(parsed.VP8, nil, &opts, cl.wrap(sink))
	default:
		err = vp8.DecodeRegion(parsed.VP8, nil, &opts, crop, cl.wrap(sink))
	}
	if err != nil {
		return nil, err
	}
	return result(), nil
}

// lazyAlpha decodes an ALPH chunk incrementally, up to the rows the VP8
// decoder is about to emit (libwebp's VP8DecompressAlphaRows): rows are
// decoded (lossless streams through vp8l.AlphaDecoder) and unfiltered in
// batches. Alpha dithering needs the whole plane, so with it the plane is
// decoded in one pass, as libwebp does.
type lazyAlpha struct {
	alph      []byte
	w, h      int
	crop      image.Rectangle
	dithering int

	started bool
	hdr     alpha.Header
	data    []byte             // payload after the header byte
	dec     *vp8l.AlphaDecoder // nil for uncompressed alpha
	pp      *[]byte            // pooled plane, if any
	spare   *[]byte            // pooled plane taken by prefetch, not yet used
	plane   []byte             // rows [0, rows) are final
	rows    int
	err     error

	// helperOK lets the first get start a helper goroutine (UseThreads);
	// while async is set the helper owns dec, pp, spare, plane and rows,
	// and get reads the published rows from hp.
	helperOK bool
	async    bool
	hp       *alphaHelper
}

// newLazyAlpha returns the alpha decoder of a w x h frame whose output
// region is crop. It must be released (on the calling goroutine).
func newLazyAlpha(alph []byte, w, h int, crop image.Rectangle, c *Config) *lazyAlpha {
	a := &lazyAlpha{alph: alph, w: w, h: h, crop: crop, dithering: c.AlphaDitheringStrength}
	// With UseThreads the plane is first needed on the worker goroutine,
	// while release runs on this one: a sync.Pool Get on another P misses
	// the plane put back by the previous decode. Take a pooled plane here
	// (without allocating one: the allocation stays lazy, see grow).
	a.prefetch()
	if c.UseThreads {
		// Likewise the lossless alpha decoder (vp8l pools its state): set
		// it up here, so that it comes from and returns to this goroutine's
		// pool. An error is kept for the first get, which keeps the VP8
		// errors that come before the first emitted row first.
		a.start()
		// The first get may hand the plane to a helper goroutine
		// (alpha_helper.go).
		a.helperOK = true
	}
	return a
}

// get returns the alpha plane with at least its first rowEnd rows decoded
// (the slice may change between calls).
func (a *lazyAlpha) get(rowEnd int) ([]byte, error) {
	if a.err != nil {
		return nil, a.err
	}
	if !a.started {
		if a.start(); a.err != nil {
			return nil, a.err
		}
	}
	rowEnd = min(rowEnd, a.h)
	if a.async {
		return a.getAsync(rowEnd)
	}
	if rowEnd <= a.rows {
		return a.plane, nil
	}
	if a.helperOK {
		a.helperOK = false
		if a.useHelper() && a.startHelper() {
			return a.getAsync(rowEnd)
		}
	}
	if a.hdr.PreProcessing == alpha.PreprocessingLevels && a.dithering > 0 {
		rowEnd = a.h // dequantization smooths the whole plane at once
	}
	if a.err = a.decodeRows(rowEnd); a.err != nil {
		return nil, a.err
	}
	return a.plane, nil
}

// start initializes the alpha decoder, keeping a failure in a.err.
func (a *lazyAlpha) start() {
	a.started = true
	a.err = a.init()
}

// init validates the ALPH header like libwebp's ALPHInit, which
// VP8DecompressAlphaRows reports as VP8_STATUS_OUT_OF_MEMORY on failure.
func (a *lazyAlpha) init() error {
	if len(a.alph) <= 1 {
		return status.New(utils.Vp8StatusOutOfMemory, "alpha decoder initialization failed: alpha chunk too short")
	}
	hdr, err := alpha.ParseHeader(a.alph[0])
	if err != nil {
		return status.New(utils.Vp8StatusOutOfMemory, "alpha decoder initialization failed: invalid ALPH header")
	}
	a.hdr = hdr
	a.data = a.alph[1:]
	if hdr.Compression == alpha.CompressionNone {
		if len(a.data) < a.w*a.h {
			return status.New(utils.Vp8StatusOutOfMemory, "alpha decoder initialization failed: alpha chunk too short")
		}
		return nil
	}
	a.dec, err = vp8l.NewAlphaDecoder(a.data, a.w, a.h)
	return err
}

// decodeRows decodes and unfilters rows [a.rows, rowEnd).
func (a *lazyAlpha) decodeRows(rowEnd int) error {
	w := a.w
	a.grow(rowEnd)
	y0 := a.rows
	if a.dec == nil {
		copy(a.plane[y0*w:rowEnd*w], a.data[y0*w:rowEnd*w])
	} else if err := a.dec.DecodeRows(a.plane[:rowEnd*w], rowEnd); err != nil {
		if status.CodeOf(err) == utils.Vp8StatusOutOfMemory {
			return err
		}
		return status.Bitstream("could not decode alpha data: " + err.Error())
	}
	if a.hdr.Filter != alpha.FilterNone {
		var prev []byte
		if y0 > 0 {
			prev = a.plane[(y0-1)*w : y0*w]
		}
		for y := y0; y < rowEnd; y++ {
			row := a.plane[y*w : (y+1)*w]
			alpha.UnfilterRow(a.hdr.Filter, prev, row, row)
			prev = row
		}
	}
	a.rows = rowEnd
	if rowEnd == a.h && a.hdr.PreProcessing == alpha.PreprocessingLevels && a.dithering > 0 {
		crop := a.crop
		win := a.plane[crop.Min.Y*w+crop.Min.X:]
		if err := alpha.Dequantize(win, crop.Dx(), crop.Dy(), w, min(a.dithering, 100)); err != nil {
			return err
		}
	}
	return nil
}

// minAlphaBand is the smallest allocation (in bytes) of a growing alpha
// plane.
const minAlphaBand = 4 << 20

// grow makes the plane hold at least rowEnd rows. Planes up to
// maxPooledPlane bytes are taken whole from planePool. Bigger ones start
// with minAlphaBand bytes and grow by 4x (straight to the full plane once
// that is at least half of it), so that a huge frame whose VP8 data ends
// early (or a crop near the top) does not allocate the full plane, while
// a complete decode allocates little more than the plane itself.
func (a *lazyAlpha) grow(rowEnd int) {
	w, h := a.w, a.h
	if len(a.plane) >= rowEnd*w {
		return
	}
	if w*h <= maxPooledPlane {
		if a.spare != nil {
			a.pp, a.spare = a.spare, nil
			*a.pp = (*a.pp)[:w*h]
		} else {
			a.pp = getPlane(w * h)
		}
		a.plane = *a.pp
		return
	}
	rows := max(rowEnd, 4*len(a.plane)/w, minAlphaBand/w)
	if 2*rows >= h {
		rows = h
	}
	np := make([]byte, rows*w)
	copy(np, a.plane[:a.rows*w])
	a.plane = np
}

// prefetch takes a plane big enough for the whole alpha plane from
// planePool, if the plane is poolable and the pool has one; it allocates
// nothing.
func (a *lazyAlpha) prefetch() {
	n := a.w * a.h
	if n > maxPooledPlane {
		return
	}
	if p, ok := planePool.Get().(*[]byte); ok {
		if cap(*p) >= n {
			a.spare = p
		} else {
			planePool.Put(p)
		}
	}
}

func (a *lazyAlpha) release() {
	a.stopHelper()
	if a.spare != nil {
		planePool.Put(a.spare)
		a.spare = nil
	}
	if a.dec != nil {
		a.dec.Close()
		a.dec = nil
	}
	if a.pp != nil {
		if cap(*a.pp) <= maxPooledPlane {
			planePool.Put(a.pp)
		}
		a.pp = nil
	}
	a.plane = nil
}

// maxPooledPlane bounds the alpha planes kept in planePool, so that one huge
// image does not pin its alpha plane in memory.
const maxPooledPlane = 16 << 20

// yuvAdapter forwards the Y'CbCr rows of the VP8 decoder (and the matching
// alpha rows) to a scaling sink, which rescales the planes separately like
// libwebp.
type yuvAdapter struct {
	sink        pipeline.YUVSink
	alpha       *lazyAlpha // nil for opaque images
	top, left   int        // crop origin in the alpha plane
	alphaStride int
	ctx         context.Context // DecodeContext, may be nil
}

func (a *yuvAdapter) PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error {
	if err := canceled(a.ctx); err != nil {
		return err
	}
	var ap []byte
	if a.alpha != nil {
		plane, err := a.alpha.get(a.top + y + n)
		if err != nil {
			return err
		}
		ap = plane[(a.top+y)*a.alphaStride+a.left:]
	}
	return a.sink.CommitYUV(y, n, ys, yStride, us, vs, uvStride, ap, a.alphaStride)
}

func (a *yuvAdapter) NeedRows() int { return a.sink.NeedRows() }

var planePool sync.Pool // of *[]byte

func getPlane(n int) *[]byte {
	if p, ok := planePool.Get().(*[]byte); ok && cap(*p) >= n {
		*p = (*p)[:n]
		return p
	}
	b := make([]byte, n)
	return &b
}
