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

package decoder

import (
	"image"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// lazyAlpha decodes an ALPH chunk incrementally, up to the rows the VP8
// decoder is about to emit (libwebp's VP8DecompressAlphaRows): rows are
// decoded (lossless streams through vp8l.AlphaDecoder) and unfiltered in
// batches.
//
// Unlike libwebp, which keeps the whole alpha plane, only a ring of the
// last decoded rows is kept (ringRows; a few hundred kilobytes instead of
// one byte per pixel), since the VP8 decoder asks for rows top to bottom
// and never goes back. A request's rows that wrap around the end of the
// ring are joined by copying the wrapped part into tail rows after the
// ring. The rows decoded for one request may be decoded in two calls (at
// the end of the ring), which gives the same result and the same errors as
// one call, except for a call that reaches the last row of the plane
// (libwebp only reports running out of data before the last pixel): the
// ring is placed so that the last row of the plane is its last row, and a
// request for the last row that does not fit in the ring switches to a
// plane.
//
// The whole plane is kept when it is not larger than the ring, and with
// alpha dithering, which smooths the whole plane at once (it is then
// decoded in one pass, as libwebp does). An uncompressed plane without a
// filter is the chunk itself.
type lazyAlpha struct {
	alph      []byte
	w, h      int
	crop      image.Rectangle
	dithering int
	threads   bool

	started bool
	hdr     alpha.Header
	data    []byte             // payload after the header byte
	dec     *vp8l.AlphaDecoder // nil for uncompressed alpha
	pp      *[]byte            // pooled buffer, if any
	spare   *[]byte            // pooled buffer taken by prefetch, not yet used

	// buf holds the decoded rows: with ring > 0 frame row y is row
	// (y+off)%ring of buf, followed by alphaTailRows tail rows; otherwise
	// frame row y is row y-base of buf (a plane).
	buf       []byte
	ring, off int
	base      int
	rows      int // rows [0, rows) are decoded
	err       error

	// helperOK lets the first get start a helper goroutine (UseThreads);
	// while async is set the helper owns dec, pp, spare, buf and rows,
	// and get reads the published rows from hp.
	helperOK bool
	async    bool
	hp       *alphaHelper
}

var (
	// alphaRingBytes is the size of the ring of a sequential decode; it
	// has at least minAlphaRingRows rows. Variables for tests.
	alphaRingBytes   = 256 << 10
	minAlphaRingRows = 64
)

// alphaTailRows bounds the rows of a request served from a ring (the VP8
// decoder asks for at most 16+8+1 rows at a time).
const alphaTailRows = 32

// ringRows returns the number of ring rows for a sequential decode, or for
// one with a helper goroutine (which works up to two bands ahead, see
// alpha_helper.go), or 0 if the plane is not larger than the ring.
func (a *lazyAlpha) ringRows(helper bool) int {
	r := max(minAlphaRingRows, (alphaRingBytes+a.w-1)/a.w)
	if helper {
		step, band := helperRows(a.w)
		r = max(r, 2*band+step+2*alphaTailRows)
	}
	if r+alphaTailRows >= a.h {
		return 0
	}
	return r
}

// wholePlane reports whether the plane must be decoded in one pass (alpha
// dithering of a quantized plane).
func (a *lazyAlpha) wholePlane() bool {
	return a.hdr.PreProcessing == alpha.PreprocessingLevels && a.dithering > 0
}

// newLazyAlpha returns the alpha decoder of a w x h frame whose output
// region is crop. It must be released (on the calling goroutine).
func newLazyAlpha(alph []byte, w, h int, crop image.Rectangle, c *Config) *lazyAlpha {
	a := &lazyAlpha{alph: alph, w: w, h: h, crop: crop, dithering: c.AlphaDitheringStrength, threads: c.UseThreads}
	// With UseThreads the buffer is first needed on the worker goroutine,
	// while release runs on this one: a sync.Pool Get on another P misses
	// the buffer put back by the previous decode. Take a pooled buffer
	// here (without allocating one: the allocation stays lazy, see alloc).
	a.prefetch()
	if c.UseThreads {
		// Likewise the lossless alpha decoder (vp8l pools its state): set
		// it up here, so that it comes from and returns to this goroutine's
		// pool. An error is kept for the first get, which keeps the VP8
		// errors that come before the first emitted row first.
		a.start()
		// The first get may hand the decoding to a helper goroutine
		// (alpha_helper.go).
		a.helperOK = true
	}
	return a
}

// get returns the alpha samples of frame rows [rowStart, rowEnd), decoding
// the plane up to rowEnd, as a window of the plane (vp8.Options.Alpha):
// frame row y is plane[(y-base)*w:]. The window is valid until the next
// call; rowStart must not decrease from one call to the next.
func (a *lazyAlpha) get(rowStart, rowEnd int) (plane []byte, base int, err error) {
	if a.err != nil {
		return nil, 0, a.err
	}
	if !a.started {
		if a.start(); a.err != nil {
			return nil, 0, a.err
		}
	}
	rowEnd = min(rowEnd, a.h)
	rowStart = min(rowStart, rowEnd)
	if a.async {
		return a.getAsync(rowStart, rowEnd)
	}
	if rowEnd > a.rows {
		if a.helperOK {
			a.helperOK = false
			if a.useHelper() && a.startHelper() {
				return a.getAsync(rowStart, rowEnd)
			}
		}
		end := rowEnd
		if a.wholePlane() {
			end = a.h // dequantization smooths the whole plane at once
		}
		if a.err = a.decodeRequest(rowStart, end); a.err != nil {
			return nil, 0, a.err
		}
	}
	return a.window(a.buf, rowStart, rowEnd)
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
		if hdr.Filter == alpha.FilterNone && !a.wholePlane() {
			// The plane is the chunk itself.
			a.buf, a.ring, a.base, a.rows = a.data, 0, 0, a.h
		}
		return nil
	}
	a.dec, err = vp8l.NewAlphaDecoder(a.data, a.w, a.h)
	return err
}

// alloc sets up the buffer before the first rows are decoded: a ring of
// ringRows(helper) rows, or the plane.
func (a *lazyAlpha) alloc(helper bool) {
	if a.buf != nil {
		return
	}
	w := a.w
	r := 0
	if !a.wholePlane() {
		r = a.ringRows(helper)
	}
	if r == 0 {
		a.grow(1)
		return
	}
	n := (r + alphaTailRows) * w
	if a.spare != nil && cap(*a.spare) >= n {
		a.pp, a.spare = a.spare, nil
		*a.pp = (*a.pp)[:n]
	} else {
		a.pp = getPlane(n)
	}
	a.buf = *a.pp
	a.ring = r
	// The last row of the plane is the last row of the ring, so that the
	// request that reaches it is decoded in one call (see lazyAlpha).
	a.off = (r - a.h%r) % r
}

// decodeRequest decodes the rows [a.rows, rowEnd) of the request for rows
// [rowStart, rowEnd). With a ring, a request that the ring cannot serve
// (more rows than the tail can join, or a call for the last row of the
// plane that does not fit in the ring in one piece) first switches to a
// plane of the rows from rowStart on.
func (a *lazyAlpha) decodeRequest(rowStart, rowEnd int) error {
	a.alloc(false)
	if a.ring > 0 && (rowEnd-rowStart > min(alphaTailRows, a.ring-1) || rowEnd == a.h && rowEnd-a.rows > a.ring-1) {
		if err := a.toPlane(rowStart); err != nil {
			return err
		}
	}
	return a.decodeRows(rowEnd)
}

// toPlane replaces the ring with a plane holding the rows from
// min(rowStart, a.rows-1) (the row before the next one to decode, which
// the unfilter needs) to the end of the frame.
func (a *lazyAlpha) toPlane(rowStart int) error {
	w := a.w
	keep := min(rowStart, a.rows)
	if a.rows > 0 {
		keep = min(keep, a.rows-1)
	}
	if keep < a.rows-(a.ring-1) {
		// Requests never go back that far (rowStart does not decrease).
		return status.InvalidParam("alpha rows no longer available")
	}
	np := make([]byte, (a.h-keep)*w)
	for y := keep; y < a.rows; y++ {
		copy(np[(y-keep)*w:][:w], a.rowAt(y))
	}
	a.buf, a.ring, a.off, a.base = np, 0, 0, keep
	return nil
}

// rowAt returns decoded frame row y (which the buffer must hold).
func (a *lazyAlpha) rowAt(y int) []byte {
	if a.ring > 0 {
		return a.buf[(y+a.off)%a.ring*a.w:][:a.w]
	}
	return a.buf[(y-a.base)*a.w:][:a.w]
}

// window returns the rows [rowStart, rowEnd) of buf (a.buf, or the buffer
// a helper published) as a plane window for get.
func (a *lazyAlpha) window(buf []byte, rowStart, rowEnd int) ([]byte, int, error) {
	if a.ring == 0 {
		return buf, a.base, nil
	}
	w := a.w
	p := (rowStart + a.off) % a.ring
	if k := p + rowEnd - rowStart - a.ring; k > 0 {
		if k > alphaTailRows {
			return nil, 0, status.InvalidParam("alpha request too large")
		}
		// The rows wrap around the ring: join them in the tail rows.
		copy(buf[a.ring*w:(a.ring+k)*w], buf[:k*w])
	}
	return buf[p*w:], rowStart, nil
}

// decodeRows decodes and unfilters rows [a.rows, rowEnd) into the buffer.
func (a *lazyAlpha) decodeRows(rowEnd int) error {
	w := a.w
	for a.rows < rowEnd {
		y0 := a.rows
		n := rowEnd - y0
		var dst []byte
		if a.ring > 0 {
			p := (y0 + a.off) % a.ring
			n = min(n, a.ring-p) // the next call continues at the start of the ring
			if p == 0 && n == a.ring && y0 > 0 {
				n-- // keep the previous row for the unfilter
			}
			dst = a.buf[p*w : (p+n)*w]
		} else {
			if a.base == 0 {
				a.grow(rowEnd)
			}
			dst = a.buf[(y0-a.base)*w : (rowEnd-a.base)*w]
		}
		if a.dec == nil {
			copy(dst, a.data[y0*w:(y0+n)*w])
		} else if err := a.dec.DecodeRowsAt(dst, y0, y0+n); err != nil {
			if status.CodeOf(err) == utils.Vp8StatusOutOfMemory {
				return err
			}
			return status.Bitstream("could not decode alpha data: " + err.Error())
		}
		if a.hdr.Filter != alpha.FilterNone {
			var prev []byte
			if y0 > 0 {
				prev = a.rowAt(y0 - 1)
			}
			for i := range n {
				row := dst[i*w : (i+1)*w]
				alpha.UnfilterRow(a.hdr.Filter, prev, row, row)
				prev = row
			}
		}
		a.rows = y0 + n
	}
	if rowEnd == a.h && a.wholePlane() {
		crop := a.crop
		win := a.buf[crop.Min.Y*w+crop.Min.X:]
		if err := alpha.Dequantize(win, crop.Dx(), crop.Dy(), w, min(a.dithering, 100)); err != nil {
			return err
		}
	}
	return nil
}

// minAlphaBand is the smallest allocation (in bytes) of a growing alpha
// plane.
const minAlphaBand = 4 << 20

// grow makes the plane (base 0) hold at least rowEnd rows. Planes up to
// maxPooledPlane bytes are taken whole from planePool. Bigger ones start
// with minAlphaBand bytes and grow by 4x (straight to the full plane once
// that is at least half of it), so that a huge frame whose VP8 data ends
// early (or a crop near the top) does not allocate the full plane, while
// a complete decode allocates little more than the plane itself.
func (a *lazyAlpha) grow(rowEnd int) {
	w, h := a.w, a.h
	if len(a.buf) >= rowEnd*w {
		return
	}
	if w*h <= maxPooledPlane {
		if a.spare != nil && cap(*a.spare) >= w*h {
			a.pp, a.spare = a.spare, nil
			*a.pp = (*a.pp)[:w*h]
		} else {
			a.pp = getPlane(w * h)
		}
		a.buf = *a.pp
		return
	}
	rows := max(rowEnd, 4*len(a.buf)/w, minAlphaBand/w)
	if 2*rows >= h {
		rows = h
	}
	np := make([]byte, rows*w)
	copy(np, a.buf[:a.rows*w])
	a.buf = np
}

// prefetch takes a buffer from planePool that is big enough for the
// expected layout (the ring, or the plane), if the pool has one; it
// allocates nothing.
func (a *lazyAlpha) prefetch() {
	n := a.w * a.h
	if len(a.alph) > 1 {
		if hdr, err := alpha.ParseHeader(a.alph[0]); err == nil {
			a.hdr = hdr // for wholePlane; init sets it again
			if hdr.Compression == alpha.CompressionNone && hdr.Filter == alpha.FilterNone && !a.wholePlane() {
				return // the plane is the chunk itself
			}
			if r := a.ringRows(a.threads); r > 0 && !a.wholePlane() {
				n = (r + alphaTailRows) * a.w
			}
		}
	}
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
	a.buf = nil
}

// maxPooledPlane bounds the alpha planes kept in planePool, so that one huge
// image does not pin its alpha plane in memory.
const maxPooledPlane = 16 << 20

var planePool sync.Pool // of *[]byte

func getPlane(n int) *[]byte {
	if p, ok := planePool.Get().(*[]byte); ok && cap(*p) >= n {
		*p = (*p)[:n]
		return p
	}
	b := make([]byte, n)
	return &b
}
