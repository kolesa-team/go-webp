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

package vp8l

// Algorithm adapted from libwebp (BSD-3-Clause): DecodeImageData,
// DecodeAlphaData and ProcessRows from src/dec/vp8l_dec.c.

import (
	"encoding/binary"

	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// prefixValue decodes an LZ77 length/distance prefix symbol, taking its
// extra bits from val. It returns the value and the number of bits used.
func prefixValue(sym int, val uint64) (int, uint) {
	if sym < 4 {
		return sym + 1, 0
	}
	eb := uint(sym-2) >> 1
	off := (2 + sym&1) << eb
	return off + int(val&(uint64(1)<<eb-1)) + 1, eb
}

func errTruncated() error { return status.NotEnoughData("vp8l: truncated pixel data") }

// Color cache maintenance.
//
// libwebp inserts every decoded pixel into the color cache: at the end of
// each row, after each backward reference and before each cache lookup.
// Only lookups read the cache, so decodePixels defers the insertions: the
// pixels buf[lastCached:p] are pending, and they are inserted in order (a
// forward loop) before the next lookup, before the window slides (which
// moves them) or, for caches of more than lazyCacheMaxBits bits, at the
// same points as libwebp.
//
// Small caches (palette images use 2 to 4 bits) are settled backwards
// instead when many pixels are pending: the last pending pixel hashing to a
// slot is the value that slot ends up with, so the scan can stop as soon as
// every slot has been seen, usually after a handful of pixels instead of
// the whole range. Both orders leave the cache in the same state.
var (
	// lazyCacheMaxBits is the largest cache (in bits) maintained lazily; at
	// most 6 (settleCache tracks the slots in a uint64). A variable so that
	// tests can compare against eager maintenance (0).
	lazyCacheMaxBits = 4
	// lazyCacheMinRun is the smallest pending range settled backwards;
	// shorter ranges are inserted forwards. A variable for tests.
	lazyCacheMinRun = 256
)

// insertCache inserts px into the color cache in order. shift is
// 32-cacheBits.
func insertCache(cache *[1 << maxCacheBits]uint32, px []uint32, shift uint) {
	for _, c := range px {
		cache[(c*colorCacheMult)>>(shift&31)&(1<<maxCacheBits-1)] = c
	}
}

// settleCache leaves a color cache of at most 64 slots (shift >= 26) in
// the state inserting px in order would: scanning backwards, the first
// pixel seen for a slot is its final value, and the scan stops once every
// slot is settled.
//
// It is small enough to be inlined: a call on the cache lookup path of
// decodePixels makes the register allocator spill around it even when the
// branch is not taken, which costs more than 10% on photos.
func settleCache(cache *[1 << maxCacheBits]uint32, px []uint32, shift uint) {
	full := ^uint64(0) >> (64 - 1<<(32-shift&31)) // one bit per slot
	var seen uint64
	for i := len(px) - 1; i >= 0 && seen != full; i-- {
		c := px[i]
		h := (c * colorCacheMult) >> (shift & 31) & 63
		if seen>>h&1 == 0 {
			seen |= 1 << h
			cache[h] = c
		}
	}
}

// decodePixels entropy-decodes a w*h image into buf.
//
// For level0, rows are passed through the inverse transforms to the output
// every numCacheRows rows and decoding stops after d.needRows rows. With
// sliding, buf is smaller than the image and acts as an LZ77 window of at
// least window pixels (plus the rows not yet emitted). The loop position is
// kept in st, so that a level-0 decode can be resumed with a larger
// d.needRows (incremental alpha decoding).
func (d *decoder) decodePixels(md *metadata, buf []uint32, w, h int, level0, sliding bool, window int, st *pixState) error {
	groups := md.groups
	total := w * h
	lastPixel := total
	if level0 {
		lastPixel = w * d.needRows
	}
	var cache *[1 << maxCacheBits]uint32
	var cacheShift uint
	var cacheSize int
	// eager: insert the pending pixels at the same points as libwebp,
	// otherwise only before lookups and window slides (see above).
	eager := false
	if md.cacheBits > 0 {
		if d.colorCache == nil {
			d.colorCache = new([1 << maxCacheBits]uint32)
		}
		cache = d.colorCache
		cacheSize = 1 << md.cacheBits
		if !st.started {
			clear(cache[:cacheSize])
		}
		cacheShift = 32 - uint(md.cacheBits)
		eager = md.cacheBits > min(lazyCacheMaxBits, 6)
	}
	const lenCodeLimit = numLiteralCodes + numLengthCodes
	cacheLimit := lenCodeLimit + cacheSize
	mask := md.huffMask

	data := d.br.data
	bpos, val, nbits := d.br.pos, d.br.val, d.br.nbits

	st.started = true
	base, p, col, row, lastCached := st.base, st.p, st.col, st.row, st.lastCached
	var g *htreeGroup
	if base+p < lastPixel {
		g = md.group(groups, col, row)
	}
	var err error
outer:
	for {
		stop := lastPixel - base
		if sliding {
			stop = min(stop, len(buf)-maxCopyLength)
		}
		if a := d.async; a != nil && level0 && a.guard > 0 {
			// Do not overwrite the rows the helper still reads (copies
			// write up to maxCopyLength pixels past stop).
			stop = min(stop, a.guard-maxCopyLength)
		}
		for p < stop {
			if col&mask == 0 {
				g = md.group(groups, col, row)
			}
			if g.isTrivialCode {
				buf[p] = g.literalARB
			} else {
				if nbits < 32 {
					if bpos+8 <= len(data) {
						val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
						bpos += int((63 - nbits) >> 3)
						nbits |= 56
					} else {
						val, nbits, bpos = refillSlow(data, val, nbits, bpos)
					}
				}
				var code int
				if g.usePacked {
					pc := g.packed[val&(packedSize-1)]
					if pc.bits < specialMarker {
						val >>= pc.bits
						nbits -= uint(pc.bits)
						buf[p] = pc.value
						code = -1
					} else {
						n := pc.bits - specialMarker
						val >>= n
						nbits -= uint(n)
						code = int(pc.value)
					}
				} else {
					c, n := readSym(g.trees[treeGreen], val)
					val >>= n
					nbits -= n
					code = c
				}
				switch {
				case code < 0:
					// Literal fully decoded from the packed table.
				case code < numLiteralCodes:
					if g.isTrivialLiteral {
						buf[p] = g.literalARB | uint32(code)<<8
						break
					}
					red, n := readSym(g.trees[treeRed], val)
					val >>= n
					nbits -= n
					if nbits < 32 {
						if bpos+8 <= len(data) {
							val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
							bpos += int((63 - nbits) >> 3)
							nbits |= 56
						} else {
							val, nbits, bpos = refillSlow(data, val, nbits, bpos)
						}
					}
					blue, n := readSym(g.trees[treeBlue], val)
					val >>= n
					nbits -= n
					alpha, n := readSym(g.trees[treeAlpha], val)
					val >>= n
					nbits -= n
					buf[p] = uint32(alpha)<<24 | uint32(red)<<16 | uint32(code)<<8 | uint32(blue)
				case code < lenCodeLimit:
					length, n := prefixValue(code-numLiteralCodes, val)
					val >>= n
					nbits -= n
					if nbits < 32 {
						if bpos+8 <= len(data) {
							val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
							bpos += int((63 - nbits) >> 3)
							nbits |= 56
						} else {
							val, nbits, bpos = refillSlow(data, val, nbits, bpos)
						}
					}
					distSym, n := readSym(g.trees[treeDist], val)
					val >>= n
					nbits -= n
					if nbits < 32 {
						if bpos+8 <= len(data) {
							val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
							bpos += int((63 - nbits) >> 3)
							nbits |= 56
						} else {
							val, nbits, bpos = refillSlow(data, val, nbits, bpos)
						}
					}
					distCode, n := prefixValue(distSym, val)
					val >>= n
					nbits -= n
					dist := planeCodeToDistance(w, distCode)
					// Like libwebp, stop as soon as the data is exhausted:
					// zero bits past the end decode as valid copies and would
					// otherwise fill the rest of a (possibly huge) frame.
					if isEOS(len(data), bpos, nbits) {
						err = errTruncated()
						break outer
					}
					abs := base + p
					if dist > p || length > total-abs {
						err = status.Bitstream("vp8l: invalid backward reference")
						break outer
					}
					copyBlock32(buf, p, dist, length)
					p += length
					col += length
					for col >= w {
						col -= w
						row++
						if level0 && row%numCacheRows == 0 {
							if err = d.processRows(buf, base, row); err != nil {
								break outer
							}
						}
					}
					if col&mask != 0 {
						g = md.group(groups, col, row)
					}
					if eager {
						insertCache(cache, buf[lastCached:p], cacheShift)
						lastCached = p
					}
					continue
				case code < cacheLimit:
					if !eager && p-lastCached >= lazyCacheMinRun {
						settleCache(cache, buf[lastCached:p], cacheShift)
					} else {
						insertCache(cache, buf[lastCached:p], cacheShift)
					}
					lastCached = p
					buf[p] = cache[(code-lenCodeLimit)&(1<<maxCacheBits-1)]
				default:
					err = status.Bitstream("vp8l: invalid green symbol")
					break outer
				}
			}
			p++
			col++
			if col >= w {
				col = 0
				row++
				if level0 && row%numCacheRows == 0 {
					if err = d.processRows(buf, base, row); err != nil {
						break outer
					}
				}
				if eager {
					insertCache(cache, buf[lastCached:p], cacheShift)
					lastCached = p
				}
				if bpos > len(data) && isEOS(len(data), bpos, nbits) {
					err = errTruncated()
					break outer
				}
			}
		}
		if base+p >= lastPixel {
			break
		}
		if a := d.async; a != nil && level0 && a.guard > 0 && p < len(buf)-maxCopyLength {
			// Stopped at the guard: wait for the helper, then go on.
			if err = a.waitGuard(); err != nil {
				break
			}
			continue
		}
		// Slide the window: keep the LZ77 window and the rows not yet
		// emitted. The pending cache pixels move, so settle them first.
		if cache != nil && !eager && p-lastCached >= lazyCacheMinRun {
			settleCache(cache, buf[lastCached:p], cacheShift)
		} else if cache != nil {
			insertCache(cache, buf[lastCached:p], cacheShift)
		}
		keep := min(base+p-window, d.lastRow*w)
		if a := d.async; a != nil && level0 {
			if err = a.beforeSlide(p - (keep - base)); err != nil {
				break
			}
		}
		n := copy(buf, buf[keep-base:p])
		base = keep
		p = n
		lastCached = p
	}
	d.br.pos, d.br.val, d.br.nbits = bpos, val, nbits
	*st = pixState{base: base, p: p, col: col, row: row, lastCached: lastCached, started: true}
	if err != nil {
		return err
	}
	if d.br.eos() {
		return errTruncated()
	}
	if level0 {
		return d.processRows(buf, base, row)
	}
	return nil
}

// processRows runs the inverse transforms over the decoded rows
// [d.lastRow, row) and emits them. The last inverse transform is fused with
// the output conversion when it is subtract-green or color indexing.
func (d *decoder) processRows(buf []uint32, base, row int) error {
	row = min(row, d.needRows)
	if d.async != nil {
		return d.sendRows(base, row, false)
	}
	y0, t0 := d.lastRow, d.probeStart()
	// A resumed decode (AlphaDecoder) may have more than numCacheRows
	// decoded rows pending: emit them in batches that fit d.rows.
	for row-d.lastRow > numCacheRows {
		if err := d.processBatch(buf, base, d.lastRow+numCacheRows); err != nil {
			return err
		}
	}
	if err := d.processBatch(buf, base, row); err != nil {
		return err
	}
	d.probeDone(t0, y0)
	return nil
}

// processBatch is processRows for at most numCacheRows rows.
func (d *decoder) processBatch(buf []uint32, base, row int) error {
	y0 := d.lastRow
	n := row - y0
	if n <= 0 {
		return nil
	}
	w := d.xsize
	argb := buf[y0*w-base : row*w-base]
	d.lastRow = row
	return d.emitRows(argb, y0, row)
}

// emitRows runs the inverse transforms over rows [y0, row) (at most
// numCacheRows rows, the level-0 pixels argb) and emits them. Rows must
// arrive in order. In a two-stage decode it runs on the helper goroutine,
// so it must only use the decoder state reserved to it: d.rows, the fused
// palette tables and the output.
func (d *decoder) emitRows(argb []uint32, y0, row int) error {
	n := row - y0
	fw := d.width

	// Pick the transform fused into the output step, if any.
	var final *transform
	last := 0
	if d.nTransforms > 0 {
		t := &d.transforms[0]
		switch {
		case t.kind == subtractGreenTransform:
			last = 1
			if d.out == outRGBA {
				final = t
			}
		case t.kind == colorIndexingTransform && d.out == outRGBA:
			last = 1
			final = t
		}
	}
	if d.nTransforms > last {
		out := d.rows[fw : fw+fw*n]
		src := argb
		inCache := false
		for i := d.nTransforms - 1; i >= last; i-- {
			t := &d.transforms[i]
			switch t.kind {
			case predictorTransform:
				inversePredictor(t, y0, row, src, d.rows, fw)
				if row != t.ysize {
					// Keep the last row as the top row of the next batch.
					copy(d.rows[fw-t.xsize:fw], d.rows[fw+(n-1)*t.xsize:fw+n*t.xsize])
				}
			case crossColorTransform:
				inverseCrossColor(t, y0, row, src, out)
			case subtractGreenTransform:
				addGreen(src[:n*t.xsize], out)
			case colorIndexingTransform:
				if inCache && t.bits > 0 {
					// Move the bundled indices to the end of the rows so
					// they can be expanded in place.
					outLen := n * t.xsize
					inLen := n * subSampleSize(t.xsize, t.bits)
					s := out[outLen-inLen : outLen]
					copy(s, out[:inLen])
					inverseColorIndex(t, y0, row, s, out)
				} else {
					inverseColorIndex(t, y0, row, src, out)
				}
			}
			src = out
			inCache = true
		}
		argb = out
	}

	if d.out == outGreen {
		dst := d.alpha[(y0-d.alphaRow)*fw : (row-d.alphaRow)*fw]
		argb = argb[:len(dst)]
		for i, c := range argb {
			dst[i] = byte(c >> 8)
		}
		return nil
	}
	b := d.sink.Buffer(y0, n)
	stride := d.sink.Stride()
	if stride < 4*fw || len(b) < (n-1)*stride+4*fw {
		return status.InvalidParam("vp8l: sink buffer too small")
	}
	switch {
	case final == nil:
		for y := 0; y < n; y++ {
			emitRGBA(argb[y*fw:(y+1)*fw], b[y*stride:y*stride+4*fw])
		}
	case final.kind == subtractGreenTransform:
		for y := 0; y < n; y++ {
			emitAddGreen(argb[y*fw:(y+1)*fw], b[y*stride:y*stride+4*fw])
		}
	default:
		pw := subSampleSize(fw, final.bits)
		for y := 0; y < n; y++ {
			d.emitPalette(final, argb[y*pw:(y+1)*pw], b[y*stride:y*stride+4*fw])
		}
	}
	return d.sink.Commit(y0, n)
}

// toRGBA converts an ARGB pixel to the little-endian word of its R,G,B,A
// bytes.
func toRGBA(c uint32) uint32 {
	return c&0xff00ff00 | c>>16&0xff | (c&0xff)<<16
}

// emitRGBAGo is the reference of emitRGBA (dsp_*.go): it converts the ARGB
// pixels src to RGBA bytes in dst (as many pixels as fit), two pixels per
// 64-bit word.
func emitRGBAGo(src []uint32, dst []byte) {
	for len(src) >= 2 && len(dst) >= 8 {
		c := uint64(src[0]) | uint64(src[1])<<32
		binary.LittleEndian.PutUint64(dst, swapRB2(c))
		src, dst = src[2:], dst[8:]
	}
	if len(src) > 0 && len(dst) >= 4 {
		binary.LittleEndian.PutUint32(dst, toRGBA(src[0]))
	}
}

// swapRB2 swaps the red and blue bytes of the two ARGB pixels in c.
func swapRB2(c uint64) uint64 {
	return c&0xff00ff00ff00ff00 | c>>16&0x000000ff000000ff | c&0x000000ff000000ff<<16
}

// emitAddGreenGo is the reference of emitAddGreen (dsp_*.go), which fuses
// the inverse subtract-green transform with emitRGBA.
func emitAddGreenGo(src []uint32, dst []byte) {
	for len(src) >= 2 && len(dst) >= 8 {
		c := uint64(src[0]) | uint64(src[1])<<32
		g := c & 0x0000ff000000ff00
		rb := (c&0x00ff00ff00ff00ff + (g>>8 | g<<8)) & 0x00ff00ff00ff00ff
		binary.LittleEndian.PutUint64(dst, swapRB2(c&0xff00ff00ff00ff00|rb))
		src, dst = src[2:], dst[8:]
	}
	if len(src) > 0 && len(dst) >= 4 {
		c := src[0]
		g := c >> 8 & 0xff
		rb := (c&0x00ff00ff + (g<<16 | g)) & 0x00ff00ff
		binary.LittleEndian.PutUint32(dst, toRGBA(c&0xff00ff00|rb))
	}
}

// emitPalette fuses the inverse color-indexing transform with emitRGBA.
// src is one row of (possibly bundled) indices, dst one RGBA row.
func (d *decoder) emitPalette(t *transform, src []uint32, dst []byte) {
	pal := d.paletteRGBA(t)
	if t.bits == 0 {
		for _, c := range src {
			if len(dst) < 4 {
				return
			}
			binary.LittleEndian.PutUint32(dst, pal[c>>8&0xff])
			dst = dst[4:]
		}
		return
	}
	// Bundled indices: expand a whole packed byte at once from a table of
	// the 256 possible byte values.
	tab := d.bundleTable(t)
	wpb := 1 << t.bits >> 1 // words per packed byte
	for _, c := range src {
		k := int(c>>8&0xff) * wpb
		for _, v := range tab[k : k+wpb] {
			if len(dst) < 8 {
				if len(dst) >= 4 {
					binary.LittleEndian.PutUint32(dst, uint32(v))
				}
				return
			}
			binary.LittleEndian.PutUint64(dst, v)
			dst = dst[8:]
		}
	}
}

// paletteRGBA returns the palette of t converted to RGBA words,
// alpha-premultiplied with d.premulPal.
func (d *decoder) paletteRGBA(t *transform) *[256]uint32 {
	if !d.palReady {
		for i, c := range t.data {
			d.pal[i] = toRGBA(c)
		}
		if d.premulPal {
			var b [4 * 256]byte
			n := len(t.data)
			for i, c := range d.pal[:n] {
				binary.LittleEndian.PutUint32(b[4*i:], c)
			}
			resample.PremultiplyRGBA(b[:4*n])
			for i := range d.pal[:n] {
				d.pal[i] = binary.LittleEndian.Uint32(b[4*i:])
			}
		}
		d.palReady = true
	}
	return &d.pal
}

// bundleTable returns, for each packed index byte, the RGBA words of the
// 1<<t.bits pixels it holds, two pixels per little-endian 64-bit word
// (1<<t.bits/2 words per byte).
func (d *decoder) bundleTable(t *transform) []uint64 {
	if d.bundleReady {
		return d.bundle
	}
	pal := d.paletteRGBA(t)
	ppb := 1 << t.bits
	bpp := uint(8 >> t.bits)
	mask := 1<<bpp - 1
	wpb := ppb / 2
	n := 256 * wpb
	if cap(d.bundle) < n {
		d.bundle = make([]uint64, 256*4)
	}
	tab := d.bundle[:n]
	for v := 0; v < 256; v++ {
		e := tab[v*wpb : (v+1)*wpb]
		for k := range e {
			lo := pal[v>>(uint(2*k)*bpp)&mask]
			hi := pal[v>>(uint(2*k+1)*bpp)&mask]
			e[k] = uint64(lo) | uint64(hi)<<32
		}
	}
	d.bundle = tab
	d.bundleReady = true
	return tab
}

// decodeAlpha8Rows continues the 8-bit alpha path (only transform: color
// indexing; red, blue and alpha codes trivial; no color cache) up to row
// upTo: the indices are decoded as bytes into d.ib (the whole image or a
// sliding window) and expanded into d.alpha in row batches, like libwebp's
// DecodeAlphaData + ExtractPalettedAlphaRows.
func (d *decoder) decodeAlpha8Rows(upTo int) error {
	md := &d.md0
	w := d.xsize
	st := &d.px
	limit := w * upTo
	for st.base+st.p < limit {
		stop := limit - st.base
		if d.sliding {
			stop = min(stop, len(d.ib)-maxCopyLength)
		}
		var err error
		if d.inTail {
			err = d.decodeTail8(md, stop)
		} else {
			err = d.decodePixels8(md, stop)
		}
		if err != nil {
			return err
		}
		if st.base+st.p >= limit || (d.inTail && d.tr.eos) {
			break
		}
		// Slide the window: emit the complete rows, then keep the LZ77
		// window and the rows not yet emitted.
		d.extractAlpha8(st.row)
		keep := min(st.base+st.p-d.window, d.lastRow*w)
		n := copy(d.ib, d.ib[keep-st.base:st.p])
		st.base = keep
		st.p = n
	}
	d.extractAlpha8(min(st.row, upTo))
	if d.inTail {
		// libwebp's DecodeAlphaData: running out of data is an error only
		// before the last pixel of the image is decoded.
		d.tr.eos = d.tr.atEnd()
		if d.tr.eos && st.base+st.p < w*d.height {
			return errTruncated()
		}
	}
	return nil
}

// extractAlpha8 expands the indices of rows [d.lastRow, row) into d.alpha.
func (d *decoder) extractAlpha8(row int) {
	y0 := d.lastRow
	if row <= y0 {
		return
	}
	w, fw, base := d.xsize, d.width, d.px.base
	inverseColorIndexAlpha(&d.transforms[0], d.alphaTab, row-y0, d.ib[y0*w-base:row*w-base], d.alpha[(y0-d.alphaRow)*fw:(row-d.alphaRow)*fw])
	d.lastRow = row
}

// decodeTail8 continues decodePixels8 up to window position stop with
// libwebp's exact end-of-stream behavior (DecodeAlphaData): the bits read
// past the end are the ones libwebp's reader produces, and decoding stops
// (d.tr.eos) when the end of the stream is reached.
func (d *decoder) decodeTail8(md *metadata, stop int) error {
	groups := md.groups
	mask := md.huffMask
	st := &d.px
	buf := d.ib
	w := d.xsize
	total := w*d.height - st.base // pixels left from the window start
	p, col, row := st.p, st.col, st.row
	tr := &d.tr
	g := md.group(groups, col, row)
	var err error
	for !tr.eos && p < stop {
		if col&mask == 0 {
			g = md.group(groups, col, row)
		}
		if g.isTrivialCode {
			buf[p] = byte(g.literalARB >> 8)
			p++
			col++
			if col >= w {
				col = 0
				row++
			}
			continue // libwebp does not re-check the end of stream here
		}
		tr.fill()
		code := tr.readSym(g.trees[treeGreen])
		switch {
		case code < numLiteralCodes:
			buf[p] = byte(code)
			p++
			col++
			if col >= w {
				col = 0
				row++
			}
		case code < numLiteralCodes+numLengthCodes:
			length := tr.copyValue(code - numLiteralCodes)
			distSym := tr.readSym(g.trees[treeDist])
			tr.fill()
			dist := planeCodeToDistance(w, tr.copyValue(distSym))
			if dist > p || length > total-p {
				if tr.atEnd() {
					err = errTruncated()
				} else {
					err = status.Bitstream("vp8l: invalid backward reference")
				}
				break
			}
			copyBlock8(buf, p, dist, length)
			p += length
			col += length
			for col >= w {
				col -= w
				row++
			}
			if p < total && col&mask != 0 {
				g = md.group(groups, col, row)
			}
		default:
			if tr.atEnd() {
				err = errTruncated()
			} else {
				err = status.Bitstream("vp8l: invalid green symbol")
			}
		}
		if err != nil {
			break
		}
		tr.eos = tr.atEnd()
	}
	st.p, st.col, st.row = p, col, row
	return err
}

// decodePixels8 is decodePixels for 8-bit index data (no color cache, no
// transforms, trivial red/blue/alpha codes), up to window position stop.
// Close to the end of the data it hands over to decodeTail8.
func (d *decoder) decodePixels8(md *metadata, stop int) error {
	groups := md.groups
	mask := md.huffMask
	st := &d.px
	buf := d.ib
	w := d.xsize
	total := w*d.height - st.base // pixels left from the window start
	data := d.br.data
	bpos, val, nbits := d.br.pos, d.br.val, d.br.nbits
	p, col, row := st.p, st.col, st.row
	g := md.group(groups, col, row)
	limit := eosLimit(len(data))
	tailPos := len(data) - 8
	var err error
loop:
	for p < stop {
		if bpos >= tailPos && bpos*8-int(nbits)+64 > limit {
			// Close to the end of data: finish with an exact emulation of
			// libwebp's bit reader.
			d.br.pos, d.br.val, d.br.nbits = bpos, val, nbits
			st.p, st.col, st.row = p, col, row
			consumed := bpos*8 - int(nbits)
			if consumed > eosLimit(len(data)) {
				return errTruncated()
			}
			d.inTail = true
			d.tr = newTailReader(data, consumed)
			return d.decodeTail8(md, stop)
		}
		if col&mask == 0 {
			g = md.group(groups, col, row)
		}
		if g.isTrivialCode {
			buf[p] = byte(g.literalARB >> 8)
		} else {
			if nbits < 32 {
				if bpos+8 <= len(data) {
					val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
					bpos += int((63 - nbits) >> 3)
					nbits |= 56
				} else {
					val, nbits, bpos = refillSlow(data, val, nbits, bpos)
				}
			}
			code, n := readSym(g.trees[treeGreen], val)
			val >>= n
			nbits -= n
			if code < numLiteralCodes {
				buf[p] = byte(code)
			} else if code < numLiteralCodes+numLengthCodes {
				length, n := prefixValue(code-numLiteralCodes, val)
				val >>= n
				nbits -= n
				if nbits < 32 {
					if bpos+8 <= len(data) {
						val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
						bpos += int((63 - nbits) >> 3)
						nbits |= 56
					} else {
						val, nbits, bpos = refillSlow(data, val, nbits, bpos)
					}
				}
				distSym, n := readSym(g.trees[treeDist], val)
				val >>= n
				nbits -= n
				if nbits < 32 {
					if bpos+8 <= len(data) {
						val |= binary.LittleEndian.Uint64(data[bpos:]) << nbits
						bpos += int((63 - nbits) >> 3)
						nbits |= 56
					} else {
						val, nbits, bpos = refillSlow(data, val, nbits, bpos)
					}
				}
				distCode, n := prefixValue(distSym, val)
				val >>= n
				nbits -= n
				dist := planeCodeToDistance(w, distCode)
				if dist > p || length > total-p {
					if isEOS(len(data), bpos, nbits) {
						err = errTruncated()
					} else {
						err = status.Bitstream("vp8l: invalid backward reference")
					}
					break loop
				}
				copyBlock8(buf, p, dist, length)
				p += length
				col += length
				for col >= w {
					col -= w
					row++
				}
				if col&mask != 0 {
					g = md.group(groups, col, row)
				}
				continue
			} else {
				err = status.Bitstream("vp8l: invalid green symbol")
				break loop
			}
		}
		p++
		col++
		if col >= w {
			col = 0
			row++
		}
	}
	d.br.pos, d.br.val, d.br.nbits = bpos, val, nbits
	st.p, st.col, st.row = p, col, row
	return err
}
