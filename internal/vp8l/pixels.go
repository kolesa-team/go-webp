package vp8l

// Algorithm adapted from libwebp (BSD-3-Clause): DecodeImageData,
// DecodeAlphaData and ProcessRows from src/dec/vp8l_dec.c.

import (
	"encoding/binary"

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
		// Slide the window: keep the LZ77 window and the rows not yet
		// emitted. The pending cache pixels move, so settle them first.
		if cache != nil && !eager && p-lastCached >= lazyCacheMinRun {
			settleCache(cache, buf[lastCached:p], cacheShift)
		} else if cache != nil {
			insertCache(cache, buf[lastCached:p], cacheShift)
		}
		keep := min(base+p-window, d.lastRow*w)
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
	// A resumed decode (AlphaDecoder) may have more than numCacheRows
	// decoded rows pending: emit them in batches that fit d.rows.
	for row-d.lastRow > numCacheRows {
		if err := d.processBatch(buf, base, d.lastRow+numCacheRows); err != nil {
			return err
		}
	}
	return d.processBatch(buf, base, row)
}

// processBatch is processRows for at most numCacheRows rows.
func (d *decoder) processBatch(buf []uint32, base, row int) error {
	y0 := d.lastRow
	n := row - y0
	if n <= 0 {
		return nil
	}
	w := d.xsize
	fw := d.width
	argb := buf[y0*w-base : row*w-base]
	d.lastRow = row

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
		dst := d.alpha[y0*fw : row*fw]
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

func emitRGBA(src []uint32, dst []byte) {
	for _, c := range src {
		if len(dst) < 4 {
			return
		}
		binary.LittleEndian.PutUint32(dst, toRGBA(c))
		dst = dst[4:]
	}
}

// emitAddGreen fuses the inverse subtract-green transform with emitRGBA.
func emitAddGreen(src []uint32, dst []byte) {
	for _, c := range src {
		if len(dst) < 4 {
			return
		}
		g := c >> 8 & 0xff
		rb := (c&0x00ff00ff + (g<<16 | g)) & 0x00ff00ff
		binary.LittleEndian.PutUint32(dst, toRGBA(c&0xff00ff00|rb))
		dst = dst[4:]
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
	n := 4 << t.bits // bytes per packed byte
	for _, c := range src {
		e := tab[(c>>8&0xff)*uint32(n):]
		if len(dst) < n {
			copy(dst, e[:len(dst)])
			return
		}
		copy(dst[:n], e[:n])
		dst = dst[n:]
	}
}

// paletteRGBA returns the palette of t converted to RGBA words.
func (d *decoder) paletteRGBA(t *transform) *[256]uint32 {
	if !d.palReady {
		for i, c := range t.data {
			d.pal[i] = toRGBA(c)
		}
		d.palReady = true
	}
	return &d.pal
}

// bundleTable returns, for each packed index byte, the RGBA bytes of the
// 1<<t.bits pixels it holds.
func (d *decoder) bundleTable(t *transform) []byte {
	if d.bundleReady {
		return d.bundle
	}
	pal := d.paletteRGBA(t)
	ppb := 1 << t.bits
	bpp := uint(8 >> t.bits)
	mask := 1<<bpp - 1
	n := 256 * ppb * 4
	if cap(d.bundle) < n {
		d.bundle = make([]byte, 256*8*4)
	}
	tab := d.bundle[:n]
	for v := 0; v < 256; v++ {
		e := tab[v*ppb*4 : (v+1)*ppb*4]
		for k := 0; k < ppb; k++ {
			binary.LittleEndian.PutUint32(e[4*k:], pal[v>>(uint(k)*bpp)&mask])
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
	inverseColorIndexAlpha(&d.transforms[0], row-y0, d.ib[y0*w-base:row*w-base], d.alpha[y0*fw:row*fw])
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
