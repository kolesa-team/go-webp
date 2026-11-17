package vp8l

import (
	"encoding/binary"
	"errors"
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Palette images decoded as bytes.
//
// When color indexing is the only transform and every Huffman group codes
// the same constant alpha, red and blue (libwebp's encoder writes the
// palette indices as 0xff000000 | index<<8), every level-0 pixel is
// arb | index<<8 and only its index byte varies. decodeIdx8 then decodes the
// index bytes into a byte window (d.ib) instead of a 32-bit one: a quarter
// of the memory (1.6 MB instead of 6.5 MB for a 3000-pixel-wide image), and
// of the bytes moved by LZ77 copies and window slides. The rows are
// expanded straight from the bytes into RGBA by emitIdx8.
//
// The color cache (cache8) holds the index of each slot, flagged with 0x100
// once the slot is set; d.hash8 maps an index to the slot of its pixel. A
// lookup of a slot that was never set yields the pixel 0, which has no index
// form unless arb is 0: valid streams never do that, but when one does,
// widenIdx8 converts the window and the cache to 32 bits and decodePixels
// goes on from the same position, so the output is always the one of the
// 32-bit path.

var (
	// idx8Enabled enables the byte path. A variable so that tests can
	// compare it with the 32-bit path.
	idx8Enabled = true
	// idx8TestWidenAt, when > 0, makes decodeIdx8 switch to 32 bits at the
	// first pixel boundary at or after this pixel index (tests).
	idx8TestWidenAt = 0
	// idx8Widens counts the switches to 32 bits (tests).
	idx8Widens atomic.Int64
)

// errWiden is returned by decodeIdx8 when the 32-bit path must take over.
var errWiden = errors.New("vp8l: switch to 32-bit pixels")

// cacheSet flags a set slot of cache8.
const cacheSet = 0x100

// idx8ARB reports whether the level-0 pixels can be decoded as index bytes
// (see above) and returns their constant alpha, red and blue.
func (d *decoder) idx8ARB() (uint32, bool) {
	if !idx8Enabled || d.out != outRGBA || d.nTransforms != 1 || d.transforms[0].kind != colorIndexingTransform {
		return 0, false
	}
	groups := d.md0.groups
	if len(groups) == 0 {
		return 0, false
	}
	// A trivial code also holds the green of its only literal.
	arb := groups[0].literalARB &^ 0xff00
	for i := range groups {
		g := &groups[i]
		if !g.isTrivialLiteral || g.literalARB&^0xff00 != arb {
			return 0, false
		}
	}
	return arb, true
}

// initIdx8 sets up the byte path for pixels arb | index<<8: the index
// window and the color cache.
func (d *decoder) initIdx8(arb uint32) {
	d.idx8, d.idxARB = true, arb
	d.initBuffer8()
	bits := d.md0.cacheBits
	if bits == 0 {
		return
	}
	if d.cache8 == nil {
		d.cache8 = new([1 << maxCacheBits]uint16)
	}
	shift := uint(32 - bits)
	for i := range d.hash8 {
		d.hash8[i] = uint16((arb | uint32(i)<<8) * colorCacheMult >> shift)
	}
	// The cache starts with the pixel 0 in every slot, index 0 when arb
	// is 0, and not representable otherwise.
	unset := uint16(0)
	if arb == 0 {
		unset = cacheSet
	}
	s := d.cache8[:1<<bits]
	for i := range s {
		s[i] = unset
	}
}

// insertCache8 inserts the index bytes px into the color cache in order.
func insertCache8(cache *[1 << maxCacheBits]uint16, hash *[256]uint16, px []byte) {
	for _, b := range px {
		cache[hash[b]&(1<<maxCacheBits-1)] = cacheSet | uint16(b)
	}
}

// settleCache8 is settleCache for index bytes, for a cache of full (a mask
// of one bit per slot, at most 64 slots). A run of equal bytes changes
// nothing after its last byte, so runs are skipped 8 bytes at a time.
func settleCache8(cache *[1 << maxCacheBits]uint16, hash *[256]uint16, px []byte, full uint64) {
	var seen uint64
	for i := len(px) - 1; i >= 0 && seen != full; i-- {
		b := px[i]
		if h := hash[b] & 63; seen>>h&1 == 0 {
			seen |= 1 << h
			cache[h] = cacheSet | uint16(b)
		}
		if i > 0 && px[i-1] == b {
			rep := uint64(b) * 0x0101010101010101
			for i >= 8 && binary.LittleEndian.Uint64(px[i-8:i]) == rep {
				i -= 8
			}
		}
	}
}

// widenIdx8 switches a level-0 decode from index bytes to 32-bit pixels at
// the current position: the window and the color cache are converted, and
// decodePixels takes over.
func (d *decoder) widenIdx8() error {
	idx8Widens.Add(1)
	if a := d.async; a != nil {
		// The jobs in flight read d.ib.
		for a.inFlight > 0 {
			if err := a.waitOne(); err != nil {
				return err
			}
		}
	}
	d.initBuffer32()
	arb := d.idxARB
	src := d.ib[:d.px.p]
	dst := d.buf[:len(src)]
	for i, b := range src {
		dst[i] = arb | uint32(b)<<8
	}
	if bits := d.md0.cacheBits; bits > 0 {
		if d.colorCache == nil {
			d.colorCache = new([1 << maxCacheBits]uint32)
		}
		for i, v := range d.cache8[:1<<bits] {
			c := uint32(0)
			if v&cacheSet != 0 {
				c = arb | uint32(v&0xff)<<8
			}
			d.colorCache[i] = c
		}
	}
	d.idx8 = false
	return nil
}

// decodeIdx8 is decodePixels (level 0) for index bytes: it decodes the
// w*h image into the byte window buf, with the same end-of-stream behavior
// and the same row batches. It returns errWiden when the 32-bit path must
// take over, with the position in st and the bit reader at the start of the
// pixel to decode next.
func (d *decoder) decodeIdx8(md *metadata, buf []byte, w, h int, sliding bool, window int, st *pixState) error {
	groups := md.groups
	total := w * h
	lastPixel := w * d.needRows
	cache := d.cache8
	hash := &d.hash8
	cacheBits := md.cacheBits
	cacheSize := 0
	eager := false
	settleFull := uint64(0) // one bit per slot
	minRun := max(lazyCacheMinRun, 1)
	if cacheBits > 0 {
		cacheSize = 1 << cacheBits
		eager = cacheBits > min(lazyCacheMaxBits, 6)
		settleFull = ^uint64(0) >> ((64 - 1<<(min(cacheBits, 6))) & 63)
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
	settle := false // a lookup needs the pending pixels settled first
outer:
	for {
		stop := lastPixel - base
		if sliding {
			stop = min(stop, len(buf)-maxCopyLength)
		}
		if a := d.async; a != nil && a.guard > 0 {
			// Do not overwrite the rows the helper still reads.
			stop = min(stop, a.guard-maxCopyLength)
		}
		if idx8TestWidenAt > 0 {
			stop = min(stop, idx8TestWidenAt-base)
		}
	inner:
		for p < stop {
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
				// The green symbol is consumed in each case below, so that
				// errWiden leaves the bit reader before it.
				var code int
				var n uint
				if g.usePacked {
					pc := g.packed[val&(packedSize-1)]
					if pc.bits < specialMarker {
						// A literal: red, blue and alpha take no bits.
						code, n = int(pc.value>>8&0xff), uint(pc.bits)
					} else {
						code, n = int(pc.value), uint(pc.bits-specialMarker)
					}
				} else {
					code, n = readSym(g.trees[treeGreen], val)
				}
				switch {
				case code < numLiteralCodes:
					val >>= n
					nbits -= n
					buf[p] = byte(code)
				case code < lenCodeLimit:
					val >>= n
					nbits -= n
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
					if isEOS(len(data), bpos, nbits) {
						err = errTruncated()
						break outer
					}
					abs := base + p
					if dist > p || length > total-abs {
						err = status.Bitstream("vp8l: invalid backward reference")
						break outer
					}
					copyBlock8(buf, p, dist, length)
					p += length
					col += length
					for col >= w {
						col -= w
						row++
						if row%numCacheRows == 0 {
							if err = d.processIdx8(base, row); err != nil {
								break outer
							}
						}
					}
					if col&mask != 0 {
						g = md.group(groups, col, row)
					}
					if eager {
						insertCache8(cache, hash, buf[lastCached:p])
						lastCached = p
					}
					continue
				case code < cacheLimit:
					if !eager && p-lastCached >= minRun {
						// Settle outside this loop, where the call does not
						// slow down the loop, then decode the symbol again.
						settle = true
						break inner
					}
					insertCache8(cache, hash, buf[lastCached:p])
					lastCached = p
					v := cache[(code-lenCodeLimit)&(1<<maxCacheBits-1)]
					if v&cacheSet == 0 {
						// The pixel 0 of a slot never set.
						err = errWiden
						break outer
					}
					val >>= n
					nbits -= n
					buf[p] = byte(v)
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
				if row%numCacheRows == 0 {
					if err = d.processIdx8(base, row); err != nil {
						break outer
					}
				}
				if eager {
					insertCache8(cache, hash, buf[lastCached:p])
					lastCached = p
				}
				if bpos > len(data) && isEOS(len(data), bpos, nbits) {
					err = errTruncated()
					break outer
				}
			}
		}
		if settle {
			settle = false
			settleCache8(cache, hash, buf[lastCached:p], settleFull)
			lastCached = p
			continue
		}
		if base+p >= lastPixel {
			break
		}
		if idx8TestWidenAt > 0 && base+p >= idx8TestWidenAt {
			err = errWiden
			break
		}
		if a := d.async; a != nil && a.guard > 0 && p < len(buf)-maxCopyLength {
			// Stopped at the guard: wait for the helper, then go on.
			if err = a.waitGuard(); err != nil {
				break
			}
			continue
		}
		// Slide the window (see decodePixels).
		if cacheBits > 0 {
			if !eager && p-lastCached >= minRun {
				settleCache8(cache, hash, buf[lastCached:p], settleFull)
			} else {
				insertCache8(cache, hash, buf[lastCached:p])
			}
		}
		keep := min(base+p-window, d.lastRow*w)
		if a := d.async; a != nil {
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
	return d.processIdx8(base, row)
}

// processIdx8 is processRows for index bytes.
func (d *decoder) processIdx8(base, row int) error {
	row = min(row, d.needRows)
	if d.async != nil {
		return d.sendRows(base, row, false)
	}
	y0 := d.lastRow
	if row <= y0 {
		return nil
	}
	w := d.xsize
	idx := d.ib[y0*w-base : row*w-base]
	d.lastRow = row
	return d.emitIdx8(idx, y0, row)
}

// emitIdx8 is emitRows for index bytes: it expands the (possibly bundled)
// indices of rows [y0, row) into RGBA rows of the sink.
func (d *decoder) emitIdx8(idx []byte, y0, row int) error {
	n := row - y0
	fw := d.width
	b := d.sink.Buffer(y0, n)
	stride := d.sink.Stride()
	if stride < 4*fw || len(b) < (n-1)*stride+4*fw {
		return status.InvalidParam("vp8l: sink buffer too small")
	}
	t := &d.transforms[0]
	pw := d.xsize
	for y := 0; y < n; y++ {
		d.emitPaletteIdx8(t, idx[y*pw:(y+1)*pw], b[y*stride:y*stride+4*fw])
	}
	return d.sink.Commit(y0, n)
}

// emitPaletteIdx8 is emitPalette for one row of index bytes.
func (d *decoder) emitPaletteIdx8(t *transform, src, dst []byte) {
	if t.bits == 0 {
		lookupRGBA(d.paletteRGBA(t), src, dst)
		return
	}
	expandBundledRGBA(d.bundleTable(t), t.bits, src, dst)
}

// lookupRGBA writes the RGBA words pal[src[i]] to dst (as many pixels as
// fit), 8 pixels per step.
func lookupRGBA(pal *[256]uint32, src, dst []byte) {
	n := min(len(src), len(dst)/4)
	src, dst = src[:n], dst[:4*n]
	i := 0
	for ; i+8 <= n; i += 8 {
		v := binary.LittleEndian.Uint64(src[i : i+8])
		o := dst[4*i : 4*i+32]
		binary.LittleEndian.PutUint64(o[0:], uint64(pal[byte(v)])|uint64(pal[byte(v>>8)])<<32)
		binary.LittleEndian.PutUint64(o[8:], uint64(pal[byte(v>>16)])|uint64(pal[byte(v>>24)])<<32)
		binary.LittleEndian.PutUint64(o[16:], uint64(pal[byte(v>>32)])|uint64(pal[byte(v>>40)])<<32)
		binary.LittleEndian.PutUint64(o[24:], uint64(pal[byte(v>>48)])|uint64(pal[byte(v>>56)])<<32)
	}
	for ; i < n; i++ {
		binary.LittleEndian.PutUint32(dst[4*i:], pal[src[i]])
	}
}

// expandBundledRGBA expands one row of bundled palette indices (each byte
// of src holds 1<<bits of them, bits 1 to 3) into the RGBA bytes dst, from
// tab, the pixels of each packed byte two per word (see bundleTable).
func expandBundledRGBA(tab []uint64, bits int, src, dst []byte) {
	n := 4 << bits // bytes per packed byte
	full := min(len(dst)/n, len(src))
	switch bits {
	case 1:
		t := tab[:256]
		d := dst[:8*full]
		for k, b := range src[:full] {
			binary.LittleEndian.PutUint64(d[8*k:], t[b])
		}
	case 2:
		t := tab[:512]
		d := dst[:16*full]
		for k, b := range src[:full] {
			e := t[2*int(b) : 2*int(b)+2]
			o := d[16*k : 16*k+16]
			binary.LittleEndian.PutUint64(o[0:], e[0])
			binary.LittleEndian.PutUint64(o[8:], e[1])
		}
	default:
		t := tab[:1024]
		d := dst[:32*full]
		for k, b := range src[:full] {
			e := t[4*int(b) : 4*int(b)+4]
			o := d[32*k : 32*k+32]
			binary.LittleEndian.PutUint64(o[0:], e[0])
			binary.LittleEndian.PutUint64(o[8:], e[1])
			binary.LittleEndian.PutUint64(o[16:], e[2])
			binary.LittleEndian.PutUint64(o[24:], e[3])
		}
	}
	if x := full * n; x < len(dst) && full < len(src) {
		// The last packed byte of the row holds fewer pixels.
		e := tab[int(src[full])<<bits>>1:]
		for j := 0; x+4 <= len(dst); j++ {
			binary.LittleEndian.PutUint32(dst[x:], uint32(e[j>>1]>>(32*(j&1))))
			x += 4
		}
	}
}
