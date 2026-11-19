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

// Algorithm adapted from libwebp (BSD-3-Clause): inverse transforms
// (src/dsp/lossless.c).

import "encoding/binary"

const (
	predictorTransform     = 0
	crossColorTransform    = 1
	subtractGreenTransform = 2
	colorIndexingTransform = 3
)

// transform is one decoded image transform.
type transform struct {
	kind  int
	bits  int
	xsize int // width of the image the transform applies to
	ysize int
	data  []uint32 // sub-image (predictor, cross color) or palette
}

func subSampleSize(size, bits int) int {
	return (size + 1<<bits - 1) >> bits
}

// addPixels adds two ARGB pixels per channel, modulo 256.
func addPixels(a, b uint32) uint32 {
	ag := (a & 0xff00ff00) + (b & 0xff00ff00)
	rb := (a & 0x00ff00ff) + (b & 0x00ff00ff)
	return ag&0xff00ff00 | rb&0x00ff00ff
}

func average2(a, b uint32) uint32 {
	return ((a^b)&0xfefefefe)>>1 + a&b
}

// Two pixels per 64-bit word (SWAR): the predictors that do not depend on
// the left pixel process the row two pixels at a time.

// load2 returns the pixels s[x] and s[x+1] as one word.
func load2(s []uint32, x int) uint64 {
	p := s[x : x+2 : x+2]
	return uint64(p[0]) | uint64(p[1])<<32
}

// store2 stores the two pixels of v at s[x] and s[x+1].
func store2(s []uint32, x int, v uint64) {
	p := s[x : x+2 : x+2]
	p[0], p[1] = uint32(v), uint32(v>>32)
}

// addPixels2 is addPixels for two pixels per word.
func addPixels2(a, b uint64) uint64 {
	ag := (a & 0xff00ff00ff00ff00) + (b & 0xff00ff00ff00ff00)
	rb := (a & 0x00ff00ff00ff00ff) + (b & 0x00ff00ff00ff00ff)
	return ag&0xff00ff00ff00ff00 | rb&0x00ff00ff00ff00ff
}

// average2x2 is average2 for two pixels per word.
func average2x2(a, b uint64) uint64 {
	return ((a^b)&0xfefefefefefefefe)>>1 + a&b
}

// absDiff returns |a-b| for channel values a, b.
func absDiff(a, b uint32) int32 {
	d := int32(a) - int32(b)
	m := d >> 31
	return (d ^ m) - m
}

// predictLeft is predictor mode 1 (L) over in[x0:x1]: a running sum of the
// residuals. Per-channel addition modulo 256 is associative, so 4 pixels at
// a time are summed among themselves first, off the critical path, and the
// serial dependency on the left pixel costs one addPixels per 4 pixels
// instead of one per pixel. in may alias out (each group of 4 is read before
// it is written).
func predictLeft(in, out []uint32, x0, x1 int) {
	in = in[:x1]
	out = out[:x1]
	l := out[x0-1]
	x := x0
	for ; x+4 <= x1; x += 4 {
		s := in[x : x+4 : x+4]
		a := s[0]
		s2 := addPixels(a, s[1])
		s3 := addPixels(s2, s[2])
		s4 := addPixels(s3, s[3])
		o := out[x : x+4 : x+4]
		o[0] = addPixels(a, l)
		o[1] = addPixels(s2, l)
		o[2] = addPixels(s3, l)
		l = addPixels(s4, l)
		o[3] = l
	}
	for ; x < x1; x++ {
		l = addPixels(in[x], l)
		out[x] = l
	}
}

// clampByte clamps a to [0, 255] without branches.
func clampByte(a int32) int32 { return min(max(a, 0), 255) }

// predictAddSubFull is predictor mode 12 (ClampedAddSubtractFull) over
// in[x0:x1]. The left pixel is kept as four channel values, so that the
// serial dependency per pixel is one add, clamp, add and mask per channel,
// with the channels in parallel; everything from the top row is computed
// off that path. in may alias out.
func predictAddSubFull(in, out, up []uint32, x0, x1 int) {
	in, out, up = in[:x1], out[:x1], up[:x1]
	l := out[x0-1]
	la, lr, lg, lb := int32(l>>24), int32(l>>16&0xff), int32(l>>8&0xff), int32(l&0xff)
	for x := x0; x < x1; x++ {
		t, tl, c := up[x], up[x-1], in[x]
		la = (clampByte(la+int32(t>>24)-int32(tl>>24)) + int32(c>>24)) & 0xff
		lr = (clampByte(lr+int32(t>>16&0xff)-int32(tl>>16&0xff)) + int32(c>>16&0xff)) & 0xff
		lg = (clampByte(lg+int32(t>>8&0xff)-int32(tl>>8&0xff)) + int32(c>>8&0xff)) & 0xff
		lb = (clampByte(lb+int32(t&0xff)-int32(tl&0xff)) + int32(c&0xff)) & 0xff
		out[x] = uint32(la)<<24 | uint32(lr)<<16 | uint32(lg)<<8 | uint32(lb)
	}
}

// predictAddSubHalf is predictor mode 13 (ClampedAddSubtractHalf) over
// in[x0:x1]: per channel clip(avg + (avg - tl) / 2) with avg the floor
// average of the left and top pixels, the division truncating toward zero
// as in C. Channel-parallel like predictAddSubFull. in may alias out.
func predictAddSubHalf(in, out, up []uint32, x0, x1 int) {
	in, out, up = in[:x1], out[:x1], up[:x1]
	l := out[x0-1]
	la, lr, lg, lb := int32(l>>24), int32(l>>16&0xff), int32(l>>8&0xff), int32(l&0xff)
	for x := x0; x < x1; x++ {
		t, tl, c := up[x], up[x-1], in[x]
		a := (la + int32(t>>24)) >> 1
		la = (clampByte(a+(a-int32(tl>>24))/2) + int32(c>>24)) & 0xff
		a = (lr + int32(t>>16&0xff)) >> 1
		lr = (clampByte(a+(a-int32(tl>>16&0xff))/2) + int32(c>>16&0xff)) & 0xff
		a = (lg + int32(t>>8&0xff)) >> 1
		lg = (clampByte(a+(a-int32(tl>>8&0xff))/2) + int32(c>>8&0xff)) & 0xff
		a = (lb + int32(t&0xff)) >> 1
		lb = (clampByte(a+(a-int32(tl&0xff))/2) + int32(c&0xff)) & 0xff
		out[x] = uint32(la)<<24 | uint32(lr)<<16 | uint32(lg)<<8 | uint32(lb)
	}
}

// predictRun adds predictor mode to the residuals in[x0:x1] and stores the
// result in out. up is the previous output row, extended by one pixel (the
// first pixel of out) so that the top-right of the last column is valid.
// in may alias out.
func predictRun(mode uint32, in, out, up []uint32, x0, x1 int) {
	in = in[:x1]
	out = out[:x1]
	up = up[:x1+1]
	switch mode {
	case 1: // L
		predictLeft(in, out, x0, x1)
	case 2: // T
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), load2(up, x)))
		}
		if x < x1 {
			out[x] = addPixels(in[x], up[x])
		}
	case 3: // TR
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), load2(up, x+1)))
		}
		if x < x1 {
			out[x] = addPixels(in[x], up[x+1])
		}
	case 4: // TL
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), load2(up, x-1)))
		}
		if x < x1 {
			out[x] = addPixels(in[x], up[x-1])
		}
	case 5:
		l := out[x0-1]
		for x := x0; x < x1; x++ {
			l = addPixels(in[x], average2(average2(l, up[x+1]), up[x]))
			out[x] = l
		}
	case 6:
		l := out[x0-1]
		for x := x0; x < x1; x++ {
			l = addPixels(in[x], average2(l, up[x-1]))
			out[x] = l
		}
	case 7:
		l := out[x0-1]
		for x := x0; x < x1; x++ {
			l = addPixels(in[x], average2(l, up[x]))
			out[x] = l
		}
	case 8:
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), average2x2(load2(up, x-1), load2(up, x))))
		}
		if x < x1 {
			out[x] = addPixels(in[x], average2(up[x-1], up[x]))
		}
	case 9:
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), average2x2(load2(up, x), load2(up, x+1))))
		}
		if x < x1 {
			out[x] = addPixels(in[x], average2(up[x], up[x+1]))
		}
	case 10:
		l := out[x0-1]
		for x := x0; x < x1; x++ {
			l = addPixels(in[x], average2(average2(l, up[x-1]), average2(up[x], up[x+1])))
			out[x] = l
		}
	case 11:
		// libwebp Select: T if the sum over the channels of |l-tl| is at
		// most that of |t-tl|, else L. Both sums are added to the residual
		// before the choice, which shortens the dependency on l.
		l := out[x0-1]
		for x := x0; x < x1; x++ {
			t, tl := up[x], up[x-1]
			pt := absDiff(t>>24, tl>>24) + absDiff(t>>16&0xff, tl>>16&0xff) +
				absDiff(t>>8&0xff, tl>>8&0xff) + absDiff(t&0xff, tl&0xff)
			pl := absDiff(l>>24, tl>>24) + absDiff(l>>16&0xff, tl>>16&0xff) +
				absDiff(l>>8&0xff, tl>>8&0xff) + absDiff(l&0xff, tl&0xff)
			c := in[x]
			withT, withL := addPixels(c, t), addPixels(c, l)
			l = withL
			if pl <= pt {
				l = withT
			}
			out[x] = l
		}
	case 12:
		predictAddSubFull(in, out, up, x0, x1)
	case 13:
		predictAddSubHalf(in, out, up, x0, x1)
	default: // 0, and 14/15 which libwebp treats as black too
		x := x0
		for ; x+2 <= x1; x += 2 {
			store2(out, x, addPixels2(load2(in, x), 0xff000000ff000000))
		}
		if x < x1 {
			out[x] = addPixels(in[x], 0xff000000)
		}
	}
}

// inversePredictor undoes the predictor transform for rows [y0, y1). in
// holds the residual rows; the output rows start at c[o], and c[o-width:o]
// must hold the previous output row (when y0 > 0).
func inversePredictor(t *transform, y0, y1 int, in, c []uint32, o int) {
	width := t.xsize
	i := 0
	if y0 == 0 {
		out := c[o : o+width]
		inr := in[:width]
		out[0] = addPixels(inr[0], 0xff000000)
		predictLeft(inr, out, 1, width)
		o += width
		i += width
		y0++
	}
	bits := t.bits
	tilesPerRow := subSampleSize(width, bits)
	for y := y0; y < y1; y++ {
		modes := t.data[(y>>bits)*tilesPerRow : (y>>bits+1)*tilesPerRow]
		out := c[o : o+width]
		up := c[o-width : o+1]
		inr := in[i : i+width]
		out[0] = addPixels(inr[0], up[0])
		predictRow(modes, inr, out, up, bits)
		o += width
		i += width
	}
}

// predictRowGo is the reference of predictRow (dsp_*.go): it undoes the
// predictor transform for the pixels 1 to len(out)-1 of a row whose pixel
// 0 is already done, with the mode of each tile of 1<<bits pixels in
// modes. up is the previous output row extended by out[0] (see
// predictRun). in may alias out.
func predictRowGo(modes, in, out, up []uint32, bits int) {
	width := len(out)
	tileW := 1 << bits
	x := 1
	for tx := 0; x < width; tx++ {
		xEnd := min(x&^(tileW-1)+tileW, width)
		predictRun(modes[tx]>>8&0xf, in, out, up, x, xEnd)
		x = xEnd
	}
}

func colorTransformDelta(pred, color int8) int32 {
	return (int32(pred) * int32(color)) >> 5
}

// inverseCrossColor undoes the cross-color transform for rows [y0, y1).
// in may alias out.
func inverseCrossColor(t *transform, y0, y1 int, in, out []uint32) {
	width := t.xsize
	bits := t.bits
	tilesPerRow := subSampleSize(width, bits)
	i := 0
	for y := y0; y < y1; y++ {
		mults := t.data[(y>>bits)*tilesPerRow : (y>>bits+1)*tilesPerRow]
		crossColorRow(mults, in[i:i+width], out[i:i+width], bits)
		i += width
	}
}

// crossColorRowGo is the reference of crossColorRow (dsp_*.go): it undoes
// the cross-color transform on one row src into dst, with the multipliers
// of each tile of 1<<bits pixels in mults. src may alias dst.
func crossColorRowGo(mults, src, dst []uint32, bits int) {
	width := len(src)
	dst = dst[:width]
	tileW := 1 << bits
	for tx, m := range mults {
		g2r := int8(m)
		g2b := int8(m >> 8)
		r2b := int8(m >> 16)
		x0 := tx << bits
		if x0 >= width {
			break
		}
		x1 := min(x0+tileW, width)
		s := src[x0:x1]
		d := dst[x0:x1]
		d = d[:len(s)]
		for k, argb := range s {
			green := int8(argb >> 8)
			r := int32(argb>>16) + colorTransformDelta(g2r, green)
			r &= 0xff
			b := int32(argb) + colorTransformDelta(g2b, green) + colorTransformDelta(r2b, int8(r))
			b &= 0xff
			d[k] = argb&0xff00ff00 | uint32(r)<<16 | uint32(b)
		}
	}
}

// addGreenGo is the reference of addGreen (dsp_*.go): it undoes the
// subtract-green transform. in may alias out.
func addGreenGo(in, out []uint32) {
	out = out[:len(in)]
	for i, argb := range in {
		g := argb >> 8 & 0xff
		rb := (argb&0x00ff00ff + (g<<16 | g)) & 0x00ff00ff
		out[i] = argb&0xff00ff00 | rb
	}
}

// inverseColorIndex maps palette indices (possibly bundled) of rows
// [y0, y1) to colors. in must not overlap the part of out written before it
// is read (see processRows).
func inverseColorIndex(t *transform, y0, y1 int, in, out []uint32) {
	width := t.xsize
	pal := t.data
	if t.bits == 0 {
		n := (y1 - y0) * width
		in = in[:n]
		out = out[:n]
		pal = pal[:256]
		for i, p := range in {
			out[i] = pal[p>>8&0xff]
		}
		return
	}
	bpp := uint(8 >> t.bits)
	ppb := 1 << t.bits
	mask := uint32(1)<<bpp - 1
	pal = pal[:mask+1]
	pw := subSampleSize(width, t.bits)
	i, o := 0, 0
	for y := y0; y < y1; y++ {
		src := in[i : i+pw]
		dst := out[o : o+width]
		x := 0
		for _, p := range src {
			packed := p >> 8 & 0xff
			n := min(ppb, width-x)
			for k := 0; k < n; k++ {
				dst[x] = pal[packed&mask]
				packed >>= bpp
				x++
			}
		}
		i += pw
		o += width
	}
}

// alphaTables holds the lookup tables of the 8-bit alpha path: the green
// channel of each palette color, and, for bundled indices, the output bytes
// of each packed index byte.
type alphaTables struct {
	pal8   [256]byte
	bundle [256]uint64 // byte k: pal8 of the k-th index of the packed byte
}

// init builds the tables of the color-indexing transform t.
func (a *alphaTables) init(t *transform) {
	clear(a.pal8[:])
	for i, c := range t.data {
		a.pal8[i] = byte(c >> 8)
	}
	if t.bits == 0 {
		return
	}
	bpp := uint(8 >> t.bits)
	ppb := 1 << t.bits
	mask := 1<<bpp - 1
	for v := range a.bundle {
		var e uint64
		for k := 0; k < ppb; k++ {
			e |= uint64(a.pal8[v>>(uint(k)*bpp)&mask]) << (8 * k)
		}
		a.bundle[v] = e
	}
}

// inverseColorIndexAlpha is the 8-bit variant of inverseColorIndex used for
// alpha planes: in holds the green (index) bytes of height rows (possibly
// bundled), out receives the green channel of their palette colors. in and
// out must not overlap.
func inverseColorIndexAlpha(t *transform, tab *alphaTables, height int, in, out []byte) {
	width := t.xsize
	if t.bits == 0 {
		n := width * height
		lookupBytes(&tab.pal8, in[:n], out[:n])
		return
	}
	pw := subSampleSize(width, t.bits)
	in = in[:pw*height]
	out = out[:width*height]
	for y := 0; y < height; y++ {
		expandBundledRow(tab, t.bits, in[y*pw:(y+1)*pw], out[y*width:(y+1)*width])
	}
}

// lookupBytesGo is the reference of lookupBytes (dsp_*.go): it sets
// out[i] = pal[in[i]] (len(out) >= len(in)), 8 bytes per step.
func lookupBytesGo(pal *[256]byte, in, out []byte) {
	out = out[:len(in)]
	i := 0
	for ; i+8 <= len(in); i += 8 {
		v := binary.LittleEndian.Uint64(in[i:])
		r := uint64(pal[byte(v)]) | uint64(pal[byte(v>>8)])<<8 |
			uint64(pal[byte(v>>16)])<<16 | uint64(pal[byte(v>>24)])<<24 |
			uint64(pal[byte(v>>32)])<<32 | uint64(pal[byte(v>>40)])<<40 |
			uint64(pal[byte(v>>48)])<<48 | uint64(pal[v>>56])<<56
		binary.LittleEndian.PutUint64(out[i:], r)
	}
	for ; i < len(in); i++ {
		out[i] = pal[in[i]]
	}
}

// expandBundledRowGo is the reference of expandBundledRow (dsp_*.go): it
// expands one row of bundled indices (each byte of src holds 1<<bits of
// them, bits 1 to 3) into the output bytes dst, with one 2-, 4- or 8-byte
// store per packed byte from a.bundle.
func expandBundledRowGo(a *alphaTables, bits int, src, dst []byte) {
	tab := &a.bundle
	ppb := 1 << bits
	full := min(len(dst)>>bits, len(src)) // packed bytes expanded whole
	switch bits {
	case 1:
		d := dst[:2*full]
		for k, b := range src[:full] {
			binary.LittleEndian.PutUint16(d[2*k:], uint16(tab[b]))
		}
	case 2:
		d := dst[:4*full]
		for k, b := range src[:full] {
			binary.LittleEndian.PutUint32(d[4*k:], uint32(tab[b]))
		}
	default:
		d := dst[:8*full]
		for k, b := range src[:full] {
			binary.LittleEndian.PutUint64(d[8*k:], tab[b])
		}
	}
	if x := full * ppb; x < len(dst) && full < len(src) {
		// The last packed byte of the row holds fewer pixels.
		e := tab[src[full]]
		for ; x < len(dst); x++ {
			dst[x] = byte(e)
			e >>= 8
		}
	}
}
