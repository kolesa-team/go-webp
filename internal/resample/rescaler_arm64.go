//go:build arm64 && !purego

package resample

import (
	"encoding/binary"
	"unsafe"
)

// NEON kernels (rescaler_arm64.s). The Go wrappers below check every
// length the assembly relies on; the column-table invariants they depend
// on (spans of at least one pixel that tile the source row, see initCols)
// are established by Init and checked by TestColumnTable, and initArch
// only hands the kernels the destination pixels whose loads stay inside
// the source row.

// shrinkRow4NEON imports destination pixels [0, n) of a 4-channel shrink:
// frow[0:4n] (and acc[0:4n] when acc is not nil) from src, starting at
// source pixel 0. It reads cols[0:2n] and, for each destination pixel, the
// 16 bytes at the start of its span: every span must start at least 4
// pixels before the end of src, and hold at most maxShrink4Span pixels.
//
//go:noescape
func shrinkRow4NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)

// shrinkRow1NEON is shrinkRow4NEON for 1 channel: every span must start at
// least 16 bytes before the end of src and hold at most maxShrink1Span
// pixels.
//
//go:noescape
func shrinkRow1NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)

// shrinkRow1T{16,32,64}NEON import destination pixels [0, 4*groups) of a
// 1-channel shrink with spans of at most 16 pixels, from the records at
// tab (see buildShrink1Records). They read, per group, the 16, 32 or 64
// source bytes starting at its first span.
//
//go:noescape
func shrinkRow1T16NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)

//go:noescape
func shrinkRow1T32NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)

//go:noescape
func shrinkRow1T64NEON(frow, acc *uint32, src *byte, tab *byte, groups int, xSub, fx uint32)

const (
	// maxShrink4Span is the longest span shrinkRow4NEON handles: it sums
	// each channel in 16-bit lanes, two pixels per lane and per 16 bytes.
	maxShrink4Span = 511
	// maxShrink1Span is the longest span shrinkRow1NEON handles: it sums
	// in 16-bit lanes, two bytes per lane and per 16 bytes.
	maxShrink1Span = 2047
	// maxShrink1TSpan is the longest span the TBL kernels handle: four
	// spans fit their largest (64-byte) window.
	maxShrink1TSpan = 16
	// shrink1RecordSize is the size of a TBL kernel record, in bytes.
	shrink1RecordSize = 96
)

// Kernel selection, see initArch.
const (
	kindGo = iota
	kindShrink4
	kindShrink1
	kindShrink1T16
	kindShrink1T32
	kindShrink1T64
	kindExpand4
)

// archState records which assembly kernel a rescaler uses and how far:
// the kernel handles destination pixels [0, asmOuts), whose spans cover
// source pixels [0, asmSrc); the Go kernels finish the row.
type archState struct {
	kind    uint8
	asmOuts int
	asmSrc  int
	records []byte // kindShrink1T*: one record per 4 destination pixels
}

// archTableSize is the room for the TBL kernel records, in uint32 cells.
func archTableSize(dstWidth, numChannels int) int {
	if numChannels != 1 {
		return 0
	}
	return (dstWidth + 3) / 4 * shrink1RecordSize / 4
}

// initArch picks the kernel for r's geometry; tab holds archTableSize
// cells.
func (r *Rescaler) initArch(tab []uint32) {
	a := &r.arch
	cols := r.cols
	if r.xExpand {
		// The left pixels never decrease; the kernel loads each one with
		// its right neighbour.
		if r.numChannels == 4 && int(cols[2*r.dstWidth-2])+2 <= r.srcWidth {
			a.kind, a.asmOuts, a.asmSrc = kindExpand4, r.dstWidth, r.srcWidth
		}
		return
	}
	if r.numChannels != 1 && r.numChannels != 4 {
		return
	}
	nmax := 0
	for x := 0; x < r.dstWidth; x++ {
		nmax = max(nmax, int(cols[2*x]))
	}
	// lead counts the destination pixels whose span starts at least room
	// pixels before the end of the source row.
	lead := func(room int) (x, start int) {
		for x < r.dstWidth && start+room <= r.srcWidth {
			start += int(cols[2*x])
			x++
		}
		return x, start
	}
	switch {
	case r.numChannels == 4:
		if nmax <= maxShrink4Span {
			a.kind = kindShrink4
			a.asmOuts, a.asmSrc = lead(4)
		}
	case nmax <= maxShrink1TSpan:
		r.initShrink1Records(tab)
	case nmax <= maxShrink1Span:
		a.kind = kindShrink1
		a.asmOuts, a.asmSrc = lead(16)
	}
	if a.asmOuts == 0 {
		*a = archState{}
	}
}

// initShrink1Records builds the records of the TBL kernels (see
// rescaler_arm64.s) for the leading groups of 4 destination pixels whose
// window fits in the source row, and picks the smallest window holding
// every group's spans.
func (r *Rescaler) initShrink1Records(tab []uint32) {
	a := &r.arch
	cols := r.cols
	groups := r.dstWidth / 4
	window := 16
	for g := 0; g < groups; g++ {
		sum := int(cols[8*g] + cols[8*g+2] + cols[8*g+4] + cols[8*g+6])
		for window < sum {
			window *= 2
		}
	}
	g, w := 0, 0
	for ; g < groups && w+window <= r.srcWidth; g++ {
		w += int(cols[8*g] + cols[8*g+2] + cols[8*g+4] + cols[8*g+6])
	}
	if g == 0 {
		return
	}
	n := g * shrink1RecordSize
	if 4*len(tab) < n {
		panic("resample: short record table")
	}
	rec := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(tab))), n)
	buildShrink1Records(rec, cols[:8*g])
	a.records = rec
	a.asmOuts, a.asmSrc = 4*g, w
	switch window {
	case 16:
		a.kind = kindShrink1T16
	case 32:
		a.kind = kindShrink1T32
	default:
		a.kind = kindShrink1T64
	}
}

// buildShrink1Records fills one 96-byte record per 4 destination pixels of
// cols (see rescaler_arm64.s): TBL indices of each span inside the window
// that starts at the group's first span, TBL indices widening the last
// pixel of each span to a word, and the four -accum values.
func buildShrink1Records(rec []byte, cols []uint32) {
	for g := 0; g < len(cols)/8; g++ {
		r := rec[g*shrink1RecordSize : (g+1)*shrink1RecordSize]
		off := 0
		for j := 0; j < 4; j++ {
			n, na := int(cols[8*g+2*j]), cols[8*g+2*j+1]
			idx := r[16*j : 16*j+16]
			for k := range idx {
				idx[k] = 0xff
				if k < n {
					idx[k] = byte(off + k)
				}
			}
			r[64+4*j], r[65+4*j], r[66+4*j], r[67+4*j] = byte(off+n-1), 0xff, 0xff, 0xff
			binary.LittleEndian.PutUint32(r[80+4*j:], na)
			off += n
		}
	}
}

func (r *Rescaler) shrinkRow(src []byte, acc []uint32) {
	switch r.arch.kind {
	case kindShrink4:
		r.shrinkRow4(src, acc)
	case kindShrink1, kindShrink1T16, kindShrink1T32, kindShrink1T64:
		r.shrinkRow1(src, acc)
	default:
		r.shrinkRowRef(src, acc)
	}
}

// asmRows returns src, frow and acc cut to the row lengths, and the
// pointer to pass for acc (nil when not accumulating). It panics if the
// kernel geometry does not fit the row.
func (r *Rescaler) asmRows(src []byte, acc []uint32) ([]byte, []uint32, []uint32, *uint32) {
	ch, a := r.numChannels, &r.arch
	src = src[:ch*r.srcWidth]
	frow := r.frow[:ch*r.dstWidth]
	if a.asmOuts < 1 || a.asmOuts > r.dstWidth || a.asmSrc > r.srcWidth {
		panic("resample: bad assembly geometry")
	}
	var ap *uint32
	if acc != nil {
		acc = acc[:ch*r.dstWidth]
		ap = &acc[0]
	}
	return src, frow, acc, ap
}

// shrinkRow4 runs shrinkRow4NEON on the first asmOuts destination pixels
// and shrinkRow4Go on the rest.
func (r *Rescaler) shrinkRow4(src []byte, acc []uint32) {
	n, s := r.arch.asmOuts, r.arch.asmSrc
	xSub, fx := uint32(r.xSub), r.fxScale
	// The last span the kernel handles starts at most at pixel
	// srcWidth-4 (initArch), so its 16-byte load is inside src; all other
	// reads are inside the spans, src[:4*s].
	src, frow, acc, ap := r.asmRows(src, acc)
	cols := r.cols[:2*r.dstWidth]
	shrinkRow4NEON(&frow[0], ap, &src[0], &cols[0], n, xSub, fx)
	if n == r.dstWidth {
		return
	}
	// Resume at destination pixel n: its carry is the rounded part of the
	// last source pixel of pixel n-1 that belongs to it.
	var carry [4]uint32
	na := cols[2*n-1]
	for c := range carry {
		carry[c] = multFix(uint32(src[4*(s-1)+c])*na, fx)
	}
	if acc != nil {
		acc = acc[4*n:]
	}
	shrinkRow4Go(frow[4*n:], acc, src[4*s:], cols[2*n:], xSub, fx, carry)
}

// shrinkRow1 runs the 1-channel kernel on the first asmOuts destination
// pixels and shrinkRow1Go on the rest.
func (r *Rescaler) shrinkRow1(src []byte, acc []uint32) {
	a := &r.arch
	n, s := a.asmOuts, a.asmSrc
	xSub, fx := uint32(r.xSub), r.fxScale
	// initArch only hands the kernels destination pixels whose loads
	// (16 bytes from the span start, or the group's window) end inside
	// src; all other reads are inside the spans, src[:s].
	src, frow, acc, ap := r.asmRows(src, acc)
	cols := r.cols[:2*r.dstWidth]
	if a.kind == kindShrink1 {
		shrinkRow1NEON(&frow[0], ap, &src[0], &cols[0], n, xSub, fx)
	} else {
		if n%4 != 0 || len(a.records) != n/4*shrink1RecordSize {
			panic("resample: bad record table")
		}
		switch a.kind {
		case kindShrink1T16:
			shrinkRow1T16NEON(&frow[0], ap, &src[0], &a.records[0], n/4, xSub, fx)
		case kindShrink1T32:
			shrinkRow1T32NEON(&frow[0], ap, &src[0], &a.records[0], n/4, xSub, fx)
		default:
			shrinkRow1T64NEON(&frow[0], ap, &src[0], &a.records[0], n/4, xSub, fx)
		}
	}
	if n == r.dstWidth {
		return
	}
	carry := multFix(uint32(src[s-1])*cols[2*n-1], fx)
	if acc != nil {
		acc = acc[n:]
	}
	shrinkRow1Go(frow[n:], acc, src[s:], cols[2*n:], xSub, fx, carry)
}

// expandRow4NEON imports a 4-channel horizontal expansion of n
// destination pixels: frow[0:4n] (and acc[0:4n] when acc is not nil). It
// reads cols[0:2n] and the 8 bytes (left and right pixel) at each left
// source pixel, which must be at most the last but one of src.
//
//go:noescape
func expandRow4NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xAdd uint32)

func (r *Rescaler) expandRow(src []byte, acc []uint32) {
	if r.arch.kind != kindExpand4 {
		r.expandRowRef(src, acc)
		return
	}
	// initArch checked that the left pixels end before the last one, so
	// every 8-byte load is inside src.
	src, frow, acc, ap := r.asmRows(src, acc)
	cols := r.cols[:2*r.dstWidth]
	expandRow4NEON(&frow[0], ap, &src[0], &cols[0], r.dstWidth, uint32(r.xAdd))
}

// The export kernels process n values, n > 0 and a multiple of 8; the
// wrappers below run the Go kernels on the rest of the row.

//go:noescape
func exportShrinkNEON(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)

//go:noescape
func exportShrinkZeroNEON(dst *byte, irow *uint32, n int, fxy uint32)

//go:noescape
func exportExpandNEON(dst *byte, frow *uint32, n int, fy uint32)

//go:noescape
func exportExpandMixNEON(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)

func exportExpand(dst []byte, frow []uint32, fy uint32) {
	n := len(dst)
	frow = frow[:n]
	if m := n &^ 7; m > 0 {
		exportExpandNEON(&dst[0], &frow[0], m, fy)
		dst, frow = dst[m:], frow[m:]
	}
	exportExpandGo(dst, frow, fy)
}

func exportExpandMix(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	n := len(dst)
	frow, irow = frow[:n], irow[:n]
	if m := n &^ 7; m > 0 {
		exportExpandMixNEON(&dst[0], &frow[0], &irow[0], m, a, b, fy)
		dst, frow, irow = dst[m:], frow[m:], irow[m:]
	}
	exportExpandMixGo(dst, frow, irow, a, b, fy)
}

func exportShrink(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	n := len(dst)
	irow, frow = irow[:n], frow[:n]
	if m := n &^ 7; m > 0 {
		exportShrinkNEON(&dst[0], &irow[0], &frow[0], m, yscale, fxy)
		dst, irow, frow = dst[m:], irow[m:], frow[m:]
	}
	exportShrinkGo(dst, irow, frow, yscale, fxy)
}

func exportShrinkZero(dst []byte, irow []uint32, fxy uint32) {
	n := len(dst)
	irow = irow[:n]
	if m := n &^ 7; m > 0 {
		exportShrinkZeroNEON(&dst[0], &irow[0], m, fxy)
		dst, irow = dst[m:], irow[m:]
	}
	exportShrinkZeroGo(dst, irow, fxy)
}

// premultiplyNEON is PremultiplyRGBA on n bytes, n > 0 and a multiple of
// 16.
//
//go:noescape
func premultiplyNEON(row *byte, n int)

func premultiplyRow(row []byte) {
	if n := len(row) &^ 15; n > 0 {
		premultiplyNEON(&row[0], n)
		row = row[n:]
	}
	multRGBA(row, &premulScale)
}

func unpremultiplyRow(row []byte) { multRGBA(row, &unpremulScale) }
