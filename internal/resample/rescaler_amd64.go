//go:build amd64 && !purego

package resample

import (
	"encoding/binary"
	"unsafe"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// SSE4.1 and AVX2 kernels (rescaler_amd64.s), bit-exact with the Go ones.
// The level is picked at run time: x86Level (the machine's best level;
// tests lower it to run every level against the Go kernels). A rescaler
// records the level it was initialized with, so that its kernels and its
// record tables always agree.
//
// The Go wrappers below check every length the assembly relies on; the
// column-table invariants they depend on (spans of at least one pixel that
// tile the source row, see initCols) are established by Init and checked
// by TestColumnTable, and initArch only hands the kernels the destination
// pixels whose loads stay inside the source row.

// x86Level is the SIMD level of the kernels.
var x86Level = cpuinfo.Max

// Kernel selection, see initArch.
const (
	kindGo = iota
	kindShrink4
	kindShrink4W8
	kindShrink4W16
	kindShrink1
	kindShrink1P
	kindShrink1S
	kindExpand1
	kindExpand4
)

const (
	// maxShrink4Span is the longest span shrinkRow4SSE41 handles: it sums
	// each channel in 16-bit lanes, two pixels per lane and per 16 bytes.
	maxShrink4Span = 511
	// maxShrink4W8Span and maxShrink4W16Span are the longest spans of the
	// windowed AVX2 kernels: one or two 32-byte windows per span.
	maxShrink4W8Span  = 8
	maxShrink4W16Span = 16
	// maxShrink1PSpan and maxShrink1SSpan are the longest spans of the
	// record kernels: two spans of 8 bytes (P) or one of 16 bytes (S)
	// per 16-byte window.
	maxShrink1PSpan = 8
	maxShrink1SSpan = 16
	// shrink1PRecordSize and shrink1SRecordSize are the record sizes, in
	// bytes, per 8 destination pixels (see buildShrink1P/SRecords).
	shrink1PRecordSize = 128
	shrink1SRecordSize = 192
	// expand1RecordSize is the record size, in bytes, of the 1-channel
	// expansion per 8 destination pixels (see buildExpand1Records).
	expand1RecordSize = 96
)

// archState records which assembly kernel a rescaler uses and how far:
// the kernel handles destination pixels [0, asmOuts), whose spans cover
// source pixels [0, asmSrc); the Go kernels finish the row.
type archState struct {
	kind    uint8
	level   cpuinfo.Level
	asmOuts int
	asmSrc  int
	records []byte // kindShrink1P/S, kindExpand1: one record per 8 destination pixels
}

// archTableSize is the room for the record tables, in uint32 cells.
func archTableSize(dstWidth, numChannels int) int {
	if numChannels != 1 {
		return 0
	}
	return dstWidth / 8 * shrink1SRecordSize / 4
}

// initArch picks the kernel for r's geometry at the current x86Level; tab
// holds archTableSize cells.
func (r *Rescaler) initArch(tab []uint32) {
	lvl := x86Level
	if lvl < cpuinfo.SSE41 {
		return
	}
	a := &r.arch
	a.level = lvl
	cols := r.cols
	if r.xExpand {
		// The left pixels never decrease; the kernel loads each one with
		// its right neighbour.
		switch {
		case r.numChannels == 4 && int(cols[2*r.dstWidth-2])+2 <= r.srcWidth:
			a.kind, a.asmOuts, a.asmSrc = kindExpand4, r.dstWidth, r.srcWidth
		case r.numChannels == 1 && r.srcWidth >= 16 && r.dstWidth >= 8:
			r.initExpand1Records(tab)
		}
		if a.kind == kindGo {
			*a = archState{}
		}
		return
	}
	if r.numChannels != 1 && r.numChannels != 4 {
		*a = archState{}
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
		// The windowed kernels do pairs of destination pixels and read
		// 8 or 16 pixels from each span start.
		switch {
		case lvl >= cpuinfo.AVX2 && nmax <= maxShrink4W8Span:
			a.kind = kindShrink4W8
			a.asmOuts, a.asmSrc = lead(8)
		case lvl >= cpuinfo.AVX2 && nmax <= maxShrink4W16Span:
			a.kind = kindShrink4W16
			a.asmOuts, a.asmSrc = lead(16)
		}
		if a.asmOuts%2 != 0 {
			a.asmOuts--
			a.asmSrc -= int(cols[2*a.asmOuts])
		}
		if a.asmOuts == 0 && nmax <= maxShrink4Span {
			a.kind = kindShrink4
			a.asmOuts, a.asmSrc = lead(4)
		}
	case nmax <= maxShrink1SSpan && r.srcWidth >= 16 && r.dstWidth >= 8:
		r.initShrink1Records(tab, nmax)
	default:
		a.kind = kindShrink1
		a.asmOuts, a.asmSrc = lead(16)
	}
	if a.asmOuts == 0 {
		*a = archState{}
	}
}

// initShrink1Records builds the records of the 1-channel record kernels
// for every whole block of 8 destination pixels.
func (r *Rescaler) initShrink1Records(tab []uint32, nmax int) {
	a := &r.arch
	blocks := r.dstWidth / 8
	size := shrink1SRecordSize
	a.kind = kindShrink1S
	if nmax <= maxShrink1PSpan {
		size = shrink1PRecordSize
		a.kind = kindShrink1P
	}
	n := blocks * size
	if 4*len(tab) < n {
		panic("resample: short record table")
	}
	rec := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(tab))), n)
	cols := r.cols[:16*blocks]
	if a.kind == kindShrink1P {
		buildShrink1PRecords(rec, cols, r.srcWidth)
	} else {
		buildShrink1SRecords(rec, cols, r.srcWidth)
	}
	a.records = rec
	a.asmOuts = 8 * blocks
	for x := 0; x < a.asmOuts; x++ {
		a.asmSrc += int(cols[2*x])
	}
}

// buildShrink1PRecords fills one 128-byte record per 8 destination pixels
// of cols (spans of at most 8 pixels, a source row of srcWidth >= 16
// bytes). The 8 pixels make two groups of 4 (0-3 and 4-7), each group two
// pairs, A (pixels 0, 1) and B (pixels 2, 3), each pair a 16-byte window
// of the source row that holds both spans:
//
//	[  0: 32) PSHUFB indices of pair A of groups 0 and 1
//	[ 32: 64) PSHUFB indices of pair B of groups 0 and 1
//	[ 64: 96) the 8 -accum values (na), as words
//	[ 96:112) window offsets: A0, A1, B0, B1, as words
//	[112:128) zero
//
// The indices put the first span of a pair right-aligned in bytes 0-7 and
// the second one in bytes 8-15 (0x80, i.e. zero, elsewhere), so that
// PSADBW sums each span and its last pixel is the top byte of its half.
func buildShrink1PRecords(rec []byte, cols []uint32, srcWidth int) {
	start := 0
	for b := 0; b < len(cols)/16; b++ {
		r := rec[b*shrink1PRecordSize : (b+1)*shrink1PRecordSize]
		clear(r)
		for j := 0; j < 8; j += 2 {
			grp, pair := j/4, (j%4)/2
			w := min(start, srcWidth-16)
			idx := r[32*pair+16*grp : 32*pair+16*grp+16]
			for h := 0; h < 2; h++ {
				n := int(cols[2*(8*b+j+h)])
				for k := 0; k < 8; k++ {
					idx[8*h+k] = 0x80
					if k >= 8-n {
						idx[8*h+k] = byte(start - w + k - (8 - n))
					}
				}
				binary.LittleEndian.PutUint32(r[64+4*(j+h):], cols[2*(8*b+j+h)+1])
				start += n
			}
			binary.LittleEndian.PutUint32(r[96+8*pair+4*grp:], uint32(w))
		}
	}
}

// buildShrink1SRecords fills one 192-byte record per 8 destination pixels
// of cols (spans of at most 16 pixels, a source row of srcWidth >= 16
// bytes), one 16-byte window of the source row per pixel:
//
//	[  0:128) PSHUFB indices of pixel j at 32*(j%4) + 16*(j/4)
//	[128:160) the 8 -accum values (na), as words
//	[160:192) the 8 window offsets, as words
//
// The indices put the span right-aligned in the window (0x80, i.e. zero,
// before it), so that PSADBW sums it and its last pixel is byte 15.
func buildShrink1SRecords(rec []byte, cols []uint32, srcWidth int) {
	start := 0
	for b := 0; b < len(cols)/16; b++ {
		r := rec[b*shrink1SRecordSize : (b+1)*shrink1SRecordSize]
		for j := 0; j < 8; j++ {
			n := int(cols[2*(8*b+j)])
			w := min(start, srcWidth-16)
			idx := r[32*(j%4)+16*(j/4) : 32*(j%4)+16*(j/4)+16]
			for k := range idx {
				idx[k] = 0x80
				if k >= 16-n {
					idx[k] = byte(start - w + k - (16 - n))
				}
			}
			binary.LittleEndian.PutUint32(r[128+4*j:], cols[2*(8*b+j)+1])
			binary.LittleEndian.PutUint32(r[160+4*j:], uint32(w))
			start += n
		}
	}
}

// initExpand1Records builds the records of the 1-channel expansion kernels
// for every whole block of 8 destination pixels.
func (r *Rescaler) initExpand1Records(tab []uint32) {
	a := &r.arch
	blocks := r.dstWidth / 8
	n := blocks * expand1RecordSize
	if 4*len(tab) < n {
		panic("resample: short record table")
	}
	rec := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(tab))), n)
	buildExpand1Records(rec, r.cols[:16*blocks], r.srcWidth)
	a.kind = kindExpand1
	a.records = rec
	a.asmOuts, a.asmSrc = 8*blocks, r.srcWidth
}

// buildExpand1Records fills one 96-byte record per 8 destination pixels of
// cols (a 1-channel expansion of a source row of srcWidth >= 16 bytes):
//
//	[ 0:32) PSHUFB indices: pixel j's left source pixel, as word j
//	[32:64) the 8 weights (w), as words
//	[64:68) the offset of the 16-byte source window the indices refer to
//	[68:96) zero
//
// The left pixels of 8 destination pixels span at most 8 source pixels
// (they advance by at most one per destination pixel when expanding), so
// they and their right neighbours fit one 16-byte window; the right
// neighbours' indices are the left ones plus one.
func buildExpand1Records(rec []byte, cols []uint32, srcWidth int) {
	for b := 0; b < len(cols)/16; b++ {
		r := rec[b*expand1RecordSize : (b+1)*expand1RecordSize]
		clear(r)
		w := min(int(cols[16*b]), srcWidth-16)
		for j := 0; j < 8; j++ {
			l := int(cols[2*(8*b+j)])
			r[4*j], r[4*j+1], r[4*j+2], r[4*j+3] = byte(l-w), 0x80, 0x80, 0x80
			binary.LittleEndian.PutUint32(r[32+4*j:], cols[2*(8*b+j)+1])
		}
		binary.LittleEndian.PutUint32(r[64:], uint32(w))
	}
}

func (r *Rescaler) shrinkRow(src []byte, acc []uint32) {
	switch r.arch.kind {
	case kindShrink4, kindShrink4W8, kindShrink4W16:
		r.shrinkRow4(src, acc)
	case kindShrink1, kindShrink1P, kindShrink1S:
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

// shrinkRow4 runs the 4-channel kernel on the first asmOuts destination
// pixels and shrinkRow4Go on the rest.
func (r *Rescaler) shrinkRow4(src []byte, acc []uint32) {
	n, s := r.arch.asmOuts, r.arch.asmSrc
	xSub, fx := uint32(r.xSub), r.fxScale
	// The last span the kernel handles starts at most at pixel
	// srcWidth-4 (16-byte loads), srcWidth-8 (W8, 32-byte loads) or
	// srcWidth-16 (W16, two 32-byte loads), see initArch; all other reads
	// are inside the spans, src[:4*s].
	src, frow, acc, ap := r.asmRows(src, acc)
	cols := r.cols[:2*r.dstWidth]
	switch r.arch.kind {
	case kindShrink4W8, kindShrink4W16:
		if n%2 != 0 {
			panic("resample: odd pair count")
		}
		if r.arch.kind == kindShrink4W8 {
			shrink4W8AVX2(&frow[0], ap, &src[0], &cols[0], n/2, xSub, fx)
		} else {
			shrink4W16AVX2(&frow[0], ap, &src[0], &cols[0], n/2, xSub, fx)
		}
	default:
		shrinkRow4SSE41(&frow[0], ap, &src[0], &cols[0], n, xSub, fx)
	}
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
	src, frow, acc, ap := r.asmRows(src, acc)
	cols := r.cols[:2*r.dstWidth]
	switch a.kind {
	case kindShrink1:
		// initArch only hands the kernel destination pixels whose span
		// starts at least 16 bytes before the end of src; all other reads
		// are inside the spans, src[:s].
		shrinkRow1SSE41(&frow[0], ap, &src[0], &cols[0], n, xSub, fx)
	default:
		// The records only hold windows inside a source row of
		// r.srcWidth >= 16 bytes (buildShrink1P/SRecords).
		size := shrink1SRecordSize
		if a.kind == kindShrink1P {
			size = shrink1PRecordSize
		}
		if n%8 != 0 || len(a.records) != n/8*size || len(src) < 16 {
			panic("resample: bad record table")
		}
		blocks := n / 8
		switch {
		case a.kind == kindShrink1P && a.level >= cpuinfo.AVX2:
			shrink1PAVX2(&frow[0], ap, &src[0], &a.records[0], blocks, xSub, fx)
		case a.kind == kindShrink1P:
			shrink1PSSE41(&frow[0], ap, &src[0], &a.records[0], blocks, xSub, fx)
		case a.level >= cpuinfo.AVX2:
			shrink1SAVX2(&frow[0], ap, &src[0], &a.records[0], blocks, xSub, fx)
		default:
			shrink1SSSE41(&frow[0], ap, &src[0], &a.records[0], blocks, xSub, fx)
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

// shrinkRow4SSE41 imports destination pixels [0, n) of a 4-channel
// shrink: frow[0:4n] (and acc[0:4n] when acc is not nil) from src,
// starting at source pixel 0. It reads cols[0:2n] and, for each
// destination pixel, the 16 bytes at the start of its span: every span
// must start at least 4 pixels before the end of src, and hold at most
// maxShrink4Span pixels.
//
//go:noescape
func shrinkRow4SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)

// shrink4W8AVX2 and shrink4W16AVX2 are shrinkRow4SSE41 for pairs of
// destination pixels [0, 2*pairs), reading 32 (W8) or 64 (W16) bytes from
// each span start: every span must start at least 8 (W8) or 16 (W16)
// pixels before the end of src and hold at most maxShrink4W8Span or
// maxShrink4W16Span pixels.
//
//go:noescape
func shrink4W8AVX2(frow, acc *uint32, src *byte, cols *uint32, pairs int, xSub, fx uint32)

//go:noescape
func shrink4W16AVX2(frow, acc *uint32, src *byte, cols *uint32, pairs int, xSub, fx uint32)

// shrinkRow1SSE41 is shrinkRow4SSE41 for 1 channel and spans of any
// length: every span must start at least 16 bytes before the end of src.
//
//go:noescape
func shrinkRow1SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)

// shrink1P{SSE41,AVX2} and shrink1S{SSE41,AVX2} import destination pixels
// [0, 8*blocks) of a 1-channel shrink from the records at rec (see
// buildShrink1PRecords and buildShrink1SRecords). They read the 16-byte
// windows of src the records name.
//
//go:noescape
func shrink1PSSE41(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)

//go:noescape
func shrink1PAVX2(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)

//go:noescape
func shrink1SSSE41(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)

//go:noescape
func shrink1SAVX2(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)

// expandRow4SSE41 imports a 4-channel horizontal expansion of n
// destination pixels: frow[0:4n] (and acc[0:4n] when acc is not nil). It
// reads cols[0:2n] and the 8 bytes (left and right pixel) at each left
// source pixel, which must be at most the last but one of src.
//
//go:noescape
func expandRow4SSE41(frow, acc *uint32, src *byte, cols *uint32, n int, xAdd uint32)

// expand1SSE41 and expand1AVX2 import destination pixels [0, 8*blocks) of
// a 1-channel horizontal expansion from the records at rec (see
// buildExpand1Records). They read the 16-byte windows of src the records
// name.
//
//go:noescape
func expand1SSE41(frow, acc *uint32, src *byte, rec *byte, blocks int, xAdd uint32)

//go:noescape
func expand1AVX2(frow, acc *uint32, src *byte, rec *byte, blocks int, xAdd uint32)

func (r *Rescaler) expandRow(src []byte, acc []uint32) {
	a := &r.arch
	switch a.kind {
	case kindExpand4:
		// initArch checked that the left pixels end before the last one,
		// so every 8-byte load is inside src.
		src, frow, _, ap := r.asmRows(src, acc)
		cols := r.cols[:2*r.dstWidth]
		expandRow4SSE41(&frow[0], ap, &src[0], &cols[0], r.dstWidth, uint32(r.xAdd))
	case kindExpand1:
		// The records only hold windows inside a source row of
		// r.srcWidth >= 16 bytes (buildExpand1Records).
		n := a.asmOuts
		src, frow, acc, ap := r.asmRows(src, acc)
		if n%8 != 0 || len(a.records) != n/8*expand1RecordSize || len(src) < 16 {
			panic("resample: bad record table")
		}
		if a.level >= cpuinfo.AVX2 {
			expand1AVX2(&frow[0], ap, &src[0], &a.records[0], n/8, uint32(r.xAdd))
		} else {
			expand1SSE41(&frow[0], ap, &src[0], &a.records[0], n/8, uint32(r.xAdd))
		}
		if n == r.dstWidth {
			return
		}
		if acc != nil {
			acc = acc[n:]
		}
		expandRow1Go(frow[n:], acc, src, r.cols[2*n:2*r.dstWidth], uint32(r.xAdd))
	default:
		r.expandRowRef(src, acc)
	}
}

// The export kernels process n values, n > 0 and a multiple of 8; the
// wrappers below run the Go kernels on the rest of the row.

//go:noescape
func exportShrinkSSE41(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)

//go:noescape
func exportShrinkAVX2(dst *byte, irow, frow *uint32, n int, yscale, fxy uint32)

//go:noescape
func exportShrinkZeroSSE41(dst *byte, irow *uint32, n int, fxy uint32)

//go:noescape
func exportShrinkZeroAVX2(dst *byte, irow *uint32, n int, fxy uint32)

//go:noescape
func exportExpandSSE41(dst *byte, frow *uint32, n int, fy uint32)

//go:noescape
func exportExpandAVX2(dst *byte, frow *uint32, n int, fy uint32)

//go:noescape
func exportExpandMixSSE41(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)

//go:noescape
func exportExpandMixAVX2(dst *byte, frow, irow *uint32, n int, a, b, fy uint32)

func exportExpand(dst []byte, frow []uint32, fy uint32) {
	n := len(dst)
	frow = frow[:n]
	if m := n &^ 7; m > 0 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			exportExpandAVX2(&dst[0], &frow[0], m, fy)
		} else {
			exportExpandSSE41(&dst[0], &frow[0], m, fy)
		}
		dst, frow = dst[m:], frow[m:]
	}
	exportExpandGo(dst, frow, fy)
}

func exportExpandMix(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	n := len(dst)
	frow, irow = frow[:n], irow[:n]
	if m := n &^ 7; m > 0 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			exportExpandMixAVX2(&dst[0], &frow[0], &irow[0], m, a, b, fy)
		} else {
			exportExpandMixSSE41(&dst[0], &frow[0], &irow[0], m, a, b, fy)
		}
		dst, frow, irow = dst[m:], frow[m:], irow[m:]
	}
	exportExpandMixGo(dst, frow, irow, a, b, fy)
}

func exportShrink(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	n := len(dst)
	irow, frow = irow[:n], frow[:n]
	if m := n &^ 7; m > 0 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			exportShrinkAVX2(&dst[0], &irow[0], &frow[0], m, yscale, fxy)
		} else {
			exportShrinkSSE41(&dst[0], &irow[0], &frow[0], m, yscale, fxy)
		}
		dst, irow, frow = dst[m:], irow[m:], frow[m:]
	}
	exportShrinkGo(dst, irow, frow, yscale, fxy)
}

func exportShrinkZero(dst []byte, irow []uint32, fxy uint32) {
	n := len(dst)
	irow = irow[:n]
	if m := n &^ 7; m > 0 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			exportShrinkZeroAVX2(&dst[0], &irow[0], m, fxy)
		} else {
			exportShrinkZeroSSE41(&dst[0], &irow[0], m, fxy)
		}
		dst, irow = dst[m:], irow[m:]
	}
	exportShrinkZeroGo(dst, irow, fxy)
}

// premultiply{SSE41,AVX2} are PremultiplyRGBA on n bytes, n > 0 and a
// multiple of 16.
//
//go:noescape
func premultiplySSE41(row *byte, n int)

//go:noescape
func premultiplyAVX2(row *byte, n int)

func premultiplyRow(row []byte) {
	if n := len(row) &^ 15; n > 0 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			premultiplyAVX2(&row[0], n)
		} else {
			premultiplySSE41(&row[0], n)
		}
		row = row[n:]
	}
	multRGBA(row, &premulScale)
}

// unpremultiplySSE41 is UnpremultiplyRGBA on n bytes, n > 0 and a
// multiple of 16, with the scales of table (unpremulScale).
//
//go:noescape
func unpremultiplySSE41(row *byte, n int, table *[256]uint32)

func unpremultiplyRow(row []byte) {
	if n := len(row) &^ 15; n > 0 && x86Level >= cpuinfo.SSE41 {
		unpremultiplySSE41(&row[0], n, &unpremulScale)
		row = row[n:]
	}
	multRGBA(row, &unpremulScale)
}
