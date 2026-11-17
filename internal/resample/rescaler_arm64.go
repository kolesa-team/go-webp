//go:build arm64 && !purego

package resample

// NEON kernels (rescaler_arm64.s). The Go wrappers below check every
// length the assembly relies on; the column-table invariants they depend
// on (spans of at least one pixel that tile the source row, see initCols)
// are established by Init and checked by TestColumnTable.

// shrinkRow4NEON imports destination pixels [0, n) of a 4-channel shrink:
// frow[0:4n] (and acc[0:4n] when acc is not nil) from src, starting at
// source pixel 0. It reads cols[0:2n] and, for each destination pixel, the
// 16 bytes at the start of its span: every span must start at least 4
// pixels before the end of src, and hold at most maxShrink4Span pixels.
//
//go:noescape
func shrinkRow4NEON(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)

// maxShrink4Span is the longest span shrinkRow4NEON handles: it sums each
// channel in 16-bit lanes, two pixels per lane and per 16 bytes.
const maxShrink4Span = 511

// Kernel selection, see initArch.
const (
	kindGo = iota
	kindShrink4
)

// archState records which assembly kernel a rescaler uses and how far:
// the kernel handles destination pixels [0, asmOuts), whose spans cover
// source pixels [0, asmSrc); the Go kernels finish the row.
type archState struct {
	kind    uint8
	asmOuts int
	asmSrc  int
}

func archTableSize(dstWidth, numChannels int) int { return 0 }

// initArch picks the kernel for r's geometry.
func (r *Rescaler) initArch(tab []uint32) {
	a := &r.arch
	if r.xExpand {
		return
	}
	cols := r.cols
	nmax := 0
	for x := 0; x < r.dstWidth; x++ {
		nmax = max(nmax, int(cols[2*x]))
	}
	switch r.numChannels {
	case 4:
		if nmax > maxShrink4Span {
			return
		}
		// Each span start must leave room for a 16-byte load.
		x, start := 0, 0
		for x < r.dstWidth && start+4 <= r.srcWidth {
			start += int(cols[2*x])
			x++
		}
		if x > 0 {
			a.kind, a.asmOuts, a.asmSrc = kindShrink4, x, start
		}
	}
}

func (r *Rescaler) shrinkRow(src []byte, acc []uint32) {
	switch r.arch.kind {
	case kindShrink4:
		r.shrinkRow4(src, acc)
	default:
		r.shrinkRowRef(src, acc)
	}
}

// shrinkRow4 runs shrinkRow4NEON on the first asmOuts destination pixels
// and shrinkRow4Go on the rest.
func (r *Rescaler) shrinkRow4(src []byte, acc []uint32) {
	n, s := r.arch.asmOuts, r.arch.asmSrc
	xSub, fx := uint32(r.xSub), r.fxScale
	// Lengths the kernel relies on. The last span it handles starts at
	// most at pixel srcWidth-4 (initArch), so its 16-byte load is inside
	// src; all other reads are inside the spans, src[:4*s].
	src = src[:4*r.srcWidth]
	cols := r.cols[:2*r.dstWidth]
	frow := r.frow[:4*r.dstWidth]
	if n < 1 || n > r.dstWidth || 4*s > len(src) {
		panic("resample: bad assembly geometry")
	}
	var ap *uint32
	if acc != nil {
		acc = acc[:4*r.dstWidth]
		ap = &acc[0]
	}
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

func (r *Rescaler) expandRow(src []byte, acc []uint32) { r.expandRowRef(src, acc) }

func exportExpand(dst []byte, frow []uint32, fy uint32) { exportExpandGo(dst, frow, fy) }

func exportExpandMix(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	exportExpandMixGo(dst, frow, irow, a, b, fy)
}

func exportShrink(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	exportShrinkGo(dst, irow, frow, yscale, fxy)
}

func exportShrinkZero(dst []byte, irow []uint32, fxy uint32) { exportShrinkZeroGo(dst, irow, fxy) }

func premultiplyRow(row []byte) { multRGBA(row, &premulScale) }

func unpremultiplyRow(row []byte) { multRGBA(row, &unpremulScale) }
