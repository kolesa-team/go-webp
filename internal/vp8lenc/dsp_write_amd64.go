//go:build amd64 && !purego

package vp8lenc

import (
	"slices"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Kernels of the input conversion, the histograms of a parse and the
// entropy-coded pixel writer, written in dsp_write_amd64.s. As in
// dsp_amd64.go, the wrappers check every bound the assembly relies on, and
// the kernels are bit-exact with their Go references.

//go:noescape
func nrgbaRowSSE41(dst *uint32, src *byte, n int, exact bool) (flags int)

//go:noescape
func nrgbaRowAVX2(dst *uint32, src *byte, n int, exact bool) (flags int)

// nrgbaRow is nrgbaRowGo: 4 pixels per step with SSE4.1, 8 with AVX2 (rows
// of at least that many pixels). dst must not overlap src.
func nrgbaRow(dst []uint32, src []byte, exact bool) (hasAlpha, transparent bool) {
	n := len(dst)
	if x86Level >= cpuinfo.SSE41 && n >= 4 {
		src = src[:4*n]
		var f int
		if x86Level >= cpuinfo.AVX2 && n >= 8 {
			f = nrgbaRowAVX2(&dst[0], &src[0], n, exact)
		} else {
			f = nrgbaRowSSE41(&dst[0], &src[0], n, exact)
		}
		return f&1 != 0, f&2 != 0
	}
	return nrgbaRowGo(dst, src, exact)
}

//go:noescape
func subtractGreenSSE41(p *uint32, n int)

//go:noescape
func subtractGreenAVX2(p *uint32, n int)

// subtractGreen is subtractGreenGo: the kernels transform the pixels 4
// (SSE4.1) or 8 (AVX2) at a time, as many as fill whole blocks; the Go
// code transforms the rest. (The transform is not idempotent, so the last
// block cannot overlap pixels already transformed.)
func subtractGreen(p []uint32) {
	n := len(p)
	// (pixChunk pixels per call at most, see hcChunk.)
	m, avx2 := 0, false
	switch {
	case x86Level >= cpuinfo.AVX2 && n >= 8:
		m, avx2 = n&^7, true
	case x86Level >= cpuinfo.SSE41 && n >= 4:
		m = n &^ 3
	}
	for i := 0; i < m; i += pixChunk {
		subtractGreenChunk(&p[i], min(m-i, pixChunk), avx2)
	}
	subtractGreenGo(p[m:])
}

// subtractGreenChunk is the AVX2 kernel (n a multiple of 8) or the SSE4.1
// one (n a multiple of 4) behind a preemption point (see hcChunk).
//
//go:noinline
func subtractGreenChunk(p *uint32, n int, avx2 bool) {
	if avx2 {
		subtractGreenAVX2(p, n)
	} else {
		subtractGreenSSE41(p, n)
	}
}

//go:noescape
func addRefsSSE41(q *uint32, stride, litLen int, refs *pixOrCopy, n, xsize int, raw bool) (done int)

// addRefsChunk is addRefsSSE41 behind a preemption point (see hcChunk).
//
//go:noinline
func addRefsChunk(q *uint32, stride, litLen int, refs *pixOrCopy, n, xsize int, raw bool) (done int) {
	return addRefsSSE41(q, stride, litLen, refs, n, xsize, raw)
}

// addRefsMin is the smallest parse the addRefs kernel takes (below, clearing
// and adding up its 4 partial histograms costs more than it saves; a
// variable for the tests).
var addRefsMin = 2048

// addRefs is addRefsGo (addRefsRawDistGo when raw): the kernel adds the
// references in turn to 4 partial histograms, so that the increments of
// consecutive references do not wait for each other when they hit the
// same bins (the alpha bin of an opaque image, for one: 4.5 -> 1.6 ms for
// the literals of a 1.2 MP photo); they are then added to h. The sums are
// the Go ones (integer counts).
func (s *histoSet) addRefs(h *histo, refs []pixOrCopy, xsize int) {
	if !s.addRefsAsm(h, refs, xsize, false) {
		s.addRefsGo(h, refs, xsize)
	}
}

// addRefsRawDist is addRefsRawDistGo (see addRefs).
func (s *histoSet) addRefsRawDist(h *histo, refs []pixOrCopy) {
	if !s.addRefsAsm(h, refs, 0, true) {
		s.addRefsRawDistGo(h, refs)
	}
}

func (s *histoSet) addRefsAsm(h *histo, refs []pixOrCopy, xsize int, raw bool) bool {
	d := h.d
	L := s.litLen
	n := len(d)
	// The literal bins (below 256 and L+768) are in range when
	// L >= numLiteralCodes; the kernel checks the cache indices against n
	// and the distance codes against numDistanceCodes.
	if x86Level < cpuinfo.SSE41 || len(refs) < addRefsMin || L < numLiteralCodes+numLengthCodes ||
		n != L+3*256+numDistanceCodes || !raw && (xsize < 1 || xsize > maxDim) {
		return false
	}
	if cap(s.quad) < 4*n {
		s.quad = make([]uint32, 4*n)
	}
	q := s.quad[:4*n]
	clear(q)
	// refsChunk references per call at most (see hcChunk).
	k := 0
	for k < len(refs) {
		m := min(len(refs)-k, refsChunk)
		d := addRefsChunk(&q[0], n, L, &refs[k], m, xsize, raw)
		k += d
		if d < m {
			break
		}
	}
	q0, q1, q2, q3 := q[:n], q[n:2*n], q[2*n:3*n], q[3*n:4*n]
	for i := range d {
		d[i] += q0[i] + q1[i] + q2[i] + q3[i]
	}
	if k < len(refs) {
		// refs[k] is out of range: the Go code panics on it.
		if raw {
			s.addRefsRawDistGo(h, refs[k:])
		} else {
			s.addRefsGo(h, refs[k:], xsize)
		}
	}
	return true
}

//go:noescape
func addTileRefsSSE41(data *uint32, stride, litLen int, refs *pixOrCopy, n, xsize, hb, hx, ntiles, x0, y0 int) (done, x, y int)

// addTileRefsChunk is addTileRefsSSE41 behind a preemption point (see
// hcChunk).
//
//go:noinline
func addTileRefsChunk(data *uint32, stride, litLen int, refs *pixOrCopy, n, xsize, hb, hx, ntiles, x0, y0 int) (done, x, y int) {
	return addTileRefsSSE41(data, stride, litLen, refs, n, xsize, hb, hx, ntiles, x0, y0)
}

// addTileRefs is addTileRefsGo: the kernel adds the references to the
// histograms of their tiles as addRefsGo does, with the alpha increments
// of consecutive literals to the same bin counted in a register and added
// at once (the alpha bin of an opaque image would otherwise make every
// literal wait for the previous one).
func (s *histoSet) addTileRefs(refs []pixOrCopy, xsize, histoBits, hx int) {
	L, n, nt := s.litLen, s.stride, len(s.h)
	if x86Level < cpuinfo.SSE41 || len(refs) == 0 || nt == 0 || L < numLiteralCodes+numLengthCodes ||
		n != L+3*256+numDistanceCodes || len(s.data) != nt*n || xsize < 1 || xsize > maxDim ||
		histoBits < 0 || histoBits > 16 || hx < 1 {
		s.addTileRefsGo(refs, xsize, histoBits, hx)
		return
	}
	// The kernel adds to tile i at data[i*n:]: the histograms must be
	// these views (as init makes them).
	for i := range s.h {
		if d := s.h[i].d; len(d) != n || &d[0] != &s.data[i*n] {
			s.addTileRefsGo(refs, xsize, histoBits, hx)
			return
		}
	}
	// refsChunk references per call at most (see hcChunk), each call
	// starting at the position (x, y) where the previous one ended.
	x, y := 0, 0
	for k := 0; k < len(refs); {
		m := min(len(refs)-k, refsChunk)
		d, nx, ny := addTileRefsChunk(&s.data[0], n, L, &refs[k], m, xsize, histoBits, hx, nt, x, y)
		k, x, y = k+d, nx, ny
		if d < m {
			// refs[k] is out of range: the Go code panics on it.
			s.addTileRefsFrom(refs[k:], xsize, histoBits, hx, x, y)
			return
		}
	}
}

// imgWriter is the state of writeImageDataSSE41: the bit writer, the
// position in the image and the current tile and its codes (g, the 5 codes
// of the tile's histogram group), with the image constants.
type imgWriter struct {
	buf          *byte // the bit writer's buffer, from its start
	pos, limit   int   // its length; no reference is written past limit
	acc, nbits   uint64
	x, y         int
	tileX, tileY int
	g            *huffCode
	w, hb        int
	tileMask     int
	hx           int
	symbols      *uint16
	nsym         int
	codes        *huffCode
	ngroups      int
	len0, len4   int // the entries every group has in its codes 0 and 4
	err          int
}

//go:noescape
func writeImageDataSSE41(st *imgWriter, refs *pixOrCopy, n int) (done int)

// writeImageDataChunk is writeImageDataSSE41 behind a preemption point
// (see hcChunk).
//
//go:noinline
func writeImageDataChunk(st *imgWriter, refs *pixOrCopy, n int) (done int) {
	return writeImageDataSSE41(st, refs, n)
}

// writeImageData is writeImageDataGo: the kernel writes the bits of the
// references with the same putBits sequence, but without bound checks and
// with branchless flushes (the flush of every other literal is not
// predictable), stopping when the buffer is nearly full (it then grows here
// as the Go code would let it grow).
func writeImageData(bw *bitWriter, w int, refs []pixOrCopy, histoBits int, symbols []uint16, codes []huffCode) {
	ng := len(codes) / 5
	if x86Level < cpuinfo.SSE41 || len(refs) == 0 || w < 1 || w > maxDim || ng == 0 ||
		symbols != nil && (len(symbols) == 0 || histoBits < 0 || histoBits > 16) {
		writeImageDataGo(bw, w, refs, histoBits, symbols, codes)
		return
	}
	st := imgWriter{acc: bw.acc, nbits: uint64(bw.nbits), w: w, g: &codes[0], codes: &codes[0], ngroups: ng,
		len0: maxDim, len4: maxDim}
	// Every literal symbol must be in range in every group.
	for i := range 5 * ng {
		c := &codes[i]
		l := min(len(c.lengths), len(c.codes))
		switch i % 5 {
		case 0:
			st.len0 = min(st.len0, l)
		case 4:
			st.len4 = min(st.len4, l)
		default:
			if l < 256 {
				writeImageDataGo(bw, w, refs, histoBits, symbols, codes)
				return
			}
		}
	}
	if st.len0 < 256 || st.nbits >= 32 {
		writeImageDataGo(bw, w, refs, histoBits, symbols, codes)
		return
	}
	if symbols != nil {
		st.hb = histoBits
		st.tileMask = ^(1<<uint(histoBits) - 1)
		st.hx = subSampleSize(w, histoBits)
		st.tileX, st.tileY = -1, -1
		st.symbols = &symbols[0]
		st.nsym = len(symbols)
	}
	start, acc0, nbits0 := len(bw.buf), bw.acc, bw.nbits
	all := refs
	for len(refs) > 0 {
		// A reference writes at most 8 bytes, plus 4 bytes the kernel
		// stores at the current position whether it flushes or not.
		if cap(bw.buf)-len(bw.buf) < 16 {
			bw.buf = slices.Grow(bw.buf, 16)
		}
		b := bw.buf[:cap(bw.buf)]
		st.buf, st.pos, st.limit = &b[0], len(bw.buf), len(b)-16
		k := writeImageDataChunk(&st, &refs[0], min(len(refs), refsChunk))
		bw.buf = bw.buf[:st.pos]
		refs = refs[k:]
		if st.err != 0 {
			// A reference or a tile is out of range: the Go code panics on
			// it, after writing the same bits.
			bw.buf, bw.acc, bw.nbits = bw.buf[:start], acc0, nbits0
			writeImageDataGo(bw, w, all, histoBits, symbols, codes)
			return
		}
	}
	bw.acc, bw.nbits = st.acc, uint(st.nbits)
}
