package vp8lenc

import "sync"

// scratch holds all reusable buffers of one encoding goroutine.
type scratch struct {
	hc      hashChain
	hcBox   hashChain // matches of the box LZ77 parse
	counts  []uint16  // pixel repetition counts (box LZ77)
	refs    [4][]pixOrCopy
	cm      costModel
	costMgr costManager
	tmpSet  histoSet
	clus    clusterer
	out     histoSet
	huff    huffScratch
	lens    []uint8
	codes   []uint16
	hcodes  []huffCode

	costRing *[costRing]int64 // trace backwards costs
	ccost    []int64
	u16buf   []uint16
	cache    []uint32
	u32abuf  []uint32
	resid    []uint32
	curbuf   []uint32 // working copy of the image (encodeStream)
	rowa     []uint32
	rowb     []uint32
	idxRow   []uint8
	md       []uint8
	predMode []uint32 // mode images of all the samplings (predSearch)
	predHist []histo4 // per sampling and mode residual histograms
	predAccH []histo4 // per sampling accumulated histograms
	tile     tilePixels

	bwTmp [2]bitWriter

	sub *scratch // for sub-images encoded while the main buffers are in use

	// pixels is the largest image (in pixels) this scratch has served; its
	// buffers are sized accordingly.
	pixels int
}

// maxPooledPixels bounds the image size whose scratch buffers are kept in
// the pool: the buffers of a large encode (hundreds of MB for 12 MP) are
// left to the garbage collector instead of being retained.
const maxPooledPixels = 2 << 20

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

func getScratch() *scratch { return scratchPool.Get().(*scratch) }

// putScratch returns s to the pool, unless it served an image larger than
// maxPooledPixels: dropping it bounds the memory retained by the pool.
func putScratch(s *scratch) {
	if s.pixels > maxPooledPixels {
		return
	}
	scratchPool.Put(s)
}

// noteSize records that s is used for an image of n pixels.
func (s *scratch) noteSize(n int) {
	s.pixels = max(s.pixels, n)
}

func (s *scratch) child() *scratch {
	if s.sub == nil {
		s.sub = new(scratch)
	}
	return s.sub
}

func (s *scratch) u16(n int) []uint16 {
	if cap(s.u16buf) < n {
		s.u16buf = make([]uint16, n)
	}
	return s.u16buf[:n]
}

func (s *scratch) u32a(n int) []uint32 {
	if cap(s.u32abuf) < n {
		s.u32abuf = make([]uint32, n)
	}
	return s.u32abuf[:n]
}

// cacheBuf returns a zeroed buffer of n entries.
func (s *scratch) cacheBuf(n int) []uint32 {
	if cap(s.cache) < n {
		s.cache = make([]uint32, n)
	}
	c := s.cache[:n]
	for i := range c {
		c[i] = 0
	}
	return c
}

// curBuf returns the working image buffer of n entries (contents
// unspecified).
func (s *scratch) curBuf(n int) []uint32 {
	if cap(s.curbuf) < n {
		s.curbuf = make([]uint32, n)
	}
	return s.curbuf[:n]
}

func (s *scratch) residBuf(n int) []uint32 {
	if cap(s.resid) < n {
		s.resid = make([]uint32, n)
	}
	return s.resid[:n]
}

func (s *scratch) rowA(n int) []uint32 {
	if cap(s.rowa) < n {
		s.rowa = make([]uint32, n)
	}
	return s.rowa[:n]
}

func (s *scratch) rowB(n int) []uint32 {
	if cap(s.rowb) < n {
		s.rowb = make([]uint32, n)
	}
	return s.rowb[:n]
}

func (s *scratch) idx(n int) []uint8 {
	if cap(s.idxRow) < n {
		s.idxRow = make([]uint8, n)
	}
	return s.idxRow[:n]
}

// tmpHisto returns a cleared single histogram with the given cache bits.
func (s *scratch) tmpHisto(cacheBits int) *histo {
	s.tmpSet.init(1, cacheBits)
	return &s.tmpSet.h[0]
}

// refsBuf returns refs buffer i with capacity for at least n entries.
func (s *scratch) refsBuf(i, n int) []pixOrCopy {
	if cap(s.refs[i]) < n {
		s.refs[i] = make([]pixOrCopy, 0, n)
	}
	return s.refs[i][:0]
}

func (s *scratch) maxDiffs(n int) []uint8 {
	if cap(s.md) < n {
		s.md = make([]uint8, n)
	}
	return s.md[:n]
}

func (s *scratch) predModeBuf(n int) []uint32 {
	if cap(s.predMode) < n {
		s.predMode = make([]uint32, n)
	}
	return s.predMode[:n]
}

// predHistos returns n cleared histograms.
func (s *scratch) predHistos(n int) []histo4 {
	if cap(s.predHist) < n {
		s.predHist = make([]histo4, n)
	}
	h := s.predHist[:n]
	clear(h)
	return h
}

// predAcc returns n cleared histograms.
func (s *scratch) predAcc(n int) []histo4 {
	if cap(s.predAccH) < n {
		s.predAccH = make([]histo4, n)
	}
	h := s.predAccH[:n]
	clear(h)
	return h
}

func (s *scratch) cacheCosts(n int) []int64 {
	if cap(s.ccost) < n {
		s.ccost = make([]int64, n)
	}
	return s.ccost[:n]
}
