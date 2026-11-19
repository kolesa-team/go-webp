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

package vp8lenc

import (
	"reflect"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
)

// Reuse of the large working buffers between encodes.
//
// The scratch of an encode (10-40 bytes per pixel) and the ARGB copy of the
// image are kept between encodes in a list and a pool of internal/bufpool,
// under the process-wide budget of retained bytes shared with the rest of
// the codec (256 MiB by default, an eighth of the Go memory limit when that
// is lower). A sync.Pool alone loses them: it drops its objects at every
// other garbage collection, which the encodes themselves trigger, and it
// cannot hold the buffers of large images without leaving hundreds of MB to
// the next collection (it used to keep nothing above 2 MP).
//
// Keeping the working set of a large encode matters beyond the allocations
// it saves: the garbage collector paces itself on the heap it found live,
// which during a large encode is mostly its working set, so encodes that
// allocate their working set anew, back to back, can reach about twice the
// peak memory of one encode before a collection reclaims the previous ones.
//
//   - A scratch is kept as a whole (with all its buffers) while the budget
//     allows; scratches that do not fit still go to a sync.Pool if they
//     served at most 2 MP, as before.
//   - getScratch takes the most recently returned scratch that was not
//     sized for a much larger image, so that small encodes do not hold the
//     buffers a large one could reuse.
//   - Objects unused for 15 s are dropped at the next garbage collection.
//
// Objects are taken and returned by the goroutine that uses them.

var (
	scratchFree bufpool.List[*scratch]
	u32Pool     bufpool.Pool[uint32]
)

// sliceBytes returns the bytes of the backing array of s.
func sliceBytes[T any](s []T) int64 {
	return int64(cap(s)) * int64(reflect.TypeFor[T]().Size())
}

// bytes returns (approximately: buffers below 1 KB are left out) the
// memory retained by the buffers of s.
func (s *scratch) bytes() int64 {
	if s == nil {
		return 0
	}
	n := sliceBytes(s.hc.ol) + sliceBytes(s.hcBox.ol) + sliceBytes(s.counts) +
		sliceBytes(s.cm.lit) + sliceBytes(s.costMgr.cacheIv) + sliceBytes(s.costMgr.costCache) +
		sliceBytes(s.tmpSet.data) + sliceBytes(s.tmpSet.h) +
		sliceBytes(s.clus.orig.data) + sliceBytes(s.clus.orig.h) + sliceBytes(s.clus.symbols) +
		sliceBytes(s.clus.mapping) + sliceBytes(s.clus.queue.q) + sliceBytes(s.clus.set) +
		sliceBytes(s.clus.owned) + sliceBytes(s.clus.freeOwned) +
		sliceBytes(s.out.data) + sliceBytes(s.out.h) +
		sliceBytes(s.lens) + sliceBytes(s.codes) + sliceBytes(s.hcodes) +
		sliceBytes(s.ccost) + sliceBytes(s.u16buf) + sliceBytes(s.cache) + sliceBytes(s.u32abuf) +
		sliceBytes(s.resid) + sliceBytes(s.curbuf) + sliceBytes(s.rowa) + sliceBytes(s.rowb) +
		sliceBytes(s.idxRow) + sliceBytes(s.md) + sliceBytes(s.predMode) + sliceBytes(s.predHist) +
		sliceBytes(s.predAccH) + sliceBytes(s.predBatch) + sliceBytes(s.predWork) +
		sliceBytes(s.hcChain) + sliceBytes(s.hcMarks) + sliceBytes(s.hcExits) +
		sliceBytes(s.tile.pg) + sliceBytes(s.tile.pr) + sliceBytes(s.tile.pc) +
		sliceBytes(s.tile.tg) + sliceBytes(s.tile.tr) + sliceBytes(s.tile.tb) + sliceBytes(s.tile.tc) +
		sliceBytes(s.bwTmp[0].buf) + sliceBytes(s.bwTmp[1].buf)
	for i := range s.refs {
		n += sliceBytes(s.refs[i])
	}
	for _, h := range s.clus.owned {
		n += sliceBytes(h.d)
	}
	for i := range s.predWork {
		w := &s.predWork[i]
		n += sliceBytes(w.resid) + sliceBytes(w.rowCur) + sliceBytes(w.rowUp)
	}
	if s.costRing != nil {
		n += int64(reflect.TypeFor[[costRing]int64]().Size())
	}
	if s.tile.tabs != nil {
		n += int64(reflect.TypeFor[ccTables]().Size())
	}
	return n + int64(reflect.TypeFor[scratch]().Size()) + s.sub.bytes()
}
