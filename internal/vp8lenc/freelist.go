package vp8lenc

import (
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Reuse of the large working buffers between encodes.
//
// The scratch of an encode (10-40 bytes per pixel) and the ARGB copy of
// large images are kept between encodes in free lists sharing one budget of
// retained bytes. A sync.Pool alone loses them: it drops its objects at
// every other garbage collection, which the encodes themselves trigger, and
// it cannot hold the buffers of large images without leaving hundreds of
// MB to the next collection (it used to keep nothing above 2 MP).
//
//   - put keeps an object only while the budget allows; an object larger
//     than a quarter of the budget is never kept. (Scratches that do not
//     fit still go to a sync.Pool if they served at most 2 MP, as before.)
//   - get takes the most recently returned object that fits.
//   - Objects unused for freeMaxIdle are dropped at the next garbage
//     collection, so the memory kept after a burst of encodes is released
//     even when no encode follows (the runtime collects at least every two
//     minutes).
//
// Objects are taken and returned by the goroutine that uses them.

// freeBudget is the budget of bytes retained by the free lists: 256 MB, or
// 64 MB on 32-bit platforms (a variable for the tests).
var freeBudget int64 = 64 << 20 << (^uintptr(0) >> 63 * 2)

// freeMaxIdle is the time after which an unused object is dropped.
const freeMaxIdle = 15 * time.Second

var (
	freeRetained atomic.Int64 // bytes held by the free lists
	freeArmed    atomic.Bool  // a collection will trim the free lists
	freeEpoch    = time.Now()

	scratchFree freeList[*scratch]
	argbFree    freeList[[]uint32]
)

func freeNow() int64 { return int64(time.Since(freeEpoch)) }

type freeEntry[T any] struct {
	v     T
	bytes int64
	at    int64 // freeNow at the put
}

// freeList is a list of free objects, oldest first.
type freeList[T any] struct {
	mu      sync.Mutex
	entries []freeEntry[T]
}

// get removes and returns the most recently put object for which fit (if
// not nil) reports true.
func (l *freeList[T]) get(fit func(T) bool) (T, bool) {
	l.mu.Lock()
	for i := len(l.entries) - 1; i >= 0; i-- {
		e := l.entries[i]
		if fit != nil && !fit(e.v) {
			continue
		}
		n := copy(l.entries[i:], l.entries[i+1:])
		l.entries[i+n] = freeEntry[T]{}
		l.entries = l.entries[:i+n]
		l.mu.Unlock()
		freeRetained.Add(-e.bytes)
		return e.v, true
	}
	l.mu.Unlock()
	var zero T
	return zero, false
}

// put keeps v, which retains the given number of bytes, if the budget
// allows, and reports whether it did.
func (l *freeList[T]) put(v T, bytes int64) bool {
	limit := freeBudget
	if bytes > limit/4 {
		return false
	}
	for {
		r := freeRetained.Load()
		if r+bytes > limit {
			return false
		}
		if freeRetained.CompareAndSwap(r, r+bytes) {
			break
		}
	}
	l.mu.Lock()
	l.entries = append(l.entries, freeEntry[T]{v: v, bytes: bytes, at: freeNow()})
	l.mu.Unlock()
	if freeArmed.CompareAndSwap(false, true) {
		armFreeTrim()
	}
	return true
}

// trim drops the objects put before cutoff (all of them if cutoff < 0).
func (l *freeList[T]) trim(cutoff int64) {
	var freed int64
	l.mu.Lock()
	k := 0
	for k < len(l.entries) && (cutoff < 0 || l.entries[k].at < cutoff) {
		freed += l.entries[k].bytes
		k++
	}
	if k == len(l.entries) {
		l.entries = nil
	} else if k > 0 {
		n := copy(l.entries, l.entries[k:])
		clear(l.entries[n:])
		l.entries = l.entries[:n]
	}
	l.mu.Unlock()
	freeRetained.Add(-freed)
}

// gcSentinel is an object whose only reference is dropped at once: its
// cleanup runs after the next garbage collection. (It must not be of size
// 0, as all such objects share one address.)
type gcSentinel struct{ _ *byte }

func armFreeTrim() {
	runtime.AddCleanup(new(gcSentinel), func(struct{}) { onFreeGC() }, struct{}{})
}

// onFreeGC runs after each garbage collection while the free lists hold
// objects: it drops those unused for freeMaxIdle, and re-arms itself unless
// the lists are empty (a later put re-arms it).
func onFreeGC() {
	trimFree(max(freeNow()-int64(freeMaxIdle), 0))
	freeArmed.Store(false)
	if freeRetained.Load() > 0 && freeArmed.CompareAndSwap(false, true) {
		armFreeTrim()
	}
}

// trimFree drops the free objects put before cutoff (all if cutoff < 0).
func trimFree(cutoff int64) {
	scratchFree.trim(cutoff)
	argbFree.trim(cutoff)
}

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
