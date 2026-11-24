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

// Package bufpool keeps reusable buffers, and other objects holding large
// buffers, under one process-wide budget of retained bytes.
//
// It replaces "pool only below N pixels" rules: buffers of large images are
// reused too, while the memory the pools hold between calls stays bounded:
//
//   - A Pool keeps free buffers in size classes (four per power of two of
//     the element count). Get returns a buffer of the class that is large
//     enough, or of the next class, or allocates one of the exact size.
//   - A List keeps free objects of any type (for example the working
//     buffers of an encode, held by one structure), each with the number of
//     bytes it retains. Get returns the most recently put object that a fit
//     function accepts.
//   - All pools and lists of the process share one budget (Limit): Put drops
//     the object, leaving it to the garbage collector, when keeping it
//     would exceed the budget. The default budget is 256 MiB (64 MiB on
//     32-bit platforms), or an eighth of the Go memory limit (GOMEMLIMIT,
//     debug.SetMemoryLimit) when that is lower; the memory limit is read
//     again at most once a second, by Put and after garbage collections.
//     A single object may take the whole budget: that is what lets a large
//     encode reuse the working set of the previous one instead of
//     allocating it again before the garbage collector has reclaimed it
//     (the collector paces itself on the heap it found live, so back to
//     back allocations of a large working set can double the peak memory).
//   - Objects that stay unused for MaxIdle (15 s by default) are dropped at
//     the next garbage collection, so the memory retained after a burst of
//     work is released even if no further call comes (the runtime forces a
//     collection at least every two minutes).
//
// Only objects in a pool or list are accounted: one obtained with Get and
// never returned is simply garbage collected.
package bufpool

import (
	"math"
	"math/bits"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// DefaultLimit is the default budget of retained bytes (when no lower
	// Go memory limit is set): 256 MiB, or 64 MiB on 32-bit platforms, where
	// the address space is much smaller.
	DefaultLimit = 64 << 20 << (^uintptr(0) >> 63 * 2)
	// DefaultMaxIdle is the default time after which an unused buffer is
	// dropped.
	DefaultMaxIdle = 15 * time.Second
	// memLimitShare is the share of the Go memory limit the default budget
	// is capped at.
	memLimitShare = 8
	// limitRefresh is the time after which the default budget reads the Go
	// memory limit again.
	limitRefresh = int64(time.Second)
)

var (
	// setting is the limit set by SetLimit, or -1 for the default.
	setting  atomic.Int64
	limit    atomic.Int64 // the budget in effect
	limitAt  atomic.Int64 // now() when the default budget was last computed
	limitMu  sync.Mutex   // serializes the updates of setting and limit
	retained atomic.Int64
	maxIdle  atomic.Int64 // nanoseconds

	epoch = time.Now() // origin of the monotonic timestamps

	registry struct {
		sync.Mutex
		pools []trimmer
	}
	armOnce sync.Once
)

func init() {
	setting.Store(-1)
	limit.Store(defaultLimit())
	maxIdle.Store(int64(DefaultMaxIdle))
}

// defaultLimit returns the default budget: DefaultLimit, or an eighth of the
// Go memory limit when that is lower.
func defaultLimit() int64 {
	l := int64(DefaultLimit)
	if m := debug.SetMemoryLimit(-1); m < math.MaxInt64 {
		l = min(l, max(m/memLimitShare, 0))
	}
	return l
}

// SetLimit sets the budget of bytes retained by all pools and lists
// together and returns the previous setting. A limit of 0 disables pooling;
// a negative one restores the default budget, which follows the Go memory
// limit (SetLimit returns -1 when the default was in effect, so that
// SetLimit(SetLimit(n)) restores it). When the new budget is below the
// bytes currently retained, everything is dropped.
func SetLimit(bytes int64) int64 {
	limitMu.Lock()
	prev := setting.Swap(max(bytes, -1))
	if bytes < 0 {
		bytes = defaultLimit()
		limitAt.Store(now())
	}
	limit.Store(bytes)
	limitMu.Unlock()
	if retained.Load() > bytes {
		Trim()
	}
	return prev
}

// Limit returns the budget of retained bytes in effect.
func Limit() int64 { return limit.Load() }

// refreshLimit returns the budget in effect at the time at (a now()
// timestamp). The default budget follows the Go memory limit, read again
// once limitRefresh has passed since the last time; when it drops below
// the bytes retained, everything is dropped.
func refreshLimit(at int64) int64 {
	if setting.Load() >= 0 || at-limitAt.Load() < limitRefresh {
		return limit.Load()
	}
	limitMu.Lock()
	if setting.Load() >= 0 || at-limitAt.Load() < limitRefresh {
		limitMu.Unlock()
		return limit.Load()
	}
	limitAt.Store(at)
	l := defaultLimit()
	limit.Store(l)
	limitMu.Unlock()
	if retained.Load() > l {
		Trim()
	}
	return l
}

// Retained returns the number of bytes held by all pools and lists.
func Retained() int64 { return retained.Load() }

// SetMaxIdle sets how long an object may stay unused in a pool or list
// before a garbage collection drops it, and returns the previous value.
// d <= 0 drops every unused object at each collection.
func SetMaxIdle(d time.Duration) time.Duration {
	return time.Duration(maxIdle.Swap(int64(max(d, 0))))
}

// Trim empties all pools and lists.
func Trim() {
	trimAll(-1)
}

// now returns a monotonic timestamp in nanoseconds.
func now() int64 { return int64(time.Since(epoch)) }

// reserve accounts size more retained bytes, for an object put at the time
// at, if the budget allows it.
func reserve(size, at int64) bool {
	lim := refreshLimit(at)
	if size > lim {
		return false
	}
	for {
		r := retained.Load()
		if r+size > lim {
			return false
		}
		if retained.CompareAndSwap(r, r+size) {
			return true
		}
	}
}

type trimmer interface {
	// trim drops the objects put before the timestamp cutoff (all of
	// them if cutoff < 0).
	trim(cutoff int64)
}

func register(t trimmer) {
	registry.Lock()
	registry.pools = append(registry.pools, t)
	registry.Unlock()
	armOnce.Do(armGC)
}

func trimAll(cutoff int64) {
	registry.Lock()
	pools := registry.pools
	registry.Unlock()
	for _, p := range pools {
		p.trim(cutoff)
	}
}

// gcSentinel is allocated to detect the end of a garbage collection cycle:
// its finalizer runs once it has been collected. It holds a pointer so that
// it is not batched by the tiny allocator (whose finalizers may not run).
type gcSentinel struct{ _ *byte }

// armGC arranges for onGC to run after the next garbage collection.
//
// It sets a finalizer rather than a runtime.AddCleanup cleanup: the runtime
// queues cleanups in per-P blocks until the sweep ends, and a GOMAXPROCS
// decrease in the middle of a sweep strands the blocks of the destroyed Ps
// until GOMAXPROCS grows back (Go 1.27). A stranded cleanup would end this
// chain, and with it the trimming of idle objects, for good. Finalizers go
// to one global queue.
func armGC() {
	runtime.SetFinalizer(new(gcSentinel), func(*gcSentinel) { onGC() })
}

// onGC follows the Go memory limit, drops the objects idle for longer than
// MaxIdle and re-arms itself. It runs on the runtime's finalizer goroutine.
func onGC() {
	gcTrims.Add(1)
	at := now()
	refreshLimit(at)
	trimAll(max(at-maxIdle.Load(), 0))
	armGC()
}

// gcTrims counts the runs of onGC.
var gcTrims atomic.Int64

// GCTrims returns how many times the trimming after a garbage collection
// has run (for tests).
func GCTrims() int64 { return gcTrims.Load() }

// numClasses covers element counts up to the largest int.
const numClasses = 4 * 64

// classOf returns the size class of a buffer of n elements: n itself below
// 8, then four classes per power of two.
func classOf(n int) int {
	if n < 8 {
		return n
	}
	b := bits.Len(uint(n)) - 1 // n is in [2^b, 2^(b+1))
	return 4*b + int(n>>(b-2))&3
}

type entry[T any] struct {
	buf []T
	at  int64 // time of the Put
}

// Pool is a size-class pool of []T buffers. The zero value is ready to use;
// a Pool must not be copied after first use.
type Pool[T any] struct {
	mu         sync.Mutex
	registered bool
	// classes[c] holds the free buffers of class c (by capacity), in the
	// order they were put back: oldest first.
	classes [numClasses][]entry[T]
}

func (p *Pool[T]) elemSize() int64 { return int64(reflect.TypeFor[T]().Size()) }

// Get returns a buffer of length n. Its contents are unspecified (a reused
// buffer is not cleared). The buffer may be returned with Put when no
// longer used.
func (p *Pool[T]) Get(n int) []T {
	if n <= 0 {
		return nil
	}
	c := classOf(n)
	p.mu.Lock()
	for cc := c; cc <= c+1 && cc < numClasses; cc++ {
		st := p.classes[cc]
		for i := len(st) - 1; i >= 0; i-- {
			b := st[i].buf
			if cap(b) < n {
				continue
			}
			// Remove entry i, keeping the order (oldest first). The
			// backing array of the list is kept for the next Puts (trim
			// releases it).
			copy(st[i:], st[i+1:])
			st[len(st)-1] = entry[T]{}
			p.classes[cc] = st[:len(st)-1]
			p.mu.Unlock()
			retained.Add(-int64(cap(b)) * p.elemSize())
			return b[:n]
		}
	}
	p.mu.Unlock()
	return make([]T, n)
}

// Put returns b to the pool, if the budget allows it. b must not be used
// afterwards.
func (p *Pool[T]) Put(b []T) {
	c := cap(b)
	if c == 0 {
		return
	}
	at := now()
	if !reserve(int64(c)*p.elemSize(), at) {
		return
	}
	p.mu.Lock()
	cl := classOf(c)
	p.classes[cl] = append(p.classes[cl], entry[T]{buf: b[:0], at: at})
	reg := !p.registered
	p.registered = true
	p.mu.Unlock()
	if reg {
		register(p)
	}
}

func (p *Pool[T]) trim(cutoff int64) {
	size := p.elemSize()
	var freed int64
	p.mu.Lock()
	for c, st := range p.classes {
		k := 0
		for k < len(st) && (cutoff < 0 || st[k].at < cutoff) {
			freed += int64(cap(st[k].buf)) * size
			k++
		}
		if k < len(st) {
			if k > 0 {
				n := copy(st, st[k:])
				clear(st[n:])
				p.classes[c] = st[:n]
			}
			continue
		}
		p.classes[c] = nil // empty: release the list too
	}
	p.mu.Unlock()
	retained.Add(-freed)
}

type listEntry[T any] struct {
	v     T
	bytes int64
	at    int64 // time of the Put
}

// List is a list of free objects of type T, each retaining a given number
// of bytes (typically a structure holding several buffers, reused as a
// whole). The zero value is ready to use; a List must not be copied after
// first use.
type List[T any] struct {
	mu         sync.Mutex
	registered bool
	entries    []listEntry[T] // oldest first
}

// Get removes and returns the most recently put object for which fit (if
// not nil) reports true. ok is false when there is none.
func (l *List[T]) Get(fit func(T) bool) (v T, ok bool) {
	l.mu.Lock()
	for i := len(l.entries) - 1; i >= 0; i-- {
		e := l.entries[i]
		if fit != nil && !fit(e.v) {
			continue
		}
		n := copy(l.entries[i:], l.entries[i+1:])
		l.entries[i+n] = listEntry[T]{}
		l.entries = l.entries[:i+n]
		l.mu.Unlock()
		retained.Add(-e.bytes)
		return e.v, true
	}
	l.mu.Unlock()
	return v, false
}

// Put keeps v, which retains the given number of bytes, if the budget
// allows it, and reports whether it did. When it did not, v is left to the
// caller (or the garbage collector).
func (l *List[T]) Put(v T, bytes int64) bool {
	bytes = max(bytes, 0)
	at := now()
	if !reserve(bytes, at) {
		return false
	}
	l.mu.Lock()
	l.entries = append(l.entries, listEntry[T]{v: v, bytes: bytes, at: at})
	reg := !l.registered
	l.registered = true
	l.mu.Unlock()
	if reg {
		register(l)
	}
	return true
}

// Len returns the number of objects in the list.
func (l *List[T]) Len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.entries)
}

func (l *List[T]) trim(cutoff int64) {
	var freed int64
	l.mu.Lock()
	k := 0
	for k < len(l.entries) && (cutoff < 0 || l.entries[k].at < cutoff) {
		freed += l.entries[k].bytes
		k++
	}
	if k == len(l.entries) {
		l.entries = nil // empty: release the list too
	} else if k > 0 {
		n := copy(l.entries, l.entries[k:])
		clear(l.entries[n:])
		l.entries = l.entries[:n]
	}
	l.mu.Unlock()
	retained.Add(-freed)
}
