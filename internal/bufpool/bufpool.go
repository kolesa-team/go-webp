// Package bufpool keeps reusable buffers of any size under one
// process-wide budget of retained bytes.
//
// It replaces "pool only below N pixels" rules: buffers of large images are
// reused too, while the memory the pools hold between calls stays bounded:
//
//   - A Pool keeps free buffers in size classes (four per power of two of
//     the element count). Get returns a buffer of the class that is large
//     enough, or of the next class, or allocates one of the exact size.
//   - All pools of the process share one budget (Limit, 256 MiB by
//     default): Put drops the buffer, leaving it to the garbage collector,
//     when keeping it would exceed the budget or when it alone is larger
//     than an eighth of the budget.
//   - Buffers that stay unused for MaxIdle (15 s by default) are dropped at
//     the next garbage collection, so the memory retained after a burst of
//     work is released even if no further call comes (the runtime forces a
//     collection at least every two minutes).
//
// Only buffers in a pool are accounted: a buffer obtained with Get and
// never returned is simply garbage collected.
package bufpool

import (
	"math/bits"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// DefaultLimit is the default budget of retained bytes.
	DefaultLimit = 256 << 20
	// DefaultMaxIdle is the default time after which an unused buffer is
	// dropped.
	DefaultMaxIdle = 15 * time.Second
)

var (
	limit    atomic.Int64
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
	limit.Store(DefaultLimit)
	maxIdle.Store(int64(DefaultMaxIdle))
}

// SetLimit sets the budget of bytes retained by all pools together and
// returns the previous one. A limit <= 0 disables pooling. When the new
// limit is below the bytes currently retained, all pools are emptied.
func SetLimit(bytes int64) int64 {
	bytes = max(bytes, 0)
	prev := limit.Swap(bytes)
	if retained.Load() > bytes {
		Trim()
	}
	return prev
}

// Limit returns the budget of retained bytes.
func Limit() int64 { return limit.Load() }

// Retained returns the number of bytes held by all pools.
func Retained() int64 { return retained.Load() }

// SetMaxIdle sets how long a buffer may stay unused in a pool before a
// garbage collection drops it, and returns the previous value. d <= 0
// drops every unused buffer at each collection.
func SetMaxIdle(d time.Duration) time.Duration {
	return time.Duration(maxIdle.Swap(int64(max(d, 0))))
}

// Trim empties all pools.
func Trim() {
	trimAll(-1)
}

// now returns a monotonic timestamp in nanoseconds.
func now() int64 { return int64(time.Since(epoch)) }

// reserve accounts size more retained bytes if the budget allows it.
func reserve(size int64) bool {
	lim := limit.Load()
	if size > lim/8 {
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
	// trim drops the buffers put before the timestamp cutoff (all of
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
// its cleanup runs once it has been collected. It holds a pointer so that
// it is not batched by the tiny allocator (whose cleanups may not run).
type gcSentinel struct{ _ *byte }

// armGC arranges for onGC to run after the next garbage collection.
func armGC() {
	runtime.AddCleanup(new(gcSentinel), func(struct{}) { onGC() }, struct{}{})
}

// onGC drops the buffers idle for longer than MaxIdle and re-arms itself.
// It runs on the runtime's cleanup goroutine.
func onGC() {
	trimAll(max(now()-maxIdle.Load(), 0))
	armGC()
}

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
	if c == 0 || !reserve(int64(c)*p.elemSize()) {
		return
	}
	at := now()
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
