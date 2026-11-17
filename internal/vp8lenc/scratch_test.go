package vp8lenc

import (
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestPoolBounds checks that buffers sized for large images are kept in the
// free lists only within the budget (they used to be kept in sync.Pools
// and retain hundreds of MB after a 12 MP encode), and are dropped by the
// trimming.
func TestPoolBounds(t *testing.T) {
	defer func(b int64) { freeBudget = b; trimFree(-1) }(freeBudget)
	trimFree(-1)
	freeBudget = 64 << 20
	n := maxPooledPixels + 1
	b := getU32(n)
	if cap(b) != n {
		t.Errorf("getU32(%d): cap %d, want the exact size", n, cap(b))
	}
	putU32(b) // kept: 8 MB is within a quarter of the budget
	if b2 := getU32(n); &b2[0] != &b[0] {
		t.Error("large buffer was not reused")
	}
	if freeRetained.Load() != 0 {
		t.Errorf("retained %d bytes, want 0", freeRetained.Load())
	}
	putU32(b)
	if b2 := getU32(n / 3); cap(b2) == cap(b) {
		t.Error("buffer more than twice as large as needed was used")
	}
	trimFree(-1)
	if b2 := getU32(n); &b2[0] == &b[0] {
		t.Error("trimmed buffer was reused")
	}

	// A buffer larger than a quarter of the budget is never kept.
	big := make([]uint32, freeBudget/16+1)
	putU32(big)
	if freeRetained.Load() != 0 {
		t.Error("buffer above a quarter of the budget was kept")
	}

	// The budget bounds the retained bytes.
	var bufs [][]uint32
	for i := 0; i < 12; i++ {
		bufs = append(bufs, make([]uint32, freeBudget/4/4))
	}
	for _, b := range bufs {
		putU32(b)
	}
	if r := freeRetained.Load(); r > freeBudget || r < freeBudget/2 {
		t.Errorf("retained %d bytes with a budget of %d", r, freeBudget)
	}
	trimFree(freeNow() + 1)
	if freeRetained.Load() != 0 || len(argbFree.entries) != 0 {
		t.Errorf("trim left %d bytes in %d buffers", freeRetained.Load(), len(argbFree.entries))
	}

	// A scratch of a large image is kept within the budget only.
	s := new(scratch)
	s.noteSize(n)
	s.resid = make([]uint32, freeBudget/4/4+1)
	putScratch(s)
	for i := 0; i < 4; i++ {
		if getScratch() == s {
			t.Fatal("scratch above a quarter of the budget was kept")
		}
	}
	s.resid = make([]uint32, n)
	putScratch(s)
	if getScratch() != s {
		t.Fatal("scratch within the budget was not kept")
	}
}

// TestFreeTrimOnGC checks that the free lists are emptied by garbage
// collections once their objects are idle, and that the trimming re-arms
// itself.
func TestFreeTrimOnGC(t *testing.T) {
	defer func(b int64) { freeBudget = b; trimFree(-1) }(freeBudget)
	trimFree(-1)
	freeBudget = 64 << 20
	put := func() {
		s := new(scratch)
		s.resid = make([]uint32, 1<<20)
		// An entry put freeMaxIdle ago is idle.
		scratchFree.put(s, s.bytes())
		scratchFree.mu.Lock()
		scratchFree.entries[len(scratchFree.entries)-1].at -= int64(2 * freeMaxIdle)
		scratchFree.mu.Unlock()
	}
	for round := 0; round < 2; round++ {
		put()
		deadline := time.Now().Add(10 * time.Second)
		for freeRetained.Load() != 0 {
			if time.Now().After(deadline) {
				t.Fatalf("round %d: idle scratch not dropped by the collections", round)
			}
			runtime.GC()
			time.Sleep(time.Millisecond)
		}
	}
	// A recent entry survives the collections.
	s := new(scratch)
	s.resid = make([]uint32, 1<<20)
	scratchFree.put(s, s.bytes())
	runtime.GC()
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	if getScratch() != s {
		t.Error("recently used scratch was dropped")
	}
}

// TestFreeConcurrent hammers the free lists from several goroutines (for
// the race detector) and checks the accounting.
func TestFreeConcurrent(t *testing.T) {
	defer func(b int64) { freeBudget = b; trimFree(-1) }(freeBudget)
	trimFree(-1)
	freeBudget = 64 << 20
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				s := getScratch()
				s.resid = s.residBuf(1 << (10 + i%8))
				putScratch(s)
				b := getU32(maxPooledPixels + 1 + i%7*1000)
				putU32(b)
			}
		}()
	}
	wg.Wait()
	var sum int64
	for _, e := range scratchFree.entries {
		sum += e.bytes
	}
	for _, e := range argbFree.entries {
		sum += e.bytes
	}
	if r := freeRetained.Load(); r != sum || r > freeBudget {
		t.Errorf("retained %d bytes, entries hold %d, budget %d", r, sum, freeBudget)
	}
}

// TestScratchBytes checks the memory accounting of a scratch against a
// reflection walk of its buffers, after encodes that use every stage.
func TestScratchBytes(t *testing.T) {
	defer forceParallel()()
	sc := new(scratch)
	for _, ti := range syntheticImages() {
		for _, cfg := range []Config{{Quality: 75, Method: 4, ThreadLevel: true}, {Quality: 90, Method: 5, NearLossless: 60}} {
			w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
			argb := make([]uint32, w*h)
			for i := range argb {
				p := ti.img.Pix[(i/w)*ti.img.Stride+4*(i%w):]
				argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
			}
			encodeARGB(argb, w, h, &cfg, true, 0, sc, nil, nil)
		}
	}
	seen := map[uintptr]bool{}
	var walk func(v reflect.Value) int64
	walk = func(v reflect.Value) int64 {
		switch v.Kind() {
		case reflect.Pointer:
			if v.IsNil() || seen[v.Pointer()] {
				return 0
			}
			seen[v.Pointer()] = true
			n := walk(v.Elem())
			if v.Elem().Kind() != reflect.Struct {
				n += int64(v.Elem().Type().Size())
			}
			return n
		case reflect.Struct:
			var n int64
			if v.Type() == reflect.TypeFor[scratch]() {
				n += int64(v.Type().Size())
			}
			for i := 0; i < v.NumField(); i++ {
				n += walk(v.Field(i))
			}
			return n
		case reflect.Array:
			var n int64
			if k := v.Type().Elem().Kind(); k == reflect.Slice || k == reflect.Struct {
				for i := 0; i < v.Len(); i++ {
					n += walk(v.Index(i))
				}
			}
			return n
		case reflect.Slice:
			if v.Cap() == 0 || seen[v.Pointer()] {
				return 0
			}
			seen[v.Pointer()] = true
			n := int64(v.Cap()) * int64(v.Type().Elem().Size())
			// Elements holding their own buffers (not views of another
			// buffer of the scratch, as histo.d of a histoSet).
			switch v.Type().Elem() {
			case reflect.TypeFor[predWorker](), reflect.TypeFor[*histo]():
				for i := 0; i < v.Len(); i++ {
					n += walk(v.Index(i))
				}
			}
			return n
		}
		return 0
	}
	want := walk(reflect.ValueOf(sc))
	got := sc.bytes()
	if got > want || got < want-want/50 {
		t.Errorf("scratch.bytes() = %d, reflection walk %d", got, want)
	}
}

// TestRefsSizing checks that the backward reference buffers follow the size
// of the parse: small for flat images (they used to be sized for one token
// per pixel, 96 MB for 6 MP), grown up to at most one entry per pixel for
// noise, and always covering every pixel.
func TestRefsSizing(t *testing.T) {
	cover := func(refs []pixOrCopy) int {
		n := 0
		for _, r := range refs {
			n += r.length()
		}
		return n
	}
	const w, h = 512, 512
	n := w * h
	flat := make([]uint32, n)
	for i := range flat {
		flat[i] = 0xff000000 | uint32(i/w%4)
	}
	noise := make([]uint32, n)
	seed := uint32(1)
	for i := range noise {
		seed = seed*1664525 + 1013904223
		noise[i] = seed
	}
	sc := new(scratch)
	// The parse buffers in use (the slots holding a parse) must be sized
	// to the parse.
	check := func(name string, noise bool) {
		used := 0
		for i := range sc.refs {
			if len(sc.refs[i]) == 0 {
				continue
			}
			used++
			if c := cover(sc.refs[i]); c != n {
				t.Errorf("%s refs[%d] covers %d pixels, want %d", name, i, c, n)
			}
			if !noise && cap(sc.refs[i]) > n/16 {
				t.Errorf("%s refs[%d]: cap %d, want <= %d", name, i, cap(sc.refs[i]), n/16)
			}
			if noise && (len(sc.refs[i]) < n/2 || cap(sc.refs[i]) > n) {
				t.Errorf("%s refs[%d]: len %d cap %d, n %d", name, i, len(sc.refs[i]), cap(sc.refs[i]), n)
			}
		}
		if used < 2 {
			t.Errorf("%s: %d parse buffers in use, want at least 2", name, used)
		}
	}
	sc.hc.fill(flat, w, h, 75, false, 0)
	getBackwardRefs(flat, w, h, 75, false, false, 0, false, -1, sc, nil)
	check("flat", false)
	sc.hc.fill(noise, w, h, 75, false, 0)
	getBackwardRefs(noise, w, h, 75, false, false, 0, false, -1, sc, nil)
	check("noise", true)
}
