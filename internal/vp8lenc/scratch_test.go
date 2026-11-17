package vp8lenc

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
)

// withBudget sets the retention budget of internal/bufpool to limit with
// empty pools, and returns the function restoring it.
func withBudget(limit int64) func() {
	bufpool.Trim()
	prev := bufpool.SetLimit(limit)
	return func() {
		bufpool.SetLimit(prev)
		bufpool.Trim()
	}
}

// TestPoolBounds checks that the buffers and scratches of large images are
// kept only within the budget shared with internal/bufpool (they used to be
// kept in sync.Pools and retain hundreds of MB after a 12 MP encode), and
// are dropped by the trimming.
func TestPoolBounds(t *testing.T) {
	const budget = 64 << 20
	defer withBudget(budget)()
	n := maxPooledPixels + 1
	b := getU32(n)
	if cap(b) != n {
		t.Errorf("getU32(%d): cap %d, want the exact size", n, cap(b))
	}
	putU32(b)
	if b2 := getU32(n); &b2[0] != &b[0] {
		t.Error("large buffer was not reused")
	}
	if r := bufpool.Retained(); r != 0 {
		t.Errorf("retained %d bytes, want 0", r)
	}
	putU32(b)
	if b2 := getU32(n / 3); cap(b2) == cap(b) {
		t.Error("buffer three times as large as needed was used")
	}
	bufpool.Trim()
	if b2 := getU32(n); &b2[0] == &b[0] {
		t.Error("trimmed buffer was reused")
	}

	// A buffer larger than the budget is never kept.
	putU32(make([]uint32, budget/4+1))
	if r := bufpool.Retained(); r != 0 {
		t.Error("buffer above the budget was kept")
	}

	// The budget bounds the retained bytes.
	for i := 0; i < 12; i++ {
		putU32(make([]uint32, budget/4/8))
	}
	if r := bufpool.Retained(); r > budget || r < budget/2 {
		t.Errorf("retained %d bytes with a budget of %d", r, budget)
	}
	bufpool.Trim()
	if r := bufpool.Retained(); r != 0 {
		t.Errorf("trim left %d bytes", r)
	}

	// A scratch of a large image is kept within the budget only, and
	// whole when it fits.
	s := new(scratch)
	s.noteSize(n)
	s.resid = make([]uint32, budget/4+1)
	putScratch(s)
	for i := 0; i < 4; i++ {
		if getScratch(n) == s {
			t.Fatal("scratch above the budget was kept")
		}
	}
	s.resid = make([]uint32, budget/4*3/4)
	putScratch(s)
	if r := bufpool.Retained(); r != s.bytes() {
		t.Fatalf("retained %d bytes for a scratch of %d", r, s.bytes())
	}
	// Not handed to an image more than 4 times smaller (above 1 MP).
	if getScratch(n/5) == s {
		t.Fatal("scratch of a large image used for a small one")
	}
	if getScratch(n/3) != s {
		t.Fatal("scratch within the budget was not kept")
	}
	small := new(scratch)
	small.noteSize(minScratchFit)
	putScratch(small)
	if getScratch(1) != small {
		t.Fatal("scratch of a small image not used for any size")
	}
}

// TestFreeTrimOnGC checks that the scratches of the free list are dropped
// by garbage collections once idle.
func TestFreeTrimOnGC(t *testing.T) {
	defer withBudget(64 << 20)()
	defer bufpool.SetMaxIdle(bufpool.SetMaxIdle(0))
	s := new(scratch)
	s.resid = make([]uint32, 1<<20)
	putScratch(s)
	if bufpool.Retained() == 0 {
		t.Fatal("scratch not kept")
	}
	deadline := time.Now().Add(10 * time.Second)
	for bufpool.Retained() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("idle scratch not dropped by the collections")
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	if scratchFree.Len() != 0 {
		t.Fatal("scratch still listed")
	}
}

// TestFreeConcurrent hammers the free lists from several goroutines (for
// the race detector) and checks the accounting.
func TestFreeConcurrent(t *testing.T) {
	defer withBudget(64 << 20)()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				s := getScratch(1 << (10 + i%8))
				s.resid = s.residBuf(1 << (10 + i%8))
				putScratch(s)
				b := getU32(maxPooledPixels + 1 + i%7*1000)
				putU32(b)
			}
		}()
	}
	wg.Wait()
	if r := bufpool.Retained(); r < 0 || r > 64<<20 {
		t.Errorf("retained %d bytes, budget %d", r, 64<<20)
	}
	bufpool.Trim()
	if r := bufpool.Retained(); r != 0 {
		t.Errorf("retained %d bytes after Trim (accounting drift)", r)
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
	var sc *scratch
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
		// Without tryNoCache, a single dense parse is kept, two sparse
		// ones (bestRefs).
		if used != 1 && (noise || used != 2) {
			t.Errorf("%s: %d parse buffers in use", name, used)
		}
	}
	sc = new(scratch)
	sc.hc.fill(flat, w, h, 75, false, 0)
	getBackwardRefs(flat, w, h, 75, false, false, 0, false, -1, sc, nil)
	check("flat", false)
	sc = new(scratch)
	sc.hc.fill(noise, w, h, 75, false, 0)
	getBackwardRefs(noise, w, h, 75, false, false, 0, false, -1, sc, nil)
	check("noise", true)
}

// TestRLERefsEstimate checks rleRefsEstimate: exact below
// rleSampleMinRows rows, close to the parse on a photo, and that bestRefs
// then needs no growth of its buffer.
func TestRLERefsEstimate(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, sz := range [][2]int{{1, 1}, {1, 63}, {63, 1}, {5000, 2}, {37, 63}} {
		w, h := sz[0], sz[1]
		argb := make([]uint32, w*h)
		for i := range argb {
			switch {
			case i > 0 && rng.Intn(3) == 0:
				argb[i] = argb[i-1]
			case i >= w && rng.Intn(3) == 0:
				argb[i] = argb[i-w]
			default:
				argb[i] = uint32(rng.Intn(3))
			}
		}
		if got, want := rleRefsEstimate(argb, w, h), len(backwardRefsRle(w, w*h, argb, nil)); got != want {
			t.Errorf("%dx%d: estimate %d, RLE parse %d entries", w, h, got, want)
		}
	}
	img := downscale(loadTestJPEG(t, "source.jpg"), 1)
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb := make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	// The residuals of a photo (what the RLE parse usually wins on).
	sc := new(scratch)
	modes := make([]uint32, subSampleSize(w, 4)*subSampleSize(h, 4))
	predictorImage(w, h, 4, 4, false, 1, false, false, true, argb, modes, sc, 0)
	want := len(backwardRefsRle(w, w*h, argb, nil))
	got := rleRefsEstimate(argb, w, h)
	if d := got - want; d > want/20 || d < -want/20 {
		t.Errorf("photo residuals: estimate %d, RLE parse %d entries", got, want)
	}
	sc.hc.fill(argb, w, h, 75, false, 0)
	c := initRefsCap(argb, w, h)
	r := bestRefs(argb, w, w*h, 75, []int{lz77Standard, lz77RLE}, true, maxColorCacheBits, c, sc.cacheBuf(1<<maxColorCacheBits), sc)
	if cap(r.refs) != min(w*h, got+got/8+4096) {
		t.Errorf("photo residuals: refs buffer of %d entries for an estimate of %d (initRefsCap %d)", cap(r.refs), got, c)
	}
}

// refBestRefs is the two-buffer selection that bestRefs replaced: every
// candidate parse is kept until a better one is found. It also returns the
// winning parse type and whether the cost-based parse was tried and won.
func refBestRefs(argb []uint32, w, n, quality int, types []int, trace bool, cacheBitsMax, refsCap int, sc *scratch) (refsCandidate, string) {
	cacheBuf := make([]uint32, 1<<maxColorCacheBits)
	var bufs [2][]pixOrCopy
	best, tmp := 0, 1
	bestCost, bestType, bestCB := math.Inf(1), 0, 0
	for _, typ := range types {
		r := parseRefs(typ, w, n, argb, sc, make([]pixOrCopy, 0, refsCap))
		bufs[tmp] = r
		cb := bestCacheBits(argb, quality, r, cacheBitsMax, sc)
		applyColorCache(argb, cb, r, cacheBuf)
		if cost := histoCost(r, w, cb, trace, sc); cost < bestCost {
			best, tmp = tmp, best
			bestCost, bestType, bestCB = cost, typ, cb
		}
	}
	kind := fmt.Sprintf("type%d", bestType)
	if (bestType == lz77Standard || bestType == lz77Box) && trace {
		hc := &sc.hc
		if bestType == lz77Box {
			hc = &sc.hcBox
		}
		r := traceBackwards(w, n, argb, bestCB, hc, bufs[best], nil, sc)
		applyColorCache(argb, bestCB, r, cacheBuf)
		kind += "-trace-loses"
		if c := histoCost(r, w, bestCB, true, sc); c < bestCost {
			bufs[best] = r
			kind = fmt.Sprintf("type%d-trace-wins", bestType)
		}
	}
	return refsCandidate{bufs[best], bestCB, bestCost}, kind
}

// TestBestRefsSingleBuffer checks the parse selection of bestRefs (with a
// single buffer for dense parses, two for sparse ones) against the
// two-buffer one, on images where each parse type (and the cost-based
// parse, or not) wins.
func TestBestRefsSingleBuffer(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	type img struct {
		name string
		w, h int
		f    func(x, y int) uint32
	}
	imgs := []img{
		{"noise", 97, 61, func(x, y int) uint32 { return rng.Uint32() }},
		{"runs", 128, 40, func(x, y int) uint32 { return uint32(x/9+y/4) * 0x10203 }},
		{"repeats", 77, 50, func(x, y int) uint32 { return uint32((x*7+y*3)%23)*0x1010101 ^ uint32(rng.Intn(2)) }},
		{"rows", 64, 64, func(x, y int) uint32 { return uint32(x*x) ^ uint32(y%2)<<8 }},
		{"few", 90, 70, func(x, y int) uint32 {
			return []uint32{0xff000000, 0xffff0000, 0xff00ff00, 0xff0000ff}[(x/3+y/2+rng.Intn(2))%4]
		}},
		// Sparse parses (a second buffer).
		{"blocks", 1024, 256, func(x, y int) uint32 { return uint32(x/32*7+y/16*13) * 0x10101 }},
		{"tiles", 800, 200, func(x, y int) uint32 { return uint32((x%97)/25+(y%41)/9) * 0x30507 }},
		{"sparse-noise", 640, 256, func(x, y int) uint32 {
			if rng.Intn(300) == 0 {
				return rng.Uint32()
			}
			return uint32(x/24+y/10) * 0x20406
		}},
	}
	won := map[string]bool{}
	for _, im := range imgs {
		n := im.w * im.h
		argb := make([]uint32, n)
		for i := range argb {
			argb[i] = im.f(i%im.w, i/im.w)
		}
		qs, cbs := []int{0, 20, 25, 50, 75, 100}, []int{0, 4, maxColorCacheBits}
		if n > 1<<16 {
			qs, cbs = []int{20, 50, 100}, []int{0, maxColorCacheBits}
		}
		for _, q := range qs {
			for _, box := range []bool{false, true} {
				for _, cbMax := range cbs {
					sc := new(scratch)
					sc.hc.fill(argb, im.w, im.h, q, q == 0, 0)
					types := []int{lz77Standard, lz77RLE}
					if box {
						types = []int{lz77Box}
						sc.hcBox.fillBox(argb, im.w, n, &sc.hc, sc)
					}
					trace := q >= 25
					ref := new(scratch)
					ref.hc, ref.hcBox = sc.hc, sc.hcBox // (read only)
					refsCap := initRefsCap(argb, im.w, im.h)
					want, kind := refBestRefs(argb, im.w, n, q, types, trace, cbMax, refsCap, ref)
					cacheBuf := sc.cacheBuf(1 << maxColorCacheBits)
					got := bestRefs(argb, im.w, n, q, types, trace, cbMax, refsCap, cacheBuf, sc)
					c := refsCap
					if c < n && !box {
						e := rleRefsEstimate(argb, im.w, im.h)
						c = max(c, min(n, e+e/8+4096))
					}
					if c <= n/sparseRefs {
						kind += "-sparse"
					}

					name := fmt.Sprintf("%s q%d box=%v cb%d (%s)", im.name, q, box, cbMax, kind)
					if got.cacheBits != want.cacheBits || got.bits != want.bits || !slices.Equal(got.refs, want.refs) {
						t.Fatalf("%s: single-buffer parse differs (cache bits %d/%d, %d/%d refs)", name, got.cacheBits, want.cacheBits, len(got.refs), len(want.refs))
					}
					if p := &got.refs[:1][0]; p != &sc.refs[0][:1][0] && (cap(sc.refs[1]) == 0 || p != &sc.refs[1][:1][0]) {
						t.Fatalf("%s: parse not kept in the parse buffers", name)
					}
					if refsCap == n && cap(sc.refs[1]) != 0 {
						t.Fatalf("%s: a second buffer for a dense parse", name)
					}
					won[kind] = true
				}
			}
		}
	}
	for _, k := range []string{"type1", "type2", "type1-trace-wins", "type1-trace-loses", "type4-trace-wins",
		"type1-sparse", "type2-sparse", "type1-trace-wins-sparse", "type1-trace-loses-sparse", "type4-trace-wins-sparse"} {
		if !won[k] {
			t.Errorf("no case where %s (cases: %v)", k, won)
		}
	}
}
