package bufpool

import (
	"math/bits"
	"math/rand"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"
)

// fresh resets the global state for a test and restores it afterwards.
func fresh(t *testing.T, lim int64, idle time.Duration) {
	t.Helper()
	Trim()
	prevLimit := SetLimit(lim)
	prevIdle := SetMaxIdle(idle)
	t.Cleanup(func() {
		Trim()
		SetLimit(prevLimit)
		SetMaxIdle(prevIdle)
	})
}

func TestClassOf(t *testing.T) {
	prev := -1
	for n := 0; n < 1<<16; n++ {
		c := classOf(n)
		if c < prev || c >= numClasses {
			t.Fatalf("classOf(%d) = %d after %d", n, c, prev)
		}
		prev = c
	}
	// four classes per power of two
	for b := 3; b < bits.UintSize-2; b++ {
		lo := 1 << b
		for k := 0; k < 4; k++ {
			n := lo + k*(lo/4)
			if c := classOf(n); c != 4*b+k {
				t.Fatalf("classOf(%d) = %d, want %d", n, c, 4*b+k)
			}
			if c := classOf(n + lo/4 - 1); c != 4*b+k {
				t.Fatalf("classOf(%d) = %d, want %d", n+lo/4-1, c, 4*b+k)
			}
		}
	}
	if c := classOf(int(^uint(0) >> 1)); c >= numClasses {
		t.Fatalf("classOf(MaxInt) = %d", c)
	}
}

func TestGetPutReuse(t *testing.T) {
	fresh(t, DefaultLimit, time.Hour)
	var p Pool[uint32]
	if b := p.Get(0); b != nil {
		t.Fatal("Get(0) != nil")
	}
	p.Put(nil) // no-op

	b := p.Get(1000)
	if len(b) != 1000 || cap(b) != 1000 {
		t.Fatalf("fresh buffer: len %d cap %d, want exact size", len(b), cap(b))
	}
	b[0] = 42
	p.Put(b)
	if r := Retained(); r != 4000 {
		t.Fatalf("Retained = %d, want 4000", r)
	}
	// Same class, too large for the pooled buffer: a fresh one.
	if c := p.Get(1010); &c[0] == &b[0] {
		t.Fatal("buffer too small was returned")
	}
	// Same class, smaller: reused, contents kept.
	c := p.Get(990)
	if &c[0] != &b[0] || len(c) != 990 || c[0] != 42 {
		t.Fatal("pooled buffer not reused")
	}
	if r := Retained(); r != 0 {
		t.Fatalf("Retained after Get = %d", r)
	}
	// A buffer of the next class serves the request.
	big := make([]uint32, 1100)
	p.Put(big)
	if d := p.Get(1010); &d[0] != &big[0] {
		t.Fatal("next class not used")
	}
	// Two classes up is too wasteful.
	huge := make([]uint32, 1500)
	p.Put(huge)
	if d := p.Get(1010); &d[0] == &huge[0] {
		t.Fatal("buffer two classes up was used")
	}
	// LIFO within a class.
	x, y := make([]uint32, 2000), make([]uint32, 2000)
	p.Put(x)
	p.Put(y)
	if d := p.Get(2000); &d[0] != &y[0] {
		t.Fatal("not LIFO")
	}
}

func TestBudget(t *testing.T) {
	fresh(t, 1<<20, time.Hour) // 1 MiB, at most 128 KiB per buffer
	var p Pool[byte]
	p.Put(make([]byte, 128<<10+1))
	if Retained() != 0 {
		t.Fatal("buffer above limit/8 retained")
	}
	for i := 0; i < 10; i++ {
		p.Put(make([]byte, 100<<10))
	}
	if r := Retained(); r != 1000<<10 {
		t.Fatalf("Retained = %d", r)
	}
	p.Put(make([]byte, 100<<10)) // over budget: dropped
	if r := Retained(); r != 1000<<10 {
		t.Fatalf("Retained over budget = %d", r)
	}
	var q Pool[uint64] // the budget is shared by all pools
	q.Put(make([]uint64, 3<<10))
	if r := Retained(); r != 1000<<10+24<<10 {
		t.Fatalf("Retained with two pools = %d", r)
	}
	// Lowering the limit empties the pools.
	SetLimit(1 << 10)
	if r := Retained(); r != 0 {
		t.Fatalf("Retained after SetLimit = %d", r)
	}
	SetLimit(0) // disabled
	p.Put(make([]byte, 10))
	if r := Retained(); r != 0 {
		t.Fatalf("Retained with pooling disabled = %d", r)
	}
}

func TestTrimIdle(t *testing.T) {
	fresh(t, DefaultLimit, time.Hour)
	var p Pool[byte]
	old := make([]byte, 100)
	p.Put(old)
	// Sleep on both sides of the cutoff: trim keeps a buffer put at the
	// cutoff itself, which a coarse clock could otherwise produce.
	time.Sleep(time.Millisecond)
	cut := now()
	time.Sleep(time.Millisecond)
	young := make([]byte, 100)
	p.Put(young)
	trimAll(cut)
	if r := Retained(); r != 100 {
		t.Fatalf("Retained = %d, want 100", r)
	}
	if b := p.Get(100); &b[0] != &young[0] {
		t.Fatal("the young buffer was dropped")
	}
	if b := p.Get(100); &b[0] == &old[0] {
		t.Fatal("the idle buffer was kept")
	}
}

// TestTrimOnGC checks that garbage collections drop idle buffers without
// any further call.
func TestTrimOnGC(t *testing.T) {
	fresh(t, DefaultLimit, time.Hour)
	var p Pool[byte]
	for i := 0; i < 10; i++ {
		p.Put(make([]byte, 1<<20))
	}
	if Retained() != 10<<20 {
		t.Fatal("not retained")
	}
	SetMaxIdle(0)
	deadline := time.Now().Add(10 * time.Second)
	for Retained() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("still %d bytes retained after GCs", Retained())
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	// still armed for the following cycles
	p.Put(make([]byte, 1<<20))
	for Retained() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("GC hook not re-armed")
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
}

func TestConcurrent(t *testing.T) {
	fresh(t, 4<<20, time.Hour)
	var p8 Pool[byte]
	var p16 Pool[int16]
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(g)))
			var held [][]byte
			for i := 0; i < 2000; i++ {
				n := 1 + rng.Intn(64<<10)
				b := p8.Get(n)
				if len(b) != n {
					t.Errorf("len %d, want %d", len(b), n)
					return
				}
				for k := range b {
					b[k] = byte(g)
				}
				held = append(held, b)
				if len(held) > 4 {
					for _, h := range held[:2] {
						for _, v := range h {
							if v != byte(g) {
								t.Error("buffer shared between goroutines")
								return
							}
						}
						p8.Put(h)
					}
					held = held[2:]
				}
				s := p16.Get(1 + rng.Intn(1000))
				p16.Put(s)
				if i%500 == 0 {
					trimAll(now())
				}
			}
			for _, h := range held {
				p8.Put(h)
			}
		}()
	}
	wg.Wait()
	if r := Retained(); r < 0 || r > 4<<20 {
		t.Fatalf("Retained = %d out of [0, limit]", r)
	}
	Trim()
	if r := Retained(); r != 0 {
		t.Fatalf("Retained after Trim = %d (accounting drift)", r)
	}
}

func BenchmarkGetPut(b *testing.B) {
	Trim()
	var p Pool[byte]
	for b.Loop() {
		p.Put(p.Get(1 << 20))
	}
}

func TestDefaultLimitByWordSize(t *testing.T) {
	want := int64(256 << 20)
	if strconv.IntSize == 32 {
		want = 64 << 20
	}
	if DefaultLimit != want {
		t.Fatalf("DefaultLimit = %d, want %d on %d-bit", int64(DefaultLimit), want, strconv.IntSize)
	}
}
