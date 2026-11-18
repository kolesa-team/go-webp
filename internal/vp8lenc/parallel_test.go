package vp8lenc

import (
	"bytes"
	"errors"
	"fmt"
	"image/color"
	"math/rand"
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// takeBudget occupies every helper slot of the process-wide budget (so that
// workers.Spawn returns nil) and returns the function giving them back.
func takeBudget(t testing.TB) func() {
	t.Helper()
	n := 0
	for workers.TryAcquire() {
		n++
	}
	if workers.Spawn(func() error { return nil }) != nil {
		t.Fatal("budget not exhausted")
	}
	return func() {
		for ; n > 0; n-- {
			workers.Release()
		}
	}
}

// checkNoLeak fails if helper slots or goroutines started since before are
// still alive shortly after.
func checkNoLeak(t testing.TB, before int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if workers.InUse() == 0 && runtime.NumGoroutine() <= before {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("leak: %d helper slots in use, %d goroutines (before: %d)", workers.InUse(), runtime.NumGoroutine(), before)
		}
		time.Sleep(time.Millisecond)
	}
}

// forceParallel lowers the size thresholds of the parallel stages so that
// small test images take the parallel paths, until the returned function is
// called.
func forceParallel() func() {
	px, stripe, remap := minParallelPixels, minHCStripe, minParallelRemap
	minParallelPixels, minHCStripe, minParallelRemap = 0, 64, 0
	return func() { minParallelPixels, minHCStripe, minParallelRemap = px, stripe, remap }
}

// budgets runs f with a budget of 1 helper, of 8 helpers and with the budget
// fully taken.
func budgets(t *testing.T, f func(t *testing.T)) {
	for _, b := range []struct {
		name  string
		limit int
		taken bool
	}{{"limit1", 1, false}, {"limit8", 8, false}, {"exhausted", 4, true}} {
		t.Run(b.name, func(t *testing.T) {
			defer workers.SetLimitForTest(workers.SetLimitForTest(b.limit))
			if b.taken {
				defer takeBudget(t)()
			}
			f(t)
		})
	}
}

func TestParallelClaimsEveryItemOnce(t *testing.T) {
	before := runtime.NumGoroutine()
	budgets(t, func(t *testing.T) {
		for _, items := range []int{0, 1, 2, 7, 100} {
			for _, helpers := range []int{0, 1, 3, 16} {
				counts := make([]atomic.Int32, items)
				var perWorker [17]atomic.Int32
				parallel(items, helpers, func(worker int, claim func() (int, bool)) {
					for {
						i, ok := claim()
						if !ok {
							break
						}
						counts[i].Add(1)
						perWorker[worker].Add(1)
					}
					// claim keeps reporting false once exhausted.
					if _, ok := claim(); ok {
						t.Error("claim after exhaustion")
					}
				})
				for i := range counts {
					if c := counts[i].Load(); c != 1 {
						t.Fatalf("items %d helpers %d: item %d claimed %d times", items, helpers, i, c)
					}
				}
				for w := max(min(helpers, items-1), 0) + 1; w < len(perWorker); w++ {
					if perWorker[w].Load() != 0 {
						t.Fatalf("items %d helpers %d: unexpected worker %d", items, helpers, w)
					}
				}
			}
		}
	})
	checkNoLeak(t, before)
}

func TestParallelExhaustedBudgetRunsOnCaller(t *testing.T) {
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	defer takeBudget(t)()
	parallel(10, 3, func(worker int, claim func() (int, bool)) {
		if worker != 0 {
			t.Errorf("worker %d started with an exhausted budget", worker)
		}
		for _, ok := claim(); ok; _, ok = claim() {
		}
	})
}

func TestParallelHelperPanic(t *testing.T) {
	before := runtime.NumGoroutine()
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	boom := errors.New("boom")
	var done atomic.Int32
	func() {
		defer func() {
			r := recover()
			pe, ok := r.(*workers.PanicError)
			if !ok || !errors.Is(pe, boom) {
				t.Fatalf("recovered %v, want the helper's panic", r)
			}
		}()
		started := make(chan struct{})
		parallel(50, 3, func(worker int, claim func() (int, bool)) {
			if worker == 1 {
				close(started)
				panic(boom)
			}
			if worker == 0 {
				<-started // make sure the helper ran
			}
			for _, ok := claim(); ok; _, ok = claim() {
				done.Add(1)
			}
		})
		t.Fatal("no panic")
	}()
	checkNoLeak(t, before)
}

func TestParallelCallerPanic(t *testing.T) {
	before := runtime.NumGoroutine()
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	var helperItems atomic.Int32
	func() {
		defer func() {
			if r := recover(); r != "caller" {
				t.Fatalf("recovered %v, want the caller's panic", r)
			}
		}()
		parallel(1000, 3, func(worker int, claim func() (int, bool)) {
			if worker == 0 {
				panic("caller")
			}
			for _, ok := claim(); ok; _, ok = claim() {
				helperItems.Add(1)
				time.Sleep(10 * time.Microsecond)
			}
		})
	}()
	if n := helperItems.Load(); n >= 1000 {
		t.Errorf("helpers processed all %d items after the caller panicked", n)
	}
	checkNoLeak(t, before)
}

// TestCrewReuse runs many stages on one crew: every item is claimed once,
// the helpers are started once (while the budget allows) and reused, never
// more than asked for a stage, and stop ends them.
func TestCrewReuse(t *testing.T) {
	before := runtime.NumGoroutine()
	budgets(t, func(t *testing.T) {
		inUse := workers.InUse() // (taken by budgets)
		var c crew
		var spawned int
		for round := 0; round < 50; round++ {
			for _, st := range []struct{ items, helpers int }{{100, 7}, {3, 7}, {40, 2}, {1, 7}, {0, 3}, {9, 0}} {
				counts := make([]atomic.Int32, st.items)
				var perWorker [intraHelpers + 1]atomic.Int32
				c.run(st.items, st.helpers, func(worker int, claim func() (int, bool)) {
					for i, ok := claim(); ok; i, ok = claim() {
						counts[i].Add(1)
						perWorker[worker].Add(1)
					}
					if _, ok := claim(); ok {
						t.Error("claim after exhaustion")
					}
				})
				for i := range counts {
					if n := counts[i].Load(); n != 1 {
						t.Fatalf("%+v: item %d claimed %d times", st, i, n)
					}
				}
				for w := max(min(st.helpers, st.items-1), 0) + 1; w < len(perWorker); w++ {
					if perWorker[w].Load() != 0 {
						t.Fatalf("%+v: unexpected worker %d", st, w)
					}
				}
				if round == 0 && st.items == 100 {
					spawned = len(c.tasks)
				}
			}
			if len(c.tasks) != spawned {
				t.Fatalf("round %d: %d helpers, %d after the first stage", round, len(c.tasks), spawned)
			}
		}
		if spawned > workers.Limit() || spawned > intraHelpers {
			t.Fatalf("%d helpers with a budget of %d", spawned, workers.Limit())
		}
		c.stop()
		if workers.InUse() != inUse {
			t.Fatalf("%d helper slots in use after stop, %d before", workers.InUse(), inUse)
		}
		// A stopped crew starts again.
		c.run(10, 3, func(_ int, claim func() (int, bool)) {
			for _, ok := claim(); ok; _, ok = claim() {
			}
		})
		c.stop()
	})
	checkNoLeak(t, before)
}

// TestCrewHelperPanic checks that a panic in a helper of a crew is
// re-raised on the caller (as the helper's *workers.PanicError) once the
// stage is joined, and that stop then ends the other helpers.
func TestCrewHelperPanic(t *testing.T) {
	before := runtime.NumGoroutine()
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	boom := errors.New("boom")
	var c crew
	// A first stage starts the helpers.
	c.run(10, 3, func(_ int, claim func() (int, bool)) {
		for _, ok := claim(); ok; _, ok = claim() {
		}
	})
	func() {
		defer c.stop()
		defer func() {
			r := recover()
			pe, ok := r.(*workers.PanicError)
			if !ok || !errors.Is(pe, boom) {
				t.Fatalf("recovered %v, want the helper's panic", r)
			}
		}()
		started := make(chan struct{})
		c.run(50, 3, func(worker int, claim func() (int, bool)) {
			if worker == 2 {
				close(started)
				panic(boom)
			}
			if worker == 0 {
				<-started // make sure the helper ran
			}
			for _, ok := claim(); ok; _, ok = claim() {
			}
		})
		t.Fatal("no panic")
	}()
	checkNoLeak(t, before)
	// The crew works again after stop.
	var n atomic.Int32
	c.run(20, 3, func(_ int, claim func() (int, bool)) {
		for _, ok := claim(); ok; _, ok = claim() {
			n.Add(1)
		}
	})
	c.stop()
	if n.Load() != 20 {
		t.Fatalf("%d items after the restart, want 20", n.Load())
	}
	checkNoLeak(t, before)
}

// TestCrewCallerPanic checks that when the caller's body panics, the helpers
// stop claiming items and are joined before the panic goes on, and stop
// ends them.
func TestCrewCallerPanic(t *testing.T) {
	before := runtime.NumGoroutine()
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	var c crew
	var helperItems atomic.Int32
	func() {
		defer c.stop()
		defer func() {
			if r := recover(); r != "caller" {
				t.Fatalf("recovered %v, want the caller's panic", r)
			}
			if c.body != nil {
				t.Fatal("stage not joined")
			}
		}()
		c.run(1000, 3, func(worker int, claim func() (int, bool)) {
			if worker == 0 {
				panic("caller")
			}
			for _, ok := claim(); ok; _, ok = claim() {
				helperItems.Add(1)
				time.Sleep(10 * time.Microsecond)
			}
		})
	}()
	if n := helperItems.Load(); n >= 1000 {
		t.Errorf("helpers processed all %d items after the caller panicked", n)
	}
	checkNoLeak(t, before)
}

// TestThreadLevelAllocs checks that a ThreadLevel encode starts its helpers
// once per parallel phase (not per stage: the predictor alone runs dozens
// of stages) and leaves none running.
func TestThreadLevelAllocs(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	defer workers.SetLimitForTest(workers.SetLimitForTest(8))
	img := downscale(loadTestJPEG(t, "source.jpg"), 1) // 480x640: the stages run in parallel
	cfg := &Config{Quality: 75, Method: 4, ThreadLevel: true}
	seq := &Config{Quality: 75, Method: 4}
	if _, err := Encode(img, cfg); err != nil {
		t.Fatal(err)
	}
	par := testing.AllocsPerRun(5, func() { Encode(img, cfg) })
	one := testing.AllocsPerRun(5, func() { Encode(img, seq) })
	// About 4 per helper (goroutine, Task, channel, closure) and phase (the
	// match search, the predictor, the remap), plus a few per stage; about
	// 720 when the helpers were started for every stage.
	if par-one > 120 {
		t.Errorf("%.0f allocations with ThreadLevel, %.0f without", par, one)
	}
	if workers.InUse() != 0 {
		t.Errorf("%d helpers running after the encodes", workers.InUse())
	}
}

// TestPhasesReleaseHelpers checks that each parallel phase (the match
// search, the predictor mode search, the histogram remap) ends its helpers
// when it ends, so that they do not hold budget slots, idle, through the
// sequential stages of the encode that follow.
func TestPhasesReleaseHelpers(t *testing.T) {
	defer forceParallel()()
	defer workers.SetLimitForTest(workers.SetLimitForTest(8))
	img := threadTestImages()[0].img
	w, h := img.Rect.Dx(), img.Rect.Dy()
	n := w * h
	argb := make([]uint32, n)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	check := func(phase string, sc *scratch) {
		t.Helper()
		if len(sc.crew.start) == 0 {
			t.Fatalf("%s: no helper was started", phase)
		}
		if workers.InUse() != 0 || len(sc.crew.tasks) != 0 {
			t.Fatalf("%s: %d helpers still running after the phase", phase, workers.InUse())
		}
	}

	sc := new(scratch)
	var hc hashChain
	hc.fillPar(argb, w, h, 75, false, 0, intraHelpers, sc)
	check("match search", sc)

	sc = new(scratch)
	modes := make([]uint32, subSampleSize(w, 2)*subSampleSize(h, 2))
	predictorImage(w, h, 2, 2, false, 1, false, false, false, slices.Clone(argb), modes, sc, intraHelpers)
	check("predictor", sc)

	sc = new(scratch)
	refs := backwardRefsLz77(n, argb, &hc, nil)
	var c clusterer
	var out histoSet
	c.crew = &sc.crew
	if ng := c.buildHistoImage(w, h, refs, 75, false, 2, 0, &out, intraHelpers); ng < 2 {
		t.Fatalf("remap: %d clusters, want a parallel remap", ng)
	}
	check("remap", sc)
}

// TestThreadLevelGOMAXPROCS checks the output against the sequential one
// with the default budget (GOMAXPROCS-1 helpers) for GOMAXPROCS 1 to 8.
func TestThreadLevelGOMAXPROCS(t *testing.T) {
	defer forceParallel()()
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	imgs := threadTestImages()
	cfgs := []Config{{Quality: 75, Method: 4}, {Quality: 90, Method: 5}, {Quality: 30, Method: 3, NearLossless: 60}}
	for _, ti := range imgs {
		for _, c := range cfgs {
			want, err := Encode(ti.img, &c)
			if err != nil {
				t.Fatal(err)
			}
			for procs := 1; procs <= 8; procs++ {
				runtime.GOMAXPROCS(procs)
				c.ThreadLevel = true
				got, err := Encode(ti.img, &c)
				c.ThreadLevel = false
				if err != nil || !bytes.Equal(got, want) {
					t.Fatalf("%s %+v GOMAXPROCS %d: output differs (%v)", ti.name, c, procs, err)
				}
				if workers.InUse() != 0 {
					t.Fatalf("%d helpers running after the encode", workers.InUse())
				}
			}
		}
	}
}

// threadTestImages are images exercising every configuration path.
func threadTestImages() []testImage {
	rng := rand.New(rand.NewSource(3))
	var out []testImage
	add := func(name string, w, h int, f func(x, y int) color.NRGBA) {
		out = append(out, testImage{name, newNRGBA(w, h, f)})
	}
	add("photo", 131, 67, func(x, y int) color.NRGBA {
		v := uint8((x*x + y*y) / 51)
		return color.NRGBA{v + uint8(rng.Intn(9)), uint8(x) + uint8(rng.Intn(5)), uint8(y * 2), 255}
	})
	add("cutout", 100, 80, func(x, y int) color.NRGBA {
		a := uint8(255)
		if (x-50)*(x-50)+(y-40)*(y-40) > 35*35 {
			a = 0
		} else if (x-50)*(x-50)+(y-40)*(y-40) > 31*31 {
			a = uint8(rng.Intn(256))
		}
		return color.NRGBA{uint8(x + rng.Intn(3)), uint8(y), uint8(x ^ y), a}
	})
	add("palette", 203, 45, func(x, y int) color.NRGBA {
		return color.NRGBA{uint8((x / 7) * 40), uint8((y / 5) * 30), 0, 255}
	})
	add("odd", 1, 77, func(x, y int) color.NRGBA { return color.NRGBA{uint8(y * 3), uint8(rng.Intn(256)), 9, 255} })
	add("row", 301, 1, func(x, y int) color.NRGBA { return color.NRGBA{uint8(x), uint8(x * 3), uint8(rng.Intn(4)), 255} })
	return out
}

var threadTestConfigs = []Config{
	{Quality: 0, Method: 0},
	{Quality: 30, Method: 3},
	{Quality: 75, Method: 4},
	{Quality: 75, Method: 4, Exact: true},
	{Quality: 80, Method: 4, NearLossless: 60},
	{Quality: 90, Method: 5},
	{Quality: 100, Method: 6},
	{Quality: 75, Method: 4, ImageHint: 3},
}

// TestThreadLevelIdentical checks that ThreadLevel encodes are
// byte-identical to sequential ones whatever the helper budget, including
// when it is exhausted, and that no helper outlives an encode.
func TestThreadLevelIdentical(t *testing.T) {
	before := runtime.NumGoroutine()
	imgs := threadTestImages()
	want := map[string][]byte{}
	key := func(i, k int) string { return fmt.Sprint(i, "/", k) }
	// The brute-force level (all configurations) is slow under -race.
	skip := func(ti testImage, cfg Config) bool {
		return testing.Short() && cfg.Method == 6 && ti.img.Rect.Dx()*ti.img.Rect.Dy() > 1000 && ti.name != "palette"
	}
	for i, ti := range imgs {
		for k := range threadTestConfigs {
			cfg := threadTestConfigs[k]
			if skip(ti, cfg) {
				continue
			}
			out, err := Encode(ti.img, &cfg)
			if err != nil {
				t.Fatal(err)
			}
			want[key(i, k)] = out
		}
	}
	budgets(t, func(t *testing.T) {
		for _, force := range []bool{false, true} {
			if force {
				// The stages of small images in parallel too.
				defer forceParallel()()
			}
			for i, ti := range imgs {
				for k := range threadTestConfigs {
					cfg := threadTestConfigs[k]
					if skip(ti, cfg) {
						continue
					}
					cfg.ThreadLevel = true
					out, err := Encode(ti.img, &cfg)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(out, want[key(i, k)]) {
						t.Fatalf("%s %+v force=%v: ThreadLevel output differs", ti.name, cfg, force)
					}
				}
			}
		}
	})
	checkNoLeak(t, before)
}

// TestThreadLevelConcurrent runs ThreadLevel encodes, with every stage in
// parallel, from several goroutines sharing the budget (run with -race).
func TestThreadLevelConcurrent(t *testing.T) {
	defer workers.SetLimitForTest(workers.SetLimitForTest(3))
	defer forceParallel()()
	imgs := threadTestImages()
	cfgs := []Config{{Quality: 75, Method: 4}, {Quality: 90, Method: 5}, {Quality: 80, Method: 3, NearLossless: 60}}
	want := make([][]byte, len(imgs)*len(cfgs))
	for i, ti := range imgs {
		for k := range cfgs {
			want[i*len(cfgs)+k], _ = Encode(ti.img, &cfgs[k])
		}
	}
	errs := make(chan error, 4*len(want))
	for g := 0; g < 4; g++ {
		go func() {
			for i, ti := range imgs {
				for k := range cfgs {
					c := cfgs[k]
					c.ThreadLevel = true
					out, err := Encode(ti.img, &c)
					if err == nil && !bytes.Equal(out, want[i*len(cfgs)+k]) {
						err = fmt.Errorf("%s %+v: concurrent ThreadLevel output differs", ti.name, c)
					}
					errs <- err
				}
			}
		}()
	}
	for range 4 * len(want) {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

// TestConfigHelpersBySize checks that only images up to
// maxConfigPairPixels encode two configurations at a time, and only with
// ThreadLevel.
func TestConfigHelpersBySize(t *testing.T) {
	p := params{parallel: true}
	if got := p.configHelpersFor(maxConfigPairPixels); got != configHelpers {
		t.Errorf("at the limit: %d helpers, want %d", got, configHelpers)
	}
	if got := p.configHelpersFor(maxConfigPairPixels + 1); got != 0 {
		t.Errorf("above the limit: %d helpers, want 0", got)
	}
	p.parallel = false
	if got := p.configHelpersFor(16); got != 0 {
		t.Errorf("without ThreadLevel: %d helpers, want 0", got)
	}
}
