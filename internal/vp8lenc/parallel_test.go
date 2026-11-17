package vp8lenc

import (
	"bytes"
	"errors"
	"fmt"
	"image/color"
	"math/rand"
	"runtime"
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
