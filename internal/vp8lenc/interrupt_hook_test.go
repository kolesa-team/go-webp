package vp8lenc

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// pollPoints returns the polls (1-based) at which the tests make the
// interrupt hook fire, out of n: all of them for short encodes, a sample
// otherwise.
func pollPoints(n int64) []int64 {
	var ks []int64
	if n <= 48 {
		for k := int64(1); k <= n; k++ {
			ks = append(ks, k)
		}
		return ks
	}
	for k := int64(1); k <= 8; k++ {
		ks = append(ks, k)
	}
	for i := int64(1); i < 32; i++ {
		ks = append(ks, 8+i*(n-16)/32)
	}
	for k := n - 7; k <= n; k++ {
		ks = append(ks, k)
	}
	return ks
}

// TestInterruptHook checks Config.Interrupt and AlphaEncoder.Interrupt:
// never firing, they leave the output unchanged; firing at any of their
// polls, the encode stops with ErrInterrupted after a few more polls at
// most (the remaining candidate configurations return at once).
func TestInterruptHook(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	photo := image.NewNRGBA(image.Rect(0, 0, 41, 23))
	rnd.Read(photo.Pix)
	pal := image.NewNRGBA(image.Rect(0, 0, 41, 23))
	for i := 0; i < len(pal.Pix); i += 4 {
		v := uint8(rnd.Intn(4) * 60)
		pal.Pix[i], pal.Pix[i+1], pal.Pix[i+2], pal.Pix[i+3] = v, 255-v, v/2, 255
	}
	for _, img := range []*image.NRGBA{photo, pal} {
		for _, cfg := range []Config{
			{Quality: 75, Method: 4, NearLossless: 100},
			{Quality: 100, Method: 6, NearLossless: 100},              // brute force: all configurations
			{Quality: 90, Method: 5, NearLossless: 100, ImageHint: 3}, // several candidates
			{Quality: 50, Method: 6, NearLossless: 100, ThreadLevel: true},
			{Quality: 0, Method: 0, NearLossless: 60},
		} {
			name := fmt.Sprintf("%p q%v m%d", img, cfg.Quality, cfg.Method)
			want, err := Encode(img, &cfg)
			if err != nil {
				t.Fatal(err)
			}
			var polls atomic.Int64
			cfg.Interrupt = func() bool { polls.Add(1); return false }
			got, err := Encode(img, &cfg)
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("%s: a silent Interrupt changed the output (err %v)", name, err)
			}
			n := polls.Load()
			if n < 2 {
				t.Fatalf("%s: polled %d times", name, n)
			}
			for _, k := range pollPoints(n) {
				polls.Store(0)
				cfg.Interrupt = func() bool { return polls.Add(1) >= k }
				if _, err := Encode(img, &cfg); !errors.Is(err, ErrInterrupted) {
					t.Fatalf("%s: interrupted at poll %d of %d: got %v", name, k, n, err)
				}
				if extra := polls.Load() - k; extra > 16 {
					t.Fatalf("%s: interrupted at poll %d, polled %d more times", name, k, extra)
				}
			}
		}
	}

	plane := make([]byte, 50*30)
	rnd.Read(plane)
	var a AlphaEncoder
	defer a.Release()
	want, err := a.Encode(plane, 50, 30, 50, 4, 100)
	if err != nil {
		t.Fatal(err)
	}
	want = bytes.Clone(want)
	var polls atomic.Int64
	a.Interrupt = func() bool { polls.Add(1); return false }
	got, err := a.Encode(plane, 50, 30, 50, 4, 100)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("alpha: a silent Interrupt changed the output (err %v)", err)
	}
	for _, k := range pollPoints(polls.Load()) {
		polls.Store(0)
		a.Interrupt = func() bool { return polls.Add(1) >= k }
		if _, err := a.Encode(plane, 50, 30, 50, 4, 100); !errors.Is(err, ErrInterrupted) {
			t.Fatalf("alpha: interrupted at poll %d: got %v", k, err)
		}
	}
	// The encoder is still usable afterwards.
	a.Interrupt = nil
	got, err = a.Encode(plane, 50, 30, 50, 4, 100)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("alpha: output changed after interrupted encodes (err %v)", err)
	}
}

// callerID returns the id of the calling goroutine (tests only).
func callerID() string {
	var buf [64]byte
	s := strings.TrimPrefix(string(buf[:runtime.Stack(buf[:], false)]), "goroutine ")
	return s[:strings.IndexByte(s, ' ')]
}

// TestInterruptParallelStages fires Config.Interrupt while the ThreadLevel
// stages (hash chain stripes, predictor batches, histogram remap) run on
// helper goroutines: the hook is polled on the helpers too, a silent hook
// leaves the output byte-identical, and an interrupt at any sampled poll
// gives ErrInterrupted after a bounded number of further polls, with no
// helper left running, whatever the budget.
func TestInterruptParallelStages(t *testing.T) {
	before := runtime.NumGoroutine()
	defer forceParallel()()
	var imgs []testImage
	for _, ti := range threadTestImages() {
		if ti.name == "photo" || ti.name == "cutout" || ti.name == "palette" {
			imgs = append(imgs, ti)
		}
	}
	cfgs := []Config{{Quality: 30, Method: 3}, {Quality: 75, Method: 4}, {Quality: 100, Method: 6}}
	helperPolls := 0
	budgets(t, func(t *testing.T) {
		for _, ti := range imgs {
			for _, cfg := range cfgs {
				if testing.Short() && cfg.Method != 4 {
					continue
				}
				want, err := Encode(ti.img, &cfg)
				if err != nil {
					t.Fatal(err)
				}
				cfg.ThreadLevel = true
				caller := callerID()
				var polls, onHelpers, waits atomic.Int64
				cfg.Interrupt = func() bool {
					polls.Add(1)
					if callerID() != caller {
						onHelpers.Add(1)
					} else if onHelpers.Load() == 0 && waits.Add(1) <= 200 {
						// Let the helpers of a stage start and claim an
						// item: on a loaded machine the caller may claim
						// them all first.
						time.Sleep(20 * time.Microsecond)
					}
					return false
				}
				got, err := Encode(ti.img, &cfg)
				if err != nil || !bytes.Equal(got, want) {
					t.Fatalf("%s %+v: a silent Interrupt changed the output (err %v)", ti.name, cfg, err)
				}
				helperPolls += int(onHelpers.Load())
				n := polls.Load()
				for i, k := range pollPoints(n) {
					if testing.Short() && i%3 != 0 {
						continue
					}
					polls.Store(0)
					cfg.Interrupt = func() bool { return polls.Add(1) >= k }
					if _, err := Encode(ti.img, &cfg); !errors.Is(err, ErrInterrupted) {
						t.Fatalf("%s %+v: interrupted at poll %d of %d: got %v", ti.name, cfg, k, n, err)
					}
					// Every helper of a stage may poll once more before it
					// stops, then the remaining passes return at once; at
					// method 6 two candidate configurations run at a time.
					if extra := polls.Load() - k; extra > 2*(16+2*intraHelpers) {
						t.Fatalf("%s %+v: interrupted at poll %d, polled %d more times", ti.name, cfg, k, extra)
					}
				}
			}
		}
	})
	if helperPolls == 0 {
		t.Fatal("Interrupt was never polled on a helper goroutine")
	}
	checkNoLeak(t, before)
}
