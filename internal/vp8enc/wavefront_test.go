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

package vp8enc

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

type wavefrontCase struct {
	name  string
	img   *image.NRGBA
	sharp bool
	cfg   Config
	// p0Limit, if not zero, lowers partition0SizeLimit so that the pass is
	// redone with stronger intra4 header limits.
	p0Limit uint64
}

func (c wavefrontCase) encode(tb testing.TB) []byte {
	tb.Helper()
	if c.p0Limit != 0 {
		defer func(l uint64) { partition0SizeLimit = l }(partition0SizeLimit)
		partition0SizeLimit = c.p0Limit
	}
	pic := FromNRGBA(c.img, c.sharp, false)
	defer pic.Release()
	cfg := c.cfg
	out, err := Encode(pic, &cfg)
	if err != nil {
		tb.Fatalf("%s: %v", c.name, err)
	}
	return out
}

// wavefrontCases covers every method, the multi-pass searches, the
// partition #0 retry, the autofilter, sharp YUV, several partitions and
// picture shapes with several refresh points per row or narrow rows.
func wavefrontCases(tb testing.TB) []wavefrontCase {
	source := loadTestImage(tb, "source.jpg") // 43x57 macroblocks
	var cases []wavefrontCase
	add := func(name string, img *image.NRGBA, sharp bool, mod func(*Config)) {
		cfg := DefaultConfig()
		mod(&cfg)
		cases = append(cases, wavefrontCase{name: name, img: img, sharp: sharp, cfg: cfg})
	}
	for m := 0; m <= 6; m++ {
		add(fmt.Sprintf("source/m%d", m), source, false, func(c *Config) { c.Method = m })
	}
	add("source/m4-sharp", source, true, func(c *Config) {})
	add("source/target-size", source, false, func(c *Config) { c.TargetSize = 20000; c.Pass = 4 })
	add("source/target-psnr", source, false, func(c *Config) { c.TargetPSNR = 38; c.Pass = 4; c.Method = 6 })
	add("source/jpeg-size", source, false, func(c *Config) { c.EmulateJPEGSize = true; c.Preprocessing = 3; c.Method = 3 })
	add("source/partitions-m2", source, false, func(c *Config) { c.Partitions = 3; c.Method = 2 })
	add("source/partitions-m4", source, false, func(c *Config) { c.Partitions = 2; c.LowMemory = true; c.PartitionLimit = 70 })
	add("source/q0-m6", source, false, func(c *Config) { c.Quality = 0; c.Method = 6 })
	add("source/q100-m3", source, false, func(c *Config) { c.Quality = 100; c.Method = 3 })
	add("source/autofilter-m4", source, false, func(c *Config) { c.Autofilter = true })
	add("source/autofilter-pass3", source, false, func(c *Config) { c.Autofilter = true; c.TargetPSNR = 40; c.Pass = 3; c.Method = 5 })
	add("source/seg1-sns0", source, false, func(c *Config) { c.Segments = 1; c.SNSStrength = 0; c.Method = 5 })
	add("source/simple-filter", source, false, func(c *Config) { c.FilterType = 0; c.Autofilter = true; c.Method = 6 })
	alpha := synthImage(400, 300, 3, 21) // transparent areas
	add("alpha/m4", alpha, false, func(c *Config) {})
	add("alpha/m1", alpha, false, func(c *Config) { c.Method = 1 })
	// 125x7 macroblocks: a refresh point every 110 macroblocks, so up to
	// two per row.
	wide := synthImage(2000, 112, 1, 22)
	add("wide/m4", wide, false, func(c *Config) {})
	add("wide/m6-autofilter", wide, false, func(c *Config) { c.Method = 6; c.Autofilter = true })
	add("wide/m0", wide, false, func(c *Config) { c.Method = 0 })
	// 5 macroblocks per row: two rows at a time at most.
	narrow := synthImage(80, 1200, 2, 23)
	add("narrow/m4", narrow, false, func(c *Config) {})
	add("narrow/m2", narrow, false, func(c *Config) { c.Method = 2 })
	// The partition #0 retry, which redoes the pass with smaller intra4
	// header limits.
	for _, m := range []int{2, 4, 6} {
		cfg := DefaultConfig()
		cfg.Method = m
		cases = append(cases, wavefrontCase{name: fmt.Sprintf("source/p0-retry-m%d", m), img: source, cfg: cfg, p0Limit: 1 << 26})
	}
	return cases
}

// forceWavefrontForTest lifts the size gate of the wavefront pass for the
// rest of the test.
func forceWavefrontForTest(tb testing.TB) {
	forceWavefront = true
	tb.Cleanup(func() { forceWavefront = false })
}

// TestWavefrontIdentical checks that the wavefront coding pass
// (ThreadLevel) writes exactly the sequential pass's bytes for every
// helper budget, including none.
func TestWavefrontIdentical(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	defer func() { testHookWavefrontRow = nil }()
	limits := []int{1, 2, 3, 4, 5, 6, 7, 8}
	if testing.Short() {
		limits = []int{1, 3, 8}
	}
	// Under the race detector one budget per case, on the cases that
	// exercise different synchronization (refresh points, several passes,
	// the partition #0 retry, narrow and wide rows); the identity of every
	// case for every budget is checked without -race.
	raceCases := map[string]bool{
		"source/m4": true, "source/target-size": true, "source/autofilter-m4": true,
		"wide/m4": true, "narrow/m4": true, "source/p0-retry-m4": true,
	}
	if raceEnabled {
		limits = []int{3}
	}
	var rows atomic.Int64
	testHookWavefrontRow = func(int) { rows.Add(1) }
	for _, c := range wavefrontCases(t) {
		if testing.Short() && c.cfg.Method != 4 && c.cfg.Method != 0 {
			continue
		}
		if raceEnabled && !raceCases[c.name] {
			continue
		}
		workers.SetLimitForTest(0)
		c.cfg.ThreadLevel = false
		want := c.encode(t)
		c.cfg.ThreadLevel = true
		for _, limit := range limits {
			workers.SetLimitForTest(limit)
			rows.Store(0)
			if got := c.encode(t); !bytes.Equal(got, want) {
				t.Errorf("%s, budget %d: output differs from the sequential pass", c.name, limit)
			}
			if rows.Load() == 0 {
				t.Errorf("%s, budget %d: the wavefront pass did not run", c.name, limit)
			}
			waitHelpersIdle(t)
		}
		// Budget fully taken: the sequential pass runs.
		workers.SetLimitForTest(1)
		if !workers.TryAcquire() {
			t.Fatal("cannot take the budget")
		}
		rows.Store(0)
		if got := c.encode(t); !bytes.Equal(got, want) {
			t.Errorf("%s, no budget: output differs", c.name)
		}
		workers.Release()
		if rows.Load() != 0 {
			t.Errorf("%s, no budget: the wavefront pass ran", c.name)
		}
	}
}

// TestWavefrontGOMAXPROCS checks the output for GOMAXPROCS 1..8 with the
// default budget (GOMAXPROCS-1 helpers), and with more helpers than
// processors, under the default size gate.
func TestWavefrontGOMAXPROCS(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	source := loadTestImage(t, "source.jpg")
	for _, m := range []int{0, 4, 6} {
		if testing.Short() && m == 6 {
			continue
		}
		cfg := DefaultConfig()
		cfg.Method = m
		c := wavefrontCase{name: "source/m" + strconv.Itoa(m), img: source, cfg: cfg}
		want := c.encode(t)
		c.cfg.ThreadLevel = true
		procs := []int{1, 2, 3, 4, 5, 6, 7, 8}
		if raceEnabled {
			procs = []int{1, 3, 8}
		}
		for _, p := range procs {
			runtime.GOMAXPROCS(p)
			for _, limit := range []int{0, 6} { // 0: the default, GOMAXPROCS-1
				workers.SetLimitForTest(limit)
				if got := c.encode(t); !bytes.Equal(got, want) {
					t.Errorf("m%d GOMAXPROCS %d budget %d: output differs", m, p, limit)
				}
			}
		}
	}
}

// TestWavefrontWorkers checks which pictures are coded as a wavefront and
// on how many rows at once (the cases hold with the vector kernels and
// with the Go reference ones).
func TestWavefrontWorkers(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(16))
	defer activeEncodes.Store(activeEncodes.Load())
	activeEncodes.Store(1)
	for _, tc := range []struct{ w, h, method, want int }{
		{1, 1, 4, 1}, {256, 256, 4, 1}, {256, 256, 0, 1}, {960, 1280, 0, 1},
		{960, 1280, 4, 7}, {640, 480, 6, 5}, {4000, 3000, 4, 8}, {4000, 3000, 2, 8},
		{16, 4000, 6, 1}, {4000, 16, 6, 1}, {4000, 32, 6, 2},
		{16383, 16383, 6, 8}, {16383, 16383, 0, 8}, // no 32-bit overflow
	} {
		e := &encoder{mbW: (tc.w + 15) >> 4, mbH: (tc.h + 15) >> 4, method: tc.method}
		e.cfg.ThreadLevel = true
		if got := e.wavefrontWorkers(); got != tc.want {
			t.Errorf("%dx%d m%d: %d workers, want %d", tc.w, tc.h, tc.method, got, tc.want)
		}
		e.cfg.ThreadLevel = false
		if got := e.wavefrontWorkers(); got != 1 {
			t.Errorf("%dx%d m%d without ThreadLevel: %d workers", tc.w, tc.h, tc.method, got)
		}
	}
	// Concurrent encodes share the processors.
	e := &encoder{mbW: 4000 / 16, mbH: 3000 / 16, method: 4}
	e.cfg.ThreadLevel = true
	for _, tc := range []struct{ active, procs, want int }{{2, 16, 8}, {3, 16, 5}, {4, 16, 4}, {9, 16, 1}, {20, 16, 1}, {1, 3, 3}, {1, 1, 1}} {
		activeEncodes.Store(int32(tc.active))
		runtime.GOMAXPROCS(tc.procs)
		if got := e.wavefrontWorkers(); got != tc.want {
			t.Errorf("%d encodes on %d processors: %d workers, want %d", tc.active, tc.procs, got, tc.want)
		}
	}
	runtime.GOMAXPROCS(16)
	activeEncodes.Store(1)
	// Forced (tests): any picture of two rows or more, rows two
	// macroblocks apart at least, whatever the other encodes.
	forceWavefrontForTest(t)
	for _, tc := range []struct{ w, h, want int }{{16, 160, 1}, {64, 64, 2}, {64, 16, 1}, {640, 480, 8}} {
		e := &encoder{mbW: (tc.w + 15) >> 4, mbH: (tc.h + 15) >> 4}
		e.cfg.ThreadLevel = true
		if got := e.wavefrontWorkers(); got != tc.want {
			t.Errorf("forced %dx%d: %d workers, want %d", tc.w, tc.h, got, tc.want)
		}
	}
}

// TestWavefrontPanic checks that a panic while coding a row, on a helper or
// on the calling goroutine, surfaces on the caller of Encode once all
// helpers have stopped, and that no helper slot or goroutine leaks.
func TestWavefrontPanic(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	defer func() { testHookWavefrontRow = nil }()
	img := synthImage(512, 512, 1, 9) // 32x32 macroblocks
	cfg := DefaultConfig()
	cfg.ThreadLevel = true
	base := runtime.NumGoroutine()
	boom := errors.New("boom")
	for _, onHelper := range []bool{true, false} {
		for _, atRow := range []int{0, 7, 31} {
			caller := goroutineID()
			var fired atomic.Bool
			testHookWavefrontRow = func(row int) {
				if row < atRow || (goroutineID() == caller) == onHelper || !fired.CompareAndSwap(false, true) {
					return
				}
				panic(boom)
			}
			func() {
				defer func() {
					r := recover()
					if !fired.Load() {
						// The chosen goroutine coded no row from atRow on.
						if r != nil {
							t.Fatalf("unexpected panic %v", r)
						}
						return
					}
					if onHelper {
						pe, ok := r.(*workers.PanicError)
						if !ok || !errors.Is(pe, boom) {
							t.Fatalf("helper panic at row %d: recovered %v, want *workers.PanicError(boom)", atRow, r)
						}
					} else if r != boom {
						t.Fatalf("caller panic at row %d: recovered %v, want boom", atRow, r)
					}
					if n := workers.InUse(); n != 0 {
						t.Fatalf("%d helpers still running after Encode returned", n)
					}
				}()
				pic := FromNRGBA(img, false, false)
				defer pic.Release()
				Encode(pic, &cfg)
			}()
		}
	}
	testHookWavefrontRow = nil
	// The encoder recovers: a later encode is unaffected.
	want := encodeNRGBA(t, img, DefaultConfig())
	if got := encodeNRGBA(t, img, cfg); !bytes.Equal(got, want) {
		t.Fatal("encode after a panic differs")
	}
	for i := 0; runtime.NumGoroutine() > base; i++ {
		if i == 500 {
			t.Fatalf("goroutines leaked: %d > %d", runtime.NumGoroutine(), base)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// TestWavefrontInterrupt fires Config.Interrupt at various polls of a
// ThreadLevel encode (analysis bands and wavefront rows, on helpers or on
// the caller): Encode returns ErrInterrupted with every helper joined.
func TestWavefrontInterrupt(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	img := synthImage(640, 480, 3, 13) // 40x30 macroblocks
	for _, pass := range []int{1, 3} {
		cfg := DefaultConfig()
		cfg.Pass, cfg.TargetSize = pass, 20000
		want := encodeNRGBA(t, img, cfg)
		cfg.ThreadLevel = true
		for _, limit := range []int{1, 4, 16} {
			workers.SetLimitForTest(limit)
			var polls atomic.Int64
			cfg.Interrupt = func() bool { polls.Add(1); return false }
			if got := encodeNRGBA(t, img, cfg); !bytes.Equal(got, want) {
				t.Fatalf("pass %d budget %d: a silent Interrupt changed the output", pass, limit)
			}
			n := polls.Load()
			if n < int64(pass*30) {
				t.Fatalf("pass %d budget %d: %d polls, want one per row and pass at least", pass, limit, n)
			}
			for _, k := range []int64{1, 2, n / 4, n / 2, n - 1, n} {
				polls.Store(0)
				cfg.Interrupt = func() bool { return polls.Add(1) >= k }
				pic := FromNRGBA(img, false, false)
				_, err := Encode(pic, &cfg)
				pic.Release()
				if !errors.Is(err, ErrInterrupted) {
					t.Fatalf("pass %d budget %d: interrupted at poll %d/%d: got %v", pass, limit, k, n, err)
				}
				if h := workers.InUse(); h != 0 {
					t.Fatalf("pass %d budget %d: %d helpers running after Encode returned", pass, limit, h)
				}
			}
		}
	}
}

// TestWavefrontInterruptLate fires Config.Interrupt on a helper polling
// for the last row after the calling goroutine has run out of rows: the
// row is not coded, so Encode must still return ErrInterrupted.
func TestWavefrontInterruptLate(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	img := synthImage(64, 32, 1, 3) // 4x2 macroblocks: two workers
	onHelper := 0
	for range 5 {
		cfg := DefaultConfig()
		cfg.ThreadLevel = true
		caller := goroutineID()
		var fired atomic.Bool
		cfg.Interrupt = func() bool {
			if fired.Load() {
				return true
			}
			if goroutineID() == caller {
				return false
			}
			// Let the calling goroutine finish its row first.
			time.Sleep(20 * time.Millisecond)
			fired.Store(true)
			return true
		}
		pic := FromNRGBA(img, false, false)
		_, err := Encode(pic, &cfg)
		pic.Release()
		if !fired.Load() {
			continue
		}
		onHelper++
		if !errors.Is(err, ErrInterrupted) {
			t.Fatalf("Interrupt fired on a helper: got %v, want ErrInterrupted", err)
		}
		if h := workers.InUse(); h != 0 {
			t.Fatalf("%d helpers running after Encode returned", h)
		}
	}
	if onHelper == 0 {
		t.Skip("no helper polled Interrupt")
	}
}

// TestWavefrontConcurrent runs many ThreadLevel encodes at once: helpers
// compete for the budget, some passes get none, and every output must be
// the sequential one.
func TestWavefrontConcurrent(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(3))
	img := synthImage(400, 300, 3, 11)
	cfgs := []Config{DefaultConfig(), DefaultConfig(), DefaultConfig()}
	cfgs[1].Method = 0
	cfgs[2].Method, cfgs[2].Autofilter = 6, true
	var wants [][]byte
	for i := range cfgs {
		wants = append(wants, encodeNRGBA(t, img, cfgs[i]))
		cfgs[i].ThreadLevel = true
	}
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 3; i++ {
				k := (g + i) % len(cfgs)
				pic := FromNRGBA(img, false, false)
				got, err := Encode(pic, &cfgs[k])
				pic.Release()
				if err != nil || !bytes.Equal(got, wants[k]) {
					errs <- fmt.Errorf("goroutine %d config %d: output differs (err %v)", g, k, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	waitHelpersIdle(t)
}

// FuzzWavefront codes small random pictures as a wavefront (the size gate
// lifted) on 1..9 workers and checks the output against the sequential
// pass.
func FuzzWavefront(f *testing.F) {
	f.Add(uint8(70), uint8(90), uint8(4), uint8(3), uint8(4), []byte{1, 2, 3})
	f.Add(uint8(255), uint8(20), uint8(6), uint8(8), uint8(1), []byte{9})
	f.Add(uint8(17), uint8(200), uint8(0), uint8(1), uint8(130), []byte{0, 255, 17, 4})
	f.Fuzz(func(t *testing.T, w8, h8, m8, limit8, flags uint8, pix []byte) {
		w, h := 1+int(w8), 1+int(h8)
		if len(pix) == 0 {
			pix = []byte{w8 ^ h8}
		}
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = pix[i%len(pix)] + uint8(i*7>>5)
		}
		cfg := DefaultConfig()
		cfg.Method = int(m8) % 7
		cfg.Segments = 1 + int(flags)%4
		cfg.Autofilter = flags&8 != 0
		cfg.FilterType = int(flags>>4) & 1
		cfg.Quality = float32(int(flags) * 100 / 255)
		if flags&0x40 != 0 {
			cfg.Pass, cfg.TargetSize = 3, w*h/4
		}
		c := wavefrontCase{name: "fuzz", img: img, cfg: cfg}
		if flags&0x80 != 0 {
			c.p0Limit = 1 << 14
		}
		want := c.encode(t)
		forceWavefrontForTest(t)
		defer workers.SetLimitForTest(workers.SetLimitForTest(1 + int(limit8)%9))
		c.cfg.ThreadLevel = true
		if got := c.encode(t); !bytes.Equal(got, want) {
			t.Fatalf("%dx%d m%d flags %#x: wavefront output differs", w, h, cfg.Method, flags)
		}
	})
}
