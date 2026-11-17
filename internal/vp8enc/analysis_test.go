package vp8enc

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// analysisResult is everything the analysis pass decides.
type analysisResult struct {
	mbInfo         []mbInfo
	preds          []uint8
	alpha, uvAlpha int
	dqm            [numMBSegments]segmentInfo
}

func analyzeOnly(img *image.NRGBA, cfg Config) analysisResult {
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	e := new(encoder)
	e.setup(pic, &cfg)
	e.analyze()
	return analysisResult{slices.Clone(e.mbInfo), slices.Clone(e.preds), e.alpha, e.uvAlpha, e.dqm}
}

// budgetCases runs f under helper budgets that give the analysis pass no
// helper at all (budget fully taken), one helper and more helpers than
// bands. ThreadLevel is set by f.
func budgetCases(t *testing.T, f func(name string)) {
	t.Helper()
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	for _, limit := range []int{1, 3, 16} {
		workers.SetLimitForTest(limit)
		f("limit" + strconv.Itoa(limit))
	}
	// budget fully taken: Spawn returns nil
	workers.SetLimitForTest(1)
	if !workers.TryAcquire() {
		t.Fatal("cannot take the budget")
	}
	f("exhausted")
	workers.Release()
}

// TestAnalysisBandsIdentical checks that the banded analysis (ThreadLevel)
// decides exactly what the sequential pass decides, whatever the number of
// helpers, and that the encoded frames are byte-identical.
func TestAnalysisBandsIdentical(t *testing.T) {
	imgs := map[string]*image.NRGBA{
		"1x1":     synthImage(1, 1, 1, 1),
		"1x300":   synthImage(1, 300, 0, 2),
		"300x1":   synthImage(300, 1, 1, 3),
		"17x33":   synthImage(17, 33, 3, 4),
		"100x75":  synthImage(100, 75, 2, 5),
		"640x481": synthImage(640, 481, 3, 6),
		"16x4100": synthImage(16, 4100, 1, 7),
	}
	if !testing.Short() {
		imgs["source"] = loadTestImage(t, "source.jpg")
	}
	for name, img := range imgs {
		for _, m := range []int{0, 2, 4, 6} {
			cfg := DefaultConfig()
			cfg.Method = m
			want := analyzeOnly(img, cfg)
			wantData := encodeNRGBA(t, img, cfg)
			cfg.ThreadLevel = true
			budgetCases(t, func(budget string) {
				got := analyzeOnly(img, cfg)
				if !slices.Equal(got.mbInfo, want.mbInfo) || !bytes.Equal(got.preds, want.preds) ||
					got.alpha != want.alpha || got.uvAlpha != want.uvAlpha || got.dqm != want.dqm {
					t.Errorf("%s m%d %s: banded analysis differs", name, m, budget)
				}
				if data := encodeNRGBA(t, img, cfg); !bytes.Equal(data, wantData) {
					t.Errorf("%s m%d %s: encoded frame differs", name, m, budget)
				}
			})
		}
	}
}

func TestAnalysisBandsCount(t *testing.T) {
	for _, tc := range []struct {
		w, h, want int
	}{
		{1, 1, 1}, {16, 16 * 300, 1}, {64 * 16, 3 * 16, 1}, {64 * 16, 6 * 16, 2},
		{960, 1280, 8}, {4000, 3000, 8}, {16 * 400, 16, 1}, {16 * 192, 32, 2},
	} {
		e := &encoder{mbW: (tc.w + 15) >> 4, mbH: (tc.h + 15) >> 4}
		e.cfg.ThreadLevel = true
		if got := e.analysisBands(); got != tc.want {
			t.Errorf("%dx%d: %d bands, want %d", tc.w, tc.h, got, tc.want)
		}
		e.cfg.ThreadLevel = false
		if got := e.analysisBands(); got != 1 {
			t.Errorf("%dx%d without ThreadLevel: %d bands", tc.w, tc.h, got)
		}
	}
}

// goroutineID returns the id of the calling goroutine (tests only).
func goroutineID() int {
	var buf [64]byte
	s := strings.TrimPrefix(string(buf[:runtime.Stack(buf[:], false)]), "goroutine ")
	id, _ := strconv.Atoi(s[:strings.IndexByte(s, ' ')])
	return id
}

// waitHelpersIdle waits until no helper goroutine of the budget runs.
func waitHelpersIdle(t *testing.T) {
	t.Helper()
	for i := 0; workers.InUse() != 0; i++ {
		if i == 500 {
			t.Fatalf("%d helper goroutines still running", workers.InUse())
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// TestAnalysisBandsPanic checks that a panic in a band, on a helper or on
// the calling goroutine, surfaces on the caller of Encode once all helpers
// have stopped, and that no helper slot or goroutine leaks.
func TestAnalysisBandsPanic(t *testing.T) {
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	defer func() { testHookAnalysisBand = nil }()
	img := synthImage(512, 512, 1, 9) // 32x32 MBs: 5 bands
	cfg := DefaultConfig()
	cfg.ThreadLevel = true
	base := runtime.NumGoroutine()
	boom := errors.New("boom")

	for _, onHelper := range []bool{true, false} {
		caller := goroutineID()
		helperStarted := make(chan struct{})
		var once sync.Once
		testHookAnalysisBand = func(int) {
			if goroutineID() != caller {
				once.Do(func() { close(helperStarted) })
				if onHelper {
					panic(boom)
				}
				return
			}
			// Make sure a helper takes part before the caller goes on.
			select {
			case <-helperStarted:
			case <-time.After(5 * time.Second):
				t.Error("no helper started")
			}
			if !onHelper {
				panic(boom)
			}
		}
		func() {
			defer func() {
				r := recover()
				if onHelper {
					pe, ok := r.(*workers.PanicError)
					if !ok || !errors.Is(pe, boom) {
						t.Fatalf("helper panic: recovered %v, want *workers.PanicError(boom)", r)
					}
				} else if r != boom {
					t.Fatalf("caller panic: recovered %v, want boom", r)
				}
			}()
			pic := FromNRGBA(img, false, false)
			defer pic.Release()
			Encode(pic, &cfg)
			t.Fatal("Encode did not panic")
		}()
		waitHelpersIdle(t)
	}
	testHookAnalysisBand = nil

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

// TestAnalysisBandsConcurrent runs many banded encodes at once (helpers
// compete for the budget, some encodes get none) and checks every output.
func TestAnalysisBandsConcurrent(t *testing.T) {
	defer workers.SetLimitForTest(workers.SetLimitForTest(3))
	img := synthImage(400, 300, 3, 11)
	cfg := DefaultConfig()
	want := encodeNRGBA(t, img, cfg)
	cfg.ThreadLevel = true
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 4; i++ {
				pic := FromNRGBA(img, false, false)
				got, err := Encode(pic, &cfg)
				pic.Release()
				if err != nil || !bytes.Equal(got, want) {
					errs <- fmt.Errorf("concurrent banded encode differs (err %v)", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	waitHelpersIdle(t)
}
