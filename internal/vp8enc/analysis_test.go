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
		"640x481": synthImage(640, 481, 3, 6), // 6 bands
	}
	methods := []int{0, 4}
	if !testing.Short() {
		imgs["16x4100"] = synthImage(16, 4100, 1, 7)
		imgs["source"] = loadTestImage(t, "source.jpg")
		methods = []int{0, 2, 4, 6}
	}
	for name, img := range imgs {
		for _, m := range methods {
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

// FuzzAnalysisBands forces every band count on small pictures (bands of a
// single macroblock row, uneven bands, more helpers than bands) and checks
// that the analysis decides what the single-band pass decides.
func FuzzAnalysisBands(f *testing.F) {
	f.Add(uint8(15), uint8(120), uint8(7), uint8(4), uint8(3), uint8(1), []byte{1, 2, 3})
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), []byte{})
	f.Add(uint8(70), uint8(255), uint8(1), uint8(0), uint8(9), uint8(3), []byte{255, 0, 17})
	f.Fuzz(func(t *testing.T, w8, h8, nb8, m8, limit8, seg8 uint8, pix []byte) {
		w, h := 1+int(w8)%80, 1+int(h8)%160
		mbH := (h + 15) >> 4
		nb := 1 + int(nb8)%min(maxAnalysisBands, mbH)
		if len(pix) == 0 {
			pix = []byte{w8 ^ h8}
		}
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = pix[i%len(pix)] + uint8(i*7>>5)
		}
		cfg := DefaultConfig()
		cfg.Method = int(m8) % 7
		cfg.Segments = 1 + int(seg8)%4
		cfg.EmulateJPEGSize = seg8&0x80 != 0
		analyzeWith := func(nb int) analysisResult {
			pic := FromNRGBA(img, false, false)
			defer pic.Release()
			e := new(encoder)
			e.setup(pic, &cfg)
			e.analyzeInBands(nb)
			return analysisResult{slices.Clone(e.mbInfo), slices.Clone(e.preds), e.alpha, e.uvAlpha, e.dqm}
		}
		want := analyzeWith(1)
		defer workers.SetLimitForTest(workers.SetLimitForTest(1 + int(limit8)%9))
		got := analyzeWith(nb)
		if !slices.Equal(got.mbInfo, want.mbInfo) || !bytes.Equal(got.preds, want.preds) ||
			got.alpha != want.alpha || got.uvAlpha != want.uvAlpha || got.dqm != want.dqm {
			t.Fatalf("%dx%d m%d, %d bands: analysis differs from the single-band pass", w, h, cfg.Method, nb)
		}
	})
}

func TestAnalysisBandsCount(t *testing.T) {
	for _, tc := range []struct {
		w, h, method, want int
	}{
		{1, 1, 4, 1}, {16, 16 * 300, 4, 1}, {64 * 16, 3 * 16, 4, 1}, {64 * 16, 6 * 16, 4, 2},
		{960, 1280, 4, 8}, {4000, 3000, 6, 8}, {16 * 400, 16, 2, 1}, {16 * 192, 32, 4, 2},
		// methods 0-1: bands of at least 512 macroblocks
		{64 * 16, 6 * 16, 0, 1}, {640, 480, 1, 2}, {960, 1280, 0, 8}, {16 * 512, 32, 0, 2},
	} {
		e := &encoder{mbW: (tc.w + 15) >> 4, mbH: (tc.h + 15) >> 4, method: tc.method}
		e.cfg.ThreadLevel = true
		if got := e.analysisBands(); got != tc.want {
			t.Errorf("%dx%d m%d: %d bands, want %d", tc.w, tc.h, tc.method, got, tc.want)
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
		callerIn := make(chan struct{})
		var once, callerOnce sync.Once
		testHookAnalysisBand = func(int) {
			if goroutineID() != caller {
				once.Do(func() { close(helperStarted) })
				if onHelper {
					panic(boom)
				}
				// Hold this band until the caller has taken one: the
				// caller starts its own bands only after spawning the
				// helpers, and on a slow machine the helpers could
				// otherwise claim all 5 bands first (at most 4 helpers
				// hold one each here, so one band is left for it).
				select {
				case <-callerIn:
				case <-time.After(5 * time.Second):
					t.Error("the caller analyzed no band")
				}
				return
			}
			callerOnce.Do(func() { close(callerIn) })
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
