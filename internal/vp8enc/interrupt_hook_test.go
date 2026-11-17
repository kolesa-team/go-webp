package vp8enc

import (
	"bytes"
	"errors"
	"image"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// TestInterruptHook checks Config.Interrupt: never firing, it leaves the
// output unchanged; firing at any of its polls, Encode stops at once with
// ErrInterrupted.
func TestInterruptHook(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 67, 45)) // 3 macroblock rows
	rand.New(rand.NewSource(1)).Read(img.Pix)
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 0xff
	}
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	for _, pass := range []int{1, 3} {
		cfg := DefaultConfig()
		cfg.Pass, cfg.TargetSize = pass, 1500
		want, err := Encode(pic, &cfg)
		if err != nil {
			t.Fatal(err)
		}
		var polls atomic.Int64
		cfg.Interrupt = func() bool { polls.Add(1); return false }
		got, err := Encode(pic, &cfg)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("pass %d: a silent Interrupt changed the output (err %v)", pass, err)
		}
		n := polls.Load()
		if n < int64(pass*3) {
			t.Fatalf("pass %d: %d polls, want at least one per macroblock row and pass", pass, n)
		}
		for k := int64(1); k <= n; k++ {
			polls.Store(0)
			cfg.Interrupt = func() bool { return polls.Add(1) >= k }
			if _, err := Encode(pic, &cfg); !errors.Is(err, ErrInterrupted) {
				t.Fatalf("pass %d: interrupted at poll %d: got %v", pass, k, err)
			}
			if polls.Load() != k {
				t.Fatalf("pass %d: interrupted at poll %d, polled %d times", pass, k, polls.Load())
			}
		}
	}
}

// TestInterruptAnalysisBands fires Config.Interrupt while the analysis pass
// runs in row bands on helper goroutines (ThreadLevel): every budget gives
// ErrInterrupted, no helper is left running, a silent hook leaves the
// output byte-identical, and an encode stopped in the analysis pass never
// reaches the token loop.
func TestInterruptAnalysisBands(t *testing.T) {
	img := synthImage(512, 512, 1, 9) // 32x32 macroblocks: 5 bands at m4
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	cfg := DefaultConfig()
	cfg.ThreadLevel = true
	want, err := Encode(pic, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	mbH := int64((img.Rect.Dy() + 15) >> 4)
	budgetCases(t, func(name string) {
		base := workers.InUse() // 1 when the test holds the whole budget
		c := cfg
		var polls atomic.Int64
		c.Interrupt = func() bool { polls.Add(1); return false }
		got, err := Encode(pic, &c)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("%s: a silent Interrupt changed the output (err %v)", name, err)
		}
		// One poll per analysis row, one after the analysis, one per
		// token-loop row (single pass).
		if n := polls.Load(); n != 2*mbH+1 {
			t.Fatalf("%s: %d polls, want %d", name, n, 2*mbH+1)
		}
		for _, k := range []int64{1, 2, mbH / 2, mbH} {
			polls.Store(0)
			c.Interrupt = func() bool { return polls.Add(1) >= k }
			if _, err := Encode(pic, &c); !errors.Is(err, ErrInterrupted) {
				t.Fatalf("%s: interrupted at poll %d: got %v", name, k, err)
			}
			// The bands polling concurrently may each see the hook fire
			// once, but none goes on to the next row or to the token loop.
			if n := polls.Load(); n > k+maxAnalysisBands {
				t.Fatalf("%s: interrupted at poll %d, polled %d times", name, k, n)
			}
			// Encode waits for its helpers before it returns.
			if n := workers.InUse(); n != base {
				t.Fatalf("%s: %d helper slots taken after Encode, want %d", name, n, base)
			}
		}
	})
}
