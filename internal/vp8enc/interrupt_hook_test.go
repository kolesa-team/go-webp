package vp8enc

import (
	"bytes"
	"errors"
	"image"
	"math/rand"
	"sync/atomic"
	"testing"
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
