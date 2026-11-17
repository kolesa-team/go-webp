package vp8lenc

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"sync/atomic"
	"testing"
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
