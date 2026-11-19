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

//go:build arm64 && !purego

package vp8enc

import (
	"bytes"
	"image"
	"math/rand"
	"testing"
)

// encodeBoth encodes pic with the NEON kernels and with the Go reference
// code (neonEnabled cleared) and fails if the bitstreams differ.
func encodeBoth(t *testing.T, img *image.NRGBA, sharp, exact bool, cfg Config) {
	t.Helper()
	enc := func(neon bool) []byte {
		neonEnabled = neon
		defer func() { neonEnabled = true }()
		pic := FromNRGBA(img, sharp, exact)
		defer pic.Release()
		c := cfg
		out, err := Encode(pic, &c)
		if err != nil {
			t.Fatalf("Encode(%+v): %v", cfg, err)
		}
		return out
	}
	a, b := enc(true), enc(false)
	if !bytes.Equal(a, b) {
		t.Fatalf("%dx%d sharp=%v exact=%v %+v: NEON output (%d bytes) differs from the Go reference (%d bytes)",
			img.Rect.Dx(), img.Rect.Dy(), sharp, exact, cfg, len(a), len(b))
	}
}

// TestEncodeNEONMatchesGo encodes random pictures with random settings
// both ways (the kernels are bit-exact, so the output must not change).
func TestEncodeNEONMatchesGo(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	n := 60
	if testing.Short() {
		n = 12
	}
	for i := 0; i < n; i++ {
		w, h := 1+rng.Intn(120), 1+rng.Intn(90)
		img := synthImage(w, h, rng.Intn(4), int64(i))
		cfg := DefaultConfig()
		cfg.Method = rng.Intn(7)
		cfg.Quality = float32(rng.Intn(101))
		cfg.SNSStrength = rng.Intn(101)
		cfg.FilterStrength = rng.Intn(101)
		cfg.Segments = 1 + rng.Intn(4)
		cfg.ThreadLevel = rng.Intn(2) == 0
		encodeBoth(t, img, rng.Intn(3) == 0, rng.Intn(2) == 0, cfg)
	}
}

func FuzzEncodeNEONMatchesGo(f *testing.F) {
	f.Add([]byte{17, 9, 75, 4, 50, 60, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{40, 40, 100, 6, 100, 0, 3, 1, 255, 0, 255, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 9 {
			return
		}
		w := 1 + int(data[0])%48
		h := 1 + int(data[1])%48
		cfg := DefaultConfig()
		cfg.Quality = float32(data[2]) / 2.55
		cfg.Method = int(data[3]) % 7
		cfg.SNSStrength = int(data[4]) % 101
		cfg.FilterStrength = int(data[5]) % 101
		cfg.Segments = 1 + int(data[6])%4
		pix := data[8:]
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = pix[i%len(pix)] + byte(i/len(pix))*data[7]
		}
		encodeBoth(t, img, data[7]&1 != 0, data[7]&2 != 0, cfg)
	})
}
