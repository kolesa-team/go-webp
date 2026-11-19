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

package vp8l

import (
	"bytes"
	"image"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// withCacheMode runs f with the given lazy color cache settings.
func withCacheMode(maxBits, minRun int, f func()) {
	oldBits, oldRun := lazyCacheMaxBits, lazyCacheMinRun
	lazyCacheMaxBits, lazyCacheMinRun = maxBits, minRun
	defer func() { lazyCacheMaxBits, lazyCacheMinRun = oldBits, oldRun }()
	f()
}

// cacheModes are the color cache settings the differential tests compare:
// libwebp's eager insertion, the default, and lazy maintenance of every
// cache it supports with a backward settle for every pending range.
var cacheModes = []struct {
	name            string
	maxBits, minRun int
}{
	{"eager", 0, 1 << 30},
	{"default", lazyCacheMaxBits, lazyCacheMinRun},
	{"lazy6-settle1", 6, 1},
	{"lazy6-forward", 6, 1 << 30},
}

func TestSettleCacheMatchesInsert(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 2000; iter++ {
		bits := 1 + rng.Intn(6)
		shift := uint(32 - bits)
		colors := make([]uint32, 1+rng.Intn(40))
		for i := range colors {
			colors[i] = rng.Uint32()
		}
		px := make([]uint32, rng.Intn(300))
		for i := 0; i < len(px); {
			run := 1 + rng.Intn(8)
			c := colors[rng.Intn(len(colors))]
			for ; run > 0 && i < len(px); run-- {
				px[i] = c
				i++
			}
		}
		var a, b [1 << maxCacheBits]uint32
		for i := range a[:1<<bits] {
			a[i] = rng.Uint32()
		}
		b = a
		insertCache(&a, px, shift)
		settleCache(&b, px, shift)
		if a != b {
			t.Fatalf("iter %d: bits %d, %d pixels: settle differs from insert", iter, bits, len(px))
		}
	}
}

// cacheBitsOf returns the color cache bits of the main image of a VP8L
// payload (with header).
func cacheBitsOf(t *testing.T, payload []byte) int {
	t.Helper()
	w, h, _, err := DecodeHeader(payload)
	must(t, err)
	d := getDecoder()
	defer putDecoder(d)
	d.br.init(payload)
	d.br.read(8)
	d.br.read(32)
	d.out = outRGBA
	must(t, d.readLevel0Header(w, h))
	return d.md0.cacheBits
}

// fewColorImage returns a w x h image of a few colors in noisy stripes,
// which the encoder codes with a palette and a small color cache.
func fewColorImage(w, h, colors int, seed int64) *image.NRGBA {
	rng := rand.New(rand.NewSource(seed))
	pal := make([][4]byte, colors)
	for i := range pal {
		pal[i] = [4]byte{byte(rng.Intn(256)), byte(rng.Intn(256)), byte(rng.Intn(256)), byte(128 + rng.Intn(128))}
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			k := (x/7 + y/5 + rng.Intn(3)) % colors
			if rng.Intn(9) == 0 {
				k = rng.Intn(colors)
			}
			copy(img.Pix[y*img.Stride+4*x:], pal[k][:])
		}
	}
	return img
}

// cacheTestStreams returns VP8L payloads (with header) using color caches
// of various sizes, including images bigger than the LZ77 window.
func cacheTestStreams(t *testing.T) map[string][]byte {
	t.Helper()
	streams := map[string][]byte{}
	for name := range losslessFixtures {
		streams[name] = findChunk(readFixture(t, "lossless", name), "VP8L")
	}
	for _, c := range []struct {
		name      string
		w, h, col int
		method    int
	}{
		{"few3-97x61", 97, 61, 3, 4},
		{"few12-301x97", 301, 97, 12, 4},
		{"few40-200x150", 200, 150, 40, 6},
		{"few200-160x120", 160, 120, 200, 4},
		{"few5-1x700", 1, 700, 5, 4},
		{"few5-700x1", 700, 1, 5, 4},
		// Bigger than the LZ77 window (the buffer slides), cache bits 2.
		{"few40-1700x1100", 1700, 1100, 40, 4},
	} {
		if testing.Short() && c.w*c.h > 1<<20 {
			continue
		}
		file, err := vp8lenc.Encode(fewColorImage(c.w, c.h, c.col, int64(c.w*c.h)), &vp8lenc.Config{Quality: 75, Method: c.method})
		must(t, err, c.name)
		p := findChunk(file, "VP8L")
		if p == nil {
			p = file
		}
		streams[c.name] = p
	}
	return streams
}

// TestColorCacheModes checks that every color cache maintenance strategy
// decodes the same pixels, for whole decodes and for incremental alpha
// decodes in random batches.
func TestColorCacheModes(t *testing.T) {
	streams := cacheTestStreams(t)
	small := 0
	for name, payload := range streams {
		bits := cacheBitsOf(t, payload)
		if bits >= 1 && bits <= 4 {
			small++
		}
		t.Logf("%s: cache bits %d", name, bits)
		var want []byte
		var wantAlpha []byte
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for _, m := range cacheModes {
			withCacheMode(m.maxBits, m.minRun, func() {
				img, err := decodeVP8L(payload)
				must(t, err, name, m.name)
				if want == nil {
					want = img.Pix
				} else if !bytes.Equal(want, img.Pix) {
					t.Errorf("%s (cache bits %d): %s differs from %s", name, bits, m.name, cacheModes[0].name)
				}
				rng := rand.New(rand.NewSource(int64(len(payload))))
				got := decodeAlphaRandomBatches(t, payload[headerSize:], w, h, rng)
				if wantAlpha == nil {
					wantAlpha = got
				} else if !bytes.Equal(wantAlpha, got) {
					t.Errorf("%s (cache bits %d): incremental alpha decode %s differs", name, bits, m.name)
				}
			})
		}
		for i := range wantAlpha {
			if wantAlpha[i] != want[4*i+1] {
				t.Fatalf("%s: alpha decode differs from the green channel at %d", name, i)
			}
		}
	}
	if small < 3 {
		t.Errorf("only %d streams use a cache of 1 to 4 bits", small)
	}
}

// decodeAlphaRandomBatches decodes an alpha stream with AlphaDecoder,
// asking for random numbers of rows (sometimes none, sometimes going back).
func decodeAlphaRandomBatches(t *testing.T, payload []byte, w, h int, rng *rand.Rand) []byte {
	t.Helper()
	a, err := NewAlphaDecoder(payload, w, h)
	must(t, err)
	defer a.Close()
	dst := make([]byte, w*h)
	for y := 0; y < h; {
		next := y + rng.Intn(40) - 5
		if rng.Intn(4) == 0 {
			next = y + rng.Intn(h+1)
		}
		next = max(next, 0)
		must(t, a.DecodeRows(dst[:min(next, h)*w], next))
		y = max(y, min(next, h))
		check(t, a.Rows() == y, "rows", a.Rows(), y)
	}
	return dst
}

// FuzzDecodeCacheModes compares libwebp's eager color cache maintenance
// with lazy maintenance (backward settle of every pending range) on
// arbitrary streams: pixels and errors must match, also when decoding the
// stream as an alpha plane in random row batches.
func FuzzDecodeCacheModes(f *testing.F) {
	addSeeds(f, func(p []byte, _, _ int) { f.Add(p, int64(0)) })
	f.Fuzz(func(t *testing.T, data []byte, seed int64) {
		w, h, _, err := DecodeHeader(data)
		if err != nil || w*h > 1<<20 {
			return
		}
		type result struct {
			pix      []byte
			err      error
			alpha    []byte
			alphaErr error
		}
		run := func(maxBits, minRun int) (r result) {
			withCacheMode(maxBits, minRun, func() {
				img := image.NewNRGBA(image.Rect(0, 0, w, h))
				r.err = Decode(data, &rowSink{img: img, need: h})
				r.pix = img.Pix
				r.alpha, r.alphaErr = fuzzAlphaBatches(data[headerSize:], w, h, seed)
			})
			return r
		}
		a := run(0, 1<<30)
		b := run(6, 1)
		if (a.err == nil) != (b.err == nil) || (a.err != nil && a.err.Error() != b.err.Error()) {
			t.Fatalf("errors differ: eager %v, lazy %v", a.err, b.err)
		}
		if !bytes.Equal(a.pix, b.pix) {
			t.Fatal("pixels differ")
		}
		if (a.alphaErr == nil) != (b.alphaErr == nil) || (a.alphaErr != nil && a.alphaErr.Error() != b.alphaErr.Error()) {
			t.Fatalf("alpha errors differ: eager %v, lazy %v", a.alphaErr, b.alphaErr)
		}
		if !bytes.Equal(a.alpha, b.alpha) {
			t.Fatal("alpha planes differ")
		}
	})
}

// fuzzAlphaBatches decodes an alpha stream in pseudo-random row batches
// derived from seed and returns the plane and the first error.
func fuzzAlphaBatches(data []byte, w, h int, seed int64) ([]byte, error) {
	a, err := NewAlphaDecoder(data, w, h)
	if err != nil {
		return nil, err
	}
	defer a.Close()
	rng := rand.New(rand.NewSource(seed))
	dst := make([]byte, w*h)
	for y := 0; y < h; {
		y = min(y+1+rng.Intn(1+h/3), h)
		if err := a.DecodeRows(dst[:y*w], y); err != nil {
			return dst, err
		}
	}
	return dst, nil
}
