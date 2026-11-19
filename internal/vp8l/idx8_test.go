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
	"encoding/binary"
	"fmt"
	"image"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// withIdx8 runs f with the byte path of palette images enabled or not, and
// with a forced switch to 32 bits at level-0 pixel widenAt (0: none).
func withIdx8(enabled bool, widenAt int, f func()) {
	oldEnabled, oldAt := idx8Enabled, idx8TestWidenAt
	idx8Enabled, idx8TestWidenAt = enabled, widenAt
	defer func() { idx8Enabled, idx8TestWidenAt = oldEnabled, oldAt }()
	f()
}

// idx8Eligible reports whether the main image of a VP8L payload (with
// header) is decoded as index bytes.
func idx8Eligible(t *testing.T, payload []byte) bool {
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
	_, ok := d.idx8ARB()
	return ok
}

// idx8Streams returns VP8L payloads (with header) of palette images: the
// fixtures and encoded images of 2 to 200 colors (bundled 8, 4, 2 and 1
// pixels per byte), of odd sizes, with and without color cache, and bigger
// than the LZ77 window.
func idx8Streams(t *testing.T) map[string][]byte {
	t.Helper()
	streams := map[string][]byte{}
	for name, p := range cacheTestStreams(t) {
		if idx8Eligible(t, p) {
			streams[name] = p
		}
	}
	for _, c := range []struct {
		w, h, colors, method int
	}{
		{1, 1, 2, 4}, {1, 77, 2, 4}, {77, 1, 3, 4}, {5, 5, 3, 6}, {33, 17, 2, 4},
		{129, 67, 3, 4}, {129, 67, 11, 6}, {67, 129, 17, 4}, {250, 40, 200, 4},
		// Bigger than the LZ77 window: 4 indices per byte, 1 per byte.
		{4100, 1700, 3, 4}, {1200, 1500, 30, 4},
	} {
		if testing.Short() && c.w*c.h > 1<<20 {
			continue
		}
		name := fmt.Sprintf("pal%d-%dx%d-m%d", c.colors, c.w, c.h, c.method)
		file, err := vp8lenc.Encode(fewColorImage(c.w, c.h, c.colors, int64(c.w+c.h)), &vp8lenc.Config{Quality: 75, Method: c.method})
		must(t, err, name)
		p := findChunk(file, "VP8L")
		if p == nil {
			p = file
		}
		streams[name] = p
	}
	n := 0
	for name, p := range streams {
		if idx8Eligible(t, p) {
			n++
		} else {
			delete(streams, name)
		}
	}
	if n < 10 {
		t.Fatalf("only %d palette streams are decoded as bytes", n)
	}
	return streams
}

// TestIdx8MatchesWide checks that decoding palette images as index bytes
// gives the same output as the 32-bit path, with every kind of sink, with a
// switch to 32 bits at various positions, and in a two-stage decode (with
// -short, started after a probe only for one switch position).
func TestIdx8MatchesWide(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	for name, payload := range idx8Streams(t) {
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for _, sc := range asyncSinks {
			var want []byte
			withIdx8(false, 0, func() {
				s, out := sc.make(w, h)
				must(t, Decode(payload, s), name, sc.name)
				want = out()
			})
			for _, at := range []int{0, 1, w - 1, 16*w + 3, w * h / 3, w*h - 1} {
				for _, m := range threadModes {
					if m.probe > 0 && testing.Short() && at != w*h/3 {
						continue
					}
					withIdx8(true, at, func() {
						withAsync(w*5+1, m.probe, func(*int) {
							s, out := sc.make(w, h)
							var err error
							if m.threads {
								err = DecodeWithOptions(payload, s, &Options{UseThreads: true})
							} else {
								err = Decode(payload, s)
							}
							must(t, err, name, sc.name, at, m.name)
							if !bytes.Equal(want, out()) {
								t.Errorf("%s %s widen at %d %s: output differs from the 32-bit path", name, sc.name, at, m.name)
							}
						})
					})
				}
			}
		}
	}
}

// TestIdx8Truncated checks that truncated palette streams fail with the
// same error, after committing the same rows, as with the 32-bit path.
func TestIdx8Truncated(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	streams := idx8Streams(t)
	for _, name := range []string{"pal200_257x129_z8.webp", "pal2_257x129_z6.webp", "pal3_1x1000_z6.webp", "few3-97x61", "pal11-129x67-m6"} {
		payload := streams[name]
		if payload == nil {
			t.Fatalf("%s: not a palette stream", name)
		}
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for n := headerSize; n < len(payload); n += 1 + len(payload)/41 {
			p := payload[:n]
			ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
			var wantErr error
			withIdx8(false, 0, func() { wantErr = Decode(p, ref) })
			for _, m := range threadModes {
				withAsync(w*3+1, m.probe, func(*int) {
					s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
					err := DecodeWithOptions(p, s, &Options{UseThreads: m.threads})
					if fmt.Sprint(err) != fmt.Sprint(wantErr) {
						t.Errorf("%s[:%d] %s: error %v, want %v", name, n, m.name, err, wantErr)
					}
					if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
						t.Errorf("%s[:%d] %s: committed %d rows, want %d (or pixels differ)", name, n, m.name, s.next, ref.next)
					}
				})
			}
		}
	}
}

// TestIdx8UnsetCacheSlot checks streams that look up a color cache slot
// never set (the pixel 0, which has no index form): the decoder switches to
// 32 bits there and decodes exactly what the 32-bit path does. The streams
// are fixtures with one bit flipped.
func TestIdx8UnsetCacheSlot(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	for _, c := range []struct {
		name     string
		pos, bit int
		fails    bool
	}{
		{"pal200_64x64_z6.webp", 679, 0, false},
		{"pal200_64x64_z6.webp", 678, 2, true},
		{"pal200_257x129_z8.webp", 828, 0, false},
	} {
		payload := append([]byte(nil), findChunk(readFixture(t, "lossless", c.name), "VP8L")...)
		payload[c.pos] ^= 1 << c.bit
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
		var wantErr error
		withIdx8(false, 0, func() { wantErr = Decode(payload, ref) })
		check(t, (wantErr != nil) == c.fails, c.name, wantErr)
		for _, m := range threadModes {
			withAsync(w*2+1, m.probe, func(*int) {
				before := idx8Widens.Load()
				s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
				err := DecodeWithOptions(payload, s, &Options{UseThreads: m.threads})
				if idx8Widens.Load() == before {
					t.Errorf("%s flip %d.%d: no switch to 32 bits", c.name, c.pos, c.bit)
				}
				if fmt.Sprint(err) != fmt.Sprint(wantErr) {
					t.Errorf("%s flip %d.%d %s: error %v, want %v", c.name, c.pos, c.bit, m.name, err, wantErr)
				}
				if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
					t.Errorf("%s flip %d.%d %s: output differs from the 32-bit path", c.name, c.pos, c.bit, m.name)
				}
			})
		}
	}
}

// TestIdx8CacheStart checks the initial color cache of the byte path: every
// slot holds the pixel 0, which is the index 0 when alpha, red and blue are
// 0 and has no index form otherwise.
func TestIdx8CacheStart(t *testing.T) {
	d := getDecoder()
	defer putDecoder(d)
	d.md0.cacheBits = 3
	d.width, d.height, d.xsize = 4, 4, 4
	for _, arb := range []uint32{0, 0xff000000, 0x00010000} {
		d.initIdx8(arb)
		want := uint16(0)
		if arb == 0 {
			want = cacheSet
		}
		for i, v := range d.cache8[:8] {
			if v != want {
				t.Errorf("arb %08x: slot %d starts as %#x, want %#x", arb, i, v, want)
			}
		}
		for i, s := range d.hash8 {
			if want := uint16((arb | uint32(i)<<8) * colorCacheMult >> 29); s != want {
				t.Fatalf("arb %08x: index %d hashes to %d, want %d", arb, i, s, want)
			}
		}
	}
}

// FuzzDecodeIdx8 compares the byte path of palette images (with a switch to
// 32 bits at a fuzzed position) with the 32-bit path on arbitrary streams:
// the same error, the same rows committed and the same pixels.
func FuzzDecodeIdx8(f *testing.F) {
	addSeeds(f, func(p []byte, _, _ int) { f.Add(p, uint16(0), uint32(0)) })
	f.Fuzz(func(t *testing.T, data []byte, need uint16, widenAt uint32) {
		w, h, _, err := DecodeHeader(data)
		if err != nil || w*h > 1<<20 {
			return
		}
		n := h
		if need > 0 {
			n = min(int(need), h)
		}
		ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: n}
		var wantErr error
		withIdx8(false, 0, func() { wantErr = Decode(data, ref) })
		withIdx8(true, int(widenAt%uint32(w*h+1)), func() {
			s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: n}
			err := Decode(data, s)
			if fmt.Sprint(err) != fmt.Sprint(wantErr) {
				t.Fatalf("error %v, want %v", err, wantErr)
			}
			if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
				t.Fatalf("committed %d rows, want %d (or pixels differ)", s.next, ref.next)
			}
		})
	})
}

// TestPaletteEmit checks the fused palette output kernels (index bytes and
// 32-bit pixels, bundled or not) against inverseColorIndex + toRGBA, for
// odd row widths and a destination shorter than the row.
func TestPaletteEmit(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for iter := 0; iter < 400; iter++ {
		bits := rng.Intn(4)
		width := 1 + rng.Intn(70)
		tr := randomPaletteTransform(rng, width, bits)
		pw := subSampleSize(width, bits)
		src := make([]byte, pw)
		src32 := make([]uint32, pw)
		for i := range src {
			src[i] = byte(rng.Intn(256))
			src32[i] = rng.Uint32()&0xffff00ff | uint32(src[i])<<8
		}
		colors := make([]uint32, width)
		inverseColorIndex(tr, 0, 1, src32, colors)
		want := make([]byte, 4*width)
		for i, c := range colors {
			binary.LittleEndian.PutUint32(want[4*i:], toRGBA(c))
		}
		short := rng.Intn(width + 1) // a destination with fewer pixels
		for _, n := range []int{width, short} {
			for _, idx8 := range []bool{false, true} {
				d := getDecoder()
				got := make([]byte, 4*n+8)
				for i := range got {
					got[i] = 0xa5
				}
				if idx8 {
					d.emitPaletteIdx8(tr, src, got[:4*n])
				} else {
					d.emitPalette(tr, src32, got[:4*n])
				}
				putDecoder(d)
				if !bytes.Equal(got[:4*n], want[:4*n]) {
					t.Fatalf("iter %d: bits %d width %d dst %d px idx8 %v: wrong pixels", iter, bits, width, n, idx8)
				}
				for _, b := range got[4*n:] {
					if b != 0xa5 {
						t.Fatalf("iter %d: bits %d width %d dst %d px idx8 %v: wrote past the destination", iter, bits, width, n, idx8)
					}
				}
			}
		}
	}
}

func BenchmarkPaletteEmit(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for bits := 0; bits <= 3; bits++ {
		const width = 3000
		tr := randomPaletteTransform(rng, width, bits)
		src := make([]byte, subSampleSize(width, bits))
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		dst := make([]byte, 4*width)
		b.Run(fmt.Sprintf("bits%d", bits), func(b *testing.B) {
			d := getDecoder()
			defer putDecoder(d)
			b.SetBytes(4 * width)
			for i := 0; i < b.N; i++ {
				d.emitPaletteIdx8(tr, src, dst)
			}
		})
	}
}

// TestSettleCache8MatchesInsert checks the backward settle of index bytes,
// which skips runs of equal bytes, against forward insertion.
func TestSettleCache8MatchesInsert(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 3000; iter++ {
		bits := 1 + rng.Intn(6)
		var hash [256]uint16
		arb := rng.Uint32() &^ 0xff00
		for i := range hash {
			hash[i] = uint16((arb | uint32(i)<<8) * colorCacheMult >> (32 - bits))
		}
		values := make([]byte, 1+rng.Intn(12))
		for i := range values {
			values[i] = byte(rng.Intn(256))
		}
		px := make([]byte, rng.Intn(400))
		for i := 0; i < len(px); {
			run := 1 + rng.Intn(8)
			if rng.Intn(4) == 0 {
				run = 1 + rng.Intn(100)
			}
			c := values[rng.Intn(len(values))]
			for ; run > 0 && i < len(px); run-- {
				px[i] = c
				i++
			}
		}
		var a, b [1 << maxCacheBits]uint16
		for i := range a[:1<<bits] {
			a[i] = uint16(rng.Intn(0x200))
		}
		b = a
		insertCache8(&a, &hash, px)
		settleCache8(&b, &hash, px, ^uint64(0)>>(64-1<<bits))
		if a != b {
			t.Fatalf("iter %d: bits %d, %d pixels: settle differs from insert", iter, bits, len(px))
		}
	}
}
