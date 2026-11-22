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
	"os"
	"path/filepath"
	"testing"
)

// discardSink accepts rows into a reused buffer, so that fuzzing large
// declared dimensions does not allocate a full image.
type discardSink struct {
	w   int
	buf []byte
}

func (s *discardSink) Buffer(y, n int) []byte {
	need := n * 4 * s.w
	if cap(s.buf) < need {
		s.buf = make([]byte, need)
	}
	return s.buf[:need]
}
func (s *discardSink) Stride() int           { return 4 * s.w }
func (s *discardSink) Commit(y, n int) error { return nil }
func (s *discardSink) NeedRows() int         { return 1 << 30 }

func addSeeds(f *testing.F, add func(payload []byte, w, h int)) {
	for _, dir := range []string{"lossless", "alpha"} {
		files, _ := filepath.Glob(filepath.Join("testdata", dir, "*.webp"))
		for _, name := range files {
			b, err := os.ReadFile(name)
			if err != nil {
				f.Fatal(err)
			}
			if p := findChunk(b, "VP8L"); p != nil {
				w, h, _, _ := DecodeHeader(p)
				add(p, w, h)
			}
			if _, p, w, h, ok := alphaFromFile(b); ok {
				hdr := []byte{signature, byte(w - 1), byte((w-1)>>8&0x3f | (h-1)<<6), byte((h - 1) >> 2), byte((h-1)>>10&0xf | 0x10)}
				add(append(hdr, p...), w, h)
			}
		}
	}
}

func FuzzDecode(f *testing.F) {
	addSeeds(f, func(p []byte, _, _ int) { f.Add(p) })
	f.Fuzz(func(t *testing.T, data []byte) {
		w, h, _, err := DecodeHeader(data)
		if err != nil || w*h > 1<<22 {
			return
		}
		_ = Decode(data, &discardSink{w: w})
	})
}

func FuzzDecodeAlpha(f *testing.F) {
	addSeeds(f, func(p []byte, w, h int) { f.Add(p[headerSize:], uint16(w), uint16(h)) })
	f.Fuzz(func(t *testing.T, data []byte, w, h uint16) {
		if w == 0 || h == 0 || int(w)*int(h) > 1<<22 {
			return
		}
		dst := make([]byte, int(w)*int(h))
		err := DecodeAlpha(data, int(w), int(h), dst)
		// Decoding in row batches (as the WebP decoder does) must give the
		// same result.
		a, err2 := NewAlphaDecoder(data, int(w), int(h))
		if err2 == nil {
			inc := make([]byte, len(dst))
			step := 1 + int(h)/7
			for y := 0; y < int(h) && err2 == nil; {
				y = min(y+step, int(h))
				err2 = a.DecodeRows(inc[:y*int(w)], y)
			}
			a.Close()
			if (err == nil) != (err2 == nil) {
				t.Fatalf("full decode: %v, incremental: %v", err, err2)
			}
			if err == nil && !bytes.Equal(dst, inc) {
				t.Fatal("incremental decode differs")
			}
		} else if err == nil {
			t.Fatalf("incremental init failed: %v", err2)
		}
	})
}
