package vp8

import (
	"image"
	"os"
	"path/filepath"
	"testing"
)

// discardSink accepts any rows into a reused buffer, so fuzzing huge declared
// dimensions does not allocate a full frame.
type discardSink struct {
	buf    []byte
	stride int
	height int
	next   int
}

func (s *discardSink) Buffer(y, n int) []byte {
	if y != s.next || n <= 0 || y+n > s.height {
		panic("sink protocol violation")
	}
	if cap(s.buf) < n*s.stride {
		s.buf = make([]byte, n*s.stride)
	}
	return s.buf[:n*s.stride]
}
func (s *discardSink) Stride() int { return s.stride }
func (s *discardSink) Commit(y, n int) error {
	s.next = y + n
	return nil
}
func (s *discardSink) NeedRows() int { return s.height }

func FuzzDecode(f *testing.F) {
	files, _ := filepath.Glob("testdata/*.webp")
	for _, p := range files {
		data, err := os.ReadFile(p)
		if err != nil {
			f.Fatal(err)
		}
		if ri, err := extractVP8(data); err == nil {
			f.Add(ri.vp8, uint8(0))
			f.Add(ri.vp8, uint8(16|4|2))
			f.Add(ri.vp8, uint8(4))
			f.Add(ri.vp8, uint8(32|8|4))
		}
	}
	f.Fuzz(func(t *testing.T, data []byte, mode uint8) {
		w, h, err := DecodeHeader(data)
		if err != nil {
			return
		}
		if w <= 0 || h <= 0 || w > 16383 || h > 16383 {
			t.Fatalf("bad dimensions %dx%d", w, h)
		}
		opts := &Options{
			BypassFiltering:   mode&1 != 0,
			NoFancyUpsampling: mode&2 != 0,
			UseThreads:        mode&4 != 0,
		}
		if mode&8 != 0 {
			opts.DitheringStrength = 100
		}
		if opts.UseThreads {
			// Small batches, so that small inputs use the helper too.
			setBatch(t, 1+47*int(mode>>5&1))
		}
		if mode&16 != 0 {
			r := image.Rect(w/3, h/3, w/3+w/2+1, h/3+h/2+1).Intersect(image.Rect(0, 0, w, h))
			r.Min.X &^= 1
			r.Min.Y &^= 1
			s := &discardSink{stride: 4 * r.Dx(), height: r.Dy()}
			if err := DecodeRegion(data, nil, opts, r, s); err == nil && s.next != r.Dy() {
				t.Fatalf("region: success but only %d of %d rows", s.next, r.Dy())
			}
			return
		}
		s := &discardSink{stride: 4 * w, height: h}
		if err := Decode(data, nil, opts, s); err == nil && s.next != h {
			t.Fatalf("success but only %d of %d rows", s.next, h)
		}
	})
}
