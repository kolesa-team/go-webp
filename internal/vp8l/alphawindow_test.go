package vp8l

import (
	"bytes"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// TestAlphaDecoderWindow decodes alpha streams with DecodeRowsAt into
// windows that hold only the rows of each batch (plus the previous row, as
// a caller that unfilters keeps it), in fresh buffers, and checks the plane
// against DecodeAlpha, on both the 8-bit (palette) and the 32-bit paths,
// with and without a sliding LZ77 window.
func TestAlphaDecoderWindow(t *testing.T) {
	type stream struct {
		name    string
		payload []byte
		w, h    int
	}
	var streams []stream
	for name := range alphaFixtures {
		hdr, payload, w, h, ok := alphaFromFile(readFixture(t, "alpha", name))
		if !ok || hdr&3 != 1 {
			t.Fatal("bad ALPH chunk")
		}
		streams = append(streams, stream{name, payload, w, h})
	}
	const bw, bh = 900, 700
	few := make([]byte, bw*bh)
	many := make([]byte, bw*bh)
	for y := 0; y < bh; y++ {
		for x := 0; x < bw; x++ {
			few[y*bw+x] = byte((x*x/37 + y*y/23) % 20 * 12)
			many[y*bw+x] = byte(x*y/97) + byte((x*7+y*13)%5)
		}
	}
	for _, p := range []struct {
		name   string
		plane  []byte
		method int
	}{{"big-palette", few, 0}, {"big-noise", many, 1}} {
		enc, err := vp8lenc.EncodeAlpha(p.plane, bw, bh, bw, p.method, 100)
		must(t, err)
		streams = append(streams, stream{p.name, enc, bw, bh})
	}
	for _, s := range streams {
		want := make([]byte, s.w*s.h)
		must(t, DecodeAlpha(s.payload, s.w, s.h, want), s.name)
		for _, batches := range [][]int{{3, 16, 1}, {16}, {17, 40}, {s.h}} {
			a, err := NewAlphaDecoder(s.payload, s.w, s.h)
			must(t, err)
			got := make([]byte, s.w*s.h)
			w := s.w
			for i, y := 0, 0; y < s.h; i++ {
				end := min(y+batches[i%len(batches)], s.h)
				first := max(y-1, 0)
				win := bytes.Repeat([]byte{0xa5}, (end-first)*w)
				copy(win, got[first*w:y*w]) // the previous row, as a caller keeps it
				must(t, a.DecodeRowsAt(win, first, end))
				check(t, bytes.Equal(win[:(y-first)*w], got[first*w:y*w]), s.name+": a row before Rows() was written")
				copy(got[y*w:end*w], win[(y-first)*w:])
				y = end
			}
			a.Close()
			if !bytes.Equal(got, want) {
				t.Fatalf("%s %v: windowed decode differs", s.name, batches)
			}
		}
		// A window that starts after Rows() is refused.
		a, err := NewAlphaDecoder(s.payload, s.w, s.h)
		must(t, err)
		check(t, a.DecodeRowsAt(make([]byte, s.w), 1, 2) != nil, s.name+": window after Rows() accepted")
		a.Close()
	}
}
