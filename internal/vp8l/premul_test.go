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
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// plainSink hides the optional interfaces of the sink it wraps.
type plainSink struct{ pipeline.Sink }

// premulSpy forwards AcceptPremultiplied and records whether the decoder
// took over the premultiplication.
type premulSpy struct {
	pipeline.Sink
	accepted bool
}

func (s *premulSpy) AcceptPremultiplied() bool {
	s.accepted = s.Sink.(pipeline.PremultipliedSink).AcceptPremultiplied()
	return s.accepted
}

// TestPalettePremultiply checks that scaled decodes of palette images, which
// premultiply the palette instead of the pixels, give the same pixels as
// premultiplying every pixel in the sink.
func TestPalettePremultiply(t *testing.T) {
	streams := map[string][]byte{}
	for name := range losslessFixtures {
		streams[name] = findChunk(readFixture(t, "lossless", name), "VP8L")
	}
	for _, c := range []struct {
		name      string
		w, h, col int
	}{{"pal2", 67, 45, 2}, {"pal3", 45, 67, 3}, {"pal11", 90, 33, 11}, {"pal100", 70, 70, 100}} {
		file, err := vp8lenc.Encode(fewColorImage(c.w, c.h, c.col, 7), &vp8lenc.Config{Quality: 75, Method: 4})
		must(t, err)
		if p := findChunk(file, "VP8L"); p != nil {
			file = p
		}
		streams[c.name] = file
	}
	palettes := 0
	for name, payload := range streams {
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for _, sc := range [][2]int{{max(w/3, 1), max(h/3, 1)}, {w + 5, h/2 + 1}} {
			p := pipeline.Params{SrcW: w, SrcH: h, ScaleW: sc[0], ScaleH: sc[1], Premultiply: true}
			ref, refImg, err := pipeline.NewSink(p)
			must(t, err)
			must(t, Decode(payload, plainSink{ref}), name)
			s, img, err := pipeline.NewSink(p)
			must(t, err)
			spy := &premulSpy{Sink: s}
			must(t, Decode(payload, spy), name)
			if !bytes.Equal(refImg().Pix, img().Pix) {
				t.Errorf("%s scaled to %v: palette premultiplication changes the output", name, sc)
			}
			if spy.accepted {
				palettes++
			}
		}
	}
	if palettes < 8 {
		t.Errorf("premultiplied palette used in only %d decodes", palettes)
	}
}
