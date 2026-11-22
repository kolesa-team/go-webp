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

package pipeline

import (
	"bytes"
	"image"
	"math/rand"
	"testing"
)

// TestScaleSinkReleasesWork checks that a scaling sink returns its
// rescaler buffer to the pool once done, and that late or mixed-up calls
// after that (or after switching to the YUV path) neither touch the
// buffer, which another decode may own by then, nor change the output.
func TestScaleSinkReleasesWork(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	const W, H = 40, 30
	src := randomImage(r, W, H)
	yp, uv := make([]byte, W*H), make([]byte, (W/2)*(H/2))
	r.Read(yp)
	r.Read(uv)
	p := Params{SrcW: W, SrcH: H, ScaleW: 13, ScaleH: 7, Premultiply: true}
	other := Params{SrcW: 300, SrcH: 20, ScaleW: 100, ScaleH: 5}

	// RGBA path, then late RGBA and YUV calls.
	s, img, err := NewSink(p)
	if err != nil {
		t.Fatalf("NewSink(%+v): %v", p, err)
	}
	feed(t, s, src, 7, r, true)
	ss := s.(*scaleSink)
	if ss.work != nil {
		t.Errorf("RGBA path: work = %p after the last row, want nil (released)", ss.work)
	}
	want := clonePix(img())
	// Another sink takes (and scribbles over) the pooled buffer.
	o, _, err := NewSink(other)
	if err != nil {
		t.Fatalf("NewSink(%+v): %v", other, err)
	}
	feed(t, o, randomImage(r, 300, 20), 3, r, true)
	s.Buffer(H-1, 1)
	if err := s.Commit(H-1, 1); err != nil {
		t.Fatalf("RGBA path: late Commit(%d, 1): %v", H-1, err)
	}
	if err := s.(YUVSink).CommitYUV(0, 2, yp, W, uv, uv, W/2, nil, 0); err != nil {
		t.Fatalf("RGBA path: late CommitYUV(0, 2): %v", err)
	}
	if got := img().Pix; !bytes.Equal(want, got) {
		t.Errorf("RGBA path: late calls changed the output (len %d, want len %d)", len(got), len(want))
	}

	// YUV path, then late YUV and RGBA calls.
	s, img, err = NewSink(Params{SrcW: W, SrcH: H, ScaleW: 13, ScaleH: 7})
	if err != nil {
		t.Fatalf("NewSink(%dx%d -> 13x7): %v", W, H, err)
	}
	ys := s.(YUVSink)
	for y := 0; y < H; y += 6 {
		if err := ys.CommitYUV(y, min(6, H-y), yp[y*W:], W, uv[(y/2)*(W/2):], uv[(y/2)*(W/2):], W/2, nil, 0); err != nil {
			t.Fatalf("YUV path: CommitYUV(%d, %d): %v", y, min(6, H-y), err)
		}
	}
	if work := s.(*scaleSink).work; work != nil {
		t.Errorf("YUV path: work = %p after the last row, want nil (released)", work)
	}
	want = clonePix(img())
	o, _, err = NewSink(other)
	if err != nil {
		t.Fatalf("NewSink(%+v): %v", other, err)
	}
	feed(t, o, randomImage(r, 300, 20), 3, r, true)
	if err := ys.CommitYUV(H-2, 2, yp[(H-2)*W:], W, uv[(H/2-1)*(W/2):], uv[(H/2-1)*(W/2):], W/2, nil, 0); err != nil {
		t.Fatalf("YUV path: late CommitYUV(%d, 2): %v", H-2, err)
	}
	s.Buffer(0, 2)
	if err := s.Commit(0, 2); err != nil {
		t.Fatalf("YUV path: late Commit(0, 2): %v", err)
	}
	if got := img().Pix; !bytes.Equal(want, got) {
		t.Errorf("YUV path: late calls changed the output (len %d, want len %d)", len(got), len(want))
	}
}

func clonePix(img *image.NRGBA) []byte { return append([]byte(nil), img.Pix...) }
