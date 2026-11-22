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
	"reflect"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/resample"
)

// dirtyImage returns an image whose Pix has the given length and capacity,
// filled with a pattern that no test picture produces by chance in every
// byte.
func dirtyImage(n, c int) *image.NRGBA {
	pix := make([]byte, c)
	for i := range pix {
		pix[i] = byte(0xa5 ^ i)
	}
	return &image.NRGBA{Pix: pix[:n], Stride: 7, Rect: image.Rect(3, 4, 5, 6)}
}

// TestTransformsReuse checks that every sink (direct, flip, crop, scale and
// their combinations) produces the same image into a reused, dirty buffer
// as into a fresh one, that the buffer is used when it is large enough,
// and that Params.Dst itself is never modified.
func TestTransformsReuse(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for iter := 0; iter < 300; iter++ {
		w, h := 1+r.Intn(40), 1+r.Intn(40)
		src := randomImage(r, w, h)
		p := Params{SrcW: w, SrcH: h, Flip: r.Intn(2) == 0, Premultiply: r.Intn(2) == 0}
		crop := src.Rect
		if r.Intn(3) != 0 {
			x, y := r.Intn(w), r.Intn(h)
			crop = image.Rect(x, y, x+1+r.Intn(w-x), y+1+r.Intn(h-y))
			if r.Intn(4) == 0 {
				crop.Min.X, crop.Max.X = 0, w
			}
			p.Crop = crop
		}
		want := cropRef(src, crop)
		if r.Intn(2) == 0 {
			p.ScaleW, p.ScaleH = 1+r.Intn(50), 1+r.Intn(50)
			sw, sh, ok := resample.ScaledDimensions(crop.Dx(), crop.Dy(), p.ScaleW, p.ScaleH)
			if !ok {
				t.Fatalf("ScaledDimensions(%d, %d, %d, %d) failed", crop.Dx(), crop.Dy(), p.ScaleW, p.ScaleH)
			}
			want = scaleRef(t, want, sw, sh, p.Premultiply)
		}
		if p.Flip {
			want = flipRef(want)
		}
		n := len(want.Pix)
		for _, c := range []int{n, n + 1 + r.Intn(64), n / 2} {
			dst := dirtyImage(min(c, 5), c)
			before := *dst
			p.Dst = dst
			s, img, err := NewSink(p)
			if err != nil {
				t.Fatalf("NewSink(%+v): %v", p, err)
			}
			feed(t, s, src, 0, r, r.Intn(2) == 0)
			got := img()
			if want.Rect != got.Rect {
				t.Fatalf("params %+v cap %d: Rect = %v, want %v", p, c, got.Rect, want.Rect)
			}
			if want.Stride != got.Stride {
				t.Fatalf("params %+v cap %d: Stride = %d, want %d", p, c, got.Stride, want.Stride)
			}
			if !bytes.Equal(want.Pix, got.Pix) {
				t.Fatalf("params %+v cap %d: Pix differs from the reference (len %d, want len %d)", p, c, len(got.Pix), len(want.Pix))
			}
			if !reflect.DeepEqual(before, *dst) {
				t.Errorf("params %+v cap %d: Dst = {Pix %p len %d, Stride %d, Rect %v}, want {Pix %p len %d, Stride %d, Rect %v} (Dst must not be modified)",
					p, c, dst.Pix, len(dst.Pix), dst.Stride, dst.Rect, before.Pix, len(before.Pix), before.Stride, before.Rect)
			}
			if c >= n && n > 0 {
				if &dst.Pix[:1][0] != &got.Pix[0] {
					t.Errorf("params %+v cap %d: output Pix %p, want the Dst buffer %p (the buffer must be reused)", p, c, &got.Pix[0], &dst.Pix[:1][0])
				}
				if c != cap(got.Pix) {
					t.Errorf("params %+v cap %d: cap(Pix) = %d, want %d (the capacity must be kept)", p, c, cap(got.Pix), c)
				}
			} else if &dst.Pix[:1][0] == &got.Pix[0] {
				t.Errorf("params %+v cap %d: output Pix reuses the too small Dst buffer", p, c)
			}
		}
	}
}

func TestNewOutput(t *testing.T) {
	img := newOutput(3, 2, nil)
	if want := image.Rect(0, 0, 3, 2); want != img.Rect {
		t.Errorf("newOutput(3, 2, nil).Rect = %v, want %v", img.Rect, want)
	}
	if img.Stride != 12 {
		t.Errorf("newOutput(3, 2, nil).Stride = %d, want 12", img.Stride)
	}
	if len(img.Pix) != 24 {
		t.Errorf("newOutput(3, 2, nil): len(Pix) = %d, want 24", len(img.Pix))
	}

	empty := &image.NRGBA{}
	img = newOutput(1, 1, empty)
	if len(img.Pix) != 4 {
		t.Errorf("newOutput(1, 1, empty): len(Pix) = %d, want 4", len(img.Pix))
	}
	if empty.Pix != nil {
		t.Errorf("newOutput(1, 1, empty) set empty.Pix (len %d, cap %d), want nil", len(empty.Pix), cap(empty.Pix))
	}

	big := &image.NRGBA{Pix: make([]byte, 0, 100)}
	img = newOutput(5, 5, big)
	if len(img.Pix) != 100 {
		t.Errorf("newOutput(5, 5, big): len(Pix) = %d, want 100", len(img.Pix))
	}
	if &big.Pix[:1][0] != &img.Pix[0] {
		t.Errorf("newOutput(5, 5, big): Pix %p, want the big.Pix buffer %p", &img.Pix[0], &big.Pix[:1][0])
	}
	if len(big.Pix) != 0 {
		t.Errorf("newOutput(5, 5, big): len(big.Pix) = %d, want 0", len(big.Pix))
	}
}
