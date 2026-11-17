package pipeline

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/resample"
)

// TestAcceptPremultiplied checks the PremultipliedSink contract of the
// scaling sink: rows delivered premultiplied after AcceptPremultiplied give
// the same image as non-premultiplied rows premultiplied by the sink.
func TestAcceptPremultiplied(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for iter := 0; iter < 20; iter++ {
		w, h := 1+r.Intn(70), 1+r.Intn(70)
		src := randomImage(r, w, h)
		p := Params{SrcW: w, SrcH: h, ScaleW: 1 + r.Intn(50), ScaleH: 1 + r.Intn(50), Flip: r.Intn(2) == 0, Premultiply: true}
		if r.Intn(2) == 0 {
			x, y := r.Intn(w), r.Intn(h)
			p.Crop.Min.X, p.Crop.Min.Y = x, y
			p.Crop.Max.X, p.Crop.Max.Y = x+1+r.Intn(w-x), y+1+r.Intn(h-y)
		}

		ref, refImg, err := NewSink(p)
		if err != nil {
			t.Fatal(err)
		}
		feed(t, ref, src, 1+r.Intn(20), r, false)

		pre := *src
		pre.Pix = append([]byte(nil), src.Pix...)
		resample.PremultiplyRGBA(pre.Pix)
		s, img, err := NewSink(p)
		if err != nil {
			t.Fatal(err)
		}
		ps, ok := s.(PremultipliedSink)
		if !ok || !ps.AcceptPremultiplied() {
			t.Fatal("scaling sink with Premultiply must accept premultiplied rows")
		}
		feed(t, s, &pre, 1+r.Intn(20), r, false)
		if !bytes.Equal(refImg().Pix, img().Pix) {
			t.Fatalf("iter %d: %+v: premultiplied rows give a different image", iter, p)
		}
	}
}

func TestAcceptPremultipliedRefused(t *testing.T) {
	// Without Premultiply there is nothing to take over.
	s, _, err := NewSink(Params{SrcW: 8, SrcH: 8, ScaleW: 4, ScaleH: 4})
	if err != nil {
		t.Fatal(err)
	}
	if s.(PremultipliedSink).AcceptPremultiplied() {
		t.Error("accepted premultiplied rows without Premultiply")
	}
	// Once rows were delivered, the mode cannot change.
	s, _, err = NewSink(Params{SrcW: 8, SrcH: 8, ScaleW: 4, ScaleH: 4, Premultiply: true})
	if err != nil {
		t.Fatal(err)
	}
	s.Buffer(0, 1)
	if s.(PremultipliedSink).AcceptPremultiplied() {
		t.Error("accepted premultiplied rows after the first Buffer call")
	}
	// Sinks that do not premultiply do not implement the interface.
	for _, p := range []Params{{SrcW: 8, SrcH: 8}, {SrcW: 8, SrcH: 8, Flip: true}, {SrcW: 8, SrcH: 8, Premultiply: true}} {
		s, _, err := NewSink(p)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := s.(PremultipliedSink); ok {
			t.Errorf("%+v: unexpected PremultipliedSink", p)
		}
	}
}
