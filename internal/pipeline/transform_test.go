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
	"runtime"
	"slices"
	"strconv"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func randomImage(r *rand.Rand, w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	r.Read(img.Pix)
	for i := 3; i < len(img.Pix); i += 4 {
		switch r.Intn(3) {
		case 0:
			img.Pix[i] = 255
		case 1:
			img.Pix[i] = 0
		}
	}
	return img
}

// feed pushes src through s like a decoder would, in batches of the given
// size (0 = random sizes), stopping at NeedRows unless all is set.
func feed(t testing.TB, s Sink, src *image.NRGBA, batch int, r *rand.Rand, all bool) {
	t.Helper()
	h := src.Rect.Dy()
	limit := s.NeedRows()
	if all {
		limit = h
	}
	w4 := src.Rect.Dx() * 4
	for y := 0; y < limit; {
		n := batch
		if n == 0 {
			n = 1 + r.Intn(20)
		}
		n = min(n, h-y)
		buf := s.Buffer(y, n)
		stride := s.Stride()
		if len(buf) < (n-1)*stride+w4 {
			t.Fatalf("Buffer(%d, %d): len = %d, want >= %d", y, n, len(buf), (n-1)*stride+w4)
		}
		for i := 0; i < n; i++ {
			copy(buf[i*stride:i*stride+w4], src.Pix[(y+i)*src.Stride:])
		}
		if err := s.Commit(y, n); err != nil {
			t.Fatalf("Commit(%d, %d): %v", y, n, err)
		}
		y += n
	}
}

func cropRef(src *image.NRGBA, c image.Rectangle) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, c.Dx(), c.Dy()))
	for y := 0; y < c.Dy(); y++ {
		copy(out.Pix[y*out.Stride:y*out.Stride+c.Dx()*4], src.Pix[(c.Min.Y+y)*src.Stride+c.Min.X*4:])
	}
	return out
}

func flipRef(img *image.NRGBA) *image.NRGBA {
	out := image.NewNRGBA(img.Rect)
	h := img.Rect.Dy()
	for y := 0; y < h; y++ {
		copy(out.Pix[y*out.Stride:(y+1)*out.Stride], img.Pix[(h-1-y)*img.Stride:])
	}
	return out
}

func scaleRef(t testing.TB, src *image.NRGBA, w, h int, premult bool) *image.NRGBA {
	t.Helper()
	rs, err := resample.New(src.Rect.Dx(), src.Rect.Dy(), w, h, 4)
	if err != nil {
		t.Fatalf("resample.New(%d, %d, %d, %d, 4): %v", src.Rect.Dx(), src.Rect.Dy(), w, h, err)
	}
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	row := make([]byte, src.Rect.Dx()*4)
	for y := 0; y < src.Rect.Dy(); y++ {
		copy(row, src.Pix[y*src.Stride:])
		if premult {
			resample.PremultiplyRGBA(row)
		}
		rs.ImportRow(row)
		for rs.HasPendingOutput() {
			o := out.Pix[rs.DstY()*out.Stride : rs.DstY()*out.Stride+w*4]
			rs.ExportRow(o)
			if premult {
				resample.UnpremultiplyRGBA(o)
			}
		}
	}
	return out
}

func TestNoTransformIsDirect(t *testing.T) {
	s, img, err := NewSink(Params{SrcW: 5, SrcH: 3})
	if err != nil {
		t.Fatalf("NewSink(5x3): %v", err)
	}
	d, ok := s.(*lazyDirect)
	if !ok {
		t.Fatalf("NewSink(5x3) = %T, want *lazyDirect", s)
	}
	buf := d.Buffer(0, 3)
	if &d.img.Pix[0] != &buf[0] {
		t.Errorf("Buffer(0, 3) = %p, want the output image %p (rows are written in place)", &buf[0], &d.img.Pix[0])
	}
	if got := img(); d.img != got {
		t.Errorf("img() = %p, want the sink's image %p", got, d.img)
	}
	if got, want := img().Rect, image.Rect(0, 0, 5, 3); want != got {
		t.Errorf("img().Rect = %v, want %v", got, want)
	}
	// A crop covering the whole frame is not a crop.
	s, _, err = NewSink(Params{SrcW: 5, SrcH: 3, Crop: image.Rect(0, 0, 5, 3)})
	if err != nil {
		t.Fatalf("NewSink(5x3, full-frame crop): %v", err)
	}
	if _, ok := s.(*lazyDirect); !ok {
		t.Errorf("NewSink(5x3, full-frame crop) = %T, want *lazyDirect", s)
	}
}

// TestLazyOutput checks that no sink allocates the output image before the
// decoder hands it its first rows.
func TestLazyOutput(t *testing.T) {
	for _, p := range []Params{
		{SrcW: 16383, SrcH: 16383},
		{SrcW: 16383, SrcH: 16383, Flip: true},
		{SrcW: 16383, SrcH: 16383, Crop: image.Rect(2, 2, 16000, 16000)},
		{SrcW: 16383, SrcH: 16383, ScaleW: 16000, ScaleH: 16000},
	} {
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		s, img, err := NewSink(p)
		if err != nil {
			t.Fatalf("NewSink(%+v): %v", p, err)
		}
		runtime.ReadMemStats(&m1)
		if got := m1.TotalAlloc - m0.TotalAlloc; got >= uint64(1<<20) {
			t.Errorf("%+v: NewSink allocated %d bytes, want < %d", p, got, uint64(1<<20))
		}
		_ = s
		_ = img
	}
}

func TestTransforms(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for iter := 0; iter < 400; iter++ {
		w, h := 1+r.Intn(40), 1+r.Intn(40)
		src := randomImage(r, w, h)
		p := Params{SrcW: w, SrcH: h, Flip: r.Intn(2) == 0, Premultiply: r.Intn(2) == 0}
		crop := src.Rect
		if r.Intn(3) != 0 {
			x, y := r.Intn(w), r.Intn(h)
			crop = image.Rect(x, y, x+1+r.Intn(w-x), y+1+r.Intn(h-y))
			if r.Intn(4) == 0 { // full width
				crop.Min.X, crop.Max.X = 0, w
			}
			p.Crop = crop
		}
		want := cropRef(src, crop)
		if r.Intn(2) == 0 {
			p.ScaleW, p.ScaleH = r.Intn(50), 1+r.Intn(50)
			if r.Intn(4) == 0 {
				p.ScaleW, p.ScaleH = p.ScaleH, 0
			}
			sw, sh, ok := resample.ScaledDimensions(crop.Dx(), crop.Dy(), p.ScaleW, p.ScaleH)
			if !ok {
				t.Fatalf("ScaledDimensions(%d, %d, %d, %d) failed", crop.Dx(), crop.Dy(), p.ScaleW, p.ScaleH)
			}
			want = scaleRef(t, want, sw, sh, p.Premultiply)
		}
		if p.Flip {
			want = flipRef(want)
		}
		for _, batch := range []int{1, 16, 0, h} {
			s, img, err := NewSink(p)
			if err != nil {
				t.Fatalf("NewSink(%+v): %v", p, err)
			}
			wantNeed := h
			if !p.Crop.Empty() {
				wantNeed = crop.Max.Y
			}
			if got := s.NeedRows(); wantNeed != got {
				t.Errorf("params %+v: NeedRows() = %d, want %d", p, got, wantNeed)
			}
			feed(t, s, src, batch, r, batch == 0 && r.Intn(2) == 0)
			if got := img(); !reflect.DeepEqual(want, got) {
				t.Fatalf("params %+v batch %d: output (Rect %v, Stride %d, len(Pix) %d) differs from the reference (Rect %v, Stride %d, len(Pix) %d)",
					p, batch, got.Rect, got.Stride, len(got.Pix), want.Rect, want.Stride, len(want.Pix))
			}
		}
	}
}

func TestFlipInPlace(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	src := randomImage(r, 7, 9)
	s, img, err := NewSink(Params{SrcW: 7, SrcH: 9, Flip: true})
	if err != nil {
		t.Fatalf("NewSink(7x9, flip): %v", err)
	}
	// The decoder writes straight into the output image.
	buf := s.Buffer(0, 9)
	if want, got := &img().Pix[0], &buf[0]; !reflect.DeepEqual(want, got) {
		t.Errorf("Buffer(0, 9) = %p, want the output image %p", got, want)
	}
	feed(t, s, src, 3, r, false)
	if want, got := flipRef(src), img(); !reflect.DeepEqual(want, got) {
		t.Errorf("output (Rect %v, Stride %d, len(Pix) %d) differs from the flipped source (Rect %v, Stride %d, len(Pix) %d)",
			got.Rect, got.Stride, len(got.Pix), want.Rect, want.Stride, len(want.Pix))
	}
}

func TestInvalidParams(t *testing.T) {
	for _, p := range []Params{
		{SrcW: 0, SrcH: 5},
		{SrcW: 5, SrcH: -1},
		{SrcW: 10, SrcH: 10, Crop: image.Rect(-1, 0, 5, 5)},
		{SrcW: 10, SrcH: 10, Crop: image.Rect(0, 0, 11, 5)},
		{SrcW: 10, SrcH: 10, Crop: image.Rect(5, 5, 10, 11)},
		{SrcW: 10, SrcH: 10, Crop: image.Rect(10, 0, 12, 5)},
		{SrcW: 10, SrcH: 10, ScaleW: -1, ScaleH: 5},
		{SrcW: 10, SrcH: 10, ScaleW: 1 << 30, ScaleH: 1},
	} {
		_, _, err := NewSink(p)
		if got := status.CodeOf(err); utils.Vp8StatusInvalidParam != got {
			t.Errorf("NewSink(%+v): status %v (err %v), want %v", p, got, err, utils.Vp8StatusInvalidParam)
		}
	}
	_, _, err := NewSink(Params{SrcW: 10, SrcH: 10, ScaleW: 1 << 20, ScaleH: 1 << 20})
	if got := status.CodeOf(err); utils.Vp8StatusOutOfMemory != got {
		t.Errorf("NewSink(10x10 -> %dx%d): status %v (err %v), want %v", 1<<20, 1<<20, got, err, utils.Vp8StatusOutOfMemory)
	}
}

// TestAllocationLimit checks that output images and rescalers above
// libwebp's WEBP_MAX_ALLOCABLE_MEMORY are refused with OutOfMemory instead
// of being allocated (on 32-bit targets image.NewNRGBA would panic), using
// a lowered limit.
func TestAllocationLimit(t *testing.T) {
	if strconv.IntSize == 64 {
		if want := uint64(1) << 34; want != maxAllocableMemory {
			t.Errorf("maxAllocableMemory = %d, want %d", maxAllocableMemory, want)
		}
	} else if want := uint64(1)<<31 - 1<<16; want != maxAllocableMemory {
		t.Errorf("maxAllocableMemory = %d, want %d", maxAllocableMemory, want)
	}
	defer func(old uint64) { maxOutputBytes = old }(maxOutputBytes)
	maxOutputBytes = 40 * 30 * 4

	for _, p := range []Params{
		{SrcW: 40, SrcH: 31},
		{SrcW: 41, SrcH: 30, Flip: true},
		{SrcW: 100, SrcH: 100, Crop: image.Rect(0, 0, 41, 30)},
		{SrcW: 10, SrcH: 10, ScaleW: 40, ScaleH: 31},
		{SrcW: 10, SrcH: 10, ScaleW: 41, ScaleH: 0}, // height derived: 41
		// The rescaler state (36 bytes per output column) is too large,
		// although the 4 bytes/pixel output fits.
		{SrcW: 10, SrcH: 10, ScaleW: 134, ScaleH: 1},
	} {
		_, _, err := NewSink(p)
		if got := status.CodeOf(err); utils.Vp8StatusOutOfMemory != got {
			t.Errorf("NewSink(%+v): status %v (err %v), want %v", p, got, err, utils.Vp8StatusOutOfMemory)
		}
	}
	for _, p := range []Params{
		{SrcW: 40, SrcH: 30},
		{SrcW: 100, SrcH: 100, Crop: image.Rect(10, 10, 50, 40), Flip: true},
		{SrcW: 10, SrcH: 10, ScaleW: 40, ScaleH: 30},
		{SrcW: 10, SrcH: 10, ScaleW: 133, ScaleH: 1},
	} {
		sink, result, err := NewSink(p)
		if err != nil {
			t.Fatalf("NewSink(%+v): %v", p, err)
		}
		if sink == nil {
			t.Fatalf("NewSink(%+v) returned a nil sink", p)
		}
		if result == nil {
			t.Fatalf("NewSink(%+v) returned a nil image function", p)
		}
	}
}

func TestCropFromV1(t *testing.T) {
	if got, want := CropFromV1(image.Rect(10, 20, 100, 50)), image.Rect(10, 20, 110, 70); want != got {
		t.Errorf("CropFromV1(%v) = %v, want %v", image.Rect(10, 20, 100, 50), got, want)
	}
}

func TestSnapCropForYUV(t *testing.T) {
	if got, want := SnapCropForYUV(image.Rect(3, 5, 8, 11)), image.Rect(2, 4, 7, 10); want != got {
		t.Errorf("SnapCropForYUV(%v) = %v, want %v", image.Rect(3, 5, 8, 11), got, want)
	}
	if got, want := SnapCropForYUV(image.Rect(2, 4, 4, 6)), image.Rect(2, 4, 4, 6); want != got {
		t.Errorf("SnapCropForYUV(%v) = %v, want %v", image.Rect(2, 4, 4, 6), got, want)
	}
	if got := SnapCropForYUV(image.Rectangle{}); !got.Empty() {
		t.Errorf("SnapCropForYUV(empty) = %v, want an empty rectangle", got)
	}
}

func BenchmarkSink(b *testing.B) {
	const w, h = 1024, 768
	src := randomImage(rand.New(rand.NewSource(3)), w, h)
	for _, c := range []struct {
		name string
		p    Params
	}{
		{"Direct", Params{SrcW: w, SrcH: h}},
		{"Flip", Params{SrcW: w, SrcH: h, Flip: true}},
		{"Crop", Params{SrcW: w, SrcH: h, Crop: image.Rect(100, 100, 600, 500)}},
		{"Scale", Params{SrcW: w, SrcH: h, ScaleW: 300, ScaleH: 225}},
		{"ScalePremultiplied", Params{SrcW: w, SrcH: h, ScaleW: 300, ScaleH: 225, Premultiply: true}},
		{"CropScaleFlip", Params{SrcW: w, SrcH: h, Crop: image.Rect(100, 100, 600, 500), ScaleW: 200, Flip: true}},
	} {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(w * h * 4)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				s, _, err := NewSink(c.p)
				if err != nil {
					b.Fatal(err)
				}
				for y := 0; y < s.NeedRows(); y += 16 {
					n := min(16, h-y)
					buf := s.Buffer(y, n)
					copy(buf, src.Pix[y*src.Stride:(y+n)*src.Stride])
					_ = s.Commit(y, n)
				}
			}
		})
	}
}

// yuvRef rescales full planes in one go and converts them, as a reference
// for the streaming YUV path.
func yuvRef(t testing.TB, W, H int, yp, up, vp, ap []byte, crop image.Rectangle, dw, dh int, flip bool) *image.NRGBA {
	t.Helper()
	cw, ch := crop.Dx(), crop.Dy()
	uw, cuw, cuh := (W+1)/2, (cw+1)/2, (ch+1)/2
	plane := func(src []byte, stride, x, y, w, h int) []byte {
		out := make([]byte, 0, w*h)
		for j := 0; j < h; j++ {
			out = append(out, src[(y+j)*stride+x:(y+j)*stride+x+w]...)
		}
		return out
	}
	scale := func(p []byte, w, h int) []byte {
		t.Helper()
		rs, err := resample.New(w, h, dw, dh, 1)
		if err != nil {
			t.Fatalf("resample.New(%d, %d, %d, %d, 1): %v", w, h, dw, dh, err)
		}
		out := make([]byte, dw*dh)
		for y := 0; y < h; y++ {
			rs.ImportRow(p[y*w:])
			for rs.HasPendingOutput() {
				rs.ExportRow(out[rs.DstY()*dw:])
			}
		}
		return out
	}
	sy := scale(plane(yp, W, crop.Min.X, crop.Min.Y, cw, ch), cw, ch)
	su := scale(plane(up, uw, crop.Min.X/2, crop.Min.Y/2, cuw, cuh), cuw, cuh)
	sv := scale(plane(vp, uw, crop.Min.X/2, crop.Min.Y/2, cuw, cuh), cuw, cuh)
	var sa []byte
	if ap != nil {
		sa = scale(plane(ap, W, crop.Min.X, crop.Min.Y, cw, ch), cw, ch)
	}
	img := image.NewNRGBA(image.Rect(0, 0, dw, dh))
	for i := range sy {
		yh := int(sy[i]) * 19077 >> 8
		u, v := int(su[i]), int(sv[i])
		o := img.Pix[i*4:]
		o[0] = clip8YUV(yh + (v*26149)>>8 - 14234)
		o[1] = clip8YUV(yh - (u*6419)>>8 - (v*13320)>>8 + 8708)
		o[2] = clip8YUV(yh + (u*33050)>>8 - 17685)
		o[3] = 255
		if sa != nil {
			o[3] = sa[i]
		}
	}
	if flip {
		img = flipRef(img)
	}
	return img
}

func TestYUVScaling(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for iter := 0; iter < 300; iter++ {
		W, H := 1+r.Intn(50), 1+r.Intn(50)
		uw, uh := (W+1)/2, (H+1)/2
		yp, up, vp := make([]byte, W*H), make([]byte, uw*uh), make([]byte, uw*uh)
		r.Read(yp)
		r.Read(up)
		r.Read(vp)
		var ap []byte
		if r.Intn(2) == 0 {
			ap = make([]byte, W*H)
			r.Read(ap)
		}
		p := Params{SrcW: W, SrcH: H, Flip: r.Intn(2) == 0, ScaleW: 1 + r.Intn(60), ScaleH: 1 + r.Intn(60)}
		crop := image.Rect(0, 0, W, H)
		if r.Intn(2) == 0 {
			x, y := r.Intn(W), r.Intn(H)
			crop = SnapCropForYUV(image.Rect(x, y, x+1+r.Intn(W-x), y+1+r.Intn(H-y)))
			p.Crop = crop
		}
		want := yuvRef(t, W, H, yp, up, vp, ap, crop, p.ScaleW, p.ScaleH, p.Flip)
		for _, batch := range []int{0, 1, 2, 3, 16, H} {
			s, img, err := NewSink(p)
			if err != nil {
				t.Fatalf("%dx%d: NewSink(%+v): %v", W, H, p, err)
			}
			ys := s.(YUVSink)
			for y := 0; y < s.NeedRows(); {
				n := batch
				if n == 0 {
					n = 1 + r.Intn(9)
				}
				n = min(n, H-y)
				var a []byte
				if ap != nil {
					a = ap[y*W:]
				}
				if err := ys.CommitYUV(y, n, yp[y*W:], W, up[(y/2)*uw:], vp[(y/2)*uw:], uw, a, W); err != nil {
					t.Fatalf("%dx%d %+v batch %d: CommitYUV(%d, %d): %v", W, H, p, batch, y, n, err)
				}
				y += n
			}
			if got := img(); !reflect.DeepEqual(want, got) {
				t.Fatalf("%dx%d %+v batch %d: output (Rect %v, Stride %d, len(Pix) %d) differs from the reference (Rect %v, Stride %d, len(Pix) %d)",
					W, H, p, batch, got.Rect, got.Stride, len(got.Pix), want.Rect, want.Stride, len(want.Pix))
			}
		}
	}
}

func TestYUVErrors(t *testing.T) {
	s, _, err := NewSink(Params{SrcW: 4, SrcH: 4, ScaleW: 2, ScaleH: 2})
	if err != nil {
		t.Fatalf("NewSink(4x4 -> 2x2): %v", err)
	}
	ys := s.(YUVSink)
	y, uv, a := make([]byte, 16), make([]byte, 4), make([]byte, 16)
	if err := ys.CommitYUV(0, 1, y, 4, uv, uv, 2, a, 4); err != nil {
		t.Fatalf("CommitYUV(0, 1) with alpha: %v", err)
	}
	if err := ys.CommitYUV(1, 1, y, 4, uv, uv, 2, nil, 4); err == nil {
		t.Error("CommitYUV(1, 1) without alpha after a row with alpha succeeded, want an error (alpha disappeared)")
	}
	if err := ys.CommitYUV(3, 1, y, 4, uv, uv, 2, a, 4); err == nil {
		t.Error("CommitYUV(3, 1) after row 1 succeeded, want an error (rows skipped)")
	}
}

func FuzzSink(f *testing.F) {
	f.Add(uint8(10), uint8(10), int8(1), int8(1), int8(5), int8(5), uint8(3), uint8(4), true, false, int64(1))
	f.Fuzz(func(t *testing.T, w, h uint8, x0, y0, x1, y1 int8, sw, sh uint8, flip, premult bool, seed int64) {
		W, H := int(w%64)+1, int(h%64)+1
		p := Params{SrcW: W, SrcH: H, Crop: image.Rect(int(x0), int(y0), int(x1), int(y1)),
			ScaleW: int(sw % 80), ScaleH: int(sh % 80), Flip: flip, Premultiply: premult}
		run := func(p Params) (*image.NRGBA, bool) {
			s, img, err := NewSink(p)
			if err != nil {
				return nil, false
			}
			r := rand.New(rand.NewSource(seed))
			feed(t, s, randomImage(r, W, H), 0, r, r.Intn(2) == 0)
			return img(), true
		}
		out, ok := run(p)
		if !ok {
			return
		}
		if out == nil || out.Rect.Dx() <= 0 || out.Rect.Dy() <= 0 {
			t.Fatal("bad output image")
		}
		// A reused dirty buffer that is just large enough gives the same
		// image: every output byte is written.
		p.Dst = dirtyImage(0, len(out.Pix))
		if got, _ := run(p); got == nil || !bytes.Equal(out.Pix, got.Pix) || &got.Pix[0] != &p.Dst.Pix[:1][0] {
			t.Fatal("output into a reused buffer differs")
		}
	})
}

func TestCarry(t *testing.T) {
	batch := func(k int) []byte { return []byte{byte(k)} }
	rows := func(rs [][]byte) []int {
		var out []int
		for _, r := range rs {
			out = append(out, int(r[0]))
		}
		return out
	}
	// Old carry holds rows 3,4; one consumed; batch supplies up to row 7.
	old := [][]byte{{3}, {4}}
	if got, want := rows(carry(old, 1, 5, 7, batch)), []int{4, 5, 6}; !slices.Equal(want, got) {
		t.Errorf("carry(rows 3-4, 1, 5, 7) = rows %v, want %v", got, want)
	}
	// Everything carried and two batch rows consumed.
	if got, want := rows(carry([][]byte{{3}, {4}}, 4, 5, 9, batch)), []int{7, 8}; !slices.Equal(want, got) {
		t.Errorf("carry(rows 3-4, 4, 5, 9) = rows %v, want %v", got, want)
	}
	// Nothing left.
	if got := carry([][]byte{{3}}, 1, 4, 4, batch); len(got) != 0 {
		t.Errorf("carry(row 3, 1, 4, 4) = %d rows, want none", len(got))
	}
	if got, want := rows(carry(nil, 0, 2, 3, batch)), []int{2}; !slices.Equal(want, got) {
		t.Errorf("carry(nil, 0, 2, 3) = rows %v, want %v", got, want)
	}
}
