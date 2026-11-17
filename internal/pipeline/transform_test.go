package pipeline

import (
	"bytes"
	"image"
	"math/rand"
	"runtime"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
		require.GreaterOrEqual(t, len(buf), (n-1)*stride+w4)
		for i := 0; i < n; i++ {
			copy(buf[i*stride:i*stride+w4], src.Pix[(y+i)*src.Stride:])
		}
		require.NoError(t, s.Commit(y, n))
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
	rs, err := resample.New(src.Rect.Dx(), src.Rect.Dy(), w, h, 4)
	require.NoError(t, err)
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
	require.NoError(t, err)
	d, ok := s.(*lazyDirect)
	require.True(t, ok)
	buf := d.Buffer(0, 3)
	assert.Same(t, &d.img.Pix[0], &buf[0], "rows are written in place")
	assert.Same(t, d.img, img())
	assert.Equal(t, image.Rect(0, 0, 5, 3), img().Rect)
	// A crop covering the whole frame is not a crop.
	s, _, err = NewSink(Params{SrcW: 5, SrcH: 3, Crop: image.Rect(0, 0, 5, 3)})
	require.NoError(t, err)
	assert.IsType(t, &lazyDirect{}, s)
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
		require.NoError(t, err)
		runtime.ReadMemStats(&m1)
		assert.Less(t, m1.TotalAlloc-m0.TotalAlloc, uint64(1<<20), "%+v", p)
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
			require.True(t, ok)
			want = scaleRef(t, want, sw, sh, p.Premultiply)
		}
		if p.Flip {
			want = flipRef(want)
		}
		for _, batch := range []int{1, 16, 0, h} {
			s, img, err := NewSink(p)
			require.NoError(t, err)
			wantNeed := h
			if !p.Crop.Empty() {
				wantNeed = crop.Max.Y
			}
			assert.Equal(t, wantNeed, s.NeedRows())
			feed(t, s, src, batch, r, batch == 0 && r.Intn(2) == 0)
			require.Equal(t, want, img(), "params %+v batch %d", p, batch)
		}
	}
}

func TestFlipInPlace(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	src := randomImage(r, 7, 9)
	s, img, err := NewSink(Params{SrcW: 7, SrcH: 9, Flip: true})
	require.NoError(t, err)
	// The decoder writes straight into the output image.
	buf := s.Buffer(0, 9)
	assert.Equal(t, &img().Pix[0], &buf[0])
	feed(t, s, src, 3, r, false)
	assert.Equal(t, flipRef(src), img())
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
		assert.Equal(t, utils.Vp8StatusInvalidParam, status.CodeOf(err), "%+v", p)
	}
	_, _, err := NewSink(Params{SrcW: 10, SrcH: 10, ScaleW: 1 << 20, ScaleH: 1 << 20})
	assert.Equal(t, utils.Vp8StatusOutOfMemory, status.CodeOf(err))
}

// TestAllocationLimit checks that output images and rescalers above
// libwebp's WEBP_MAX_ALLOCABLE_MEMORY are refused with OutOfMemory instead
// of being allocated (on 32-bit targets image.NewNRGBA would panic), using
// a lowered limit.
func TestAllocationLimit(t *testing.T) {
	if strconv.IntSize == 64 {
		assert.Equal(t, uint64(1)<<34, maxAllocableMemory)
	} else {
		assert.Equal(t, uint64(1)<<31-1<<16, maxAllocableMemory)
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
		assert.Equal(t, utils.Vp8StatusOutOfMemory, status.CodeOf(err), "%+v", p)
	}
	for _, p := range []Params{
		{SrcW: 40, SrcH: 30},
		{SrcW: 100, SrcH: 100, Crop: image.Rect(10, 10, 50, 40), Flip: true},
		{SrcW: 10, SrcH: 10, ScaleW: 40, ScaleH: 30},
		{SrcW: 10, SrcH: 10, ScaleW: 133, ScaleH: 1},
	} {
		sink, result, err := NewSink(p)
		require.NoError(t, err, "%+v", p)
		require.NotNil(t, sink)
		require.NotNil(t, result)
	}
}

func TestCropFromV1(t *testing.T) {
	assert.Equal(t, image.Rect(10, 20, 110, 70), CropFromV1(image.Rect(10, 20, 100, 50)))
}

func TestSnapCropForYUV(t *testing.T) {
	assert.Equal(t, image.Rect(2, 4, 7, 10), SnapCropForYUV(image.Rect(3, 5, 8, 11)))
	assert.Equal(t, image.Rect(2, 4, 4, 6), SnapCropForYUV(image.Rect(2, 4, 4, 6)))
	assert.True(t, SnapCropForYUV(image.Rectangle{}).Empty())
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
		rs, err := resample.New(w, h, dw, dh, 1)
		require.NoError(t, err)
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
			require.NoError(t, err)
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
				require.NoError(t, ys.CommitYUV(y, n, yp[y*W:], W, up[(y/2)*uw:], vp[(y/2)*uw:], uw, a, W))
				y += n
			}
			require.Equal(t, want, img(), "%dx%d %+v batch %d", W, H, p, batch)
		}
	}
}

func TestYUVErrors(t *testing.T) {
	s, _, err := NewSink(Params{SrcW: 4, SrcH: 4, ScaleW: 2, ScaleH: 2})
	require.NoError(t, err)
	ys := s.(YUVSink)
	y, uv, a := make([]byte, 16), make([]byte, 4), make([]byte, 16)
	require.NoError(t, ys.CommitYUV(0, 1, y, 4, uv, uv, 2, a, 4))
	assert.Error(t, ys.CommitYUV(1, 1, y, 4, uv, uv, 2, nil, 4), "alpha disappeared")
	assert.Error(t, ys.CommitYUV(3, 1, y, 4, uv, uv, 2, a, 4), "rows skipped")
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
	assert.Equal(t, []int{4, 5, 6}, rows(carry(old, 1, 5, 7, batch)))
	// Everything carried and two batch rows consumed.
	assert.Equal(t, []int{7, 8}, rows(carry([][]byte{{3}, {4}}, 4, 5, 9, batch)))
	// Nothing left.
	assert.Empty(t, carry([][]byte{{3}}, 1, 4, 4, batch))
	assert.Equal(t, []int{2}, rows(carry(nil, 0, 2, 3, batch)))
}
