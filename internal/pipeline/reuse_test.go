package pipeline

import (
	"bytes"
	"image"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
			require.True(t, ok)
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
			require.NoError(t, err)
			feed(t, s, src, 0, r, r.Intn(2) == 0)
			got := img()
			require.Equal(t, want.Rect, got.Rect, "params %+v cap %d", p, c)
			require.Equal(t, want.Stride, got.Stride)
			require.True(t, bytes.Equal(want.Pix, got.Pix), "params %+v cap %d", p, c)
			assert.Equal(t, before, *dst, "Dst must not be modified")
			if c >= n && n > 0 {
				assert.Same(t, &dst.Pix[:1][0], &got.Pix[0], "the buffer must be reused")
				assert.Equal(t, c, cap(got.Pix), "the capacity must be kept")
			} else {
				assert.NotSame(t, &dst.Pix[:1][0], &got.Pix[0])
			}
		}
	}
}

func TestNewOutput(t *testing.T) {
	img := newOutput(3, 2, nil)
	assert.Equal(t, image.Rect(0, 0, 3, 2), img.Rect)
	assert.Equal(t, 12, img.Stride)
	assert.Len(t, img.Pix, 24)

	empty := &image.NRGBA{}
	img = newOutput(1, 1, empty)
	assert.Len(t, img.Pix, 4)
	assert.Nil(t, empty.Pix)

	big := &image.NRGBA{Pix: make([]byte, 0, 100)}
	img = newOutput(5, 5, big)
	assert.Len(t, img.Pix, 100)
	assert.Same(t, &big.Pix[:1][0], &img.Pix[0])
	assert.Len(t, big.Pix, 0)
}
