package pipeline

import (
	"image"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.NoError(t, err)
	feed(t, s, src, 7, r, true)
	ss := s.(*scaleSink)
	assert.Nil(t, ss.work)
	want := clonePix(img())
	// Another sink takes (and scribbles over) the pooled buffer.
	o, _, err := NewSink(other)
	require.NoError(t, err)
	feed(t, o, randomImage(r, 300, 20), 3, r, true)
	s.Buffer(H-1, 1)
	require.NoError(t, s.Commit(H-1, 1))
	require.NoError(t, s.(YUVSink).CommitYUV(0, 2, yp, W, uv, uv, W/2, nil, 0))
	assert.Equal(t, want, img().Pix)

	// YUV path, then late YUV and RGBA calls.
	s, img, err = NewSink(Params{SrcW: W, SrcH: H, ScaleW: 13, ScaleH: 7})
	require.NoError(t, err)
	ys := s.(YUVSink)
	for y := 0; y < H; y += 6 {
		require.NoError(t, ys.CommitYUV(y, min(6, H-y), yp[y*W:], W, uv[(y/2)*(W/2):], uv[(y/2)*(W/2):], W/2, nil, 0))
	}
	assert.Nil(t, s.(*scaleSink).work)
	want = clonePix(img())
	o, _, err = NewSink(other)
	require.NoError(t, err)
	feed(t, o, randomImage(r, 300, 20), 3, r, true)
	require.NoError(t, ys.CommitYUV(H-2, 2, yp[(H-2)*W:], W, uv[(H/2-1)*(W/2):], uv[(H/2-1)*(W/2):], W/2, nil, 0))
	s.Buffer(0, 2)
	require.NoError(t, s.Commit(0, 2))
	assert.Equal(t, want, img().Pix)
}

func clonePix(img *image.NRGBA) []byte { return append([]byte(nil), img.Pix...) }
