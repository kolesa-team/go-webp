package webp

import (
	"bytes"
	"fmt"
	"image"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
)

// oddSizes are picture sizes that are not multiples of the 16x16 macroblock
// (and single rows or columns).
var oddSizes = [][2]int{{1, 1}, {1, 37}, {37, 1}, {17, 33}, {33, 18}, {130, 3}}

// encodedSynth returns synthetic pictures of the given size encoded lossy,
// lossy with alpha and lossless.
func encodedSynth(t *testing.T, w, h int) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	lossless, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 6)
	require.NoError(t, err)
	lossy, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	require.NoError(t, err)
	for _, s := range synthImages(w, h) {
		for name, o := range map[string]*encoder.Options{"lossy": lossy, "lossless": lossless} {
			buf := &bytes.Buffer{}
			require.NoError(t, Encode(buf, s.img, o))
			out[fmt.Sprintf("%s-%s-%dx%d", s.name, name, w, h)] = buf.Bytes()
		}
	}
	return out
}

// decodeOptionsFor returns option sets valid for a w x h picture.
func decodeOptionsFor(w, h int) []*decoder.Options {
	opts := []*decoder.Options{
		nil,
		{Flip: true},
		{UseThreads: true, NoFancyUpsampling: true},
		{Scale: image.Rect(0, 0, 2*w+1, h)},
		{Scale: image.Rect(0, 0, (w+1)/2, (h+2)/3), Flip: true},
	}
	if w > 2 && h > 2 {
		opts = append(opts,
			&decoder.Options{Crop: image.Rect(1, 1, w-2, h-2)},
			&decoder.Options{Crop: image.Rect(1, 1, w-1, h-1), Flip: true, AlphaDitheringStrength: 50},
			&decoder.Options{Crop: image.Rect(0, 1, w, h-1), Scale: image.Rect(0, 0, w, 2*h)},
		)
	}
	return opts
}

// TestDecodeIntoOddSizes compares DecodeInto with a dirty, reused buffer
// against Decode for odd picture sizes and all sinks.
func TestDecodeIntoOddSizes(t *testing.T) {
	sizes := oddSizes
	if testing.Short() {
		sizes = sizes[:4]
	}
	dst := &image.NRGBA{Pix: bytes.Repeat([]byte{0xa5, 0x5a, 0x33}, 20000)}
	for _, sz := range sizes {
		for name, data := range encodedSynth(t, sz[0], sz[1]) {
			for _, o := range decodeOptionsFor(sz[0], sz[1]) {
				want, err := Decode(bytes.NewReader(data), o)
				require.NoError(t, err, "%s %+v", name, o)
				got, err := DecodeInto(bytes.NewReader(data), dst, o)
				require.NoError(t, err, "%s %+v", name, o)
				require.Same(t, dst, got)
				require.Equal(t, want, got, "%s %+v", name, o)
				require.Equal(t, 60000, cap(got.Pix), "buffer not reused")
			}
		}
	}
}
