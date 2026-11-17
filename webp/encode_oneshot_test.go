package webp

import (
	"bytes"
	"context"
	"image"
	"image/draw"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/encoder"
)

// TestEncodeOneShotIdentical checks that the one-shot Encode, which recycles
// the snapshot of the source, writes the same bytes as an Encoder that is
// never released, for the source types with a pooled snapshot.
func TestEncodeOneShotIdentical(t *testing.T) {
	jpg := loadJPEG(t, "source.jpg") // *image.YCbCr
	rgba := image.NewRGBA(image.Rect(0, 0, 301, 217))
	draw.Draw(rgba, rgba.Rect, jpg, image.Point{100, 200}, draw.Src)
	gray := image.NewGray(image.Rect(0, 0, 1, 77))
	draw.Draw(gray, gray.Rect, jpg, image.Point{300, 300}, draw.Src)
	lossy, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	require.NoError(t, err)
	lossless, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 2)
	require.NoError(t, err)
	for round := 0; round < 2; round++ { // the second round reuses pooled storage
		for _, src := range []image.Image{jpg, rgba, gray} {
			for _, o := range []*encoder.Options{lossy, lossless} {
				e, err := encoder.NewEncoder(src, o)
				require.NoError(t, err)
				var want, got bytes.Buffer
				require.NoError(t, e.Encode(&want))
				require.NoError(t, Encode(&got, src, o))
				require.Equal(t, want.Bytes(), got.Bytes(), "%T lossless=%v", src, o.Lossless)
				got.Reset()
				require.NoError(t, EncodeContext(context.Background(), &got, src, o))
				require.Equal(t, want.Bytes(), got.Bytes(), "%T lossless=%v", src, o.Lossless)
			}
		}
	}
}
