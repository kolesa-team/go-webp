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
	skipUnderRace(t)
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
