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
	"fmt"
	"image"
	"os"
	"testing"
	"time"

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

func TestDecodeContext(t *testing.T) {
	data, err := os.ReadFile("../test_data/images/m4_q75.webp")
	require.NoError(t, err)
	want, err := Decode(bytes.NewReader(data), nil)
	require.NoError(t, err)
	got, err := DecodeContext(context.Background(), bytes.NewReader(data), nil)
	require.NoError(t, err)
	require.Equal(t, want, got)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &countingReader{r: bytes.NewReader(data)}
	_, err = DecodeContext(ctx, r, nil)
	require.ErrorIs(t, err, context.Canceled)
	require.Zero(t, r.n, "the reader must not be read once cancelled")

	dctx, dcancel := context.WithTimeout(context.Background(), -time.Second)
	defer dcancel()
	dec, err := decoder.NewDecoder(bytes.NewReader(data), &decoder.Options{UseThreads: true})
	require.NoError(t, err)
	dst := &image.NRGBA{}
	_, err = dec.DecodeIntoContext(dctx, dst)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, &image.NRGBA{}, dst)
	img, err := dec.DecodeIntoContext(context.Background(), dst)
	require.NoError(t, err)
	require.Same(t, dst, img)
	require.Equal(t, want, img)
}
