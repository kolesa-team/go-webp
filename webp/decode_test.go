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
	"errors"
	"fmt"
	"image"
	"os"
	"reflect"
	"testing"
	"time"

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
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions: %v", err)
	}
	lossy, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	for _, s := range synthImages(w, h) {
		for name, o := range map[string]*encoder.Options{"lossy": lossy, "lossless": lossless} {
			buf := &bytes.Buffer{}
			if err := Encode(buf, s.img, o); err != nil {
				t.Fatalf("Encode(%s %s %dx%d): %v", s.name, name, w, h, err)
			}
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
				if err != nil {
					t.Fatalf("Decode(%s %+v): %v", name, o, err)
				}
				got, err := DecodeInto(bytes.NewReader(data), dst, o)
				if err != nil {
					t.Fatalf("DecodeInto(%s %+v): %v", name, o, err)
				}
				if got != dst {
					t.Fatalf("DecodeInto(%s %+v) returned %p, want dst %p", name, o, got, dst)
				}
				if !reflect.DeepEqual(want, got) {
					t.Fatalf("DecodeInto(%s %+v) = %v (len(Pix) %d), differs from Decode = %v", name, o, got.Rect, len(got.Pix), want.Bounds())
				}
				if c := cap(got.Pix); c != 60000 {
					t.Fatalf("%s %+v: cap(Pix) = %d, want 60000: buffer not reused", name, o, c)
				}
			}
		}
	}
}

func TestDecodeContext(t *testing.T) {
	data, err := os.ReadFile("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}
	want, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	got, err := DecodeContext(context.Background(), bytes.NewReader(data), nil)
	if err != nil {
		t.Fatalf("DecodeContext: %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("DecodeContext = %v, differs from Decode = %v", got.Bounds(), want.Bounds())
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &countingReader{r: bytes.NewReader(data)}
	_, err = DecodeContext(ctx, r, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("DecodeContext(canceled) error = %v, want %v", err, context.Canceled)
	}
	if r.n != 0 {
		t.Fatalf("DecodeContext(canceled) read %d bytes, want 0: the reader must not be read once cancelled", r.n)
	}

	dctx, dcancel := context.WithTimeout(context.Background(), -time.Second)
	defer dcancel()
	dec, err := decoder.NewDecoder(bytes.NewReader(data), &decoder.Options{UseThreads: true})
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	dst := &image.NRGBA{}
	_, err = dec.DecodeIntoContext(dctx, dst)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("DecodeIntoContext(expired) error = %v, want %v", err, context.DeadlineExceeded)
	}
	if !reflect.DeepEqual(&image.NRGBA{}, dst) {
		t.Fatalf("DecodeIntoContext(expired) changed dst: Rect %v, Stride %d, len(Pix) %d, want a zero image.NRGBA", dst.Rect, dst.Stride, len(dst.Pix))
	}
	img, err := dec.DecodeIntoContext(context.Background(), dst)
	if err != nil {
		t.Fatalf("DecodeIntoContext: %v", err)
	}
	if img != dst {
		t.Fatalf("DecodeIntoContext returned %p, want dst %p", img, dst)
	}
	if !reflect.DeepEqual(want, img) {
		t.Fatalf("DecodeIntoContext = %v (len(Pix) %d), differs from Decode = %v", img.Rect, len(img.Pix), want.Bounds())
	}
}
