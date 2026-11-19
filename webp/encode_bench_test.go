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
	"image"
	"image/draw"
	"io"
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/v2/encoder"
)

// loadLogo decodes the 3000x2000 transparent logo (lossy encodes of it carry
// an ALPH chunk).
func loadLogo(b testing.TB) *image.NRGBA {
	b.Helper()
	data, err := os.ReadFile("../test_data/images/webp-logo-lossless.webp")
	if err != nil {
		b.Fatal(err)
	}
	img, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		b.Fatal(err)
	}
	return img.(*image.NRGBA)
}

func benchmarkEncodeImage(b *testing.B, img image.Image, o *encoder.Options) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Encode(io.Discard, img, o); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeLossyAlpha encodes a transparent picture lossy: VP8 frame
// plus compressed ALPH chunk, sequentially.
func BenchmarkEncodeLossyAlpha(b *testing.B) {
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	benchmarkEncodeImage(b, loadLogo(b), o)
}

// BenchmarkEncodeLossyAlphaThreads is BenchmarkEncodeLossyAlpha with
// ThreadLevel, which codes the alpha plane concurrently with the VP8 frame.
func BenchmarkEncodeLossyAlphaThreads(b *testing.B) {
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	o.ThreadLevel = true
	benchmarkEncodeImage(b, loadLogo(b), o)
}

// BenchmarkEncodeRGBA encodes an opaque *image.RGBA (what png.Decode returns
// for RGB files) lossy through the one-shot Encode.
func BenchmarkEncodeRGBA(b *testing.B) {
	src := loadJPEG(b, "source.jpg")
	img := image.NewRGBA(src.Bounds())
	draw.Draw(img, img.Rect, src, src.Bounds().Min, draw.Src)
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	benchmarkEncodeImage(b, img, o)
}

// BenchmarkEncoderNoRelease uses NewEncoder and Encode without Release, as
// code written for v1 does.
func BenchmarkEncoderNoRelease(b *testing.B) {
	img := loadJPEG(b, "source.jpg")
	o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e, err := encoder.NewEncoder(img, o)
		if err != nil {
			b.Fatal(err)
		}
		if err := e.Encode(io.Discard); err != nil {
			b.Fatal(err)
		}
	}
}
