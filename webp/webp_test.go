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
	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/encoder"
	"golang.org/x/image/webp"
	"image"
	"image/jpeg"
	"io/ioutil"
	"os"
	"testing"
)

func TestEncode(t *testing.T) {
	t.Run("encode lossy", func(t *testing.T) {
		r, err := os.Open("../test_data/images/source.jpg")
		if err != nil {
			t.Fatal(err)
		}

		img, err := jpeg.Decode(r)
		if err != nil {
			t.Fatal(err)
		}

		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			t.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("encode lossless", func(t *testing.T) {
		r, err := os.Open("../test_data/images/source.jpg")
		if err != nil {
			t.Fatal(err)
		}

		img, err := jpeg.Decode(r)
		if err != nil {
			t.Fatal(err)
		}

		options, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 4)
		if err != nil {
			t.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDecode(t *testing.T) {
	r, err := os.Open("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}

	if img, err := Decode(r, &decoder.Options{
		BypassFiltering:   true,
		NoFancyUpsampling: true,

		Crop: image.Rectangle{},
		Scale: image.Rectangle{
			Max: image.Point{
				X: 400,
				Y: 300,
			},
		},

		UseThreads:             true,
		Flip:                   true,
		DitheringStrength:      1,
		AlphaDitheringStrength: 1,
	}); err != nil {
		t.Fatal(err)
	} else if img == nil {
		t.Fatal("img is empty")
	} else if img.Bounds().Max.X != 400 || img.Bounds().Max.Y != 300 {
		t.Fatal("img is invalid")
	}
}

func BenchmarkDecodeLossy(b *testing.B) {
	data, err := ioutil.ReadFile("../test_data/images/webp-logo-lossy.webp")
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		dec, err := decoder.NewDecoder(bytes.NewBuffer(data), &decoder.Options{})
		if err != nil {
			b.Fatal(err)
		}

		if _, err := dec.Decode(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeXImageLossy(b *testing.B) {
	data, err := ioutil.ReadFile("../test_data/images/webp-logo-lossy.webp")
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		_, err = webp.Decode(bytes.NewBuffer(data))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLossless(b *testing.B) {
	data, err := ioutil.ReadFile("../test_data/images/webp-logo-lossless.webp")
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		dec, err := decoder.NewDecoder(bytes.NewBuffer(data), &decoder.Options{})
		if err != nil {
			b.Fatal(err)
		}

		if _, err := dec.Decode(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeXImageLossless(b *testing.B) {
	data, err := ioutil.ReadFile("../test_data/images/webp-logo-lossless.webp")
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		_, err = webp.Decode(bytes.NewBuffer(data))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	r, err := os.Open("../test_data/images/source.jpg")
	if err != nil {
		b.Fatal(err)
	}

	img, err := jpeg.Decode(r)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			b.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			b.Fatal(err)
		}
	}
}
