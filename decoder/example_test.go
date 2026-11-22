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

package decoder_test

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"os"
	"time"

	"github.com/kolesa-team/go-webp/v2/decoder"
)

// Decoding many pictures into one reused image: after the buffer has grown
// to the largest picture, decoding allocates almost nothing.
func ExampleDecoder_DecodeInto() {
	var img *image.NRGBA // nil: the first decode allocates it
	for _, name := range []string{"m4_q75.webp", "100x150_lossy.webp", "webp-logo-lossless.webp"} {
		data, err := os.ReadFile("../test_data/images/" + name)
		if err != nil {
			panic(err)
		}
		dec, err := decoder.NewDecoder(bytes.NewReader(data), nil)
		if err != nil {
			panic(err)
		}
		// The result is img itself (or the image allocated for it):
		// process it before the next decode overwrites it.
		if img, err = dec.DecodeInto(img); err != nil {
			panic(err)
		}
		fmt.Println(name, img.Bounds().Size())
	}
	// Output:
	// m4_q75.webp (675,900)
	// 100x150_lossy.webp (100,150)
	// webp-logo-lossless.webp (3000,2000)
}

// Bounding the time spent on one picture.
func ExampleDecoder_DecodeContext() {
	data, err := os.ReadFile("../test_data/images/m4_q75.webp")
	if err != nil {
		panic(err)
	}
	dec, err := decoder.NewDecoder(bytes.NewReader(data), nil)
	if err != nil {
		panic(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	img, err := dec.DecodeContext(ctx)
	if err != nil {
		panic(err) // errors.Is(err, context.DeadlineExceeded) after a minute
	}
	fmt.Println(img.Bounds().Size())
	// Output: (675,900)
}

// Decoding a lossy picture to its Y'CbCr planes, without the conversion to
// RGB.
func ExampleDecoder_DecodeYCbCr() {
	data, err := os.ReadFile("../test_data/images/m4_q75.webp")
	if err != nil {
		panic(err)
	}
	dec, err := decoder.NewDecoder(bytes.NewReader(data), nil)
	if err != nil {
		panic(err)
	}
	if f := dec.GetFeatures(); f.HasAlpha {
		panic("use DecodeNYCbCrA")
	}
	img, err := dec.DecodeYCbCr()
	if err != nil {
		panic(err)
	}
	fmt.Println(img.Bounds().Size(), img.SubsampleRatio, len(img.Y), len(img.Cb))
	// Output: (675,900) YCbCrSubsampleRatio420 607500 152100
}
