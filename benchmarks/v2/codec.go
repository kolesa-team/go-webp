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

// Package v2 adapts this repository's v2 (pure Go, with assembly kernels on
// arm64 and amd64) to the benchmark suite. Build with -tags purego for the
// pure Go code path that the other architectures run.
package v2

import (
	"bytes"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/benchmarks/suite"
	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/utils"
	"github.com/kolesa-team/go-webp/v2/webp"
)

// Codec is the v2 adapter.
type Codec struct{}

var _ suite.Codec = Codec{}

func (Codec) Decoder(data []byte, mode string) (func() error, bool) {
	var o decoder.Options
	switch mode {
	case "default":
	case "threads":
		o.UseThreads = true
	case "scale":
		o.Scale = image.Rect(0, 0, 320, 240)
	case "into":
		var dst *image.NRGBA
		return func() error {
			d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
			if err != nil {
				return err
			}
			dst, err = d.DecodeInto(dst)
			return err
		}, true
	case "ycbcr":
		d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
		if err != nil {
			return nil, false
		}
		f := d.GetFeatures()
		if f.Format == utils.FormatLossless {
			return nil, false // lossless pictures have no Y'CbCr planes
		}
		return func() error {
			d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
			if err != nil {
				return err
			}
			if f.HasAlpha {
				_, err = d.DecodeNYCbCrA()
			} else {
				_, err = d.DecodeYCbCr()
			}
			return err
		}, true
	default:
		return nil, false
	}
	return func() error {
		d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
		if err != nil {
			return err
		}
		_, err = d.Decode()
		return err
	}, true
}

func (Codec) Decode(data []byte) (image.Image, error) {
	return webp.Decode(bytes.NewReader(data), &decoder.Options{})
}

func (Codec) DecodeInto(data []byte, dst *image.NRGBA) (*image.NRGBA, error) {
	return webp.DecodeInto(bytes.NewReader(data), dst, &decoder.Options{})
}

func (Codec) Encode(w io.Writer, img image.Image, c suite.EncodeCase, threads bool) error {
	var o *encoder.Options
	var err error
	if c.Lossless {
		o, err = encoder.NewLosslessEncoderOptions(encoder.PresetDefault, c.Level)
	} else {
		o, err = encoder.NewLossyEncoderOptions(encoder.PresetDefault, float32(c.Level))
	}
	if err != nil {
		return err
	}
	if !c.Lossless {
		o.Method = c.Method
	}
	o.ThreadLevel = threads
	return webp.Encode(w, img, o)
}
