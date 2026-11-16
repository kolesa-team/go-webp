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

// Package webp encodes and decodes WebP images with a pure Go implementation
// of libwebp's codecs (no cgo). Importing it registers the "webp" format with
// the standard image package.
package webp

import (
	"errors"
	"image"
	"image/color"
	"io"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func init() {
	image.RegisterFormat("webp", "RIFF????WEBPVP8", quickDecode, quickDecodeConfig)
}

func quickDecode(r io.Reader) (image.Image, error) {
	return Decode(r, &decoder.Options{})
}

func quickDecodeConfig(r io.Reader) (image.Config, error) {
	return DecodeConfig(r, &decoder.Options{})
}

// Decode picture from reader
func Decode(r io.Reader, options *decoder.Options) (image.Image, error) {
	dec, err := decoder.NewDecoder(r, options)
	if err != nil {
		return nil, err
	}
	return dec.Decode()
}

// DecodeConfig returns the dimensions of a WebP picture without decoding its
// pixels. Only the headers up to the VP8/VP8L frame header are read from r;
// they are validated like decoder.NewDecoder does.
//
// For compatibility with v1 the reported ColorModel is color.RGBAModel,
// although Decode returns an *image.NRGBA.
func DecodeConfig(r io.Reader, options *decoder.Options) (image.Config, error) {
	if _, err := options.GetConfig(); err != nil {
		return image.Config{}, err
	}
	f, err := readFeatures(r)
	if err != nil {
		return image.Config{}, err
	}
	return image.Config{
		ColorModel: color.RGBAModel,
		Width:      f.Width,
		Height:     f.Height,
	}, nil
}

// GetFeatures returns the features of a WebP picture (dimensions, alpha,
// animation, lossy/lossless) reading only its headers from r.
func GetFeatures(r io.Reader) (utils.BitstreamFeatures, error) {
	f, err := readFeatures(r)
	if err != nil {
		return utils.BitstreamFeatures{}, err
	}
	return utils.BitstreamFeatures{
		Width:        f.Width,
		Height:       f.Height,
		HasAlpha:     f.HasAlpha,
		HasAnimation: f.HasAnimation,
		Format:       f.Format,
	}, nil
}

// featuresError formats like decoder.NewDecoder's errors and unwraps to the
// underlying *status.Error.
type featuresError struct{ err error }

func (e *featuresError) Error() string {
	return "cannot fetch features: " + status.CodeOf(e.err).String()
}

func (e *featuresError) Unwrap() error { return e.err }

// readFeatures reads a growing prefix of r until the WebP headers up to the
// VP8/VP8L frame header can be parsed (or r ends), and returns the same
// features and errors as container.GetFeatures on the whole file, which is
// what decoder.NewDecoder (and libwebp's WebPGetFeatures) report.
func readFeatures(r io.Reader) (container.Features, error) {
	buf := make([]byte, 0, 64)
	eof := false
	for {
		for !eof && len(buf) < cap(buf) {
			n, err := r.Read(buf[len(buf):cap(buf)])
			buf = buf[:len(buf)+n]
			if err == io.EOF {
				eof = true
			} else if err != nil {
				return container.Features{}, err
			}
		}
		if len(buf) == 0 {
			return container.Features{}, errors.New("data is empty")
		}
		f, done, err := container.GetFeaturesPrefix(buf)
		if !done && eof {
			f, err = container.GetFeatures(buf)
			done = true
		}
		if done {
			if err != nil {
				return container.Features{}, &featuresError{err: err}
			}
			return f, nil
		}
		nb := make([]byte, len(buf), 2*cap(buf))
		copy(nb, buf)
		buf = nb
	}
}

// Encode picture and write to io.Writer
func Encode(w io.Writer, src image.Image, options *encoder.Options) error {
	enc, err := encoder.NewEncoder(src, options)
	if err != nil {
		return err
	}
	return enc.Encode(w)
}
