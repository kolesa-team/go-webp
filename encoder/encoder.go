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

// Package encoder encodes images to WebP, lossy (VP8) or lossless (VP8L),
// with a pure Go implementation of libwebp's encoders.
package encoder

import (
	"errors"
	"fmt"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/vp8enc"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// maxDimension is the largest width or height of a WebP image.
const maxDimension = 16383

// Encoder stores information to encode image
type Encoder struct {
	options *Options
	config  *Config
	img     *image.NRGBA
	// ycc is set instead of img for opaque *image.YCbCr sources encoded
	// lossy: they are converted straight to the encoder's YUV 4:2:0.
	ycc *image.YCbCr
	// bounds of the source image
	bounds image.Rectangle
}

// NewEncoder return new encoder instance
func NewEncoder(src image.Image, options *Options) (e *Encoder, err error) {
	var config *Config

	if options == nil {
		options, _ = NewLossyEncoderOptions(PresetDefault, 75)
	}
	if config, err = options.GetConfig(); err != nil {
		return nil, err
	}
	if src == nil {
		return nil, errors.New("cannot encode webppicture: nil image")
	}

	e = &Encoder{options: options, config: config, bounds: src.Bounds()}
	if !validDimensions(e.bounds) {
		// Reported by Encode, as libwebp's WebPEncode does; skip converting
		// a picture that cannot be encoded.
		return e, nil
	}

	if ycc, ok := src.(*image.YCbCr); ok && !config.Lossless {
		e.ycc = ycc
	} else {
		e.img = toNRGBA(src)
	}

	return e, nil
}

func validDimensions(r image.Rectangle) bool {
	w, h := r.Dx(), r.Dy()
	return w > 0 && h > 0 && w <= maxDimension && h <= maxDimension
}

// Encode picture and flush to io.Writer
func (e *Encoder) Encode(w io.Writer) error {
	if !validDimensions(e.bounds) {
		return &encodeError{fmt.Errorf("bad dimension %dx%d", e.bounds.Dx(), e.bounds.Dy())}
	}

	var (
		out []byte
		err error
	)
	if e.config.Lossless {
		out, err = e.encodeLossless()
	} else {
		out, err = e.encodeLossy()
	}
	if err != nil {
		return &encodeError{err}
	}

	_, err = w.Write(out)
	return err
}

// encodeError reports an encoding failure with exactly v1's message,
// "cannot encode webppicture" (v1 returned it for every libwebp failure),
// and unwraps to the detailed cause.
type encodeError struct{ err error }

func (e *encodeError) Error() string { return "cannot encode webppicture" }

func (e *encodeError) Unwrap() error { return e.err }

// sliceWriter collects the container output in a single, pre-sized buffer,
// so that the destination io.Writer receives exactly one Write call.
type sliceWriter struct{ b []byte }

func (s *sliceWriter) Write(p []byte) (int, error) {
	s.b = append(s.b, p...)
	return len(p), nil
}

func (e *Encoder) encodeLossless() ([]byte, error) {
	c := e.config
	buf, err := vp8lenc.EncodeWithHeadroom(e.img, &vp8lenc.Config{
		Quality:         c.Quality,
		Method:          c.Method,
		Exact:           c.Exact,
		NearLossless:    c.NearLossless,
		UseDeltaPalette: c.UseDeltaPalette,
		LowMemory:       c.LowMemory,
		ThreadLevel:     c.ThreadLevel,
		ImageHint:       int(c.ImageHint),
	}, container.LosslessHeaderSize)
	if err != nil {
		return nil, err
	}
	// The payload was written after LosslessHeaderSize bytes of headroom:
	// add the RIFF headers in place instead of copying it.
	return container.FrameLossless(buf)
}

func (e *Encoder) encodeLossy() ([]byte, error) {
	c := e.config

	var (
		pic  *vp8enc.Picture
		alph []byte
		err  error
	)
	if e.ycc != nil {
		pic = vp8enc.FromYCbCr(e.ycc)
	} else {
		if hasTransparency(e.img) {
			if alph, err = encodeAlpha(e.img, c); err != nil {
				return nil, err
			}
		}
		dithering := vp8enc.DitheringFor(&vp8enc.Config{Quality: c.Quality, Preprocessing: c.Preprocessing})
		pic = vp8enc.FromNRGBADithered(e.img, c.UseSharpYUV || c.Preprocessing&4 != 0, c.Exact, dithering)
	}
	defer pic.Release()

	payload, err := vp8enc.Encode(pic, &vp8enc.Config{
		Quality:         c.Quality,
		Method:          c.Method,
		ImageHint:       int(c.ImageHint),
		TargetSize:      c.TargetSize,
		TargetPSNR:      c.TargetPSNR,
		Pass:            c.Pass,
		Segments:        c.Segments,
		SNSStrength:     c.SNSStrength,
		FilterStrength:  c.FilterStrength,
		FilterSharpness: c.FilterSharpness,
		FilterType:      c.FilterType,
		Autofilter:      c.Autofilter,
		Preprocessing:   c.Preprocessing,
		Partitions:      c.Partitions,
		PartitionLimit:  c.PartitionLimit,
		EmulateJPEGSize: c.EmulateJPEGSize,
		ThreadLevel:     c.ThreadLevel,
		LowMemory:       c.LowMemory,
	})
	if err != nil {
		return nil, err
	}

	sw := sliceWriter{b: make([]byte, 0, len(payload)+len(alph)+64)}
	if err := container.WriteLossy(&sw, e.bounds.Dx(), e.bounds.Dy(), payload, alph); err != nil {
		return nil, err
	}
	return sw.b, nil
}
