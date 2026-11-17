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
	"runtime"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/vp8enc"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// maxDimension is the largest width or height of a WebP image.
const maxDimension = 16383

// ErrReleased is returned by Encode when it is called after Release.
var ErrReleased = errors.New("encoder: Encode called after Release")

// Encoder stores information to encode image
type Encoder struct {
	options *Options
	config  *Config
	// img is the picture to encode. Like v1 (which imported the source into
	// a WebPPicture in NewEncoder), it is a snapshot taken by NewEncoder,
	// except for an *image.NRGBA source, which is used in place (v1 shared
	// its pixels too): changes made to any other source after NewEncoder
	// do not affect Encode.
	img *image.NRGBA
	// pic is set instead of img for *image.YCbCr sources encoded lossy:
	// they are converted in NewEncoder straight to the encoder's YUV 4:2:0
	// (cheaper than going through NRGBA).
	pic *vp8enc.Picture
	// bounds of the source image
	bounds image.Rectangle

	// mu is held for reading by every encode and for writing by Release,
	// which therefore waits for the encodes in progress.
	mu       sync.RWMutex
	released bool
	// snap is the pooled storage of the snapshot (img's pixels or pic), if
	// any. Release hands it back to the pools; if Release is never called,
	// cleanup does so once the Encoder is unreachable.
	snap    *snapshot
	cleanup runtime.Cleanup
}

// snapshot is the pooled storage of an Encoder's copy of its source. It
// must not reference the Encoder: it is the argument of the Encoder's
// cleanup.
type snapshot struct {
	pix *[]byte         // storage of Encoder.img, from pooledNRGBA
	pic *vp8enc.Picture // Encoder.pic, from vp8enc.FromYCbCr
}

func (s *snapshot) release() {
	putPix(s.pix)
	s.pix = nil
	s.pic.Release()
	s.pic = nil
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

	// The snapshot's storage comes from pools. Encode does not consume it
	// (Encode may be called several times); Release returns it, and so does
	// a cleanup, as a safety net, when the Encoder becomes unreachable.
	s := &snapshot{}
	if v, ok := src.(*image.NRGBA); ok {
		e.img = v
	} else if ycc, ok := src.(*image.YCbCr); ok && !config.Lossless {
		e.pic = vp8enc.FromYCbCr(ycc)
		s.pic = e.pic
	} else {
		e.img, s.pix = pooledNRGBA(e.bounds.Dx(), e.bounds.Dy())
		convertNRGBA(e.img, src)
	}
	if s.pix != nil || s.pic != nil {
		e.snap = s
		e.cleanup = runtime.AddCleanup(e, (*snapshot).release, s)
	}
	return e, nil
}

func validDimensions(r image.Rectangle) bool {
	w, h := r.Dx(), r.Dy()
	return w > 0 && h > 0 && w <= maxDimension && h <= maxDimension
}

// Encode picture and flush to io.Writer
func (e *Encoder) Encode(w io.Writer) error {
	out, err := e.encode(nil)
	if err != nil {
		return err
	}
	_, err = w.Write(out)
	return err
}

// Release hands the Encoder's copy of its source picture back to internal
// pools, so that later encodes can reuse the memory. Encode returns
// ErrReleased afterwards. Release waits for the encodes in progress on
// other goroutines; calling it again does nothing.
//
// Calling Release is optional: otherwise the memory is reclaimed by the
// garbage collector (and returned to the pools once the Encoder is
// unreachable), as with v1, which had no Release.
func (e *Encoder) Release() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.released {
		return
	}
	e.released = true
	if e.snap != nil {
		e.cleanup.Stop()
		e.snap.release()
		e.snap = nil
	}
	e.img, e.pic = nil, nil
}

// encode returns the WebP file. interrupt, if not nil, is polled by the
// encoders, which give up when it returns true.
func (e *Encoder) encode(interrupt func() bool) ([]byte, error) {
	// Holding the read lock until the snapshot is no longer used also keeps
	// e reachable meanwhile, so that its cleanup cannot recycle the storage
	// under the encoders.
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.released {
		return nil, ErrReleased
	}
	if !validDimensions(e.bounds) {
		return nil, &encodeError{fmt.Errorf("bad dimension %dx%d", e.bounds.Dx(), e.bounds.Dy())}
	}

	var (
		out []byte
		err error
	)
	if e.config.Lossless {
		out, err = e.encodeLossless(interrupt)
	} else {
		out, err = e.encodeLossy(interrupt)
	}
	if err != nil {
		return nil, &encodeError{err}
	}
	return out, nil
}

// interrupted reports whether err comes from an encoder that stopped
// because its interrupt hook asked it to.
func interrupted(err error) bool {
	return errors.Is(err, vp8enc.ErrInterrupted) || errors.Is(err, vp8lenc.ErrInterrupted) ||
		errors.Is(err, errAlphaInterrupted)
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

func (e *Encoder) encodeLossless(interrupt func() bool) ([]byte, error) {
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
		Interrupt:       interrupt,
	}, container.LosslessHeaderSize)
	if err != nil {
		return nil, err
	}
	// The payload was written after LosslessHeaderSize bytes of headroom:
	// add the RIFF headers in place instead of copying it.
	return container.FrameLossless(buf)
}

func (e *Encoder) encodeLossy(interrupt func() bool) ([]byte, error) {
	c := e.config
	cfg := vp8enc.Config{
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
		Interrupt:       interrupt,
	}

	var (
		payload []byte
		alph    []byte
		err     error
	)
	if e.pic != nil {
		payload, err = vp8Encode(e.pic, &cfg)
	} else {
		var at *alphaTask
		if hasTransparency(e.img) {
			if c.ThreadLevel {
				// Like libwebp's alpha worker (thread_level > 0), code the
				// alpha plane concurrently with the VP8 frame when a helper
				// goroutine is available.
				at = startAlpha(e.img, c, interrupt)
			}
			if at != nil {
				// Joins the helper on every return path, panics included.
				defer at.cancel()
			} else if alph, err = alphaEncode(e.img, c, interrupt); err != nil {
				return nil, err
			}
		}
		dithering := vp8enc.DitheringFor(&vp8enc.Config{Quality: c.Quality, Preprocessing: c.Preprocessing})
		pic := vp8enc.FromNRGBADithered(e.img, c.UseSharpYUV || c.Preprocessing&4 != 0, c.Exact, dithering)
		payload, err = vp8Encode(pic, &cfg)
		pic.Release()
		if at != nil {
			// The sequential order codes the alpha plane first, so its
			// errors take precedence, except an interruption caused by the
			// failure of the VP8 frame.
			var aerr error
			alph, aerr = at.wait(err != nil)
			if aerr != nil && (err == nil || !interrupted(aerr)) {
				return nil, aerr
			}
		}
	}
	if err != nil {
		return nil, err
	}

	sw := sliceWriter{b: make([]byte, 0, len(payload)+len(alph)+64)}
	if err := container.WriteLossy(&sw, e.bounds.Dx(), e.bounds.Dy(), payload, alph); err != nil {
		return nil, err
	}
	return sw.b, nil
}
