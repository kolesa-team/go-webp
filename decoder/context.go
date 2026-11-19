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

package decoder

import (
	"context"
	"image"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// DecodeContext is Decode with cancellation: once ctx is cancelled or its
// deadline has passed, decoding stops at the next batch of output rows (at
// most 16 rows of the frame; a lossless picture first reads its headers),
// and the error satisfies errors.Is(err, context.Canceled) (or
// context.DeadlineExceeded); its status code is VP8_STATUS_USER_ABORT.
// Helper goroutines (UseThreads) have exited when DecodeContext returns.
func (d *Decoder) DecodeContext(ctx context.Context) (image.Image, error) {
	img, err := d.decodeInto(call{ctx: ctx})
	if err != nil {
		return nil, err
	}
	return img, nil
}

// DecodeIntoContext is DecodeInto with cancellation, see DecodeContext.
func (d *Decoder) DecodeIntoContext(ctx context.Context, dst *image.NRGBA) (*image.NRGBA, error) {
	return d.decodeInto(call{dst: dst, ctx: ctx})
}

// abortError is the error of a cancelled decode: it matches both the
// context's error and a VP8_STATUS_USER_ABORT status (libwebp's code for a
// decode stopped by the application).
type abortError struct {
	err error // ctx.Err()
	st  error
}

func (e *abortError) Error() string   { return e.err.Error() }
func (e *abortError) Unwrap() []error { return []error{e.err, e.st} }

// canceled returns an abortError once ctx (which may be nil) is done.
func canceled(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return &abortError{err: err, st: status.New(utils.Vp8StatusUserAbort, "decoding cancelled")}
	}
	return nil
}

// wrap returns sink, checking the context of the call after each committed
// batch of rows when it can be cancelled.
func (c call) wrap(sink pipeline.Sink) pipeline.Sink {
	if c.ctx == nil || c.ctx.Done() == nil {
		return sink
	}
	return &ctxSink{Sink: sink, ctx: c.ctx}
}

// ctxSink stops the decoder (Commit fails) once the context is done.
type ctxSink struct {
	pipeline.Sink
	ctx context.Context
}

func (s *ctxSink) Commit(y, n int) error {
	if err := s.Sink.Commit(y, n); err != nil {
		return err
	}
	return canceled(s.ctx)
}

// AcceptPremultiplied forwards pipeline.PremultipliedSink to the wrapped
// sink, so that a cancellable decode keeps the lossless decoder's
// palette-domain premultiplication (Scale of palette images).
func (s *ctxSink) AcceptPremultiplied() bool {
	p, ok := s.Sink.(pipeline.PremultipliedSink)
	return ok && p.AcceptPremultiplied()
}
