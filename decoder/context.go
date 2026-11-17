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
