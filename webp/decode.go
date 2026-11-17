package webp

import (
	"context"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/v2/decoder"
)

// DecodeInto decodes a picture from r into dst, reusing the memory of
// dst.Pix when it is large enough, and returns dst (a new image when dst is
// nil). See decoder.Decoder.DecodeInto for the exact semantics.
func DecodeInto(r io.Reader, dst *image.NRGBA, options *decoder.Options) (*image.NRGBA, error) {
	dec, err := decoder.NewDecoder(r, options)
	if err != nil {
		return nil, err
	}
	return dec.DecodeInto(dst)
}

// DecodeContext is Decode with cancellation: ctx is checked before r is
// read and between batches of output rows, see
// decoder.Decoder.DecodeContext.
func DecodeContext(ctx context.Context, r io.Reader, options *decoder.Options) (image.Image, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dec, err := decoder.NewDecoder(r, options)
	if err != nil {
		return nil, err
	}
	return dec.DecodeContext(ctx)
}
