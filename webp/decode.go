package webp

import (
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
