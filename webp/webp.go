package webp

import (
	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/encoder"
	"image"
	"io"
)

func Decode(r io.Reader, options *decoder.Options) (image.Image, error) {
	if dec, err := decoder.NewDecoder(r, options); err != nil {
		return nil, err
	} else {
		return dec.Decode()
	}
}

func Encode(w io.Writer, src image.Image, options *encoder.Options) error {
	if enc, err := encoder.NewEncoder(src, options); err != nil {
		return err
	} else {
		return enc.Encode(w)
	}
}
