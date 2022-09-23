package webp

// Importing this package allows decoding of webp image files using the standard library's image.Decode.
// It will clash with the golang.org/x/image/webp package

import (
	"image"
	"io"

	"github.com/kolesa-team/go-webp/webp"
)

func init() {
	image.RegisterFormat("webp", "RIFF????WEBPVP8", quickDecode, quickDecodeConfig)
}

func quickDecode(r io.Reader) (image.Image, error) {
	return webp.Decode(r, nil)
}

func quickDecodeConfig(r io.Reader) (image.Config, error) {
	return webp.DecodeConfig(r, nil)
}
