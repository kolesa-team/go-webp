// Package v1 adapts github.com/kolesa-team/go-webp v1.0.5 (cgo bindings to
// libwebp) to the benchmark suite. It needs cgo and libwebp with its
// pkg-config file (for example "brew install webp" or "apt install
// libwebp-dev").
package v1

import (
	"bytes"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/benchmarks/suite"
	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/encoder"
	"github.com/kolesa-team/go-webp/webp"
)

// Codec is the v1 adapter.
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
	default:
		return nil, false // DecodeInto and DecodeYCbCr are v2 additions
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

func (Codec) DecodeInto([]byte, *image.NRGBA) (*image.NRGBA, error) {
	return nil, suite.ErrUnsupported
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
