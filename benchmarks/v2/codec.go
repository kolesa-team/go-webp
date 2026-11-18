// Package v2 adapts this repository's v2 (pure Go, NEON assembly on arm64)
// to the benchmark suite. Build with -tags purego for the pure Go code path
// that amd64 and the other architectures run.
package v2

import (
	"bytes"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/benchmarks/suite"
	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/utils"
	"github.com/kolesa-team/go-webp/v2/webp"
)

// Codec is the v2 adapter.
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
	case "into":
		var dst *image.NRGBA
		return func() error {
			d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
			if err != nil {
				return err
			}
			dst, err = d.DecodeInto(dst)
			return err
		}, true
	case "ycbcr":
		d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
		if err != nil {
			return nil, false
		}
		f := d.GetFeatures()
		if f.Format == utils.FormatLossless {
			return nil, false // lossless pictures have no Y'CbCr planes
		}
		return func() error {
			d, err := decoder.NewDecoder(bytes.NewReader(data), &o)
			if err != nil {
				return err
			}
			if f.HasAlpha {
				_, err = d.DecodeNYCbCrA()
			} else {
				_, err = d.DecodeYCbCr()
			}
			return err
		}, true
	default:
		return nil, false
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

func (Codec) DecodeInto(data []byte, dst *image.NRGBA) (*image.NRGBA, error) {
	return webp.DecodeInto(bytes.NewReader(data), dst, &decoder.Options{})
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
