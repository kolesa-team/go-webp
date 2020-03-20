package decoder

/*
#cgo LDFLAGS: -lwebp
#include <webp/decode.h>
*/
import "C"
import (
	"errors"
	"image"
)

type Options struct {
	BypassFiltering        bool
	NoFancyUpsampling      bool
	Crop                   image.Rectangle
	Scale                  image.Rectangle
	UseThreads             bool
	Flip                   bool
	DitheringStrength      int
	AlphaDitheringStrength int
}

func (o *Options) GetConfig() (*C.WebPDecoderConfig, error) {
	config := C.WebPDecoderConfig{}

	if C.WebPInitDecoderConfig(&config) == 0 {
		return nil, errors.New("cannot init decoder config")
	}

	if o.BypassFiltering {
		config.options.bypass_filtering = 1
	}

	if o.NoFancyUpsampling {
		config.options.no_fancy_upsampling = 1
	}

	// проверяем надо ли кропнуть
	if o.Crop.Max.X > 0 && o.Crop.Max.Y > 0 {
		config.options.use_cropping = 1
		config.options.crop_left = C.int(o.Crop.Min.X)
		config.options.crop_top = C.int(o.Crop.Min.Y)
		config.options.crop_width = C.int(o.Crop.Max.X)
		config.options.crop_height = C.int(o.Crop.Max.Y)
	}

	// проверяем надо ли заскейлить
	if o.Scale.Max.X > 0 && o.Scale.Max.Y > 0 {
		config.options.use_scaling = 1
		config.options.scaled_width = C.int(o.Scale.Max.X)
		config.options.scaled_height = C.int(o.Scale.Max.Y)
	}

	if o.UseThreads {
		config.options.use_threads = 1
	}

	config.options.dithering_strength = C.int(o.DitheringStrength)

	if o.Flip {
		config.options.flip = 1
	}

	config.options.alpha_dithering_strength = C.int(o.AlphaDitheringStrength)

	return &config, nil
}
