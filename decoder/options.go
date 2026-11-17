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
	"image"
)

// Options specifies webp decoding parameters.
//
// Crop uses libwebp's convention: Crop.Min is the top-left corner
// (crop_left, crop_top) and Crop.Max holds the crop WIDTH and HEIGHT
// (crop_width, crop_height), not the bottom-right corner. Cropping is enabled
// when both Crop.Max.X and Crop.Max.Y are positive. For lossy images the
// corner is rounded down to even coordinates, as libwebp does.
//
// Scale.Max holds the output width and height; scaling is enabled when both
// are positive. Scaling is applied after cropping.
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

// Config is the decoding configuration built by Options.GetConfig. It mirrors
// libwebp's WebPDecoderOptions field for field (C int flags are bools). In v1
// GetConfig returned the cgo type *C.WebPDecoderConfig; v2 returns this pure
// Go equivalent.
type Config struct {
	// BypassFiltering skips the in-loop deblocking filter (lossy).
	BypassFiltering bool
	// NoFancyUpsampling uses point-sampled chroma upsampling (lossy).
	NoFancyUpsampling bool
	// UseCropping enables cropping to the CropWidth x CropHeight rectangle
	// whose top-left corner is (CropLeft, CropTop).
	UseCropping bool
	CropLeft    int
	CropTop     int
	CropWidth   int
	CropHeight  int
	// UseScaling enables rescaling of the (cropped) picture to
	// ScaledWidth x ScaledHeight.
	UseScaling   bool
	ScaledWidth  int
	ScaledHeight int
	// UseThreads decodes with helper goroutines from the process-wide
	// budget of GOMAXPROCS-1 helpers shared by all concurrent Decode and
	// Encode calls:
	//   - lossy pictures of at least 1024 macroblocks (about 512x512):
	//     one helper reconstructs, filters and emits batches of macroblock
	//     rows while the calling goroutine parses the next ones;
	//   - lossy pictures with alpha whose alpha rows down to the bottom of
	//     the output region hold 256K pixels or more: another helper
	//     decodes the alpha plane ahead of the rows that need it;
	//   - lossless pictures of at least about 256K output pixels: the
	//     calling goroutine entropy-decodes while one helper applies the
	//     inverse transforms and writes the output rows (libwebp ignores
	//     use_threads for lossless pictures; this is a go-webp extension).
	// Smaller pictures, or a call that finds no free helper slot, decode on
	// the calling goroutine. The output is identical either way. A panic
	// in a helper is re-raised on the calling goroutine.
	UseThreads bool
	// DitheringStrength in [0, 100] dithers the chroma of lossy pictures.
	DitheringStrength int
	// Flip mirrors the output vertically.
	Flip bool
	// AlphaDitheringStrength in [0, 100] smooths quantized alpha planes.
	AlphaDitheringStrength int
}

// GetConfig builds the decoder configuration (the equivalent of libwebp's
// WebPDecoderConfig options). A nil *Options yields the default
// configuration.
func (o *Options) GetConfig() (*Config, error) {
	config := &Config{}
	if o == nil {
		return config, nil
	}

	config.BypassFiltering = o.BypassFiltering
	config.NoFancyUpsampling = o.NoFancyUpsampling

	// проверяем надо ли кропнуть
	if o.Crop.Max.X > 0 && o.Crop.Max.Y > 0 {
		config.UseCropping = true
		config.CropLeft = o.Crop.Min.X
		config.CropTop = o.Crop.Min.Y
		config.CropWidth = o.Crop.Max.X
		config.CropHeight = o.Crop.Max.Y
	}

	// проверяем надо ли заскейлить
	if o.Scale.Max.X > 0 && o.Scale.Max.Y > 0 {
		config.UseScaling = true
		config.ScaledWidth = o.Scale.Max.X
		config.ScaledHeight = o.Scale.Max.Y
	}

	config.UseThreads = o.UseThreads
	config.DitheringStrength = o.DitheringStrength
	config.Flip = o.Flip
	config.AlphaDitheringStrength = o.AlphaDitheringStrength

	return config, nil
}
