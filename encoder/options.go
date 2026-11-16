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

package encoder

import (
	"errors"
	"fmt"
	"math"
)

// Default libwebp image hints
const (
	HintDefault ImageHint = iota
	HintPicture
	HintPhoto
	HintGraph
	HintLast
)

// Default libwebp presets
const (
	PresetDefault EncodingPreset = iota
	PresetPicture
	PresetPhoto
	PresetDrawing
	PresetIcon
	PresetText
)

type (
	// ImageHint hint of picture
	ImageHint int
	// EncodingPreset using Preset
	EncodingPreset int
	// Options specifies webp encoding parameters. The field semantics are
	// those of libwebp's WebPConfig.
	Options struct {
		Lossless         bool
		Quality          float32
		Method           int
		ImageHint        ImageHint
		TargetSize       int
		TargetPsnr       float32
		Segments         int
		SnsStrength      int
		FilterStrength   int
		FilterSharpness  int
		FilterType       int
		Autofilter       bool
		AlphaCompression int
		AlphaFiltering   int
		alphaQuality     int
		Pass             int
		// Disabled for compatibility with old version libwebp
		// QMin             int
		// QMax             int
		ShowCompressed  bool
		Preprocessing   int
		Partitions      int
		PartitionLimit  int
		EmulateJpegSize bool
		ThreadLevel     bool
		LowMemory       bool
		NearLossless    int
		Exact           int
		UseDeltaPalette bool
		UseSharpYuv     bool
	}
)

// Config is the validated encoding configuration built by Options.GetConfig.
// It mirrors libwebp's WebPConfig field for field (C int flags are bools).
// In v1 GetConfig returned the cgo type *C.WebPConfig; v2 returns this pure
// Go equivalent.
type Config struct {
	// Lossless selects lossless (VP8L) instead of lossy (VP8) coding.
	Lossless bool
	// Quality is in [0, 100]. For lossy coding it trades size for quality;
	// for lossless coding it is the compression effort.
	Quality float32
	// Method is the speed/quality trade-off in [0, 6] (0 = fastest).
	Method int
	// ImageHint is a hint about the picture type (lossless only).
	ImageHint ImageHint
	// TargetSize, if non-zero, is the desired output size in bytes (lossy).
	// As in libwebp it is only honored with Pass > 1.
	TargetSize int
	// TargetPSNR, if non-zero, is the desired PSNR in dB (lossy). As in
	// libwebp it is only honored with Pass > 1.
	TargetPSNR float32
	// Segments is the number of segments in [1, 4] (lossy).
	Segments int
	// SNSStrength is the spatial noise shaping strength in [0, 100] (lossy).
	SNSStrength int
	// FilterStrength is the loop filter strength in [0, 100] (lossy).
	FilterStrength int
	// FilterSharpness is the loop filter sharpness in [0, 7] (lossy).
	FilterSharpness int
	// FilterType is 0 for the simple and 1 for the strong loop filter (lossy).
	FilterType int
	// Autofilter adjusts the filter strength automatically (lossy).
	Autofilter bool
	// AlphaCompression is 0 for raw and 1 for lossless-compressed alpha.
	AlphaCompression int
	// AlphaFiltering is 0 (none), 1 (fast) or 2 (best) alpha prediction filtering.
	AlphaFiltering int
	// AlphaQuality in [0, 100]: below 100 the alpha plane is quantized.
	AlphaQuality int
	// Pass is the number of entropy-analysis passes in [1, 10] (lossy).
	Pass int
	// ShowCompressed is accepted for compatibility; it has no effect.
	ShowCompressed bool
	// Preprocessing in [0, 7]: bit 0 segment smoothing, bit 1 dithering,
	// bit 2 sharp YUV conversion (lossy).
	Preprocessing int
	// Partitions is log2 of the number of token partitions in [0, 3] (lossy).
	Partitions int
	// PartitionLimit in [0, 100] allows quality degradation to fit the
	// 512k limit on the first partition (lossy).
	PartitionLimit int
	// EmulateJPEGSize maps the quality like a JPEG encoder would (lossy).
	EmulateJPEGSize bool
	// ThreadLevel allows the encoder to use more than one goroutine.
	ThreadLevel bool
	// LowMemory reduces memory usage at some cost in speed or size.
	LowMemory bool
	// NearLossless in [0, 100] enables near-lossless preprocessing (100 = off).
	NearLossless int
	// Exact preserves the RGB values under fully transparent pixels.
	Exact bool
	// UseDeltaPalette is accepted for compatibility; it has no effect.
	UseDeltaPalette bool
	// UseSharpYUV uses the slower but sharper RGB->YUV conversion (lossy).
	UseSharpYUV bool
}

// validate reproduces libwebp's WebPValidateConfig.
func (c *Config) validate() bool {
	q := float64(c.Quality)
	switch {
	case math.IsNaN(q) || c.Quality < 0 || c.Quality > 100,
		c.TargetSize < 0,
		math.IsNaN(float64(c.TargetPSNR)) || c.TargetPSNR < 0,
		c.Method < 0 || c.Method > 6,
		c.Segments < 1 || c.Segments > 4,
		c.SNSStrength < 0 || c.SNSStrength > 100,
		c.FilterStrength < 0 || c.FilterStrength > 100,
		c.FilterSharpness < 0 || c.FilterSharpness > 7,
		c.FilterType < 0 || c.FilterType > 1,
		c.Pass < 1 || c.Pass > 10,
		c.Preprocessing < 0 || c.Preprocessing > 7,
		c.Partitions < 0 || c.Partitions > 3,
		c.PartitionLimit < 0 || c.PartitionLimit > 100,
		c.AlphaCompression < 0,
		c.AlphaFiltering < 0,
		c.AlphaQuality < 0 || c.AlphaQuality > 100,
		c.NearLossless < 0 || c.NearLossless > 100,
		c.ImageHint < 0 || c.ImageHint >= HintLast:
		return false
	}
	return true
}

// presetConfig reproduces libwebp's WebPConfigPreset (without validation).
func presetConfig(preset EncodingPreset, quality float32) Config {
	c := Config{
		Quality:          quality,
		Method:           4,
		SNSStrength:      50,
		FilterStrength:   60,
		FilterType:       1,
		Segments:         4,
		Pass:             1,
		AlphaCompression: 1,
		AlphaFiltering:   1,
		AlphaQuality:     100,
		NearLossless:     100,
	}
	switch preset {
	case PresetPicture:
		c.SNSStrength = 80
		c.FilterSharpness = 4
		c.FilterStrength = 35
		c.Preprocessing &^= 2
	case PresetPhoto:
		c.SNSStrength = 80
		c.FilterSharpness = 3
		c.FilterStrength = 30
		c.Preprocessing |= 2
	case PresetDrawing:
		c.SNSStrength = 25
		c.FilterSharpness = 6
		c.FilterStrength = 10
	case PresetIcon:
		c.SNSStrength = 0
		c.FilterStrength = 0
		c.Preprocessing &^= 2
	case PresetText:
		c.SNSStrength = 0
		c.FilterStrength = 0
		c.Preprocessing &^= 2
		c.Segments = 2
	}
	return c
}

// losslessPresets maps the lossless level (cwebp -z) to (method, quality),
// as libwebp's WebPConfigLosslessPreset.
var losslessPresets = [10]struct {
	method  int
	quality float32
}{
	{0, 0}, {1, 20}, {2, 25}, {3, 30}, {3, 50},
	{4, 50}, {4, 75}, {4, 90}, {5, 90}, {6, 100},
}

// NewLossyEncoderOptions build lossy encoding options
func NewLossyEncoderOptions(preset EncodingPreset, quality float32) (options *Options, err error) {
	c := presetConfig(preset, quality)
	if !c.validate() {
		return nil, errors.New("cannot init encoder config")
	}
	options = &Options{}
	options.sync(&c)
	return options, nil
}

// NewLosslessEncoderOptions build lossless encoding options
func NewLosslessEncoderOptions(preset EncodingPreset, level int) (options *Options, err error) {
	if options, err = NewLossyEncoderOptions(preset, 0); err != nil {
		return nil, err
	}
	if level < 0 || level >= len(losslessPresets) {
		return nil, errors.New("cannot init lossless preset")
	}
	options.Lossless = true
	options.Method = losslessPresets[level].method
	options.Quality = losslessPresets[level].quality
	return options, nil
}

func (o *Options) sync(c *Config) {
	o.Lossless = c.Lossless
	o.Quality = c.Quality
	o.Method = c.Method
	o.ImageHint = c.ImageHint
	o.TargetSize = c.TargetSize
	o.TargetPsnr = c.TargetPSNR
	o.Segments = c.Segments
	o.SnsStrength = c.SNSStrength
	o.FilterStrength = c.FilterStrength
	o.FilterSharpness = c.FilterSharpness
	o.FilterType = c.FilterType
	o.Autofilter = c.Autofilter
	o.AlphaCompression = c.AlphaCompression
	o.AlphaFiltering = c.AlphaFiltering
	o.alphaQuality = c.AlphaQuality
	o.Pass = c.Pass
	o.ShowCompressed = c.ShowCompressed
	o.Preprocessing = c.Preprocessing
	o.Partitions = c.Partitions
	o.PartitionLimit = c.PartitionLimit
	o.EmulateJpegSize = c.EmulateJPEGSize
	o.ThreadLevel = c.ThreadLevel
	o.LowMemory = c.LowMemory
	o.NearLossless = c.NearLossless
	o.Exact = 0
	if c.Exact {
		o.Exact = 1
	}
	o.UseDeltaPalette = c.UseDeltaPalette
	o.UseSharpYuv = c.UseSharpYUV
}

// GetConfig builds and validates the encoder configuration (the equivalent of
// libwebp's WebPConfig after WebPValidateConfig). A nil *Options yields the
// lossless preset of level 0.
func (o *Options) GetConfig() (*Config, error) {
	var err error
	if o == nil {
		o, err = NewLosslessEncoderOptions(PresetDefault, 0)
		if err != nil {
			return nil, fmt.Errorf("cannot validate default config: [%w]", err)
		}
	}
	c := &Config{
		Lossless:         o.Lossless,
		Quality:          o.Quality,
		Method:           o.Method,
		ImageHint:        o.ImageHint,
		TargetSize:       o.TargetSize,
		TargetPSNR:       o.TargetPsnr,
		Segments:         o.Segments,
		SNSStrength:      o.SnsStrength,
		FilterStrength:   o.FilterStrength,
		FilterSharpness:  o.FilterSharpness,
		FilterType:       o.FilterType,
		Autofilter:       o.Autofilter,
		AlphaCompression: o.AlphaCompression,
		AlphaFiltering:   o.AlphaFiltering,
		AlphaQuality:     o.alphaQuality,
		Pass:             o.Pass,
		ShowCompressed:   o.ShowCompressed,
		Preprocessing:    o.Preprocessing,
		Partitions:       o.Partitions,
		PartitionLimit:   o.PartitionLimit,
		EmulateJPEGSize:  o.EmulateJpegSize,
		ThreadLevel:      o.ThreadLevel,
		LowMemory:        o.LowMemory,
		NearLossless:     o.NearLossless,
		Exact:            o.Exact == 1,
		UseDeltaPalette:  o.UseDeltaPalette,
		UseSharpYUV:      o.UseSharpYuv,
	}
	if o.Exact < 0 || o.Exact > 1 || !c.validate() {
		return nil, errors.New("cannot validate config")
	}
	return c, nil
}
