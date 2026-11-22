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
	"math"
	"testing"
)

func TestNilOptions(t *testing.T) {
	t.Run("config from nil Options succeeds", func(t *testing.T) {
		var options *Options = nil
		cfg, err := options.GetConfig()
		if err != nil {
			t.Fatalf("GetConfig: %v", err)
		}
		if cfg == nil {
			t.Fatal("GetConfig() = nil, want a *Config")
		}
		if !cfg.Lossless {
			t.Error("Lossless = false, want true")
		}
		if cfg.Method != 0 {
			t.Errorf("Method = %d, want 0", cfg.Method)
		}
		if cfg.Quality != 0 {
			t.Errorf("Quality = %v, want 0", cfg.Quality)
		}
	})
}

func TestNewLossyEncoderOptions(t *testing.T) {
	t.Run("invalid quality", func(t *testing.T) {
		for _, q := range []float32{-1, 100.5} {
			options, err := NewLossyEncoderOptions(PresetDefault, q)
			if err == nil {
				t.Fatalf("NewLossyEncoderOptions(%v): no error", q)
			}
			if options != nil {
				t.Errorf("NewLossyEncoderOptions(%v) = %+v, want nil options", q, options)
			}
		}
	})
	t.Run("NaN quality is accepted like libwebp", func(t *testing.T) {
		// WebPValidateConfig's range checks are comparisons, false for NaN;
		// v1 (cgo) accepts it.
		options, err := NewLossyEncoderOptions(PresetDefault, float32(math.NaN()))
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions(NaN): %v", err)
		}
		_, err = options.GetConfig()
		if err != nil {
			t.Fatalf("GetConfig with a NaN Quality: %v", err)
		}
		options.TargetPsnr = float32(math.NaN())
		_, err = options.GetConfig()
		if err != nil {
			t.Fatalf("GetConfig with a NaN TargetPsnr: %v", err)
		}
	})
	t.Run("create lossy encoder options is success", func(t *testing.T) {
		options, err := NewLossyEncoderOptions(PresetDefault, 75)
		if err != nil {
			t.Fatalf("NewLossyEncoderOptions: %v", err)
		}
		if options == nil {
			t.Fatal("NewLossyEncoderOptions returned nil options")
		}

		if options.Lossless {
			t.Error("Lossless = true, want false")
		}
		if options.Quality != 75 {
			t.Errorf("Quality = %v, want 75", options.Quality)
		}

		cfg, err := options.GetConfig()
		if err != nil {
			t.Fatalf("GetConfig: %v", err)
		}
		if cfg == nil {
			t.Error("GetConfig() = nil, want a *Config")
		}
	})
}

func TestNewLosslessEncoderOptions(t *testing.T) {
	t.Run("invalid level of lossless encoder", func(t *testing.T) {
		for _, level := range []int{-1, 10} {
			options, err := NewLosslessEncoderOptions(PresetDefault, level)
			if err == nil {
				t.Fatalf("NewLosslessEncoderOptions(%d): no error", level)
			}
			if options != nil {
				t.Errorf("NewLosslessEncoderOptions(%d) = %+v, want nil options", level, options)
			}
		}
	})
	t.Run("create lossless encoder options is success", func(t *testing.T) {
		options, err := NewLosslessEncoderOptions(PresetDefault, 1)
		if err != nil {
			t.Fatalf("NewLosslessEncoderOptions: %v", err)
		}
		if options == nil {
			t.Fatal("NewLosslessEncoderOptions returned nil options")
		}

		if !options.Lossless {
			t.Error("Lossless = false, want true")
		}
		if options.Method != 1 {
			t.Errorf("Method = %d, want 1", options.Method)
		}
		if options.Quality != 20 {
			t.Errorf("Quality = %v, want 20", options.Quality)
		}

		cfg, err := options.GetConfig()
		if err != nil {
			t.Fatalf("GetConfig: %v", err)
		}
		if cfg == nil {
			t.Error("GetConfig() = nil, want a *Config")
		}
	})
	t.Run("levels", func(t *testing.T) {
		want := [][2]int{{0, 0}, {1, 20}, {2, 25}, {3, 30}, {3, 50}, {4, 50}, {4, 75}, {4, 90}, {5, 90}, {6, 100}}
		for level, mq := range want {
			options, err := NewLosslessEncoderOptions(PresetPhoto, level)
			if err != nil {
				t.Fatalf("level %d: NewLosslessEncoderOptions: %v", level, err)
			}
			if options.Method != mq[0] {
				t.Errorf("level %d: Method = %d, want %d", level, options.Method, mq[0])
			}
			if options.Quality != float32(mq[1]) {
				t.Errorf("level %d: Quality = %v, want %d", level, options.Quality, mq[1])
			}
			if options.NearLossless != 100 {
				t.Errorf("level %d: NearLossless = %d, want 100", level, options.NearLossless)
			}
		}
	})
}

// TestPresets checks the values of libwebp's WebPConfigPreset (libwebp 1.6.0,
// src/enc/config_enc.c; verified against the C library with cgo).
func TestPresets(t *testing.T) {
	base := Config{
		Quality: 42, Method: 4, SNSStrength: 50, FilterStrength: 60, FilterType: 1,
		Segments: 4, Pass: 1, AlphaCompression: 1, AlphaFiltering: 1, AlphaQuality: 100,
		NearLossless: 100,
	}
	tests := map[EncodingPreset]func(c *Config){
		PresetDefault: func(*Config) {},
		PresetPicture: func(c *Config) { c.SNSStrength, c.FilterSharpness, c.FilterStrength = 80, 4, 35 },
		PresetPhoto: func(c *Config) {
			c.SNSStrength, c.FilterSharpness, c.FilterStrength, c.Preprocessing = 80, 3, 30, 2
		},
		PresetDrawing: func(c *Config) { c.SNSStrength, c.FilterSharpness, c.FilterStrength = 25, 6, 10 },
		PresetIcon:    func(c *Config) { c.SNSStrength, c.FilterStrength = 0, 0 },
		PresetText:    func(c *Config) { c.SNSStrength, c.FilterStrength, c.Segments = 0, 0, 2 },
		99:            func(*Config) {}, // unknown presets fall back to the default
	}
	for preset, mod := range tests {
		want := base
		mod(&want)
		options, err := NewLossyEncoderOptions(preset, 42)
		if err != nil {
			t.Fatalf("preset %d: NewLossyEncoderOptions: %v", preset, err)
		}
		got, err := options.GetConfig()
		if err != nil {
			t.Fatalf("preset %d: GetConfig: %v", preset, err)
		}
		if *got != want {
			t.Errorf("preset %d: GetConfig() = %+v, want %+v", preset, *got, want)
		}
	}
}

// TestValidation checks the ranges of libwebp's WebPValidateConfig.
func TestValidation(t *testing.T) {
	type field struct {
		name     string
		set      func(o *Options, v int)
		min, max int // valid range; max < min means "no upper bound"
	}
	fields := []field{
		{"Quality", func(o *Options, v int) { o.Quality = float32(v) }, 0, 100},
		{"Method", func(o *Options, v int) { o.Method = v }, 0, 6},
		{"ImageHint", func(o *Options, v int) { o.ImageHint = ImageHint(v) }, 0, int(HintLast) - 1},
		{"TargetSize", func(o *Options, v int) { o.TargetSize = v }, 0, -1},
		{"TargetPsnr", func(o *Options, v int) { o.TargetPsnr = float32(v) }, 0, -1},
		{"Segments", func(o *Options, v int) { o.Segments = v }, 1, 4},
		{"SnsStrength", func(o *Options, v int) { o.SnsStrength = v }, 0, 100},
		{"FilterStrength", func(o *Options, v int) { o.FilterStrength = v }, 0, 100},
		{"FilterSharpness", func(o *Options, v int) { o.FilterSharpness = v }, 0, 7},
		{"FilterType", func(o *Options, v int) { o.FilterType = v }, 0, 1},
		{"AlphaCompression", func(o *Options, v int) { o.AlphaCompression = v }, 0, -1},
		{"AlphaFiltering", func(o *Options, v int) { o.AlphaFiltering = v }, 0, -1},
		{"alphaQuality", func(o *Options, v int) { o.alphaQuality = v }, 0, 100},
		{"Pass", func(o *Options, v int) { o.Pass = v }, 1, 10},
		{"Preprocessing", func(o *Options, v int) { o.Preprocessing = v }, 0, 7},
		{"Partitions", func(o *Options, v int) { o.Partitions = v }, 0, 3},
		{"PartitionLimit", func(o *Options, v int) { o.PartitionLimit = v }, 0, 100},
		{"NearLossless", func(o *Options, v int) { o.NearLossless = v }, 0, 100},
		{"Exact", func(o *Options, v int) { o.Exact = v }, 0, 1},
	}
	base, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	for _, f := range fields {
		for v := -2; v <= 120; v++ {
			o := *base
			f.set(&o, v)
			_, err := o.GetConfig()
			valid := v >= f.min && (f.max < f.min || v <= f.max)
			if (err == nil) != valid {
				t.Errorf("%s = %d: GetConfig() succeeded = %v, want %v (err: %v)", f.name, v, err == nil, valid, err)
			}
		}
	}
}
