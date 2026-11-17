package encoder

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNilOptions(t *testing.T) {
	t.Run("config from nil Options succeeds", func(t *testing.T) {
		var options *Options = nil
		cfg, err := options.GetConfig()
		require.NoError(t, err)
		require.NotNil(t, cfg)
		assert.True(t, cfg.Lossless)
		assert.Equal(t, 0, cfg.Method)
		assert.Equal(t, float32(0), cfg.Quality)
	})
}

func TestNewLossyEncoderOptions(t *testing.T) {
	t.Run("invalid quality", func(t *testing.T) {
		for _, q := range []float32{-1, 100.5} {
			options, err := NewLossyEncoderOptions(PresetDefault, q)
			require.Error(t, err)
			assert.Nil(t, options)
		}
	})
	t.Run("NaN quality is accepted like libwebp", func(t *testing.T) {
		// WebPValidateConfig's range checks are comparisons, false for NaN;
		// v1 (cgo) accepts it.
		options, err := NewLossyEncoderOptions(PresetDefault, float32(math.NaN()))
		require.NoError(t, err)
		_, err = options.GetConfig()
		require.NoError(t, err)
		options.TargetPsnr = float32(math.NaN())
		_, err = options.GetConfig()
		require.NoError(t, err)
	})
	t.Run("create lossy encoder options is success", func(t *testing.T) {
		options, err := NewLossyEncoderOptions(PresetDefault, 75)
		require.Nil(t, err)
		require.NotNil(t, options)

		assert.False(t, options.Lossless)
		assert.Equal(t, float32(75), options.Quality)

		cfg, err := options.GetConfig()
		require.NoError(t, err)
		assert.NotNil(t, cfg)
	})
}

func TestNewLosslessEncoderOptions(t *testing.T) {
	t.Run("invalid level of lossless encoder", func(t *testing.T) {
		for _, level := range []int{-1, 10} {
			options, err := NewLosslessEncoderOptions(PresetDefault, level)
			require.Error(t, err)
			assert.Nil(t, options)
		}
	})
	t.Run("create lossless encoder options is success", func(t *testing.T) {
		options, err := NewLosslessEncoderOptions(PresetDefault, 1)
		require.Nil(t, err)
		require.NotNil(t, options)

		assert.True(t, options.Lossless)
		assert.Equal(t, 1, options.Method)
		assert.Equal(t, float32(20), options.Quality)

		cfg, err := options.GetConfig()
		require.NoError(t, err)
		assert.NotNil(t, cfg)
	})
	t.Run("levels", func(t *testing.T) {
		want := [][2]int{{0, 0}, {1, 20}, {2, 25}, {3, 30}, {3, 50}, {4, 50}, {4, 75}, {4, 90}, {5, 90}, {6, 100}}
		for level, mq := range want {
			options, err := NewLosslessEncoderOptions(PresetPhoto, level)
			require.NoError(t, err)
			assert.Equal(t, mq[0], options.Method, "level %d", level)
			assert.Equal(t, float32(mq[1]), options.Quality, "level %d", level)
			assert.Equal(t, 100, options.NearLossless)
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
		require.NoError(t, err)
		got, err := options.GetConfig()
		require.NoError(t, err)
		assert.Equal(t, want, *got, "preset %d", preset)
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
	require.NoError(t, err)
	for _, f := range fields {
		for v := -2; v <= 120; v++ {
			o := *base
			f.set(&o, v)
			_, err := o.GetConfig()
			valid := v >= f.min && (f.max < f.min || v <= f.max)
			assert.Equal(t, valid, err == nil, "%s = %d", f.name, v)
		}
	}
}
