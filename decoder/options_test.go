package decoder

import (
	"image"
	"testing"
)

func TestOptions_GetConfig(t *testing.T) {
	t.Run("check crop", func(t *testing.T) {
		options := &Options{
			Crop: image.Rectangle{
				Min: image.Point{X: 100, Y: 200},
				Max: image.Point{X: 400, Y: 500},
			},
		}
		if cfg, err := options.GetConfig(); err != nil {
			t.Fatal(err)
		} else if cfg.options.use_cropping != 1 {
			t.Fatal("cropping is disabled")
		} else if cfg.options.crop_left != 100 {
			t.Fatal("crop_left is invalid")
		} else if cfg.options.crop_top != 200 {
			t.Fatal("crop_top is invalid")
		} else if cfg.options.crop_width != 400 {
			t.Fatal("crop_width is invalid")
		} else if cfg.options.crop_height != 500 {
			t.Fatal("crop_height is invalid")
		}
	})
	t.Run("check scale", func(t *testing.T) {
		options := &Options{
			Scale: image.Rectangle{
				Max: image.Point{X: 400, Y: 500},
			},
		}
		if cfg, err := options.GetConfig(); err != nil {
			t.Fatal(err)
		} else if cfg.options.use_scaling != 1 {
			t.Fatal("scaling is disabled")
		} else if cfg.options.scaled_width != 400 {
			t.Fatal("scaled_width is invalid")
		} else if cfg.options.scaled_height != 500 {
			t.Fatal("scaled_height is invalid")
		}
	})
}
