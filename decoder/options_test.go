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
		} else if !cfg.UseCropping {
			t.Fatal("cropping is disabled")
		} else if cfg.CropLeft != 100 {
			t.Fatal("crop_left is invalid")
		} else if cfg.CropTop != 200 {
			t.Fatal("crop_top is invalid")
		} else if cfg.CropWidth != 400 {
			t.Fatal("crop_width is invalid")
		} else if cfg.CropHeight != 500 {
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
		} else if !cfg.UseScaling {
			t.Fatal("scaling is disabled")
		} else if cfg.ScaledWidth != 400 {
			t.Fatal("scaled_width is invalid")
		} else if cfg.ScaledHeight != 500 {
			t.Fatal("scaled_height is invalid")
		}
	})
	t.Run("crop and scale need both dimensions", func(t *testing.T) {
		options := &Options{
			Crop:  image.Rectangle{Max: image.Point{X: 10}},
			Scale: image.Rectangle{Max: image.Point{Y: 10}},
		}
		cfg, err := options.GetConfig()
		if err != nil {
			t.Fatal(err)
		}
		if cfg.UseCropping || cfg.UseScaling {
			t.Fatal("cropping/scaling must stay disabled")
		}
	})
	t.Run("nil options", func(t *testing.T) {
		var options *Options
		cfg, err := options.GetConfig()
		if err != nil || cfg == nil {
			t.Fatal("nil options must give the default config")
		}
	})
}
