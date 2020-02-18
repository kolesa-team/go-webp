package webp

import (
	"bytes"
	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/encoder"
	"image"
	"image/jpeg"
	"io/ioutil"
	"os"
	"testing"
)

func TestEncode(t *testing.T) {
	t.Run("encode lossy", func(t *testing.T) {
		r, err := os.Open("../test_data/images/source.jpg")
		if err != nil {
			t.Fatal(err)
		}

		img, err := jpeg.Decode(r)
		if err != nil {
			t.Fatal(err)
		}

		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			t.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("encode lossless", func(t *testing.T) {
		r, err := os.Open("../test_data/images/source.jpg")
		if err != nil {
			t.Fatal(err)
		}

		img, err := jpeg.Decode(r)
		if err != nil {
			t.Fatal(err)
		}

		options, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 4)
		if err != nil {
			t.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDecode(t *testing.T) {
	r, err := os.Open("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}

	if img, err := Decode(r, &decoder.Options{
		BypassFiltering:   true,
		NoFancyUpsampling: true,

		Crop: image.Rectangle{},
		Scale: image.Rectangle{
			Max: image.Point{
				X: 400,
				Y: 300,
			},
		},

		UseThreads:             true,
		Flip:                   true,
		DitheringStrength:      1,
		AlphaDitheringStrength: 1,
	}); err != nil {
		t.Fatal(err)
	} else if img == nil {
		t.Fatal("img is empty")
	} else if img.Bounds().Max.X != 400 || img.Bounds().Max.Y != 300 {
		t.Fatal("img is invalid")
	}
}

func BenchmarkDecode(b *testing.B) {
	data, err := ioutil.ReadFile("../test_data/images/m4_q75.webp")
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		dec, err := decoder.NewDecoder(bytes.NewBuffer(data), &decoder.Options{})
		if err != nil {
			b.Fatal(err)
		}

		if _, err := dec.Decode(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	r, err := os.Open("../test_data/images/source.jpg")
	if err != nil {
		b.Fatal(err)
	}

	img, err := jpeg.Decode(r)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
		if err != nil {
			b.Fatal(err)
		}

		if err = Encode(ioutil.Discard, img, options); err != nil {
			b.Fatal(err)
		}
	}
}
