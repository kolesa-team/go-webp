package decoder

import (
	"os"
	"testing"

	"github.com/kolesa-team/go-webp/utils"
)

func TestNewDecoder(t *testing.T) {
	t.Run("create success", func(t *testing.T) {
		file, err := os.Open("../test_data/images/m4_q75.webp")
		if err != nil {
			t.Fatal(err)
		}

		if _, err := NewDecoder(file, &Options{}); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("empty file", func(t *testing.T) {
		file, err := os.Open("../test_data/images/invalid.webp")
		if err != nil {
			t.Fatal(err)
		}

		if _, err := NewDecoder(file, &Options{}); err == nil {
			t.Fatal(err)
		}
	})
}

func TestDecoder_GetFeatures(t *testing.T) {
	file, err := os.Open("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}

	dec, err := NewDecoder(file, &Options{})
	if err != nil {
		t.Fatal(err)
	}

	features := dec.GetFeatures()

	if features.Width != 675 || features.Height != 900 {
		t.Fatal("incorrect dimensions")
	}

	if features.Format != utils.FormatLossy {
		t.Fatal("file format is invalid")
	}

	if features.HasAlpha {
		t.Fatal("file has_alpha is invalid")
	}

	if features.HasAlpha {
		t.Fatal("file has_animation is invalid")
	}
}
