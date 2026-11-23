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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"image"
	"os"
	"strconv"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func TestNewDecoder(t *testing.T) {
	t.Run("create success", func(t *testing.T) {
		file, err := os.Open("../test_data/images/m4_q75.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()

		if d, err := NewDecoder(file, &Options{
			BypassFiltering:   true,
			NoFancyUpsampling: true,
			Scale: image.Rectangle{
				Max: image.Point{
					X: 300,
					Y: 300,
				},
			},
			UseThreads:             true,
			Flip:                   true,
			DitheringStrength:      0,
			AlphaDitheringStrength: 0,
		}); err != nil {
			t.Fatal(err)
		} else if img, err := d.Decode(); err != nil {
			t.Fatal(err)
		} else if img.Bounds().Dx() != 300 || img.Bounds().Dy() != 300 {
			t.Fatal("invalid decoding result")
		}
	})
	t.Run("empty file", func(t *testing.T) {
		file, err := os.Open("../test_data/images/invalid.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()

		_, err = NewDecoder(file, &Options{
			Crop: image.Rectangle{
				Max: image.Point{
					X: 150,
					Y: 150,
				},
			},
		})
		if want := "data is empty"; err == nil || err.Error() != want {
			t.Fatalf("NewDecoder(invalid.webp): error %v, want %q", err, want)
		}
	})
	t.Run("not a webp file", func(t *testing.T) {
		data, err := os.ReadFile("../test_data/images/logo.jpg")
		if err != nil {
			t.Fatal(err)
		}
		_, err = NewDecoder(bytes.NewReader(data), nil)
		if want := "cannot fetch features: VP8_STATUS_BITSTREAM_ERROR"; err == nil || err.Error() != want {
			t.Fatalf("NewDecoder(logo.jpg): error %v, want %q", err, want)
		}
		if code := status.CodeOf(err); code != utils.Vp8StatusBitstreamError {
			t.Errorf("NewDecoder(logo.jpg): code %v, want %v", code, utils.Vp8StatusBitstreamError)
		}
	})
	t.Run("reader without size", func(t *testing.T) {
		data := readFile(t, "m4_q75.webp")
		d, err := NewDecoder(iotest.HalfReader(bytes.NewReader(data)), nil)
		if err != nil {
			t.Fatalf("NewDecoder(HalfReader): %v", err)
		}
		if got := d.GetFeatures().Width; got != 675 {
			t.Errorf("Width = %d, want 675", got)
		}
	})
}

func TestDecoder_GetFeatures(t *testing.T) {
	file, err := os.Open("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

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

	if features.HasAnimation {
		t.Fatal("file has_animation is invalid")
	}

	tests := []struct {
		file string
		want utils.BitstreamFeatures
	}{
		{"webp-logo-lossless.webp", utils.BitstreamFeatures{Width: 3000, Height: 2000, HasAlpha: true, Format: utils.FormatLossless}},
		{"100x150_lossy.webp", utils.BitstreamFeatures{Width: 100, Height: 150, HasAlpha: true, Format: utils.FormatLossy}},
		{"100x150_lossless.webp", utils.BitstreamFeatures{Width: 100, Height: 150, HasAlpha: false, Format: utils.FormatLossless}},
	}
	for _, tt := range tests {
		dec, err := NewDecoder(bytes.NewReader(readFile(t, tt.file)), nil)
		if err != nil {
			t.Fatalf("NewDecoder(%s): %v", tt.file, err)
		}
		f := dec.GetFeatures()
		if f.Width != tt.want.Width {
			t.Errorf("%s: Width = %d, want %d", tt.file, f.Width, tt.want.Width)
		}
		if f.Height != tt.want.Height {
			t.Errorf("%s: Height = %d, want %d", tt.file, f.Height, tt.want.Height)
		}
		if f.Format != tt.want.Format {
			t.Errorf("%s: Format = %v, want %v", tt.file, f.Format, tt.want.Format)
		}
	}
}

func TestDecoder_NilOptions(t *testing.T) {
	file, err := os.Open("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	d, err := NewDecoder(file, nil)
	if err != nil {
		t.Fatalf("NewDecoder(m4_q75.webp, nil): %v", err)
	}
	img, err := d.Decode()
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got, want := img.Bounds(), image.Rect(0, 0, 675, 900); got != want {
		t.Errorf("Bounds = %v, want %v", got, want)
	}
}

func readFile(t testing.TB, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("../test_data/images/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func decode(t testing.TB, data []byte, o *Options) *image.NRGBA {
	t.Helper()
	d, err := NewDecoder(bytes.NewReader(data), o)
	if err != nil {
		t.Fatalf("NewDecoder(%+v): %v", o, err)
	}
	img, err := d.Decode()
	if err != nil {
		t.Fatalf("Decode(%+v): %v", o, err)
	}
	return img.(*image.NRGBA)
}

// optionsFromDwebp translates dwebp command line flags.
func optionsFromDwebp(t *testing.T, flags string) *Options {
	o := &Options{}
	f := strings.Fields(flags)
	atoi := func(s string) int {
		v, err := strconv.Atoi(s)
		if err != nil {
			t.Fatalf("flags %q: %v", flags, err)
		}
		return v
	}
	for i := 0; i < len(f); i++ {
		switch f[i] {
		case "-nofancy":
			o.NoFancyUpsampling = true
		case "-nofilter":
			o.BypassFiltering = true
		case "-flip":
			o.Flip = true
		case "-mt":
			o.UseThreads = true
		case "-dither":
			o.DitheringStrength = atoi(f[i+1])
			i++
		case "-crop":
			o.Crop = image.Rectangle{Min: image.Pt(atoi(f[i+1]), atoi(f[i+2])), Max: image.Pt(atoi(f[i+3]), atoi(f[i+4]))}
			i += 4
		default:
			t.Fatalf("unknown flag %q", f[i])
		}
	}
	return o
}

// TestConformance compares the decoded pixels with the output of libwebp 1.6
// ("dwebp <file> <flags> -pam", SHA-256 of the RGBA samples).
func TestConformance(t *testing.T) {
	golden := []struct{ file, flags, sum string }{
		{"m4_q75", "", "b83a8bf240134a38715a1c0ec10b11426ac602a3b22f730ebae32e6d3c43e5a3"},
		{"m4_q75", "-nofancy", "79ec8f32b67e1130a7f897de93c257598313619c04226be1f4dc7698d6102e4b"},
		{"m4_q75", "-nofilter", "f52f05902330a919da15cfca846cd6090a08fbede44a5095b5d622b9d788a747"},
		{"m4_q75", "-flip", "f101e7008b361b20fb4fb2bc1760b46b20daf8b536f500a0a0173fc7c15703f3"},
		{"m4_q75", "-crop 31 17 200 100", "ea42269d7fc95fa1da923e8763cc8215c2c8e2763efec4bf5280dfd3fa480aab"},
		{"m4_q75", "-crop 30 16 64 48 -flip", "8e1b0dfb274c11cb200669a98e5554db689d8843fea00645b3130835555d41d5"},
		{"m4_q75", "-dither 100", "b83a8bf240134a38715a1c0ec10b11426ac602a3b22f730ebae32e6d3c43e5a3"},
		{"m4_q75", "-nofancy -nofilter", "2954d575869ada0fffc64d137e537616f67756557db26a4c689b900df2fa967c"},
		{"m4_q75", "-mt", "b83a8bf240134a38715a1c0ec10b11426ac602a3b22f730ebae32e6d3c43e5a3"},
		{"webp-logo-lossy", "", "09332df8799ff0cb114f50ad320dbbe78666bccf63cbb4e1e3dfc55c2f698531"},
		{"webp-logo-lossy", "-nofancy", "c851d6ed8925e0b4fce631a9bd3186d2793af21cf33f9aad7438688ce462a529"},
		{"webp-logo-lossy", "-nofilter", "3a45b61b29c7efb7b6569c87bfd913864856f0a347ce6a187fac0e31317d1e43"},
		{"webp-logo-lossy", "-flip", "2ee360320a69088ebb8af02e6d7daf35731ea079b867f314fd187cb2f5953152"},
		{"webp-logo-lossy", "-crop 31 17 200 100", "e811891d3adaa76d7ec9ced2e17f54edda2d89eff7981a79176adebcdeada5e0"},
		{"webp-logo-lossy", "-crop 30 16 64 48 -flip", "8b2e7d51ebf3360d443eed45632fd14307b0e8b07cdcb0f1473075c0fd2a99e4"},
		{"webp-logo-lossy", "-dither 100", "ede3312669b33eaa964a680e57c4baf735bfb924cbdf4569c06da89e2049cec6"},
		{"webp-logo-lossy", "-nofancy -nofilter", "fe7577d05c0a43a7c322882c3750107e64e33b214141c9729453a0e6b81a8a37"},
		{"webp-logo-lossy", "-mt -dither 100", "ede3312669b33eaa964a680e57c4baf735bfb924cbdf4569c06da89e2049cec6"},
		{"webp-logo-lossless", "", "b0325033a6bf401450ddf6c461861f0d2133d211a114c889d7c44e35e7cd19e5"},
		{"webp-logo-lossless", "-flip", "688b4338a6c38af1dd014ecbaa382aaab30786608e6a2e07f55bbf6590aae05d"},
		{"webp-logo-lossless", "-crop 31 17 200 100", "f8c784aa6b57396e7c5e094c34d079d8252473e46e2f60593a921dbebf941fcc"},
		{"webp-logo-lossless", "-crop 30 16 64 48 -flip", "f3cc103136423a57975750907ebc1d367e2985ac6338976d4d5a439f50323f4a"},
		{"100x150_lossy", "", "d98a5ff86464c5046609c6e409bcc19579f651ddcd2675ed18bca646eeee7eeb"},
		{"100x150_lossless", "", "0946e2eb0fb9ea7ddd935efd1922bc7d1f27101c69ce6d2f5145c7ee28f1b6ba"},
	}
	for _, g := range golden {
		t.Run(g.file+g.flags, func(t *testing.T) {
			img := decode(t, readFile(t, g.file+".webp"), optionsFromDwebp(t, g.flags))
			sum := sha256.Sum256(img.Pix)
			if got := hex.EncodeToString(sum[:]); got != g.sum {
				t.Errorf("SHA-256 of the pixels = %s, want %s", got, g.sum)
			}
		})
	}
}

func TestDecoder_Transforms(t *testing.T) {
	for _, file := range []string{"webp-logo-lossless.webp", "m4_q75.webp"} {
		data := readFile(t, file)
		full := decode(t, data, &Options{NoFancyUpsampling: true})
		b := full.Bounds()

		t.Run(file+" flip", func(t *testing.T) {
			flipped := decode(t, data, &Options{NoFancyUpsampling: true, Flip: true})
			if flipped.Bounds() != b {
				t.Fatalf("flipped Bounds = %v, want %v", flipped.Bounds(), b)
			}
			for y := 0; y < b.Dy(); y++ {
				if !bytes.Equal(full.Pix[y*full.Stride:][:4*b.Dx()], flipped.Pix[(b.Dy()-1-y)*flipped.Stride:][:4*b.Dx()]) {
					t.Fatalf("row %d of the flipped picture differs from row %d of the picture", b.Dy()-1-y, y)
				}
			}
		})
		t.Run(file+" crop", func(t *testing.T) {
			// Without fancy upsampling a crop with an even corner equals
			// the same rectangle of the full picture, for both formats.
			crop := decode(t, data, &Options{NoFancyUpsampling: true, Crop: image.Rectangle{Min: image.Pt(64, 32), Max: image.Pt(101, 77)}})
			if want := image.Rect(0, 0, 101, 77); crop.Bounds() != want {
				t.Fatalf("cropped Bounds = %v, want %v", crop.Bounds(), want)
			}
			sub := full.SubImage(image.Rect(64, 32, 64+101, 32+77)).(*image.NRGBA)
			for y := 0; y < 77; y++ {
				if !bytes.Equal(sub.Pix[y*sub.Stride:][:4*101], crop.Pix[y*crop.Stride:][:4*101]) {
					t.Fatalf("row %d of the crop differs from the same rectangle of the picture", y)
				}
			}
		})
		t.Run(file+" scale", func(t *testing.T) {
			img := decode(t, data, &Options{Scale: image.Rectangle{Max: image.Pt(123, 45)}})
			if want := image.Rect(0, 0, 123, 45); img.Bounds() != want {
				t.Errorf("scaled Bounds = %v, want %v", img.Bounds(), want)
			}
			img = decode(t, data, &Options{
				Crop:  image.Rectangle{Min: image.Pt(10, 10), Max: image.Pt(200, 100)},
				Scale: image.Rectangle{Max: image.Pt(400, 50)},
				Flip:  true,
			})
			if want := image.Rect(0, 0, 400, 50); img.Bounds() != want {
				t.Errorf("cropped, scaled and flipped Bounds = %v, want %v", img.Bounds(), want)
			}
		})
		t.Run(file+" threads", func(t *testing.T) {
			img := decode(t, data, &Options{NoFancyUpsampling: true, UseThreads: true})
			if !bytes.Equal(full.Pix, img.Pix) {
				t.Fatalf("UseThreads pixels (%d bytes) differ from the sequential decode (%d bytes)", len(img.Pix), len(full.Pix))
			}
		})
	}
}

func TestDecoder_Errors(t *testing.T) {
	m4 := readFile(t, "m4_q75.webp")
	lossless := readFile(t, "webp-logo-lossless.webp")

	decodeErr := func(data []byte, o *Options) error {
		d, err := NewDecoder(bytes.NewReader(data), o)
		if err != nil {
			return err
		}
		img, err := d.Decode()
		if err == nil {
			// A typed nil pointer in the interface counts as nil too.
			if p, ok := img.(*image.NRGBA); img == nil || ok && p == nil {
				t.Errorf("Decode(%+v) returned a nil image and no error", o)
			}
		}
		return err
	}

	t.Run("crop outside of the picture", func(t *testing.T) {
		for _, data := range [][]byte{m4, lossless} {
			err := decodeErr(data, &Options{Crop: image.Rectangle{Min: image.Pt(2990, 0), Max: image.Pt(100, 100)}})
			if want := "cannot decode picture: VP8_STATUS_INVALID_PARAM"; err == nil || err.Error() != want {
				t.Fatalf("error %v, want %q", err, want)
			}
			var se *status.Error
			if !errors.As(err, &se) {
				t.Fatalf("error %v (%T) does not wrap a *status.Error", err, err)
			}
		}
	})
	t.Run("truncated", func(t *testing.T) {
		for _, data := range [][]byte{m4, lossless} {
			err := decodeErr(data[:len(data)/2], nil)
			if want := "cannot decode picture: VP8_STATUS_NOT_ENOUGH_DATA"; err == nil || err.Error() != want {
				t.Fatalf("error %v, want %q", err, want)
			}
		}
	})
	t.Run("corrupted", func(t *testing.T) {
		for _, data := range [][]byte{m4, lossless} {
			for i := 40; i < len(data); i += 997 {
				c := bytes.Clone(data)
				c[i] ^= 0x55
				_ = decodeErr(c, &Options{UseThreads: i%2 == 0}) // must not panic
			}
		}
	})
	t.Run("animation", func(t *testing.T) {
		// RIFF + VP8X with the animation flag + an empty ANIM chunk.
		anim := []byte("RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x09\x00\x00\x09\x00\x00ANIM\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00")
		anim[4] = byte(len(anim) - 8)
		d, err := NewDecoder(bytes.NewReader(anim), nil)
		if err != nil {
			t.Fatalf("NewDecoder: %v", err)
		}
		if !d.GetFeatures().HasAnimation {
			t.Error("HasAnimation = false, want true")
		}
		if got := d.GetFeatures().Width; got != 10 {
			t.Errorf("Width = %d, want 10", got)
		}
		_, err = d.Decode()
		if want := "cannot decode picture: VP8_STATUS_UNSUPPORTED_FEATURE"; err == nil || err.Error() != want {
			t.Fatalf("Decode: error %v, want %q", err, want)
		}
	})
}

func BenchmarkDecode(b *testing.B) {
	for _, file := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data := readFile(b, file)
		for _, o := range []struct {
			name string
			opts *Options
		}{
			{"default", nil},
			{"threads", &Options{UseThreads: true}},
			{"scale", &Options{Scale: image.Rectangle{Max: image.Pt(320, 240)}}},
		} {
			b.Run(file+"/"+o.name, func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					d, err := NewDecoder(bytes.NewReader(data), o.opts)
					if err != nil {
						b.Fatal(err)
					}
					if _, err := d.Decode(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
