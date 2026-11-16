package vp8lenc

import (
	"image"
	"image/color"
	"math/rand"
)

// testImage is a named synthetic test image.
type testImage struct {
	name string
	img  *image.NRGBA
}

func newNRGBA(w, h int, f func(x, y int) color.NRGBA) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetNRGBA(x, y, f(x, y))
		}
	}
	return m
}

// syntheticImages returns a deterministic set of images covering edge cases.
func syntheticImages() []testImage {
	rng := rand.New(rand.NewSource(1))
	var out []testImage
	add := func(name string, m *image.NRGBA) { out = append(out, testImage{name, m}) }

	sizes := [][2]int{{1, 1}, {1, 2}, {2, 1}, {3, 7}, {17, 33}, {1000, 1}, {1, 300}, {64, 64}, {129, 67}}
	for _, s := range sizes {
		w, h := s[0], s[1]
		add("gradient", newNRGBA(w, h, func(x, y int) color.NRGBA {
			return color.NRGBA{uint8(x * 255 / max(w, 1)), uint8(y * 255 / max(h, 1)), uint8((x + y) * 3), 255}
		}))
		add("noise", newNRGBA(w, h, func(x, y int) color.NRGBA {
			return color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255}
		}))
		add("noise-alpha", newNRGBA(w, h, func(x, y int) color.NRGBA {
			return color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256))}
		}))
		for _, nc := range []int{1, 2, 3, 4, 11, 16, 17, 200, 256} {
			pal := make([]color.NRGBA, nc)
			for i := range pal {
				pal[i] = color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(2) * 255)}
			}
			add("palette", newNRGBA(w, h, func(x, y int) color.NRGBA {
				if rng.Intn(4) == 0 {
					return pal[rng.Intn(nc)]
				}
				return pal[(x/3+y/5)%nc]
			}))
		}
	}
	// Larger structured images.
	add("photo-like", newNRGBA(300, 200, func(x, y int) color.NRGBA {
		v := uint8((x*x + y*y) / 97)
		return color.NRGBA{v + uint8(rng.Intn(6)), uint8(x) + uint8(rng.Intn(4)), uint8(y), 255}
	}))
	add("stripes", newNRGBA(257, 129, func(x, y int) color.NRGBA {
		if (x/7+y/3)%2 == 0 {
			return color.NRGBA{10, 200, 30, 255}
		}
		return color.NRGBA{uint8(x), uint8(y), 99, 128}
	}))
	add("transparent-garbage", newNRGBA(70, 70, func(x, y int) color.NRGBA {
		if (x+y)%3 == 0 {
			return color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 0}
		}
		return color.NRGBA{uint8(x * 3), uint8(y * 3), 7, 255}
	}))
	add("flat", newNRGBA(500, 300, func(x, y int) color.NRGBA { return color.NRGBA{1, 2, 3, 255} }))
	// Sub-image with non-zero origin.
	big := newNRGBA(90, 80, func(x, y int) color.NRGBA {
		return color.NRGBA{uint8(x * 2), uint8(y * 3), uint8(x ^ y), uint8(200 + x%50)}
	})
	add("subimage", big.SubImage(image.Rect(13, 7, 77, 61)).(*image.NRGBA))
	return out
}
