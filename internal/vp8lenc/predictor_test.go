package vp8lenc

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// predInputs returns predictor inputs: opaque, with fully transparent areas
// and noisy, of sizes that are not multiples of the tile sizes.
func predInputs() []struct {
	name string
	argb []uint32
	w, h int
} {
	rng := rand.New(rand.NewSource(13))
	type in = struct {
		name string
		argb []uint32
		w, h int
	}
	var out []in
	add := func(name string, w, h int, f func(x, y int) uint32) {
		a := make([]uint32, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				a[y*w+x] = f(x, y)
			}
		}
		out = append(out, in{name, a, w, h})
	}
	smooth := func(x, y int) uint32 {
		return 0xff000000 | uint32(x*3+rng.Intn(4))&0xff<<16 | uint32(y*2+x)&0xff<<8 | uint32(x^y)&0xff
	}
	add("opaque", 157, 93, smooth)
	add("transparent", 131, 77, func(x, y int) uint32 {
		if (x/9+y/7)%3 == 0 {
			return rng.Uint32() & 0x00ffffff // alpha 0, garbage RGB
		}
		if (x+y)%17 == 0 {
			return uint32(rng.Intn(256))<<24 | rng.Uint32()&0xffffff
		}
		return smooth(x, y)
	})
	add("noise", 70, 45, func(x, y int) uint32 { return rng.Uint32() })
	add("column", 1, 300, smooth)
	add("row", 300, 1, smooth)
	add("small", 5, 3, smooth)
	return out
}

// TestPredictorParallel checks that the parallel mode search gives the same
// modes and residuals as the sequential one, for opaque and transparent
// images (clean), near-lossless quantization and every budget.
func TestPredictorParallel(t *testing.T) {
	allBits, subGreens := []int{2, 3, 4, 5}, []bool{false, true}
	if testing.Short() {
		allBits, subGreens = []int{2, 5}, []bool{true}
	}
	for _, in := range predInputs() {
		for _, bits := range allBits {
			for _, maxQuant := range []int{1, 4} {
				for _, clean := range []bool{false, true} {
					for _, subGreen := range subGreens {
						run := func(helpers int) ([]uint32, []uint32, int) {
							argb := slices.Clone(in.argb)
							modes := make([]uint32, subSampleSize(in.w, bits)*subSampleSize(in.h, bits))
							b := predictorImage(in.w, in.h, bits, bits, false, maxQuant, subGreen, clean, argb, modes, new(scratch), helpers)
							return argb, modes, b
						}
						wantArgb, wantModes, wantBits := run(0)
						for _, limit := range []int{1, 8} {
							for _, helpers := range []int{1, 7} {
								func() {
									defer workers.SetLimitForTest(workers.SetLimitForTest(limit))
									argb, modes, b := run(helpers)
									if b != wantBits || !slices.Equal(modes, wantModes) || !slices.Equal(argb, wantArgb) {
										t.Fatalf("%s bits %d maxQuant %d clean %v subGreen %v limit %d helpers %d: parallel search differs",
											in.name, bits, maxQuant, clean, subGreen, limit, helpers)
									}
								}()
							}
						}
						func() {
							defer workers.SetLimitForTest(workers.SetLimitForTest(2))
							defer takeBudget(t)()
							argb, modes, b := run(7)
							if b != wantBits || !slices.Equal(modes, wantModes) || !slices.Equal(argb, wantArgb) {
								t.Fatalf("%s bits %d: search with an exhausted budget differs", in.name, bits)
							}
						}()
					}
				}
			}
		}
	}
}
