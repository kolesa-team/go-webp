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

// TestCleanFastPath checks that the exact=0 cleaning with the fast path for
// the pixels away from transparency gives the residuals and modes of the
// all-slow path, on random transparency masks (isolated pixels, first and
// last columns, whole rows), for every tile size and with each single mode.
func TestCleanFastPath(t *testing.T) {
	defer func(v bool) { cleanFastPath = v }(cleanFastPath)
	rng := rand.New(rand.NewSource(17))
	type in struct {
		name string
		w, h int
		tr   func(x, y int) bool
	}
	ins := []in{
		{"sparse", 67, 41, func(x, y int) bool { return rng.Intn(20) == 0 }},
		{"dense", 50, 33, func(x, y int) bool { return rng.Intn(2) == 0 }},
		{"edges", 45, 30, func(x, y int) bool { return x == 0 || x == 44 || y == 29 }},
		{"firstcol", 40, 20, func(x, y int) bool { return x == 0 && y%2 == 1 }},
		{"blocks", 70, 70, func(x, y int) bool { return (x/9+y/7)%3 == 0 }},
		{"1xN", 1, 50, func(x, y int) bool { return y%3 == 0 }},
		{"Nx1", 50, 1, func(x, y int) bool { return x%4 == 1 }},
		{"2x2", 2, 2, func(x, y int) bool { return x == y }},
	}
	for _, c := range ins {
		src := make([]uint32, c.w*c.h)
		for y := 0; y < c.h; y++ {
			for x := 0; x < c.w; x++ {
				p := 0xff000000 | uint32(x*5+rng.Intn(3))&0xff<<16 | uint32(y*3)&0xff<<8 | uint32(x+y)&0xff
				if c.tr(x, y) {
					p = rng.Uint32() & 0x00ffffff
				} else if rng.Intn(10) == 0 {
					p = p&0x00ffffff | uint32(1+rng.Intn(254))<<24
				}
				src[y*c.w+x] = p
			}
		}
		for _, bits := range []int{2, 3, 4, 5} {
			for _, maxBits := range []int{bits, min(bits+2, maxTransformBits)} {
				run := func(fast bool, helpers int) ([]uint32, []uint32, int) {
					cleanFastPath = fast
					argb := slices.Clone(src)
					modes := make([]uint32, subSampleSize(c.w, bits)*subSampleSize(c.h, bits))
					sc := new(scratch)
					b := predictorImage(c.w, c.h, bits, maxBits, false, 1, true, true, false, argb, modes, sc, helpers)
					sc.crew.stop()
					return argb, modes, b
				}
				wantArgb, wantModes, wantBits := run(false, 0)
				for _, helpers := range []int{0, 3} {
					argb, modes, b := run(true, helpers)
					if b != wantBits || !slices.Equal(modes, wantModes) || !slices.Equal(argb, wantArgb) {
						t.Fatalf("%s bits %d..%d helpers %d: fast clean path differs", c.name, bits, maxBits, helpers)
					}
				}
			}
		}
		// Every mode on full rows (the final residual pass).
		for mode := 0; mode < numPredModes; mode++ {
			rows := func(fast bool) []uint32 {
				prev, cur := make([]uint32, c.w+1), make([]uint32, c.w+1)
				out := make([]uint32, c.w*c.h)
				mask := make([]uint8, c.w*c.h)
				cleanMask(src, c.w, c.h, mask)
				for y := 0; y < c.h; y++ {
					copy(cur, src[y*c.w:(y+1)*c.w])
					if y > 0 {
						prev[c.w] = cur[0]
					}
					if fast {
						residualRowClean(mode, cur, prev, y, c.w, c.h, 0, c.w, mask[y*c.w:(y+1)*c.w], out[y*c.w:(y+1)*c.w])
					} else {
						residualRowNL(mode, cur, prev, y, c.w, c.h, 0, c.w, nil, 1, false, true, out[y*c.w:(y+1)*c.w])
					}
					prev, cur = cur, prev
				}
				return out
			}
			if !slices.Equal(rows(true), rows(false)) {
				t.Fatalf("%s mode %d: fast clean rows differ", c.name, mode)
			}
		}
	}
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
							sc := new(scratch)
							b := predictorImage(in.w, in.h, bits, bits, false, maxQuant, subGreen, clean, false, argb, modes, sc, helpers)
							sc.crew.stop()
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

// TestPredictorOpaque checks the opaque shortcut of the mode search: for
// opaque images, every alpha residual is 0 (for every mode, with and
// without near-lossless), so the search with opaque set chooses the same
// modes and produces the same residuals.
func TestPredictorOpaque(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	inputs := predInputs()
	noise := make([]uint32, 61*39)
	for i := range noise {
		noise[i] = 0xff000000 | rng.Uint32()
	}
	inputs = append(inputs, struct {
		name string
		argb []uint32
		w, h int
	}{"opaque noise", noise, 61, 39})
	for _, in := range inputs {
		if slices.ContainsFunc(in.argb, func(p uint32) bool { return p < 0xff000000 }) {
			continue
		}
		for mode := 0; mode < numPredModes; mode++ {
			out := make([]uint32, in.w)
			for y := 0; y < in.h; y++ {
				var up []uint32
				if y > 0 {
					up = in.argb[(y-1)*in.w : y*in.w+1]
				}
				residualRow(mode, in.argb[y*in.w:(y+1)*in.w], up, y, 0, in.w, out)
				for x, r := range out {
					if r>>24 != 0 {
						t.Fatalf("%s mode %d: alpha residual %#x at (%d,%d)", in.name, mode, r, x, y)
					}
				}
			}
		}
		for _, bits := range []int{2, 3, 5} {
			for _, maxBits := range []int{bits, min(bits+2, maxTransformBits)} {
				for _, maxQuant := range []int{1, 8} {
					for _, subGreen := range []bool{false, true} {
						run := func(opaque bool, helpers int) ([]uint32, []uint32, int) {
							argb := slices.Clone(in.argb)
							modes := make([]uint32, subSampleSize(in.w, bits)*subSampleSize(in.h, bits))
							sc := new(scratch)
							b := predictorImage(in.w, in.h, bits, maxBits, false, maxQuant, subGreen, false, opaque, argb, modes, sc, helpers)
							sc.crew.stop()
							return argb, modes, b
						}
						wantArgb, wantModes, wantBits := run(false, 0)
						for _, helpers := range []int{0, 3} {
							argb, modes, b := run(true, helpers)
							if b != wantBits || !slices.Equal(modes, wantModes) || !slices.Equal(argb, wantArgb) {
								t.Fatalf("%s bits %d..%d maxQuant %d subGreen %v helpers %d: opaque search differs",
									in.name, bits, maxBits, maxQuant, subGreen, helpers)
							}
						}
					}
				}
			}
		}
	}
}
