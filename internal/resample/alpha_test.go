package resample

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"
)

// TestPremultiplyExhaustive runs the selected premultiply and
// unpremultiply kernels on every (channel value, alpha) pair, in every
// channel position, and compares them with the Go ones.
func TestPremultiplyExhaustive(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		row := make([]byte, 0, 4*65536)
		for a := 0; a < 256; a++ {
			for x := 0; x < 256; x++ {
				row = append(row, byte(x), byte(255-x), byte(x^0x5a), byte(a))
			}
		}
		for _, inverse := range []bool{false, true} {
			got, want := slices.Clone(row), slices.Clone(row)
			if inverse {
				UnpremultiplyRGBA(got)
				multRGBA(want, &unpremulScale)
			} else {
				PremultiplyRGBA(got)
				multRGBA(want, &premulScale)
			}
			if !bytes.Equal(got, want) {
				i := 0
				for got[i] == want[i] {
					i++
				}
				t.Fatalf("inverse %v, pixel %v: got %v, want %v", inverse,
					row[i&^3:i&^3+4], got[i&^3:i&^3+4], want[i&^3:i&^3+4])
			}
		}
	})
}

// checkMultRow compares the selected premultiply and unpremultiply kernels
// with the Go ones on row, including its trailing partial pixel if any.
func checkMultRow(t testing.TB, row []byte) {
	t.Helper()
	for _, inverse := range []bool{false, true} {
		got, want := slices.Clone(row), slices.Clone(row)
		if inverse {
			UnpremultiplyRGBA(got)
			multRGBA(want, &unpremulScale)
		} else {
			PremultiplyRGBA(got)
			multRGBA(want, &premulScale)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("%d bytes, inverse %v: differs from the Go kernel", len(row), inverse)
		}
	}
}

func TestMultRows(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(16))
		for i := 0; i < 20000; i++ {
			checkMultRow(t, randomPixels(rnd, rnd.Intn(200)))
		}
	})
}

func FuzzMultRows(f *testing.F) {
	f.Add([]byte{200, 100, 50, 128, 1, 2, 3, 0, 4, 5, 6, 255, 7})
	f.Fuzz(func(t *testing.T, row []byte) { eachLevel(func() { checkMultRow(t, row) }) })
}

func BenchmarkPremultiplyRandom(b *testing.B) {
	defer restoreLevel()
	levels, setLevel := benchLevels()
	random := benchImage(4096, 1)
	opaque := benchImage(4096, 1)
	for i := 3; i < len(opaque); i += 4 {
		opaque[i] = 255
	}
	for li := 0; li <= len(levels); li++ {
		name, ref := "go", li == len(levels)
		if !ref {
			name = levels[li]
			setLevel(li)
		}
		for k, inverse := range []bool{false, true, true} {
			op := map[bool]string{false: "premultiply", true: "unpremultiply"}[inverse]
			row := random
			if k == 2 {
				op, row = "unpremultiply-opaque", opaque
			}
			b.Run(op+"/"+name, func(b *testing.B) {
				b.SetBytes(int64(len(row)))
				for i := 0; i < b.N; i++ {
					switch {
					case ref && inverse:
						multRGBA(row, &unpremulScale)
					case ref:
						multRGBA(row, &premulScale)
					case inverse:
						UnpremultiplyRGBA(row)
					default:
						PremultiplyRGBA(row)
					}
					row[3] = byte(i)
				}
			})
		}
	}
}
