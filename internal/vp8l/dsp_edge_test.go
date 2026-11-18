package vp8l

import (
	"bytes"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// Edge cases of the per-row kernels (dsp_*.go) that the random tests of
// dsp_test.go reach only by chance: every short width, every pair of
// adjacent predictor modes, channel values at the wrap-around and
// averaging boundaries, and byte slices starting at every offset modulo 16.

// edgeChannels are channel values around the 0/255 wrap and the signed
// 0x7f/0x80 boundary.
var edgeChannels = []uint32{0, 1, 2, 0x7e, 0x7f, 0x80, 0x81, 0xfd, 0xfe, 0xff}

// edgePixels returns n pixels whose channels are edge values, or random
// bytes for one pixel in four.
func edgePixels(rng *rand.Rand, n int) []uint32 {
	p := make([]uint32, n)
	for i := range p {
		if rng.Intn(4) == 0 {
			p[i] = rng.Uint32()
			continue
		}
		for c := 0; c < 4; c++ {
			p[i] |= edgeChannels[rng.Intn(len(edgeChannels))] << (8 * c)
		}
	}
	return p
}

// TestPredictRowModePairs runs every ordered pair of predictor modes in
// alternating tiles, for tile sizes 4, 8 and 16 and every width up to
// three tiles and a bit, with edge channel values.
func TestPredictRowModePairs(t *testing.T) { forEachLevel(t, testPredictRowModePairs) }

func testPredictRowModePairs(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	reps := 3
	if testing.Short() {
		reps = 1
	}
	for a := uint32(0); a < 16; a++ {
		for b := uint32(0); b < 16; b++ {
			for bits := 2; bits <= 4; bits++ {
				tileW := 1 << bits
				for width := 1; width <= 3*tileW+3; width++ {
					modes := make([]uint32, subSampleSize(width, bits))
					for i := range modes {
						m := a
						if i%2 == 1 {
							m = b
						}
						modes[i] = rng.Uint32()&^0xf00 | m<<8
					}
					for r := 0; r < reps; r++ {
						in, up := edgePixels(rng, width), edgePixels(rng, width)
						predictCheck(t, modes, in, up, bits, fmt.Sprintf("modes %d,%d", a, b))
					}
				}
			}
		}
	}
}

// TestPredictRowWide checks long rows with large tiles (bits 5-9) and
// random modes, the shape of real images, on edge channel values.
func TestPredictRowWide(t *testing.T) { forEachLevel(t, testPredictRowWide) }

func testPredictRowWide(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	iters := 2000
	if testing.Short() {
		iters = 200
	}
	for iter := 0; iter < iters; iter++ {
		bits := 5 + rng.Intn(5)
		width := 1 + rng.Intn(4100)
		modes := make([]uint32, subSampleSize(width, bits))
		for i := range modes {
			modes[i] = uint32(rng.Intn(16)) << 8
		}
		predictCheck(t, modes, edgePixels(rng, width), edgePixels(rng, width), bits, fmt.Sprintf("iter %d", iter))
	}
}

// TestByteKernelsOffsets runs the byte-oriented kernels on slices that
// start at every offset modulo 16, for every short length.
func TestByteKernelsOffsets(t *testing.T) { forEachLevel(t, testByteKernelsOffsets) }

func testByteKernelsOffsets(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	var pal [256]byte
	rng.Read(pal[:])
	for off := 0; off < 16; off++ {
		for n := 0; n <= 80; n++ {
			// lookupBytes
			inAll := make([]byte, n+off+16)
			rng.Read(inAll)
			in := inAll[off : off+n : off+n]
			want := make([]byte, n)
			lookupBytesGo(&pal, in, want)
			all := bytes.Repeat([]byte{guard}, n+32+off)
			out := all[16+off : 16+off+n : 16+off+n]
			lookupBytes(&pal, in, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("lookupBytes: offset %d, %d bytes: output differs", off, n)
			}
			for i, v := range all {
				if (i < 16+off || i >= 16+off+n) && v != guard {
					t.Fatalf("lookupBytes: offset %d, %d bytes: wrote byte %d", off, n, i-16-off)
				}
			}

			// emitRGBA, emitAddGreen into an unaligned destination.
			src := edgePixels(rng, n)
			for _, k := range []struct {
				name     string
				got, ref func([]uint32, []byte)
			}{{"emitRGBA", emitRGBA, emitRGBAGo}, {"emitAddGreen", emitAddGreen, emitAddGreenGo}} {
				want := bytes.Repeat([]byte{guard}, 4*n)
				k.ref(src, want)
				all := bytes.Repeat([]byte{guard}, 4*n+32+off)
				dst := all[16+off : 16+off+4*n : 16+off+4*n]
				k.got(src, dst)
				if !bytes.Equal(dst, want) {
					t.Fatalf("%s: offset %d, %d pixels: output differs", k.name, off, n)
				}
				for i, v := range all {
					if (i < 16+off || i >= 16+off+4*n) && v != guard {
						t.Fatalf("%s: offset %d, %d pixels: wrote byte %d", k.name, off, n, i-16-off)
					}
				}
			}
		}
	}
}

// TestExpandBundledOffsets runs the bundled-index expansions on packed
// rows and destinations starting at every offset modulo 16, with palettes
// of edge values, for widths covering zero to three asm blocks.
func TestExpandBundledOffsets(t *testing.T) { forEachLevel(t, testExpandBundledOffsets) }

func testExpandBundledOffsets(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	for bits := 1; bits <= 3; bits++ {
		pal := edgePixels(rng, 1<<(8>>bits))
		tr := &transform{kind: colorIndexingTransform, bits: bits, data: pal}
		var a alphaTables
		a.init(tr)
		d := getDecoder()
		d.palReady, d.bundleReady = false, false
		rgba, tab := d.paletteRGBA(tr), d.bundleTable(tr)
		for off := 0; off < 16; off++ {
			for width := 1; width <= 48<<bits+9; width += 1 + rng.Intn(5) {
				pw := subSampleSize(width, bits)
				srcAll := make([]byte, pw+off)
				rng.Read(srcAll)
				src := srcAll[off : off+pw : off+pw]

				want := bytes.Repeat([]byte{guard}, width)
				expandBundledRowGo(&a, bits, src, want)
				all := bytes.Repeat([]byte{guard}, width+32+off)
				got := all[16+off : 16+off+width : 16+off+width]
				expandBundledRow(&a, bits, src, got)
				if !bytes.Equal(got, want) {
					t.Fatalf("8-bit, bits %d, offset %d, width %d: output differs", bits, off, width)
				}
				for i, v := range all {
					if (i < 16+off || i >= 16+off+width) && v != guard {
						t.Fatalf("8-bit, bits %d, offset %d, width %d: wrote byte %d", bits, off, width, i-16-off)
					}
				}

				n := 4 * width
				wantRGBA := bytes.Repeat([]byte{guard}, n)
				expandBundledRGBAGo(tab, bits, src, wantRGBA)
				all = bytes.Repeat([]byte{guard}, n+32+off)
				got = all[16+off : 16+off+n : 16+off+n]
				expandBundledRGBA(rgba, tab, bits, src, got)
				if !bytes.Equal(got, wantRGBA) {
					t.Fatalf("RGBA, bits %d, offset %d, width %d: output differs", bits, off, width)
				}
				for i, v := range all {
					if (i < 16+off || i >= 16+off+n) && v != guard {
						t.Fatalf("RGBA, bits %d, offset %d, width %d: wrote byte %d", bits, off, width, i-16-off)
					}
				}
			}
		}
		putDecoder(d)
	}
}

// TestPixelKernelsShortRows checks crossColorRow and addGreen on every
// width up to 64 at every start offset modulo 4 pixels, with edge values
// and edge multipliers.
func TestPixelKernelsShortRows(t *testing.T) { forEachLevel(t, testPixelKernelsShortRows) }

func testPixelKernelsShortRows(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	edgeMult := func() uint32 {
		var m uint32
		for c := 0; c < 3; c++ {
			m |= edgeChannels[rng.Intn(len(edgeChannels))] << (8 * c)
		}
		return m | rng.Uint32()&0xff000000
	}
	for off := 0; off < 4; off++ {
		for width := 1; width <= 64; width++ {
			for bits := 2; bits <= 4; bits++ {
				for r := 0; r < 4; r++ {
					mults := make([]uint32, subSampleSize(width, bits))
					for i := range mults {
						mults[i] = edgeMult()
					}
					srcAll := edgePixels(rng, width+off)
					src := srcAll[off:]
					want := make([]uint32, width)
					crossColorRowGo(mults, src, want, bits)
					gotAll := make([]uint32, width+off+2)
					for i := range gotAll {
						gotAll[i] = 0xa5a5a5a5
					}
					got := gotAll[1+off : 1+off+width : 1+off+width]
					crossColorRow(mults, src, got, bits)
					if !slices.Equal(got, want) {
						t.Fatalf("crossColorRow: offset %d, width %d, bits %d: output differs", off, width, bits)
					}
					for i, c := range gotAll {
						if (i < 1+off || i >= 1+off+width) && c != 0xa5a5a5a5 {
							t.Fatalf("crossColorRow: offset %d, width %d: wrote outside the row", off, width)
						}
					}
				}
			}
			src := edgePixels(rng, width+off)[off:]
			want := make([]uint32, width)
			addGreenGo(src, want)
			got := make([]uint32, width+off)[off:]
			addGreen(src, got)
			if !slices.Equal(got, want) {
				t.Fatalf("addGreen: offset %d, width %d: output differs", off, width)
			}
		}
	}
}
