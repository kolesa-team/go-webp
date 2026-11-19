package resample

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"
)

// Boundary cases for the row kernels the build selects (assembly on arm64,
// and on amd64 at every SIMD level), against the Go reference kernels:
// every small geometry, spans at the kernels' limits with maximal sums,
// unaligned rows and dirty work buffers.

// TestImportKernelsSmall runs every horizontal shrink and expansion of a
// source row up to 64 pixels wide.
func TestImportKernelsSmall(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(21))
		for sw := 1; sw <= 64; sw++ {
			for dw := 1; dw <= sw+64; dw++ {
				for ch := 1; ch <= 4; ch++ {
					checkImport(t, rnd, sw, dw, ch, nil, nil)
				}
			}
		}
	})
}

// TestImportKernelsSpanLimits runs geometries whose longest span sits at
// the limits where the kernels switch strategy or their 16-bit sums are
// tightest, with rows of 255 (the largest sums) and random rows.
func TestImportKernelsSpanLimits(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(22))
		spans := []int{2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
			255, 256, 257, 510, 511, 512, 513, 514, 515, 519, 1023, 1024, 2046, 2047, 2048, 2049, 2051, 2063}
		for _, span := range spans {
			for dw := 1; dw <= 9; dw++ {
				for _, extra := range []int{-1, 0, 1, dw / 2, dw - 1} {
					sw := span*dw + extra
					if sw < dw {
						continue
					}
					for _, ch := range []int{1, 4} {
						src := bytes.Repeat([]byte{255}, sw*ch)
						compareImport(t, rnd, sw, dw, ch, src, nil)
						checkImport(t, rnd, sw, dw, ch, nil, nil)
					}
				}
			}
		}
	})
}

// TestImportKernelsHugeSpans runs 1-channel shrinks whose spans are long
// enough (over 2^32/255 pixels) for the 32-bit span sums of the Go kernels
// to wrap; the assembly sums 64-bit and must wrap the same way.
func TestImportKernelsHugeSpans(t *testing.T) {
	if testing.Short() {
		t.Skip("34 MB source rows")
	}
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(27))
		const sw = 2*(1<<32/255) + 4099
		white := bytes.Repeat([]byte{255}, sw)
		for _, dw := range []int{1, 2, 3} {
			compareImport(t, rnd, sw, dw, 1, white, nil)
			checkImport(t, rnd, sw, dw, 1, nil, nil)
		}
	})
}

// TestImportKernelsUnaligned runs the import kernels on source rows that
// start at every offset within 16 bytes, with work buffers that start at
// every word offset within 16 bytes.
func TestImportKernelsUnaligned(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(23))
		for i := 0; i < 400; i++ {
			ch := []int{1, 4}[i%2]
			sw, dw := kernelWidths(rnd, i%5 == 0)
			for off := 1; off < 16; off++ {
				buf := make([]byte, off+sw*ch)
				woff := off % 4
				wbuf := make([]uint32, woff+BufferSize(dw, ch))
				checkImport(t, rnd, sw, dw, ch, buf[off:], wbuf[woff:])
			}
		}
	})
}

// TestExportRowKernelsUnaligned runs the export kernels on rows that start
// at every offset within 16 bytes (32 bytes for the word rows).
func TestExportRowKernelsUnaligned(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(24))
		for i := 0; i < 2000; i++ {
			n := rnd.Intn(100)
			boff, woff := rnd.Intn(16), rnd.Intn(8)
			irow := randomWords(rnd, woff+n)[woff:]
			frow := randomWords(rnd, woff+n)[woff:]
			k := randomWords(rnd, 3)
			b := max(1, k[1])
			a := uint32(one - uint64(b))
			got := make([]byte, boff+n+16)[boff : boff+n]
			want := make([]byte, n)
			gi, wi := slices.Clone(irow), slices.Clone(irow)
			exportShrink(got, gi, frow, k[0], k[1])
			exportShrinkGo(want, wi, frow, k[0], k[1])
			if !bytes.Equal(got, want) || !slices.Equal(gi, wi) {
				t.Fatalf("exportShrink, %d values at offsets %d/%d: differs", n, boff, woff)
			}
			exportShrinkZero(got, gi, k[2])
			exportShrinkZeroGo(want, wi, k[2])
			if !bytes.Equal(got, want) || !slices.Equal(gi, wi) {
				t.Fatalf("exportShrinkZero, %d values at offsets %d/%d: differs", n, boff, woff)
			}
			exportExpand(got, frow, k[0])
			exportExpandGo(want, frow, k[0])
			if !bytes.Equal(got, want) {
				t.Fatalf("exportExpand, %d values at offsets %d/%d: differs", n, boff, woff)
			}
			exportExpandMix(got, frow, irow, a, b, k[2])
			exportExpandMixGo(want, frow, irow, a, b, k[2])
			if !bytes.Equal(got, want) {
				t.Fatalf("exportExpandMix, %d values at offsets %d/%d: differs", n, boff, woff)
			}
		}
	})
}

// TestExportRowKernelsExtremeScales runs the export kernels with the
// scales at 0, 1, 2^31 and 2^32-1 against rows of extreme values.
func TestExportRowKernelsExtremeScales(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		scales := []uint32{0, 1, 2, 255, 256, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, 0xffffffff}
		vals := []uint32{0, 1, 2, 127, 128, 255, 256, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff}
		n := 8*len(vals) + 5
		irow, frow := make([]uint32, n), make([]uint32, n)
		for i := range irow {
			irow[i], frow[i] = vals[i%len(vals)], vals[(i/len(vals)+i)%len(vals)]
		}
		got, want := make([]byte, n), make([]byte, n)
		for _, s0 := range scales {
			for _, s1 := range scales {
				gi, wi := slices.Clone(irow), slices.Clone(irow)
				exportShrink(got, gi, frow, s0, s1)
				exportShrinkGo(want, wi, frow, s0, s1)
				if !bytes.Equal(got, want) || !slices.Equal(gi, wi) {
					t.Fatalf("exportShrink, scales %#x %#x: differs", s0, s1)
				}
				exportExpandMix(got, frow, irow, uint32(one-uint64(max(1, s0))), max(1, s0), s1)
				exportExpandMixGo(want, frow, irow, uint32(one-uint64(max(1, s0))), max(1, s0), s1)
				if !bytes.Equal(got, want) {
					t.Fatalf("exportExpandMix, scales %#x %#x: differs", s0, s1)
				}
			}
			gi, wi := slices.Clone(irow), slices.Clone(irow)
			exportShrinkZero(got, gi, s0)
			exportShrinkZeroGo(want, wi, s0)
			if !bytes.Equal(got, want) || !slices.Equal(gi, wi) {
				t.Fatalf("exportShrinkZero, scale %#x: differs", s0)
			}
			exportExpand(got, frow, s0)
			exportExpandGo(want, frow, s0)
			if !bytes.Equal(got, want) {
				t.Fatalf("exportExpand, scale %#x: differs", s0)
			}
		}
	})
}

// TestPremultiplyUnaligned runs the premultiply kernel on rows starting at
// every offset within 16 bytes, of every length up to 80 bytes.
func TestPremultiplyUnaligned(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(25))
		for off := 0; off < 16; off++ {
			for n := 0; n <= 80; n++ {
				buf := make([]byte, off+n+16)
				row := buf[off : off+n]
				copy(row, randomPixels(rnd, n))
				tail := slices.Clone(buf[off+n:])
				checkMultRow(t, row)
				if !bytes.Equal(buf[off+n:], tail) {
					t.Fatalf("offset %d, %d bytes: wrote past the row", off, n)
				}
			}
		}
	})
}

// TestDirtyWorkBuffer reinitializes one work buffer, left dirty by a
// previous geometry, for random geometries, and checks every output row
// against the legacy rescaler, like the pooled buffers of the scaling
// sinks.
func TestDirtyWorkBuffer(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(26))
		var work []uint32
		n := 400
		if testing.Short() {
			n = 100
		}
		for i := 0; i < n; i++ {
			sw, sh, dw, dh := randomGeometry(rnd)
			ch := []int{1, 4, 1, 4, 3}[i%5]
			if size := BufferSize(dw, ch); cap(work) < size {
				work = make([]uint32, size+rnd.Intn(64))
			}
			for j := range work[:cap(work)] {
				work[:cap(work)][j] = rnd.Uint32()
			}
			src := randomPixels(rnd, sw*sh*ch)
			want := rescaleLegacy(t, src, sw, sh, dw, dh, ch)
			var rs Rescaler
			if err := rs.Init(sw, sh, dw, dh, ch, work); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, dw*dh*ch)
			imported := 0
			for imported < sh {
				imported += rs.Import(sh-imported, src[imported*sw*ch:], sw*ch)
				rs.Export(got[rs.DstY()*dw*ch:], dw*ch)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%dx%d -> %dx%d, %d channels: output differs with a dirty work buffer", sw, sh, dw, dh, ch)
			}
		}
	})
}
