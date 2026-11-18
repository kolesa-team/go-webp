//go:build darwin || linux

package vp8l

import (
	"bytes"
	"math/rand"
	"slices"
	"syscall"
	"testing"
	"unsafe"
)

// guardedBytes returns n bytes of memory placed directly after (atEnd
// false) or directly before (atEnd true) an inaccessible page, so that a
// kernel reading or writing outside the n bytes on that side faults.
func guardedBytes(t *testing.T, n int, atEnd bool) []byte {
	t.Helper()
	pg := syscall.Getpagesize()
	data := max((n+pg-1)/pg*pg, pg)
	mem, err := syscall.Mmap(-1, 0, data+2*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:pg], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[pg+data:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if atEnd {
		return mem[pg+data-n : pg+data : pg+data]
	}
	return mem[pg : pg+n : pg+n]
}

// guardedWords is guardedBytes for n pixels.
func guardedWords(t *testing.T, n int, atEnd bool) []uint32 {
	b := guardedBytes(t, 4*n, atEnd)
	if n == 0 {
		return []uint32{}
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), n)
}

// guardLengths are the row lengths of the guard-page tests: every short
// length (all tail sizes of every vector width) and a few long ones.
func guardLengths() []int {
	var ns []int
	for n := 0; n <= 140; n++ {
		ns = append(ns, n)
	}
	return append(ns, 255, 256, 257, 1000, 3001)
}

// TestKernelsGuardPages runs every per-row kernel at every SIMD level on
// buffers that sit against inaccessible pages on either side: a read or
// write outside the slices crashes the test. The outputs are also
// compared with the Go references.
func TestKernelsGuardPages(t *testing.T) { forEachLevel(t, testKernelsGuardPages) }

func testKernelsGuardPages(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, atEnd := range []bool{false, true} {
		pal := (*[256]uint32)(unsafe.Pointer(&guardedBytes(t, 1024, atEnd)[0]))
		copy(pal[:], randomPixels(rng, 256))
		pal8 := (*[256]byte)(unsafe.Pointer(&guardedBytes(t, 256, atEnd)[0]))
		rng.Read(pal8[:])
		for _, n := range guardLengths() {
			src := guardedWords(t, n, atEnd)
			copy(src, randomPixels(rng, n))
			dst := guardedBytes(t, 4*n, atEnd)
			want := make([]byte, 4*n)
			for _, k := range []struct {
				name     string
				got, ref func([]uint32, []byte)
			}{{"emitRGBA", emitRGBA, emitRGBAGo}, {"emitAddGreen", emitAddGreen, emitAddGreenGo}} {
				k.ref(src, want)
				k.got(src, dst)
				if !bytes.Equal(dst, want) {
					t.Fatalf("%s: %d pixels: output differs", k.name, n)
				}
			}

			out := guardedWords(t, n, atEnd)
			wantW := make([]uint32, n)
			addGreenGo(src, wantW)
			addGreen(src, out)
			if !slices.Equal(out, wantW) {
				t.Fatalf("addGreen: %d pixels: output differs", n)
			}

			idx := guardedBytes(t, n, atEnd)
			rng.Read(idx)
			lb := guardedBytes(t, n, atEnd)
			lookupBytes(pal8, idx, lb)
			wantB := make([]byte, n)
			lookupBytesGo(pal8, idx, wantB)
			if !bytes.Equal(lb, wantB) {
				t.Fatalf("lookupBytes: %d bytes: output differs", n)
			}
			lookupRGBA(pal, idx, dst)
			lookupRGBAGo(pal, idx, want)
			if !bytes.Equal(dst, want) {
				t.Fatalf("lookupRGBA: %d pixels: output differs", n)
			}

			if n > 0 {
				for bits := 2; bits <= 5; bits++ {
					mults := guardedWords(t, subSampleSize(n, bits), atEnd)
					copy(mults, randomPixels(rng, len(mults)))
					crossColorRow(mults, src, out, bits)
					crossColorRowGo(mults, src, wantW, bits)
					if !slices.Equal(out, wantW) {
						t.Fatalf("crossColorRow: %d pixels, bits %d: output differs", n, bits)
					}
					modes := guardedWords(t, subSampleSize(n, bits), atEnd)
					for i := range modes {
						modes[i] = uint32(rng.Intn(16)) << 8
					}
					up := guardedWords(t, n+1, atEnd)
					copy(up, randomPixels(rng, n))
					wantW[0] = addPixels(src[0], up[0])
					up[n] = wantW[0]
					out[0] = wantW[0]
					predictRowGo(modes, src, wantW, up, bits)
					predictRow(modes, src, out, up, bits)
					if !slices.Equal(out, wantW) {
						t.Fatalf("predictRow: %d pixels, bits %d: output differs", n, bits)
					}
				}
			}
		}
		for bits := 1; bits <= 3; bits++ {
			tr := &transform{kind: colorIndexingTransform, bits: bits, data: randomPixels(rng, 1<<(8>>bits))}
			var a alphaTables
			a.init(tr)
			d := getDecoder()
			d.palReady, d.bundleReady = false, false
			rgba, tab := d.paletteRGBA(tr), d.bundleTable(tr)
			for _, w := range guardLengths() {
				packed := guardedBytes(t, subSampleSize(w, bits), atEnd)
				rng.Read(packed)
				got := guardedBytes(t, w, atEnd)
				want := make([]byte, w)
				expandBundledRow(&a, bits, packed, got)
				expandBundledRowGo(&a, bits, packed, want)
				if !bytes.Equal(got, want) {
					t.Fatalf("expandBundledRow: width %d, bits %d: output differs", w, bits)
				}
				got = guardedBytes(t, 4*w, atEnd)
				want = make([]byte, 4*w)
				expandBundledRGBA(rgba, tab, bits, packed, got)
				expandBundledRGBAGo(tab, bits, packed, want)
				if !bytes.Equal(got, want) {
					t.Fatalf("expandBundledRGBA: width %d, bits %d: output differs", w, bits)
				}
			}
			putDecoder(d)
		}
	}
}
