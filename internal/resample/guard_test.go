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

//go:build (arm64 || amd64) && !purego && (darwin || linux)

package resample

import (
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// guardedBytes returns n bytes between two inaccessible pages, so that any
// access past them faults: they end right before the second page or, when
// atStart is true, start right after the first one (reads before them
// fault). free releases them.
func guardedBytes(t testing.TB, n int, atStart bool) (b []byte, free func()) {
	t.Helper()
	ps := os.Getpagesize()
	body := (n + ps - 1) / ps * ps
	size := body + 2*ps
	mem, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(mem[:ps], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(mem[size-ps:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	start := ps + body - n
	if atStart {
		start = ps
	}
	return mem[start : start+n : start+n], func() { _ = syscall.Munmap(mem) }
}

// guardedWords is guardedBytes for n uint32 cells.
func guardedWords(t testing.TB, n int, atStart bool) ([]uint32, func()) {
	b, free := guardedBytes(t, 4*n, atStart)
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(b))), n), free
}

// TestKernelsGuarded runs the import and export kernels with the source
// row, the work buffer (column tables and records included) and the
// destination row each next to an inaccessible page, after their end and
// before their start.
func TestKernelsGuarded(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(13))
		n := 6000
		if testing.Short() {
			n = 1000
		}
		for i := 0; i < n; i++ {
			ch := []int{1, 4}[i%2]
			sw, dw := kernelWidths(rnd, i%5 == 0)
			atStart := i%3 == 0
			src, freeSrc := guardedBytes(t, sw*ch, atStart)
			work, freeWork := guardedWords(t, BufferSize(dw, ch), atStart)
			checkImport(t, rnd, sw, dw, ch, src, work)
			freeSrc()
			freeWork()
		}
		for i := 0; i < n/10; i++ {
			sw, sh, dw, dh := randomGeometry(rnd)
			ch := []int{1, 4}[i%2]
			dst, free := guardedBytes(t, dw*ch, i%3 == 0)
			checkRescale(t, rnd, sw, sh, dw, dh, ch, dst)
			free()
		}
	})
}

// TestExportRowKernelsGuarded runs the export kernels on rows (bytes and
// words) next to inaccessible pages.
func TestExportRowKernelsGuarded(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(17))
		for i := 0; i < 400; i++ {
			n := 1 + rnd.Intn(200)
			atStart := i%2 == 0
			irow, f1 := guardedWords(t, n, atStart)
			frow, f2 := guardedWords(t, n, atStart)
			dst, f3 := guardedBytes(t, n, atStart)
			copy(irow, randomWords(rnd, n))
			copy(frow, randomWords(rnd, n))
			k := randomWords(rnd, 3)
			want := make([]byte, n)
			wi := append([]uint32(nil), irow...)
			exportShrinkGo(want, wi, frow, k[0], k[1])
			exportShrink(dst, irow, frow, k[0], k[1])
			if string(dst) != string(want) {
				t.Fatalf("exportShrink, %d values: differs", n)
			}
			exportShrinkZeroGo(want, wi, k[2])
			exportShrinkZero(dst, irow, k[2])
			if string(dst) != string(want) {
				t.Fatalf("exportShrinkZero, %d values: differs", n)
			}
			exportExpandGo(want, frow, k[0])
			exportExpand(dst, frow, k[0])
			if string(dst) != string(want) {
				t.Fatalf("exportExpand, %d values: differs", n)
			}
			b := max(1, k[1])
			exportExpandMixGo(want, frow, irow, uint32(one-uint64(b)), b, k[2])
			exportExpandMix(dst, frow, irow, uint32(one-uint64(b)), b, k[2])
			if string(dst) != string(want) {
				t.Fatalf("exportExpandMix, %d values: differs", n)
			}
			f1()
			f2()
			f3()
		}
	})
}

// TestMultRowsGuarded runs the premultiply and unpremultiply kernels on
// rows next to inaccessible pages.
func TestMultRowsGuarded(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rnd := rand.New(rand.NewSource(18))
		for i := 0; i < 600; i++ {
			n := rnd.Intn(300)
			row, free := guardedBytes(t, n, i%2 == 0)
			copy(row, randomPixels(rnd, n))
			want := append([]byte(nil), row...)
			multRGBA(want, &premulScale)
			PremultiplyRGBA(row)
			if string(row) != string(want) {
				t.Fatalf("premultiply, %d bytes: differs", n)
			}
			multRGBA(want, &unpremulScale)
			UnpremultiplyRGBA(row)
			if string(row) != string(want) {
				t.Fatalf("unpremultiply, %d bytes: differs", n)
			}
			free()
		}
	})
}
