//go:build arm64 && !purego && (darwin || linux)

package resample

import (
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// guardedBytes returns n bytes that end right before an inaccessible page,
// so that any read or write past them faults, and a function releasing
// them.
func guardedBytes(t testing.TB, n int) ([]byte, func()) {
	t.Helper()
	ps := os.Getpagesize()
	size := (n+ps-1)/ps*ps + ps
	mem, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(mem[size-ps:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	end := size - ps
	return mem[end-n : end : end], func() { _ = syscall.Munmap(mem) }
}

// guardedWords is guardedBytes for n uint32 cells.
func guardedWords(t testing.TB, n int) ([]uint32, func()) {
	b, free := guardedBytes(t, 4*n)
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(b))), n), free
}

// TestKernelsGuarded runs the import and export kernels with the source
// row, the work buffer (column tables included) and the destination row
// each ending at an inaccessible page.
func TestKernelsGuarded(t *testing.T) {
	rnd := rand.New(rand.NewSource(13))
	n := 6000
	if testing.Short() {
		n = 1000
	}
	for i := 0; i < n; i++ {
		ch := []int{1, 4}[i%2]
		sw, dw := kernelWidths(rnd, i%5 == 0)
		src, freeSrc := guardedBytes(t, sw*ch)
		work, freeWork := guardedWords(t, BufferSize(dw, ch))
		checkImport(t, rnd, sw, dw, ch, src, work)
		freeSrc()
		freeWork()
	}
	for i := 0; i < n/10; i++ {
		sw, sh, dw, dh := randomGeometry(rnd)
		ch := []int{1, 4}[i%2]
		dst, free := guardedBytes(t, dw*ch)
		checkRescale(t, rnd, sw, sh, dw, dh, ch, dst)
		free()
	}
}

// TestKernelCoverage checks that the geometries of TestImportKernels and
// FuzzImportKernels reach every assembly kernel.
func TestKernelCoverage(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	hits := map[uint8]int{}
	for i := 0; i < 20000; i++ {
		ch := []int{1, 4}[i%2]
		sw, dw := kernelWidths(rnd, i%5 == 0)
		rs, err := New(sw, 2, dw, 1, ch)
		if err != nil {
			t.Fatal(err)
		}
		hits[rs.arch.kind]++
	}
	for k := uint8(kindGo); k <= kindExpand4; k++ {
		if hits[k] < 100 {
			t.Errorf("kernel %d: %d hits", k, hits[k])
		}
	}
}

// TestKernelSelection checks that the assembly kernels are the ones the
// usual geometries get, so that the differential tests exercise them.
func TestKernelSelection(t *testing.T) {
	for _, c := range []struct {
		sw, dw, ch int
		kind       uint8
	}{
		{1024, 300, 4, kindShrink4},
		{3000, 320, 4, kindShrink4},
		{3000, 6, 4, kindShrink4},
		{3000, 5, 4, kindGo}, // spans of more than maxShrink4Span pixels
		{3, 1, 4, kindGo},    // narrower than one 16-byte load
		{512, 320, 1, kindShrink1T16},
		{1024, 320, 1, kindShrink1T16},
		{1500, 320, 1, kindShrink1T32},
		{3000, 320, 1, kindShrink1T64},
		{3000, 100, 1, kindShrink1},
		{5000, 2, 1, kindGo}, // spans of more than maxShrink1Span pixels
		{12, 3, 1, kindGo},   // narrower than one 16-byte window
		{3000, 320, 2, kindGo},
		{300, 1000, 4, kindExpand4},
		{1, 1000, 4, kindGo}, // one pixel wide
		{300, 1000, 1, kindGo},
	} {
		rs, err := New(c.sw, 10, c.dw, 5, c.ch)
		if err != nil {
			t.Fatal(err)
		}
		if rs.arch.kind != c.kind {
			t.Errorf("%d -> %d, %d channels: kernel %d, want %d", c.sw, c.dw, c.ch, rs.arch.kind, c.kind)
		}
	}
}
