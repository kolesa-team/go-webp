//go:build darwin || linux

package vp8

import (
	"math/rand/v2"
	"syscall"
	"testing"
	"unsafe"
)

// guardedBuf returns n bytes of memory placed directly after (atEnd false)
// or directly before (atEnd true) an inaccessible page, so that any access
// outside the n bytes on that side faults.
func guardedBuf(t *testing.T, n int, atEnd bool) []byte {
	t.Helper()
	ps := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 3*ps, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:ps], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[2*ps:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if atEnd {
		return mem[2*ps-n : 2*ps : 2*ps]
	}
	return mem[ps : ps+n : ps+n]
}

// TestReconGuardPages runs the reconstruction entry points on buffers that
// sit against inaccessible pages on either side: an out-of-bounds read or
// write in a kernel crashes the test.
func TestReconGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	var c reconCase
	for _, atEnd := range []bool{false, true} {
		bm := guardedBuf(t, yuvSize, atEnd)
		cm := guardedBuf(t, 384*2, atEnd)
		mm := guardedBuf(t, 16, atEnd)
		b := (*[yuvSize]byte)(unsafe.Pointer(&bm[0]))
		coeffs := (*[384]int16)(unsafe.Pointer(&cm[0]))
		modes := (*[16]uint8)(unsafe.Pointer(&mm[0]))
		n := reconIters(t) / 20
		for i := 0; i < n; i++ {
			randReconCase(r, &c)
			*coeffs, *modes = c.coeffs, c.modes

			*b = c.b
			want := c.b
			reconI4(b, coeffs, c.bits, modes)
			reconI4Ref(&want, &c.coeffs, c.bits, &c.modes)
			if !diffBuffers(t, "reconI4", b, &want) {
				t.FailNow()
			}
			*b = c.b
			want = c.b
			reconI16(b, coeffs, c.bits, c.mode)
			reconI16Ref(&want, &c.coeffs, c.bits, c.mode)
			if !diffBuffers(t, "reconI16", b, &want) {
				t.FailNow()
			}
			*b = c.b
			want = c.b
			reconUV(b, coeffs, c.uvBits, c.uvMode)
			reconUVRef(&want, &c.coeffs, c.uvBits, c.uvMode)
			if !diffBuffers(t, "reconUV", b, &want) {
				t.FailNow()
			}
			if *coeffs != c.coeffs || *modes != c.modes {
				t.Fatal("inputs modified")
			}
		}
	}
}
