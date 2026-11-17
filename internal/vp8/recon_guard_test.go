//go:build darwin || linux

package vp8

import (
	"bytes"
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
		in := (*[16]int16)(unsafe.Pointer(&guardedBuf(t, 32, atEnd)[0]))
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

			*in = *(*[16]int16)(c.coeffs[16*r.IntN(24):])
			whtOut := c.coeffs
			transformWHTRef(in, &whtOut)
			transformWHT(in, coeffs)
			if *coeffs != whtOut {
				t.Fatal("transformWHT differs from the reference")
			}
		}
	}
}

// TestReconGlueGuardPages runs loadEdges and storeMBRows on buffers that
// sit against inaccessible pages (cache rows sized exactly).
func TestReconGlueGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for _, atEnd := range []bool{false, true} {
		bm := guardedBuf(t, yuvSize, atEnd)
		tm := guardedBuf(t, int(unsafe.Sizeof(topSamples{})), atEnd)
		b := (*[yuvSize]byte)(unsafe.Pointer(&bm[0]))
		top := (*topSamples)(unsafe.Pointer(&tm[0]))
		for i := 0; i < 2000; i++ {
			fillPixels(r, bm)
			fillPixels(r, tm)
			want := *b
			loadEdgesRef(&want, top, 0x01020304, uint32(i&15))
			loadEdges(b, top, 0x01020304, uint32(i&15))
			if !diffBuffers(t, "loadEdges", b, &want) {
				t.FailNow()
			}
		}
		for _, g := range [][2]int{{16, 8}, {17, 9}, {160, 80}} {
			ys, uvs := g[0], g[1]
			cy := guardedBuf(t, 15*ys+16, atEnd)
			cu := guardedBuf(t, 7*uvs+8, atEnd)
			cv := guardedBuf(t, 7*uvs+8, atEnd)
			fillPixels(r, bm)
			storeMBRows(b, top, cy, cu, cv, ys, uvs)
			for j := 0; j < 16; j++ {
				if !bytes.Equal(cy[j*ys:][:16], b[yOff+j*bps:][:16]) {
					t.Fatalf("luma row %d", j)
				}
			}
			for j := 0; j < 8; j++ {
				if !bytes.Equal(cu[j*uvs:][:8], b[uOff+j*bps:][:8]) || !bytes.Equal(cv[j*uvs:][:8], b[vOff+j*bps:][:8]) {
					t.Fatalf("chroma row %d", j)
				}
			}
			if top.y != *(*[16]byte)(b[yOff+15*bps:]) || top.u != *(*[8]byte)(b[uOff+7*bps:]) || top.v != *(*[8]byte)(b[vOff+7*bps:]) {
				t.Fatal("top samples")
			}
		}
	}
}
