//go:build darwin || linux

package alpha

import (
	"bytes"
	"math/rand"
	"syscall"
	"testing"
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

// TestUnfilterGuardPages runs the unfilter kernels at every SIMD level on
// rows that sit against inaccessible pages on either side: a read or write
// outside the slices crashes the test.
func TestUnfilterGuardPages(t *testing.T) { forEachLevel(t, testUnfilterGuardPages) }

func testUnfilterGuardPages(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, atEnd := range []bool{false, true} {
		for n := 0; n <= 300; n++ {
			in := guardedBytes(t, n, atEnd)
			prev := guardedBytes(t, n, atEnd)
			out := guardedBytes(t, n, atEnd)
			rng.Read(in)
			rng.Read(prev)
			want := make([]byte, n)
			unfilterHorizontalGo(7, in, want)
			unfilterHorizontal(7, in, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("horizontal, %d bytes: output differs", n)
			}
			unfilterVerticalGo(prev, in, want)
			unfilterVertical(prev, in, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("vertical, %d bytes: output differs", n)
			}
		}
	}
}
