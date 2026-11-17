//go:build darwin || linux

package vp8

import (
	"syscall"
	"testing"
)

// guardedBytes returns an n-byte slice that ends right before an
// inaccessible page (atEnd) or starts right after one (!atEnd), so that a
// kernel reading or writing past the end (or before the start) of the
// slice faults instead of silently touching neighbouring memory.
func guardedBytes(t testing.TB, n int, atEnd bool) []byte {
	t.Helper()
	pg := syscall.Getpagesize()
	data := (n + pg - 1) / pg * pg
	if data == 0 {
		data = pg
	}
	mem, err := syscall.Mmap(-1, 0, data+2*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:pg], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[pg+data:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}
	if atEnd {
		return mem[pg+data-n : pg+data : pg+data]
	}
	return mem[pg : pg+n : pg+n]
}
