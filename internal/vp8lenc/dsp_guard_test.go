//go:build darwin || linux

package vp8lenc

import (
	"math/rand"
	"os"
	"slices"
	"syscall"
	"testing"
	"unsafe"
)

// Guard-page tests of the kernel entry points: every buffer handed to a
// kernel is placed right before (or right after) an inaccessible page, so a
// load or store one element outside the slice faults instead of reading
// the slack of a Go allocation. The results are compared with the Go
// reference on ordinary copies of the same data.

// guardedU32 returns n uint32 of fresh anonymous memory that end exactly at
// an inaccessible page (atEnd), or start right after one.
func guardedU32(tb testing.TB, n int, atEnd bool) []uint32 {
	tb.Helper()
	ps := os.Getpagesize()
	pages := max((4*n+ps-1)/ps, 1)
	mem, err := syscall.Mmap(-1, 0, (pages+2)*ps, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		tb.Fatalf("mmap: %v", err)
	}
	tb.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:ps], syscall.PROT_NONE); err != nil {
		tb.Fatalf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[(pages+1)*ps:], syscall.PROT_NONE); err != nil {
		tb.Fatalf("mprotect: %v", err)
	}
	data := mem[ps : (pages+1)*ps]
	if atEnd {
		data = data[len(data)-4*n:]
	}
	if n == 0 {
		return []uint32{}
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&data[0])), n)
}

func TestPredictRowGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100} {
			for _, atEnd := range []bool{false, true} {
				// The rows hold exactly what the kernel may read:
				// cur[x0-1:x1] and up[x0-1:x1+1] with x0 = 1.
				cur := guardedU32(t, n+1, atEnd)
				up := guardedU32(t, n+2, atEnd)
				out := guardedU32(t, n, atEnd)
				copy(cur, randPixels(rng, n+1))
				copy(up, randPixels(rng, n+2))
				want := make([]uint32, n)
				for mode := 0; mode < numPredModes; mode++ {
					predictRowGo(mode, slices.Clone(cur), slices.Clone(up), 1, n+1, want)
					predictRow(mode, cur, up, 1, n+1, out)
					if !slices.Equal(out, want) {
						t.Fatalf("n %d atEnd %v mode %d: got %08x, want %08x", n, atEnd, mode, out, want)
					}
				}
			}
		}
	})
}
