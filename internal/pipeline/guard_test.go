//go:build (arm64 || amd64) && !purego && (darwin || linux)

package pipeline

import (
	"bytes"
	"math/rand/v2"
	"os"
	"syscall"
	"testing"
)

// guardedBytes returns n bytes between two inaccessible pages, so that any
// access past them faults: they end right before the second page or, when
// atStart is true, start right after the first one. free releases them.
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

// TestYUVToRGBARowGuarded runs the YUV->RGBA kernel with every row (Y, U,
// V, alpha and the output) next to an inaccessible page, after its end and
// before its start.
func TestYUVToRGBARowGuarded(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(31, 32))
		for i := 0; i < 600; i++ {
			w := 1 + r.IntN(200)
			if i < 100 {
				w = 1 + i
			}
			atStart := i%2 == 0
			var frees []func()
			mk := func(n int) []byte {
				b, free := guardedBytes(t, n, atStart)
				frees = append(frees, free)
				for k := range b {
					b[k] = byte(r.IntN(256))
				}
				return b
			}
			ys, us, vs := mk(w), mk(w), mk(w)
			var as []byte
			if i%3 != 0 {
				as = mk(w)
			}
			out := mk(4 * w)
			want := make([]byte, 4*w)
			yuvToRGBARowGo(ys, us, vs, as, want)
			yuvToRGBARow(ys, us, vs, as, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("w=%d alpha=%v: rows differ", w, as != nil)
			}
			for _, f := range frees {
				f()
			}
		}
	})
}
