//go:build darwin || linux

package vp8lenc

import (
	"bytes"
	"math/rand"
	"os"
	"slices"
	"syscall"
	"testing"
	"unsafe"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
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

// guardedBytes returns n bytes of fresh anonymous memory that end exactly
// at an inaccessible page (atEnd), or start right after one.
func guardedBytes(tb testing.TB, n int, atEnd bool) []byte {
	tb.Helper()
	ps := os.Getpagesize()
	pages := max((n+ps-1)/ps, 1)
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
		data = data[len(data)-n:]
	}
	return data[:n:n]
}

// guardedRefs returns a guarded copy of refs (see guardedU32).
func guardedRefs(tb testing.TB, refs []pixOrCopy, atEnd bool) []pixOrCopy {
	u := guardedU32(tb, 2*len(refs), atEnd)
	g := unsafe.Slice((*pixOrCopy)(unsafe.Pointer(&u[0])), len(refs))
	copy(g, refs)
	return g
}

func TestNrgbaRowGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100} {
			for _, atEnd := range []bool{false, true} {
				for _, exact := range []bool{false, true} {
					src := guardedBytes(t, 4*n, atEnd)
					copy(src, randNRGBA(rng, n, 2))
					dst := guardedU32(t, n, atEnd)
					want := make([]uint32, n)
					wa, wt := nrgbaRowGo(want, slices.Clone(src), exact)
					ga, gt := nrgbaRow(dst, src, exact)
					if ga != wa || gt != wt || !slices.Equal(dst, want) {
						t.Fatalf("n %d atEnd %v exact %v: got %v %v %08x, want %v %v %08x", n, atEnd, exact, ga, gt, dst, wa, wt, want)
					}
				}
			}
		}
	})
}

func TestAddRefsGuard(t *testing.T) {
	defer setAddRefsMin(0)()
	rng := rand.New(rand.NewSource(5))
	forEachLevel(t, func(t *testing.T) {
		for _, atEnd := range []bool{false, true} {
			for _, raw := range []bool{false, true} {
				w := 1 + rng.Intn(100)
				refs := guardedRefs(t, randImageRefs(rng, w, 1+rng.Intn(50), 5), atEnd)
				var sg, sa histoSet
				sg.init(1, 5)
				sa.init(1, 5)
				if raw {
					sg.addRefsRawDistGo(&sg.h[0], slices.Clone(refs))
					sa.addRefsRawDist(&sa.h[0], refs)
				} else {
					sg.addRefsGo(&sg.h[0], slices.Clone(refs), w)
					sa.addRefs(&sa.h[0], refs, w)
				}
				if !slices.Equal(sa.h[0].d, sg.h[0].d) {
					t.Fatalf("atEnd %v raw %v: histograms differ", atEnd, raw)
				}
			}
		}
	})
}

// TestAddTileRefsGuard adds guarded references to tile histograms in
// guarded memory.
func TestAddTileRefsGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for it := 0; it < 8; it++ {
		atEnd := it%2 == 0
		w, h, hb := 1+rng.Intn(100), 1+rng.Intn(50), 2+rng.Intn(3)
		refs := guardedRefs(t, randImageRefs(rng, w, h, 3), atEnd)
		hx := subSampleSize(w, hb)
		nt := hx * subSampleSize(h, hb)
		var sg histoSet
		sg.init(nt, 3)
		sg.addTileRefsGo(slices.Clone(refs), w, hb, hx)
		atEveryLevel(func(l cpuinfo.Level) {
			var s histoSet
			s.init(nt, 3)
			s.data = guardedU32(t, len(s.data), atEnd)
			for i := range s.h {
				s.h[i].d = s.data[i*s.stride : (i+1)*s.stride : (i+1)*s.stride]
			}
			s.addTileRefs(refs, w, hb, hx)
			if !slices.Equal(s.data, sg.data) {
				t.Fatalf("iter %d level %v: histograms differ", it, l)
			}
		})
	}
}

// TestWriteImageDataGuard writes into a buffer whose capacity ends at an
// inaccessible page, from guarded references and entropy image.
func TestWriteImageDataGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for it := 0; it < 40; it++ {
		atEnd := it%2 == 0
		w, h := 1+rng.Intn(200), 1+rng.Intn(60)
		refs := randImageRefs(rng, w, h, 2)
		codes := randCodes(rng, 4, literalLen(2))
		hb := 2 + rng.Intn(4)
		sym := randSymbols(rng, w, h, hb, 4)
		var bg bitWriter
		writeImageDataGo(&bg, w, refs, hb, sym, codes)
		gsym := unsafe.Slice((*uint16)(unsafe.Pointer(&guardedBytes(t, 2*len(sym), atEnd)[0])), len(sym))
		copy(gsym, sym)
		grefs := guardedRefs(t, refs, atEnd)
		atEveryLevel(func(l cpuinfo.Level) {
			// The buffer can hold the whole output, or most of it (the
			// kernel stops at its end and the wrapper grows it).
			for _, extra := range []int{16, 3, -20} {
				n := max(len(bg.buf)+extra, 1)
				bw := bitWriter{buf: guardedBytes(t, n, true)[:0]}
				writeImageData(&bw, w, grefs, hb, gsym, codes)
				if !bytes.Equal(bw.buf, bg.buf) || bw.acc != bg.acc || bw.nbits != bg.nbits {
					t.Fatalf("iter %d level %v extra %d: writers differ", it, l, extra)
				}
			}
		})
	}
}
