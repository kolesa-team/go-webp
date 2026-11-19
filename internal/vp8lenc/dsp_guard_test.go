//go:build darwin || linux

package vp8lenc

import (
	"bytes"
	"math"
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

func TestSubtractGreenGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 33, 100} {
			for _, atEnd := range []bool{false, true} {
				p := guardedU32(t, n, atEnd)
				copy(p, randPixels(rng, n))
				want := slices.Clone(p)
				subtractGreenGo(want)
				subtractGreen(p)
				if !slices.Equal(p, want) {
					t.Fatalf("n %d atEnd %v: got %08x, want %08x", n, atEnd, p, want)
				}
			}
		}
	})
}

func TestPopulationCostGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{17, 18, 31, 32, 33, 40, 256, 280, 281, 1304} {
			for _, atEnd := range []bool{false, true} {
				p := guardedU32(t, n, atEnd)
				copy(p, randStreaky(rng, n))
				want := populationCostGo(slices.Clone(p))
				if got := populationCost(p); got != want {
					t.Fatalf("n %d atEnd %v: got %v, want %v", n, atEnd, got, want)
				}
			}
		}
	})
}

func TestAccEntropyInitGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	words := int(unsafe.Sizeof(accEntropy{}) / 4)
	forEachLevel(t, func(t *testing.T) {
		for _, atEnd := range []bool{false, true} {
			hu := guardedU32(t, 256, atEnd)
			h := (*[256]int32)(unsafe.Pointer(&hu[0]))
			randHisto(rng, h[:])
			a := (*accEntropy)(unsafe.Pointer(&guardedU32(t, words, atEnd)[0]))
			var want accEntropy
			want.initGo(h)
			a.init(h)
			if d := accDiff(a, &want); d != "" {
				t.Fatalf("atEnd %v: %s", atEnd, d)
			}
		}
	})
}

func TestCCHistoGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 31, 33, 100} {
			for _, atEnd := range []bool{false, true} {
				b0, g0, r0, c0 := randCC(rng, n, 0)
				gi8 := func(s []int8) []int8 {
					u := guardedBytes(t, n, atEnd)
					g := unsafe.Slice((*int8)(unsafe.Pointer(&u[0])), n)
					copy(g, s)
					return g
				}
				b, g, r := gi8(b0), gi8(g0), gi8(r0)
				c := unsafe.Slice((*int32)(unsafe.Pointer(&guardedU32(t, n, atEnd)[0])), n)
				copy(c, c0)
				hu := guardedU32(t, 256, atEnd)
				histo := (*[256]int32)(unsafe.Pointer(&hu[0]))
				var want [256]int32
				ccHistoBlueGo(&want, b0, g0, r0, c0, -7, 33)
				ccHistoBlue(histo, b, g, r, c, -7, 33)
				if *histo != want {
					t.Fatalf("n %d atEnd %v: blue histograms differ", n, atEnd)
				}
				*histo, want = [256]int32{}, [256]int32{}
				ccHistoRedGo(&want, r0, g0, c0, 101)
				ccHistoRed(histo, r, g, c, 101)
				if *histo != want {
					t.Fatalf("n %d atEnd %v: red histograms differ", n, atEnd)
				}
			}
		}
	})
}

func TestLz77ReachGuard(t *testing.T) {
	defer setLz77ReachMin(4)()
	rng := rand.New(rand.NewSource(13))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{5, 6, 7, 8, 9, 13, 100} {
			for _, atEnd := range []bool{false, true} {
				ol := guardedU32(t, n, atEnd)
				copy(ol, randOl(rng, n))
				want := lz77ReachGo(slices.Clone(ol), 0, n-1, n, 3)
				if got := lz77Reach(ol, 0, n-1, n, 3); got != want {
					t.Fatalf("n %d atEnd %v: got %d, want %d", n, atEnd, got, want)
				}
			}
		}
	})
}

// guardedU64 returns n uint64 of guarded memory (see guardedU32).
func guardedU64(tb testing.TB, n int, atEnd bool) []uint64 {
	u := guardedU32(tb, 2*n, atEnd)
	return unsafe.Slice((*uint64)(unsafe.Pointer(&u[0])), n)
}

// guardedPixels returns a guarded copy of p, with runs, repeated rows and
// matches reaching the last pixel (the match length loads stop at size-1).
func guardedPixels(tb testing.TB, rng *rand.Rand, n, xsize int, atEnd bool) []uint32 {
	p := guardedU32(tb, n, atEnd)
	copy(p, randHashImage(rng, n, xsize))
	if rng.Intn(2) == 0 {
		// A run up to the end: maximal matches at distances 1 and xsize.
		c := p[n-1]
		for i := n - 1 - rng.Intn(n); i < n; i++ {
			p[i] = c
		}
	}
	return p
}

// TestHashChainRunGuard runs the search on guarded pixels, chain and ol
// (and marks, as fillStripes passes them).
func TestHashChainRunGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{3, 4, 5, 6, 7, 8, 9, 13, 17, 64, 65, 100, 1000, 5000} {
			for _, atEnd := range []bool{false, true} {
				for _, q := range []int{0, 50, 100} {
					low := rng.Intn(3) == 0
					xsize := 1 + rng.Intn(min(n, 40))
					argb := guardedPixels(t, rng, n, xsize, atEnd)
					window := windowSizeForHashChain(q, xsize)
					search := func(argb []uint32, ol []uint32, marks []uint64, useGo bool) (int, int) {
						s := hcSearcher{argb: argb, ol: ol, chain: ol, xsize: xsize, size: n,
							iterMax: maxItersForQuality(q), window: window, lowEffort: low}
						buildChain(argb, ol)
						ol[0], ol[n-1] = 0, 0
						if useGo {
							return s.runGo(n-2, 0, 1, marks)
						}
						return s.run(n-2, 0, 1, marks)
					}
					want := make([]uint32, n)
					wmarks := make([]uint64, (n+63)/64)
					wl, wn := search(slices.Clone(argb), want, wmarks, true)
					ol := guardedU32(t, n, atEnd)
					marks := guardedU64(t, (n+63)/64, atEnd)
					clear(marks)
					gl, gn := search(argb, ol, marks, false)
					if gl != wl || gn != wn || !slices.Equal(ol, want) || !slices.Equal(marks, wmarks) {
						t.Fatalf("n %d atEnd %v q %d low %v xsize %d: searches differ", n, atEnd, q, low, xsize)
					}
				}
			}
		}
	})
}

// TestBackwardRefsRleGuard parses guarded pixels into a guarded reference
// buffer (which fills up, so the parse also continues in a grown one).
func TestBackwardRefsRleGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 100, 1000, 5000} {
			for _, atEnd := range []bool{false, true} {
				xsize := 1 + rng.Intn(min(n, 40))
				argb := guardedPixels(t, rng, n, xsize, atEnd)
				for _, c := range []int{1, 7, n} {
					want := backwardRefsRleGo(xsize, n, slices.Clone(argb), make([]pixOrCopy, 0, c))
					refs := guardedRefs(t, make([]pixOrCopy, c), atEnd)[:0]
					got := backwardRefsRle(xsize, n, argb, refs)
					if !slices.Equal(got, want) {
						t.Fatalf("n %d atEnd %v xsize %d cap %d: parses differ", n, atEnd, xsize, c)
					}
				}
			}
		}
	})
}

// TestBestCacheBitsGuard counts guarded references of guarded pixels, the
// last reference ending at the last pixel.
func TestBestCacheBitsGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	sc := getScratch(0)
	defer putScratch(sc)
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{1, 2, 5, 17, 100, 1000, 5000} {
			for _, atEnd := range []bool{false, true} {
				argb := guardedPixels(t, rng, n, 1+rng.Intn(min(n, 40)), atEnd)
				if rng.Intn(2) == 0 {
					for k := range argb {
						argb[k] &= 0x01030507
					}
				}
				refs := guardedRefs(t, randRefs(rng, n, rng.Intn(2) == 0), atEnd)
				for _, maxBits := range []int{1, 4, maxColorCacheBits} {
					want := bestCacheBitsGo(slices.Clone(argb), 75, slices.Clone(refs), maxBits, sc)
					if got := bestCacheBits(argb, 75, refs, maxBits, sc); got != want {
						t.Fatalf("n %d atEnd %v maxBits %d: got %d, want %d", n, atEnd, maxBits, got, want)
					}
				}
			}
		}
	})
}

// TestCombinedEntropyGuard runs combinedPopulationCost on two guarded
// operands and combinedShannonEntropy on guarded histograms.
func TestCombinedEntropyGuard(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	words := int(unsafe.Sizeof(accEntropy{}) / 4)
	forEachLevel(t, func(t *testing.T) {
		for _, n := range []int{17, 18, 31, 32, 33, 40, 256, 280, 281, 1304, 4000} {
			for _, atEnd := range []bool{false, true} {
				a, b := guardedU32(t, n, atEnd), guardedU32(t, n, atEnd)
				copy(a, randStreaky(rng, n))
				copy(b, randStreaky(rng, n))
				want := combinedPopulationCostGo(slices.Clone(a), slices.Clone(b))
				if got := combinedPopulationCost(a, b); math.Float64bits(got) != math.Float64bits(want) {
					t.Fatalf("n %d atEnd %v: got %v, want %v", n, atEnd, got, want)
				}
			}
		}
		for _, atEnd := range []bool{false, true} {
			xu, yu := guardedU32(t, 256, atEnd), guardedU32(t, 256, atEnd)
			x, y := (*[256]int32)(unsafe.Pointer(&xu[0])), (*[256]int32)(unsafe.Pointer(&yu[0]))
			randHisto(rng, x[:])
			randHisto(rng, y[:])
			acc := (*accEntropy)(unsafe.Pointer(&guardedU32(t, words, atEnd)[0]))
			acc.initGo(y)
			xc, yc, ac := *x, *y, *acc
			want := combinedShannonEntropyGo(&xc, &yc, &ac)
			if got := combinedShannonEntropy(x, y, acc); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("atEnd %v: got %v, want %v", atEnd, got, want)
			}
		}
	})
}
