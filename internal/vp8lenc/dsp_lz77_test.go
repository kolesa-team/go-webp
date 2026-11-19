package vp8lenc

import (
	"encoding/binary"
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of lz77Reach against lz77ReachGo and of the parse of
// backwardRefsLz77 at every kernel level.

// randOl returns n hash chain entries (distance<<12 | length) whose
// lengths are short (around minLength), long, up to the maximum, or runs
// of decreasing lengths (as a long match gives), with random distances.
func randOl(rng *rand.Rand, n int) []uint32 {
	ol := make([]uint32, n)
	kind := rng.Intn(4)
	for i := 0; i < n; i++ {
		var l int
		switch kind {
		case 0:
			l = rng.Intn(7)
		case 1:
			l = rng.Intn(64)
		case 2:
			l = rng.Intn(4096)
		default:
			if rng.Intn(8) == 0 {
				for l = 1 + rng.Intn(300); l > 0 && i < n; l-- {
					ol[i] = uint32(rng.Intn(1<<20))<<maxLengthBits | uint32(l)
					i++
				}
				i--
				continue
			}
			l = rng.Intn(6)
		}
		ol[i] = uint32(rng.Intn(1<<20))<<maxLengthBits | uint32(l)
	}
	return ol
}

func TestLz77ReachLevels(t *testing.T) {
	defer setLz77ReachMin(4)()
	rng := rand.New(rand.NewSource(20))
	iters := 200000
	if testing.Short() {
		iters = 20000
	}
	forEachLevel(t, func(t *testing.T) {
		for k := 0; k < iters; k++ {
			size := 1 + rng.Intn(600)
			ol := randOl(rng, size)
			i := rng.Intn(size)
			jMax := i + rng.Intn(size-i)
			n := size
			switch rng.Intn(4) {
			case 0:
				n = 1 + rng.Intn(size+8)
			case 1:
				n = jMax + 1 + rng.Intn(5000)
			}
			l := rng.Intn(4096)
			want := lz77ReachGo(ol, i, jMax, n, l)
			if got := lz77Reach(ol, i, jMax, n, l); got != want {
				t.Fatalf("ol %v i %d jMax %d n %d: got %d, want %d", ol[i:jMax+1], i, jMax, n, got, want)
			}
		}
	})
}

// TestBackwardRefsLz77Levels compares the parses of hash chains of random
// images and of the photo at every level with the Go level.
func TestBackwardRefsLz77Levels(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	sc := getScratch(0)
	defer putScratch(sc)
	check := func(argb []uint32, xsize int) {
		for _, q := range []int{0, 50, 100} {
			sc.hc.fill(argb, xsize, len(argb)/xsize, q, q == 0, 0)
			var want []pixOrCopy
			atEveryLevel(func(l cpuinfo.Level) {
				got := slices.Clone(backwardRefsLz77(len(argb), argb, &sc.hc, nil))
				if l == cpuinfo.Go {
					want = got
				} else if !slices.Equal(got, want) {
					t.Fatalf("level %v n %d xsize %d q %d: parses differ", l, len(argb), xsize, q)
				}
			})
		}
	}
	iters := 40
	if testing.Short() {
		iters = 8
	}
	for k := 0; k < iters; k++ {
		xsize := 1 + rng.Intn(300)
		ysize := 1 + rng.Intn(100)
		check(randHashImage(rng, xsize*ysize, xsize), xsize)
	}
	if !testing.Short() {
		w, argb, _, _ := photoParse(t)
		check(argb, w)
	}
}

func FuzzLz77Reach(f *testing.F) {
	f.Add([]byte{9, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0, 30, 0, 0, 0}, uint16(0), uint16(4), uint16(9))
	f.Fuzz(func(t *testing.T, data []byte, i, jMax, n uint16) {
		ol := make([]uint32, len(data)/4)
		for k := range ol {
			ol[k] = binary.LittleEndian.Uint32(data[4*k:])
		}
		if len(ol) == 0 || int(i) >= len(ol) || int(jMax) >= len(ol) || n == 0 {
			return
		}
		defer setLz77ReachMin(4)()
		want := lz77ReachGo(ol, int(i), int(jMax), int(n), 7)
		atEveryLevel(func(l cpuinfo.Level) {
			if got := lz77Reach(ol, int(i), int(jMax), int(n), 7); got != want {
				t.Fatalf("level %v: got %d, want %d", l, got, want)
			}
		})
	})
}

// BenchmarkBackwardRefsLz77 parses the level 0 hash chain of the photo and
// of a graphic (the logo: long matches).
func BenchmarkBackwardRefsLz77(b *testing.B) {
	w, photo, _, _ := photoParse(b)
	logo := downscale(loadTestJPEG(b, "logo.jpg"), 1)
	lw := logo.Rect.Dx()
	la := make([]uint32, lw*logo.Rect.Dy())
	for i := range la {
		p := logo.Pix[(i/lw)*logo.Stride+4*(i%lw):]
		la[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	for _, im := range []struct {
		name string
		argb []uint32
		w    int
	}{{"photo", photo, w}, {"logo", la, lw}} {
		var hc hashChain
		hc.fill(im.argb, im.w, len(im.argb)/im.w, 0, true, 0)
		refs := make([]pixOrCopy, 0, len(im.argb))
		for _, l := range testLevels() {
			b.Run(im.name+"/"+l.String(), func(b *testing.B) {
				defer setLevel(l)()
				for i := 0; i < b.N; i++ {
					refs = backwardRefsLz77(len(im.argb), im.argb, &hc, refs)
				}
			})
		}
	}
}
