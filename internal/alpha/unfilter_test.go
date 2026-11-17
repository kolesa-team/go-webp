package alpha

import (
	"bytes"
	"math/rand"
	"testing"
)

// Differential tests of the unfilter kernels (unfilter_*.go) against their
// Go references. On arm64 without the purego tag they check the NEON
// assembly; elsewhere both sides are the same Go code.

const guardByte = 0xa5

// guardedRow returns n bytes surrounded by 16 guard bytes on each side, and
// the n bytes.
func guardedRow(n int) (all, mid []byte) {
	all = bytes.Repeat([]byte{guardByte}, n+32)
	return all, all[16 : 16+n : 16+n]
}

func checkGuardBytes(t *testing.T, all []byte, what string) {
	t.Helper()
	n := len(all) - 32
	for i, b := range all {
		if (i < 16 || i >= 16+n) && b != guardByte {
			t.Fatalf("%s: wrote outside the row (byte %d)", what, i-16)
		}
	}
}

// unfilterCheck compares both kernels with their references on one row,
// out of place and in place.
func unfilterCheck(t *testing.T, pred byte, prev, in []byte) {
	t.Helper()
	n := len(in)
	want := make([]byte, n)
	unfilterHorizontalGo(pred, in, want)
	all, got := guardedRow(n)
	unfilterHorizontal(pred, in, got)
	if !bytes.Equal(got, want) {
		t.Fatalf("horizontal, %d bytes, pred %d: output differs", n, pred)
	}
	checkGuardBytes(t, all, "horizontal")
	inPlace := bytes.Clone(in)
	unfilterHorizontal(pred, inPlace, inPlace)
	if !bytes.Equal(inPlace, want) {
		t.Fatalf("horizontal, %d bytes: in-place output differs", n)
	}

	unfilterVerticalGo(prev, in, want)
	all, got = guardedRow(n)
	unfilterVertical(prev, in, got)
	if !bytes.Equal(got, want) {
		t.Fatalf("vertical, %d bytes: output differs", n)
	}
	checkGuardBytes(t, all, "vertical")
	inPlace = bytes.Clone(in)
	unfilterVertical(prev, inPlace, inPlace)
	if !bytes.Equal(inPlace, want) {
		t.Fatalf("vertical, %d bytes: in-place output differs", n)
	}
}

func TestUnfilterKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 50000; iter++ {
		n := rng.Intn(100)
		if iter%100 == 0 {
			n = rng.Intn(5000)
		}
		in := make([]byte, n)
		prev := make([]byte, n+rng.Intn(3)) // prev may be longer
		rng.Read(in)
		rng.Read(prev)
		unfilterCheck(t, byte(rng.Intn(256)), prev, in)
	}
}

// TestUnfilterRowKernels checks UnfilterRow, which picks the kernels, on
// whole planes against a per-pixel model of libwebp's unfilters.
func TestUnfilterRowKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 300; iter++ {
		w, h := 1+rng.Intn(300), 1+rng.Intn(5)
		plane := make([]byte, w*h)
		rng.Read(plane)
		for _, f := range []int{FilterHorizontal, FilterVertical} {
			want := bytes.Clone(plane)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					// The first row is left-predicted from 0, the first
					// column of the other rows predicted from above.
					var pred byte
					switch {
					case x == 0 && y == 0:
					case x > 0 && (y == 0 || f == FilterHorizontal):
						pred = want[y*w+x-1]
					default:
						pred = want[(y-1)*w+x]
					}
					want[y*w+x] += pred
				}
			}
			got := bytes.Clone(plane)
			Unfilter(f, got, w, h, w)
			if !bytes.Equal(got, want) {
				t.Fatalf("filter %d, %dx%d: output differs", f, w, h)
			}
		}
	}
}

func FuzzUnfilterKernels(f *testing.F) {
	f.Add([]byte("0123456789abcdefghijklmnopqrstuvwxyz"), byte(7))
	f.Fuzz(func(t *testing.T, data []byte, pred byte) {
		n := len(data) / 2
		unfilterCheck(t, pred, data[n:], data[:n])
	})
}

// BenchmarkUnfilterKernels times the kernels against their Go references
// on rows of 3000 bytes.
func BenchmarkUnfilterKernels(b *testing.B) {
	const n = 3000
	rng := rand.New(rand.NewSource(1))
	in, prev, out := make([]byte, n), make([]byte, n), make([]byte, n)
	rng.Read(in)
	rng.Read(prev)
	for _, k := range []struct {
		name string
		f    func()
	}{
		{"horizontal/go", func() { unfilterHorizontalGo(1, in, out) }},
		{"horizontal/asm", func() { unfilterHorizontal(1, in, out) }},
		{"vertical/go", func() { unfilterVerticalGo(prev, in, out) }},
		{"vertical/asm", func() { unfilterVertical(prev, in, out) }},
		{"gradient/go", func() { gradientUnfilter(prev, in, out) }},
	} {
		b.Run(k.name, func(b *testing.B) {
			b.SetBytes(n)
			for i := 0; i < b.N; i++ {
				k.f()
			}
		})
	}
}
