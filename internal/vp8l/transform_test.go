package vp8l

import (
	"math/rand"
	"slices"
	"testing"
)

// predictLeftRef is predictor mode 1 pixel by pixel.
func predictLeftRef(in, out []uint32, x0, x1 int) {
	l := out[x0-1]
	for x := x0; x < x1; x++ {
		l = addPixels(in[x], l)
		out[x] = l
	}
}

func TestPredictLeft(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 2000; iter++ {
		n := 2 + rng.Intn(80)
		x0 := 1 + rng.Intn(n-1)
		x1 := x0 + rng.Intn(n-x0+1)
		in := make([]uint32, n)
		for i := range in {
			in[i] = rng.Uint32()
		}
		want := make([]uint32, n)
		want[x0-1] = rng.Uint32()
		got := slices.Clone(want)
		predictLeftRef(in, want, x0, x1)
		predictLeft(in, got, x0, x1)
		if !slices.Equal(want, got) {
			t.Fatalf("n %d [%d, %d): output differs", n, x0, x1)
		}
		// In place (in aliases out), as inversePredictor does for a
		// predictor after another transform.
		inPlace := slices.Clone(in)
		inPlace[x0-1] = want[x0-1]
		predictLeft(inPlace, inPlace, x0, x1)
		if !slices.Equal(want[x0-1:x1], inPlace[x0-1:x1]) {
			t.Fatalf("n %d [%d, %d): in-place output differs", n, x0, x1)
		}
	}
}

func BenchmarkPredictLeft(b *testing.B) {
	const w = 3000
	in, out := make([]uint32, w), make([]uint32, w)
	for i := range in {
		in[i] = uint32(i) * 0x9e3779b9
	}
	b.Run("ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predictLeftRef(in, out, 1, w)
		}
	})
	b.Run("new", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predictLeft(in, out, 1, w)
		}
	})
}

// emitRef converts ARGB pixels to RGBA bytes one pixel at a time, adding
// green to red and blue first with addGreen.
func emitRef(src []uint32, dst []byte, addGreen bool) {
	for _, c := range src {
		if len(dst) < 4 {
			return
		}
		if addGreen {
			g := c >> 8 & 0xff
			c = c&0xff00ff00 | (c&0x00ff00ff+(g<<16|g))&0x00ff00ff
		}
		dst[0], dst[1], dst[2], dst[3] = byte(c>>16), byte(c>>8), byte(c), byte(c>>24)
		dst = dst[4:]
	}
}

func TestEmit(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 1000; iter++ {
		src := make([]uint32, rng.Intn(40))
		for i := range src {
			src[i] = rng.Uint32()
		}
		// dst may be shorter or longer than the row.
		n := 4 * len(src)
		if iter%3 == 0 {
			n = max(n+rng.Intn(9)-4, 0)
		}
		for _, ag := range []bool{false, true} {
			want := make([]byte, n)
			rng.Read(want)
			got := slices.Clone(want)
			emitRef(src, want, ag)
			if ag {
				emitAddGreen(src, got)
			} else {
				emitRGBA(src, got)
			}
			if !slices.Equal(want, got) {
				t.Fatalf("%d pixels, %d bytes, add green %v: output differs", len(src), n, ag)
			}
		}
	}
}

func BenchmarkEmit(b *testing.B) {
	const w = 3000
	src, dst := make([]uint32, w), make([]byte, 4*w)
	for i := range src {
		src[i] = uint32(i) * 0x9e3779b9
	}
	for _, c := range []struct {
		name string
		f    func([]uint32, []byte)
	}{
		{"rgba/ref", func(s []uint32, d []byte) { emitRef(s, d, false) }},
		{"rgba/new", emitRGBA},
		{"addgreen/ref", func(s []uint32, d []byte) { emitRef(s, d, true) }},
		{"addgreen/new", emitAddGreen},
	} {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(4 * w)
			for i := 0; i < b.N; i++ {
				c.f(src, dst)
			}
		})
	}
}
