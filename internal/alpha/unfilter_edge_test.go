package alpha

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestUnfilterKernelsOffsets runs both unfilter kernels on rows whose in,
// out and prev slices start at every offset modulo 16, for every length up
// to 100, with bytes at the 0/255 wrap, and every pred value at one length.
func TestUnfilterKernelsOffsets(t *testing.T) { forEachLevel(t, testUnfilterKernelsOffsets) }

func testUnfilterKernelsOffsets(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	edge := []byte{0, 1, 0x7f, 0x80, 0xfe, 0xff}
	fill := func(b []byte) {
		for i := range b {
			if rng.Intn(2) == 0 {
				b[i] = edge[rng.Intn(len(edge))]
			} else {
				b[i] = byte(rng.Intn(256))
			}
		}
	}
	for off := 0; off < 16; off++ {
		for n := 0; n <= 100; n++ {
			inAll := make([]byte, n+off)
			prevAll := make([]byte, n+(off*7)%16)
			fill(inAll)
			fill(prevAll)
			in := inAll[off:]
			prev := prevAll[(off*7)%16:]
			pred := edge[rng.Intn(len(edge))]

			want := make([]byte, n)
			unfilterHorizontalGo(pred, in, want)
			all := bytes.Repeat([]byte{guardByte}, n+32+off)
			out := all[16+off : 16+off+n : 16+off+n]
			unfilterHorizontal(pred, in, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("horizontal: offset %d, %d bytes: output differs", off, n)
			}
			for i, v := range all {
				if (i < 16+off || i >= 16+off+n) && v != guardByte {
					t.Fatalf("horizontal: offset %d, %d bytes: wrote byte %d", off, n, i-16-off)
				}
			}

			unfilterVerticalGo(prev, in, want)
			all = bytes.Repeat([]byte{guardByte}, n+32+off)
			out = all[16+off : 16+off+n : 16+off+n]
			unfilterVertical(prev, in, out)
			if !bytes.Equal(out, want) {
				t.Fatalf("vertical: offset %d, %d bytes: output differs", off, n)
			}
			for i, v := range all {
				if (i < 16+off || i >= 16+off+n) && v != guardByte {
					t.Fatalf("vertical: offset %d, %d bytes: wrote byte %d", off, n, i-16-off)
				}
			}
		}
	}
	in := make([]byte, 77)
	fill(in)
	for pred := 0; pred < 256; pred++ {
		want := make([]byte, len(in))
		unfilterHorizontalGo(byte(pred), in, want)
		got := make([]byte, len(in))
		unfilterHorizontal(byte(pred), in, got)
		if !bytes.Equal(got, want) {
			t.Fatalf("horizontal: pred %d: output differs", pred)
		}
	}
}
