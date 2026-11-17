package vp8lenc

import (
	"bytes"
	"math"
	"slices"
	"testing"
)

// TestPaletteInPlace checks that applying the palette in place (mutable
// source) gives the same stream as applying it to a separate buffer.
func TestPaletteInPlace(t *testing.T) {
	for _, ti := range syntheticImages() {
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		argb := make([]uint32, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				s := ti.img.Pix[y*ti.img.Stride+4*x:]
				argb[y*w+x] = uint32(s[3])<<24 | uint32(s[0])<<16 | uint32(s[1])<<8 | uint32(s[2])
			}
		}
		pal := collectPalette(argb, nil)
		if pal == nil {
			continue
		}
		for _, method := range []int{0, 4} {
			p := params{quality: 75, method: method, lowEffort: method == 0, useCache: true}
			p.histoBits = getHistoBits(method, true, w, h)
			p.transBits = getTransformBits(method, p.histoBits)
			p.paletteSorts[sortDefault] = pal
			p.paletteSorts[sortMinimizeDelta] = paletteSortMinimizeDeltas(pal, nil)
			for _, cc := range []crunchConfig{{modePalette, sortDefault}, {modePalette, sortMinimizeDelta}, {modePaletteAndSpatial, sortMinimizeDelta}} {
				src := slices.Clone(argb)
				want := encodeStream(src, false, w, h, &p, cc, 3, new(scratch), nil)
				if !slices.Equal(src, argb) {
					t.Fatalf("%s: immutable source modified", ti.name)
				}
				got := encodeStream(src, true, w, h, &p, cc, 3, new(scratch), nil)
				if !bytes.Equal(got, want) {
					t.Fatalf("%s %dx%d m%d %v: in-place palette changes the stream", ti.name, w, h, method, cc)
				}
			}
		}
	}
}

// TestOutputCapacity checks the output contract of the encoders (spare
// capacity for a RIFF padding byte), including streams larger than their
// size estimate.
func TestOutputCapacity(t *testing.T) {
	for _, ti := range syntheticImages() {
		for _, cfg := range []Config{{Quality: 0, Method: 0}, {Quality: 75, Method: 4}, {Quality: 90, Method: 5}, {Quality: 100, Method: 6}} {
			if cfg.Method == 6 && ti.img.Rect.Dx()*ti.img.Rect.Dy() > 1000 || cfg.Method == 5 && testing.Short() {
				continue
			}
			for _, headroom := range []int{0, 7} {
				out, err := EncodeWithHeadroom(ti.img, &cfg, headroom)
				if err != nil {
					t.Fatal(err)
				}
				if cap(out) <= len(out) {
					t.Fatalf("%s m%d: no spare capacity (len %d cap %d)", ti.name, cfg.Method, len(out), cap(out))
				}
				if headroom == 0 {
					continue
				}
				ref, _ := Encode(ti.img, &cfg)
				if !bytes.Equal(out[headroom:], ref) {
					t.Fatalf("%s m%d: headroom changes the stream", ti.name, cfg.Method)
				}
			}
		}
	}
	for _, bits := range []float64{0, 1, 1e6, 1e15, math.Inf(1), math.NaN()} {
		if r := mainImageReserve(bits); r < 4096 || r > 1<<31 {
			t.Fatalf("mainImageReserve(%v) = %d", bits, r)
		}
	}
	var bw bitWriter
	bw.reserve(10)
	if cap(bw.buf) < 10 {
		t.Fatal("reserve did not grow the buffer")
	}
	bw.putBits(0x5a, 8)
	c := cap(bw.buf)
	bw.reserve(c - len(bw.buf))
	if cap(bw.buf) != c {
		t.Fatal("reserve grew a large enough buffer")
	}
}
