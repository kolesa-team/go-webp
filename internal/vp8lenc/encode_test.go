// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package vp8lenc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"os"
	"testing"
)

func TestHeaderFields(t *testing.T) {
	for _, ti := range syntheticImages() {
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		out, err := Encode(ti.img, &Config{Quality: 75, Method: 4, NearLossless: 100})
		if err != nil {
			t.Fatalf("%s: %v", ti.name, err)
		}
		if len(out) < 6 || out[0] != vp8lSignature {
			t.Fatalf("%s: bad signature", ti.name)
		}
		v := binary.LittleEndian.Uint32(out[1:5])
		gw := int(v&0x3fff) + 1
		gh := int((v>>14)&0x3fff) + 1
		alpha := (v >> 28) & 1
		version := v >> 29
		if gw != w || gh != h {
			t.Fatalf("%s: header size %dx%d, want %dx%d", ti.name, gw, gh, w, h)
		}
		if version != 0 {
			t.Fatalf("%s: version %d", ti.name, version)
		}
		wantAlpha := uint32(0)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if ti.img.Pix[y*ti.img.Stride+4*x+3] != 255 {
					wantAlpha = 1
				}
			}
		}
		if alpha != wantAlpha {
			t.Fatalf("%s: alpha bit %d, want %d", ti.name, alpha, wantAlpha)
		}
	}
}

func TestDeterminism(t *testing.T) {
	imgs := syntheticImages()
	for _, lv := range []struct{ m, q int }{{0, 0}, {2, 25}, {4, 75}, {5, 90}, {6, 100}} {
		for i, ti := range imgs {
			if i%5 != 0 {
				continue
			}
			cfg := &Config{Quality: float32(lv.q), Method: lv.m, NearLossless: 100}
			a, err := Encode(ti.img, cfg)
			if err != nil {
				t.Fatal(err)
			}
			cfg.ThreadLevel = true
			b, err := Encode(ti.img, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(a, b) {
				t.Fatalf("%s m%d q%d: non-deterministic output", ti.name, lv.m, lv.q)
			}
		}
	}
}

func TestErrors(t *testing.T) {
	if _, err := Encode(nil, nil); err == nil {
		t.Fatal("expected error for nil image")
	}
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 5), image.Rect(0, 0, 5, 0), image.Rect(0, 0, 16384, 1), image.Rect(0, 0, 1, 16384),
	} {
		m := &image.NRGBA{Rect: r, Stride: 4 * r.Dx()}
		if _, err := Encode(m, nil); err == nil {
			t.Fatalf("expected error for %v", r)
		}
	}
	// Pix buffer too small.
	if _, err := Encode(&image.NRGBA{Rect: image.Rect(0, 0, 4, 4), Stride: 16, Pix: make([]byte, 10)}, nil); err == nil {
		t.Fatal("expected error for short Pix")
	}
	if _, err := EncodeAlpha(make([]byte, 3), 2, 2, 2, 4, 100); err == nil {
		t.Fatal("expected error for short plane")
	}
	if _, err := EncodeAlpha(make([]byte, 4), 2, 2, 1, 4, 100); err == nil {
		t.Fatal("expected error for bad stride")
	}
	if _, err := EncodeAlpha(nil, 0, 2, 0, 4, 100); err == nil {
		t.Fatal("expected error for zero width")
	}
}

func TestMaxDimension(t *testing.T) {
	if testing.Short() {
		t.Skip("large image")
	}
	m := image.NewNRGBA(image.Rect(0, 0, maxDim, 2))
	for i := range m.Pix {
		m.Pix[i] = uint8(i * 7)
	}
	out, err := Encode(m, &Config{Method: 0})
	if err != nil {
		t.Fatal(err)
	}
	v := binary.LittleEndian.Uint32(out[1:5])
	if int(v&0x3fff)+1 != maxDim {
		t.Fatal("bad width")
	}
}

func TestAllConfigsNoPanic(t *testing.T) {
	imgs := syntheticImages()
	cfgs := []Config{
		{Quality: 0, Method: 0}, {Quality: 100, Method: 0},
		{Quality: 20, Method: 1, NearLossless: 100}, {Quality: 30, Method: 3, NearLossless: 60},
		{Quality: 75, Method: 4, NearLossless: 100, Exact: true}, {Quality: 90, Method: 5, NearLossless: 100, LowMemory: true},
		{Quality: 100, Method: 6, NearLossless: 100, ThreadLevel: true}, {Quality: 80, Method: 4, NearLossless: 0},
		{Quality: -5, Method: 99, NearLossless: 100, ImageHint: 3, UseDeltaPalette: true},
	}
	for _, ti := range imgs {
		if raceEnabled && ti.img.Rect.Dx()*ti.img.Rect.Dy() > 40000 {
			continue // the large pictures take minutes under the race detector
		}
		for i := range cfgs {
			if raceEnabled && !cfgs[i].ThreadLevel {
				continue // no goroutines: covered by the run without -race
			}
			out, err := Encode(ti.img, &cfgs[i])
			if err != nil {
				t.Fatalf("%s cfg %d: %v", ti.name, i, err)
			}
			if len(out) < 6 {
				t.Fatalf("%s cfg %d: short output", ti.name, i)
			}
		}
		if raceEnabled {
			continue // EncodeAlpha starts no goroutines either
		}
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		plane := make([]byte, w*h)
		for k := range plane {
			plane[k] = ti.img.Pix[(k/w)*ti.img.Stride+4*(k%w)+3]
		}
		for m := 0; m <= 6; m++ {
			if _, err := EncodeAlpha(plane, w, h, w, m, 100); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestEncodeAlphaMatchesGreenImage(t *testing.T) {
	// The alpha stream is the image stream (header stripped) of an image
	// whose green channel holds the alpha values; it must be deterministic.
	w, h := 37, 21
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = uint8(i * 13 % 7 * 40)
	}
	a, err := EncodeAlpha(plane, w, h, w, 4, 50)
	if err != nil {
		t.Fatal(err)
	}
	b, err := EncodeAlpha(plane, w, h, w, 4, 50)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) || len(a) == 0 {
		t.Fatal("alpha encoding not deterministic")
	}
}

func TestPrefixEncode(t *testing.T) {
	for v := 1; v < 1<<20; v += 1 + v/64 {
		code, nb, extra := prefixEncode(v)
		// Decode like the VP8L spec.
		var got int
		if code < 4 {
			got = code + 1
		} else {
			eb := (code - 2) >> 1
			off := (2 + code&1) << uint(eb)
			got = off + extra + 1
			if eb != nb {
				t.Fatalf("v=%d: extra bits %d != %d", v, nb, eb)
			}
		}
		if got != v {
			t.Fatalf("prefixEncode(%d) roundtrip = %d", v, got)
		}
		if prefixCode(v) != code {
			t.Fatalf("prefixCode mismatch for %d", v)
		}
	}
}

func TestDistanceToPlaneCode(t *testing.T) {
	for _, xsize := range []int{1, 2, 5, 8, 9, 16, 100, 1000} {
		for dist := 1; dist < 20*xsize+50; dist++ {
			code := distanceToPlaneCode(xsize, dist)
			var got int
			if code > 120 {
				got = code - 120
			} else {
				p := int(codeToPlane[code-1])
				got = (p>>4)*xsize + 8 - p&0xf
				if got < 1 {
					got = 1
				}
			}
			if got != dist {
				t.Fatalf("xsize %d dist %d: code %d decodes to %d", xsize, dist, code, got)
			}
		}
	}
}

func TestHuffmanLengthLimit(t *testing.T) {
	// Fibonacci-like counts force deep trees.
	hist := make([]uint32, 40)
	a, b := uint32(1), uint32(1)
	for i := range hist {
		hist[i] = a
		a, b = b, a+b
		if b > 1<<30 {
			b = 1 << 30
		}
	}
	var s huffScratch
	lens := make([]uint8, len(hist))
	codes := make([]uint16, len(hist))
	c := huffCode{lens, codes}
	s.generateOptimalTree(hist, 15, lens)
	kraft := 0.0
	for _, l := range lens {
		if l == 0 || l > 15 {
			t.Fatalf("bad length %d", l)
		}
		kraft += 1 / float64(uint(1)<<l)
	}
	if kraft > 1.0000001 {
		t.Fatalf("kraft sum %v > 1", kraft)
	}
	convertBitDepthsToSymbols(&c)
}

func loadTestJPEG(tb testing.TB, name string) *image.NRGBA {
	f, err := os.Open("../../test_data/images/" + name)
	if err != nil {
		tb.Skip("test image not available:", err)
	}
	defer f.Close()
	m, _, err := image.Decode(f)
	if err != nil {
		tb.Fatal(err)
	}
	n := image.NewNRGBA(m.Bounds())
	draw.Draw(n, n.Rect, m, m.Bounds().Min, draw.Src)
	return n
}

// downscale halves the image n times (box filter).
func downscale(m *image.NRGBA, n int) *image.NRGBA {
	for ; n > 0; n-- {
		w, h := m.Rect.Dx()/2, m.Rect.Dy()/2
		o := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var s [4]int
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						p := m.Pix[(2*y+dy)*m.Stride+4*(2*x+dx):]
						for k := 0; k < 4; k++ {
							s[k] += int(p[k])
						}
					}
				}
				o.SetNRGBA(x, y, color.NRGBA{uint8(s[0] / 4), uint8(s[1] / 4), uint8(s[2] / 4), uint8(s[3] / 4)})
			}
		}
		m = o
	}
	return m
}

func BenchmarkEncode(b *testing.B) {
	photo := downscale(loadTestJPEG(b, "source.jpg"), 1)
	logo := downscale(loadTestJPEG(b, "logo.jpg"), 1)
	levels := []struct{ level, m, q int }{{0, 0, 0}, {1, 1, 20}, {3, 3, 30}, {6, 4, 75}, {9, 6, 100}}
	for _, im := range []struct {
		name string
		img  *image.NRGBA
	}{{"photo480x640", photo}, {"logo875x348", logo}} {
		for _, lv := range levels {
			if lv.level == 9 && testing.Short() {
				continue
			}
			b.Run(fmt.Sprintf("%s/z%d", im.name, lv.level), func(b *testing.B) {
				cfg := &Config{Quality: float32(lv.q), Method: lv.m, NearLossless: 100}
				b.ReportAllocs()
				var size int
				for i := 0; i < b.N; i++ {
					out, err := Encode(im.img, cfg)
					if err != nil {
						b.Fatal(err)
					}
					size = len(out)
				}
				b.ReportMetric(float64(size), "bytes")
			})
		}
	}
}

func BenchmarkEncodeAlpha(b *testing.B) {
	logo := downscale(loadTestJPEG(b, "logo.jpg"), 1)
	w, h := logo.Rect.Dx(), logo.Rect.Dy()
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = logo.Pix[(i/w)*logo.Stride+4*(i%w)+1] &^ 0x1f
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := EncodeAlpha(plane, w, h, w, 4, 100); err != nil {
			b.Fatal(err)
		}
	}
}
