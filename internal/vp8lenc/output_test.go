package vp8lenc

import (
	"bytes"
	"math"
	"math/rand"
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

// TestRefsCapEstimate checks initRefsCap against its definition (on the
// sampled rows) and that applyPalette returns the estimate of the bundled
// image on all its rows.
func TestRefsCapEstimate(t *testing.T) {
	newPixel := func(argb []uint32, w, i int) bool {
		return i == 0 || argb[i] != argb[i-1] && (i < w || argb[i] != argb[i-w])
	}
	ref := func(argb []uint32, w, h int) int {
		n := w * h
		if n <= 4096 {
			return n
		}
		est := 0
		for i := 0; i < n; i++ {
			if newPixel(argb, w, i) {
				est++
			}
		}
		return min(n, est+est/2+4096)
	}
	sampled := func(argb []uint32, w, h int) int {
		n := w * h
		if n <= 4096 || h < rleSampleMinRows {
			return ref(argb, w, h)
		}
		est, rows := 0, 0
		for y := rleSampleStep; y < h; y += rleSampleStep {
			for i := y * w; i < (y+1)*w; i++ {
				if newPixel(argb, w, i) {
					est++
				}
			}
			rows++
		}
		est = est * (h - 1) / rows
		for i := 0; i < w; i++ {
			if newPixel(argb, w, i) {
				est++
			}
		}
		return min(n, est+est/2+4096)
	}
	rng := rand.New(rand.NewSource(9))
	for _, sz := range [][2]int{{1, 5000}, {5000, 1}, {64, 64}, {65, 65}, {333, 101}, {7, 7}} {
		w, h := sz[0], sz[1]
		for _, colors := range []int{2, 4, 16, 17, 200} {
			pal := make([]uint32, colors)
			for i := range pal {
				pal[i] = rng.Uint32()
			}
			argb := make([]uint32, w*h)
			for i := range argb {
				switch {
				case i > 0 && rng.Intn(2) == 0:
					argb[i] = argb[i-1]
				case i >= w && rng.Intn(2) == 0:
					argb[i] = argb[i-w]
				default:
					argb[i] = pal[rng.Intn(colors)]
				}
			}
			if got, want := initRefsCap(argb, w, h), sampled(argb, w, h); got != want {
				t.Fatalf("%dx%d: initRefsCap %d, want %d", w, h, got, want)
			}
			if got, all := initRefsCap(argb, w, h), ref(argb, w, h); got < all-all/10 || got > all+all/10 {
				t.Errorf("%dx%d: initRefsCap %d, %d from all the rows", w, h, got, all)
			}
			palette := collectPalette(argb, nil)
			pw := subSampleSize(w, paletteXBits(len(palette)))
			for _, inPlace := range []bool{false, true} {
				src := slices.Clone(argb)
				dst := make([]uint32, pw*h)
				if inPlace {
					dst = src[:pw*h]
				}
				got := applyPalette(src, w, h, palette, dst, make([]uint8, w))
				if want := ref(dst, pw, h); got != want {
					t.Fatalf("%dx%d %d colors in place %v: applyPalette estimate %d, want %d", w, h, colors, inPlace, got, want)
				}
			}
		}
	}
}

// TestColorSetRows checks that collecting the colors row by row (as the
// conversion does) gives the palette of the whole image.
func TestColorSetRows(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, colors := range []int{1, 2, 17, 255, 256, 257, 1000} {
		for _, w := range []int{1, 3, 64} {
			argb := make([]uint32, w*50)
			for i := range argb {
				if i > 0 && rng.Intn(3) == 0 {
					argb[i] = argb[i-1] // runs across row boundaries
				} else {
					argb[i] = uint32(rng.Intn(colors)) * 0x01030507
				}
			}
			want := collectPalette(argb, nil)
			var s colorSet
			for y := 0; y < 50; y++ {
				s.add(argb[y*w : (y+1)*w])
			}
			got := s.palette(nil)
			if (got == nil) != (want == nil) || !slices.Equal(got, want) {
				t.Fatalf("%d colors, width %d: row-wise palette %d entries, whole %d", colors, w, len(got), len(want))
			}
		}
	}
}

// TestImageInfo checks that the conversion's image info (palette and
// transparency) gives the same streams as encodeARGB's own scans.
func TestImageInfo(t *testing.T) {
	for _, ti := range syntheticImages() {
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		for _, exact := range []bool{false, true} {
			cfg := &Config{Quality: 75, Method: 4, Exact: exact}
			want, err := Encode(ti.img, cfg)
			if err != nil {
				t.Fatal(err)
			}
			argb := make([]uint32, w*h)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					s := ti.img.Pix[y*ti.img.Stride+4*x:]
					if s[3] == 0 && !exact {
						continue
					}
					argb[y*w+x] = uint32(s[3])<<24 | uint32(s[0])<<16 | uint32(s[1])<<8 | uint32(s[2])
				}
			}
			got, _ := encodeARGB(argb, w, h, cfg, true, 5, nil, nil, nil)
			if !bytes.Equal(got[5:], want[5:]) {
				t.Fatalf("%s exact=%v: stream differs without image info", ti.name, exact)
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
		if r := mainImageReserve(bits); r < 4096 || int64(r) > 1<<31 {
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
