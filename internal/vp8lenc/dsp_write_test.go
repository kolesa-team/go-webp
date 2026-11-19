package vp8lenc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Differential tests of nrgbaRow, addRefs, addRefsRawDist and
// writeImageData (dsp_write_amd64.go, dsp_generic.go) against their Go
// references, at every kernel level.

// randNRGBA returns n random NRGBA pixels whose alpha is mostly 255, else
// 0 or any value, as chosen by mode (0: all opaque, 1: one non-opaque
// pixel, 2: a mix).
func randNRGBA(rng *rand.Rand, n, mode int) []byte {
	src := make([]byte, 4*n)
	rng.Read(src)
	for i := 0; i < n; i++ {
		a := byte(255)
		if mode == 2 {
			switch rng.Intn(4) {
			case 0:
				a = 0
			case 1:
				a = byte(rng.Intn(256))
			}
		}
		src[4*i+3] = a
	}
	if mode == 1 && n > 0 {
		src[4*rng.Intn(n)+3] = []byte{0, 1, 254}[rng.Intn(3)]
	}
	return src
}

// nrgbaBoth converts src at every level into a dst placed off elements
// into a buffer of sentinels, and compares with nrgbaRowGo.
func nrgbaBoth(t *testing.T, src []byte, off int, exact bool) {
	t.Helper()
	n := len(src) / 4
	want := make([]uint32, n)
	wa, wt := nrgbaRowGo(want, src, exact)
	buf := make([]uint32, n+off+4)
	for i := range buf {
		buf[i] = 0xdeadbeef
	}
	dst := buf[off : off+n : off+n]
	ga, gt := nrgbaRow(dst, src, exact)
	if ga != wa || gt != wt || !slices.Equal(dst, want) {
		t.Fatalf("n %d off %d exact %v: got %v %v %08x, want %v %v %08x", n, off, exact, ga, gt, dst, wa, wt, want)
	}
	for i, v := range buf {
		if (i < off || i >= off+n) && v != 0xdeadbeef {
			t.Fatalf("n %d off %d: element %d outside dst written", n, off, i-off)
		}
	}
}

func TestNrgbaRowLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	widths := []int{}
	for n := 1; n <= 64; n++ {
		widths = append(widths, n)
	}
	widths = append(widths, 100, 255, 1000, 4097)
	forEachLevel(t, func(t *testing.T) {
		for _, n := range widths {
			for so := 0; so < 16; so++ {
				for _, exact := range []bool{false, true} {
					for mode := 0; mode < 3; mode++ {
						// src starts so bytes into its buffer, dst so%4
						// elements into its own.
						buf := make([]byte, 4*n+so)
						src := buf[so:]
						copy(src, randNRGBA(rng, n, mode))
						nrgbaBoth(t, src, so%4, exact)
					}
				}
			}
		}
		// Every alpha value, and a single non-opaque pixel at every
		// position of rows up to 40 pixels (the overlapping last block).
		src := randNRGBA(rng, 256, 0)
		for i := 0; i < 256; i++ {
			src[4*i+3] = byte(i)
		}
		nrgbaBoth(t, src, 0, false)
		nrgbaBoth(t, src, 0, true)
		for n := 1; n <= 40; n++ {
			for p := 0; p < n; p++ {
				for _, a := range []byte{0, 1, 254} {
					src := randNRGBA(rng, n, 0)
					src[4*p+3] = a
					nrgbaBoth(t, src, 0, false)
				}
			}
		}
	})
}

func FuzzNrgbaRow(f *testing.F) {
	f.Add([]byte{1, 2, 3, 255, 4, 5, 6, 0, 7, 8, 9, 10, 1, 2, 3, 255, 0, 0, 0, 0}, false, uint8(1))
	f.Fuzz(func(t *testing.T, data []byte, exact bool, off uint8) {
		src := data[:len(data)&^3]
		forEachLevel(t, func(t *testing.T) { nrgbaBoth(t, src, int(off%8), exact) })
	})
}

// subGreenBoth transforms p at every level in place, placed off elements
// into a buffer of sentinels, and compares with subtractGreenGo.
func subGreenBoth(t *testing.T, p []uint32, off int) {
	t.Helper()
	n := len(p)
	want := slices.Clone(p)
	subtractGreenGo(want)
	buf := make([]uint32, n+off+9)
	for i := range buf {
		buf[i] = 0xdeadbeef
	}
	got := buf[off : off+n : off+n]
	copy(got, p)
	subtractGreen(got)
	if !slices.Equal(got, want) {
		t.Fatalf("n %d off %d: got %08x, want %08x", n, off, got, want)
	}
	for i, v := range buf {
		if (i < off || i >= off+n) && v != 0xdeadbeef {
			t.Fatalf("n %d off %d: element %d outside p written", n, off, i-off)
		}
	}
}

func TestSubtractGreenLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	widths := []int{0}
	for n := 1; n <= 64; n++ {
		widths = append(widths, n)
	}
	widths = append(widths, 100, 255, 1000, 4097, 65536+13)
	forEachLevel(t, func(t *testing.T) {
		for _, n := range widths {
			for off := 0; off < 16; off++ {
				subGreenBoth(t, randPixels(rng, n), off)
			}
		}
		if testing.Short() {
			return
		}
		// Every (green, red, blue) byte triple, with random alphas.
		p := make([]uint32, 1<<24)
		for i := range p {
			p[i] = uint32(i) | uint32(rng.Intn(256))<<24
		}
		subGreenBoth(t, p, 3)
	})
}

func FuzzSubtractGreen(f *testing.F) {
	f.Add([]byte{1, 2, 3, 255, 4, 5, 6, 0, 7, 8, 9, 10, 1, 2, 3, 255, 0, 0, 0, 0}, uint8(1))
	f.Fuzz(func(t *testing.T, data []byte, off uint8) {
		p := make([]uint32, len(data)/4)
		for i := range p {
			p[i] = binary.LittleEndian.Uint32(data[4*i:])
		}
		forEachLevel(t, func(t *testing.T) { subGreenBoth(t, p, int(off%16)) })
	})
}

// randImageRefs returns a random parse of a w x h image: literals (random
// pixels, often opaque), cache indices below 1<<cacheBits and copies of
// lengths 1 to 4095 at distances from 1 to the window size, the plane
// neighborhoods (1-8 rows up, up to 8 columns aside) and multiples of w
// included.
func randImageRefs(rng *rand.Rand, w, h, cacheBits int) []pixOrCopy {
	n := w * h
	var refs []pixOrCopy
	for pos := 0; pos < n; {
		switch k := rng.Intn(20); {
		case k < 12:
			p := rng.Uint32()
			if rng.Intn(3) > 0 {
				p |= 0xff000000
			}
			refs = append(refs, mkLiteral(p))
			pos++
		case k < 14 && cacheBits > 0:
			refs = append(refs, mkCache(uint32(rng.Intn(1<<cacheBits))))
			pos++
		default:
			maxLen := 16
			if k == 19 {
				maxLen = []int{300, 4095}[rng.Intn(2)]
			}
			l := 1 + rng.Intn(min(n-pos, maxLen))
			var d int
			switch rng.Intn(4) {
			case 0:
				d = 1 + rng.Intn(16)
			case 1:
				d = (1+rng.Intn(8))*w + rng.Intn(17) - 8
			case 2:
				d = w * (1 + rng.Intn(20))
			default:
				d = 1 + rng.Intn(windowSize)
			}
			refs = append(refs, mkCopy(uint32(max(d, 1)), l))
			pos += l
		}
	}
	return refs
}

func TestAddRefsLevels(t *testing.T) {
	defer setAddRefsMin(0)()
	rng := rand.New(rand.NewSource(2))
	iters := 60
	if testing.Short() {
		iters = 10
	}
	forEachLevel(t, func(t *testing.T) {
		for it := 0; it < iters; it++ {
			cacheBits := rng.Intn(maxColorCacheBits + 1)
			w := 1 + rng.Intn(300)
			h := 1 + rng.Intn(400)
			refs := randImageRefs(rng, w, h, cacheBits)
			raw := it%2 == 1
			var sg, sa histoSet
			sg.init(1, cacheBits)
			sa.init(1, cacheBits)
			for i := range sg.h[0].d {
				v := rng.Uint32() >> uint(rng.Intn(33)&31)
				sg.h[0].d[i], sa.h[0].d[i] = v, v
			}
			if raw {
				sg.addRefsRawDistGo(&sg.h[0], refs)
				sa.addRefsRawDist(&sa.h[0], refs)
			} else {
				sg.addRefsGo(&sg.h[0], refs, w)
				sa.addRefs(&sa.h[0], refs, w)
			}
			if !slices.Equal(sa.h[0].d, sg.h[0].d) {
				t.Fatalf("iter %d (w %d, %d refs, cache %d, raw %v): histograms differ", it, w, len(refs), cacheBits, raw)
			}
		}
	})
}

// TestAddRefsBadRefs checks that a cache index or a distance code out of
// range panics as in the Go code.
func TestAddRefsBadRefs(t *testing.T) {
	defer setAddRefsMin(0)()
	rng := rand.New(rand.NewSource(3))
	for _, bad := range []pixOrCopy{
		mkCache(1 << 12), mkCopy(1<<22, 5), mkCopy(0, 5),
	} {
		for _, raw := range []bool{false, true} {
			refs := randImageRefs(rng, 100, 50, 4)
			refs[len(refs)/2] = bad
			atEveryLevel(func(l cpuinfo.Level) {
				var s histoSet
				s.init(1, 4)
				defer func() {
					if recover() == nil && (raw || bad.mode() != modeCopy || bad.val != 0) {
						t.Errorf("level %v raw %v ref %v: no panic", l, raw, bad)
					}
				}()
				if raw {
					s.addRefsRawDist(&s.h[0], refs)
				} else {
					s.addRefs(&s.h[0], refs, 100)
				}
			})
		}
	}
}

func FuzzAddRefs(f *testing.F) {
	f.Add(int64(1), uint16(64), uint16(64), uint8(3), false)
	f.Fuzz(func(t *testing.T, seed int64, w, h uint16, cacheBits uint8, raw bool) {
		defer setAddRefsMin(0)()
		rng := rand.New(rand.NewSource(seed))
		cb := int(cacheBits) % (maxColorCacheBits + 1)
		refs := randImageRefs(rng, 1+int(w)%512, 1+int(h)%512, cb)
		var sg histoSet
		sg.init(1, cb)
		if raw {
			sg.addRefsRawDistGo(&sg.h[0], refs)
		} else {
			sg.addRefsGo(&sg.h[0], refs, 1+int(w)%512)
		}
		atEveryLevel(func(l cpuinfo.Level) {
			var s histoSet
			s.init(1, cb)
			if raw {
				s.addRefsRawDist(&s.h[0], refs)
			} else {
				s.addRefs(&s.h[0], refs, 1+int(w)%512)
			}
			if !slices.Equal(s.h[0].d, sg.h[0].d) {
				t.Fatalf("level %v: histograms differ", l)
			}
		})
	})
}

// tileBoth adds refs to the tile histograms of a w x h image (tiles of
// 1<<hb pixels) with addTileRefsGo and addTileRefs at every level, from
// the same random counts, and compares.
func tileBoth(t *testing.T, rng *rand.Rand, refs []pixOrCopy, w, h, hb, cacheBits int) {
	t.Helper()
	hx := subSampleSize(w, hb)
	nt := hx * subSampleSize(h, hb)
	var sg histoSet
	sg.init(nt, cacheBits)
	for i := range sg.data {
		sg.data[i] = rng.Uint32() >> uint(rng.Intn(33)&31)
	}
	start := slices.Clone(sg.data)
	sg.addTileRefsGo(refs, w, hb, hx)
	atEveryLevel(func(l cpuinfo.Level) {
		var s histoSet
		s.init(nt, cacheBits)
		copy(s.data, start)
		s.addTileRefs(refs, w, hb, hx)
		if !slices.Equal(s.data, sg.data) {
			t.Fatalf("level %v (w %d h %d hb %d, %d refs, cache %d): histograms differ", l, w, h, hb, len(refs), cacheBits)
		}
	})
}

func TestAddTileRefsLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	iters := 200
	if testing.Short() {
		iters = 30
	}
	for it := 0; it < iters; it++ {
		cacheBits := rng.Intn(maxColorCacheBits + 1)
		w := 1 + rng.Intn([]int{8, 64, 700}[rng.Intn(3)])
		h := 1 + rng.Intn(1+20000/w)
		refs := randImageRefs(rng, w, h, cacheBits)
		if it%5 == 0 {
			// Opaque runs of literals (the pending alpha count).
			for i := range refs {
				if rng.Intn(4) > 0 {
					refs[i] = mkLiteral(rng.Uint32() | 0xff000000)
				}
			}
			refs = refs[:min(len(refs), w*h)]
		}
		tileBoth(t, rng, refs, w, h, 2+rng.Intn(8), cacheBits)
	}
}

// TestAddTileRefsBadRefs checks that a tile, a cache index or a distance
// code out of range panics as in the Go code.
func TestAddTileRefsBadRefs(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	w, h, hb := 60, 40, 3
	for k := 0; k < 3; k++ {
		refs := randImageRefs(rng, w, h, 2)
		switch k {
		case 0:
			refs = append(refs, refs...) // past the last tile
		case 1:
			refs[len(refs)/2] = mkCache(1 << 12)
		case 2:
			refs[len(refs)/2] = mkCopy(1<<22, 3)
		}
		atEveryLevel(func(l cpuinfo.Level) {
			defer func() {
				if recover() == nil {
					t.Errorf("level %v case %d: no panic", l, k)
				}
			}()
			hx := subSampleSize(w, hb)
			var s histoSet
			s.init(hx*subSampleSize(h, hb), 2)
			s.addTileRefs(refs, w, hb, hx)
		})
	}
}

func FuzzAddTileRefs(f *testing.F) {
	f.Add(int64(1), uint16(64), uint16(64), uint8(3), uint8(4))
	f.Fuzz(func(t *testing.T, seed int64, w, h uint16, cacheBits, hb uint8) {
		rng := rand.New(rand.NewSource(seed))
		cb := int(cacheBits) % (maxColorCacheBits + 1)
		ww, hh := 1+int(w)%700, 1+int(h)%300
		tileBoth(t, rng, randImageRefs(rng, ww, hh, cb), ww, hh, 2+int(hb)%8, cb)
	})
}

// randCodes returns ng groups of 5 codes (litLen symbols for code 0, 256
// for codes 1-3, 40 for code 4) with lengths 0 to 15 and codes below
// 1<<length; some codes have a single symbol, cleared as the encoder does.
func randCodes(rng *rand.Rand, ng, litLen int) []huffCode {
	codes := make([]huffCode, 5*ng)
	for i := range codes {
		n := []int{litLen, 256, 256, 256, numDistanceCodes}[i%5]
		c := huffCode{lengths: make([]uint8, n), codes: make([]uint16, n)}
		maxLen := []int{15, 8, 3, 1}[rng.Intn(4)]
		for j := range c.lengths {
			l := rng.Intn(maxLen + 1)
			c.lengths[j] = uint8(l)
			c.codes[j] = uint16(rng.Intn(1 << uint(l)))
		}
		if rng.Intn(8) == 0 {
			clear(c.lengths)
			clear(c.codes)
		}
		codes[i] = c
	}
	return codes
}

// writeBoth writes refs with writeImageDataGo and writeImageData from the
// same bit writer state (pre bytes, nbits pending bits, capacity extra
// bytes of spare room) and compares the writers.
func writeBoth(t *testing.T, w int, refs []pixOrCopy, hb int, symbols []uint16, codes []huffCode, pre, nbits, room int, rng *rand.Rand) {
	t.Helper()
	start := make([]byte, pre, pre+room)
	rng.Read(start)
	acc := rng.Uint64() & (1<<uint(nbits) - 1)
	bg := bitWriter{buf: slices.Clone(start), acc: acc, nbits: uint(nbits)}
	writeImageDataGo(&bg, w, refs, hb, symbols, codes)
	atEveryLevel(func(l cpuinfo.Level) {
		ba := bitWriter{buf: append(make([]byte, 0, pre+room), start...), acc: acc, nbits: uint(nbits)}
		writeImageData(&ba, w, refs, hb, symbols, codes)
		if !bytes.Equal(ba.buf, bg.buf) || ba.acc != bg.acc || ba.nbits != bg.nbits {
			t.Fatalf("level %v (w %d, %d refs, hb %d, %d groups, pre %d, nbits %d, room %d): writers differ (len %d/%d, acc %x/%x, nbits %d/%d)",
				l, w, len(refs), hb, len(codes)/5, pre, nbits, room, len(ba.buf), len(bg.buf), ba.acc, bg.acc, ba.nbits, bg.nbits)
		}
	})
}

// randSymbols returns the entropy image of a w x h image with tiles of
// 1<<hb pixels: ng groups, in patches.
func randSymbols(rng *rand.Rand, w, h, hb, ng int) []uint16 {
	s := make([]uint16, subSampleSize(w, hb)*subSampleSize(h, hb))
	for i := range s {
		if i > 0 && rng.Intn(3) == 0 {
			s[i] = s[i-1]
		} else {
			s[i] = uint16(rng.Intn(ng))
		}
	}
	return s
}

func TestWriteImageDataLevels(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	iters := 300
	if testing.Short() {
		iters = 40
	}
	for it := 0; it < iters; it++ {
		cacheBits := rng.Intn(maxColorCacheBits + 1)
		w := 1 + rng.Intn([]int{8, 64, 700}[rng.Intn(3)])
		h := 1 + rng.Intn(1+20000/w)
		refs := randImageRefs(rng, w, h, cacheBits)
		ng := 1 + rng.Intn(12)
		codes := randCodes(rng, ng, literalLen(cacheBits))
		var symbols []uint16
		hb := 0
		if ng > 1 || rng.Intn(2) == 0 {
			hb = 2 + rng.Intn(8)
			symbols = randSymbols(rng, w, h, hb, ng)
		}
		room := []int{0, 3, 100, 100000}[rng.Intn(4)]
		writeBoth(t, w, refs, hb, symbols, codes, rng.Intn(9), rng.Intn(32), room, rng)
	}
}

// TestWriteImageDataBadRefs checks that a symbol, a tile or a group out of
// range panics as in the Go code.
func TestWriteImageDataBadRefs(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	w, h := 50, 40
	for k := 0; k < 5; k++ {
		refs := randImageRefs(rng, w, h, 3)
		codes := randCodes(rng, 3, literalLen(3))
		hb := 3
		symbols := randSymbols(rng, w, h, hb, 3)
		switch k {
		case 0:
			refs[len(refs)/2] = mkCache(1 << 4)
		case 1:
			refs[len(refs)/2] = mkCopy(1<<22, 3)
		case 2:
			refs = append(refs, mkLiteral(0)) // past the last tile
			refs = append(refs, refs...)
		case 3:
			symbols[len(symbols)/2] = 3
		case 4:
			refs[len(refs)/2] = mkCopy(5, 5000)
		}
		atEveryLevel(func(l cpuinfo.Level) {
			defer func() {
				if recover() == nil {
					t.Errorf("level %v case %d: no panic", l, k)
				}
			}()
			var bw bitWriter
			writeImageData(&bw, w, refs, hb, symbols, codes)
		})
	}
}

func FuzzWriteImageData(f *testing.F) {
	f.Add(int64(1), uint16(40), uint16(30), uint8(3), uint8(4), uint8(2))
	f.Fuzz(func(t *testing.T, seed int64, w, h uint16, cacheBits, hb, ng uint8) {
		rng := rand.New(rand.NewSource(seed))
		cb := int(cacheBits) % (maxColorCacheBits + 1)
		ww, hh := 1+int(w)%700, 1+int(h)%300
		refs := randImageRefs(rng, ww, hh, cb)
		groups := 1 + int(ng)%16
		codes := randCodes(rng, groups, literalLen(cb))
		var symbols []uint16
		b := 0
		if hb%10 >= 2 {
			b = int(hb % 10)
			symbols = randSymbols(rng, ww, hh, b, groups)
		} else {
			codes = codes[:5]
		}
		writeBoth(t, ww, refs, b, symbols, codes, rng.Intn(9), rng.Intn(32), rng.Intn(64), rng)
	})
}

// photoParse returns the level 0 parse of the photo and its codes (one
// histogram), as the encoder writes it.
func photoParse(tb testing.TB) (w int, argb []uint32, refs []pixOrCopy, codes []huffCode) {
	img := loadTestJPEG(tb, "source.jpg")
	w, h := img.Rect.Dx(), img.Rect.Dy()
	argb = make([]uint32, w*h)
	for i := range argb {
		p := img.Pix[(i/w)*img.Stride+4*(i%w):]
		argb[i] = uint32(p[3])<<24 | uint32(p[0])<<16 | uint32(p[1])<<8 | uint32(p[2])
	}
	sc := getScratch(0)
	defer putScratch(sc)
	sc.hc.fill(argb, w, h, 0, true, 0)
	refs = slices.Clone(backwardRefsLz77(len(argb), argb, &sc.hc, nil))
	var s histoSet
	s.init(1, 0)
	s.addRefsGo(&s.h[0], refs, w)
	codes = buildCodes(&s, sc)
	for i := range codes {
		codes[i] = huffCode{lengths: slices.Clone(codes[i].lengths), codes: slices.Clone(codes[i].codes)}
		clearIfOneSymbol(&codes[i])
	}
	return w, argb, refs, codes
}

// BenchmarkStreamKernels runs nrgbaRow, addRefs, addTileRefs (tiles of
// 128 pixels, as at level 0) and writeImageData on the photo (its level 0
// parse) at every level.
func BenchmarkStreamKernels(b *testing.B) {
	w, argb, refs, codes := photoParse(b)
	src := make([]byte, 4*len(argb))
	for i, p := range argb {
		src[4*i], src[4*i+1], src[4*i+2], src[4*i+3] = byte(p>>16), byte(p>>8), byte(p), byte(p>>24)
	}
	dst := make([]uint32, len(argb))
	var s histoSet
	s.init(1, 0)
	bw := bitWriter{buf: make([]byte, 0, 4*len(argb))}
	for _, l := range testLevels() {
		b.Run(fmt.Sprintf("nrgba/%v", l), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				for y := 0; y < len(argb); y += w {
					nrgbaRow(dst[y:y+w], src[4*y:4*(y+w)], false)
				}
			}
		})
		b.Run(fmt.Sprintf("subGreen/%v", l), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				subtractGreen(dst)
			}
		})
		b.Run(fmt.Sprintf("addRefs/%v", l), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				s.clear(&s.h[0])
				s.addRefs(&s.h[0], refs, w)
			}
		})
		b.Run(fmt.Sprintf("tiles/%v", l), func(b *testing.B) {
			defer setLevel(l)()
			hb := 7
			hx := subSampleSize(w, hb)
			var ts histoSet
			ts.init(hx*subSampleSize(len(argb)/w, hb), 0)
			for i := 0; i < b.N; i++ {
				clear(ts.data)
				ts.addTileRefs(refs, w, hb, hx)
			}
		})
		b.Run(fmt.Sprintf("write/%v", l), func(b *testing.B) {
			defer setLevel(l)()
			for i := 0; i < b.N; i++ {
				bw.reset()
				writeImageData(&bw, w, refs, 0, nil, codes)
			}
		})
	}
}
