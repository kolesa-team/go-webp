package vp8l

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// Differential tests of the per-row kernels (dsp_*.go) against their Go
// references. On arm64 without the purego tag they check the NEON
// assembly; elsewhere both sides are the same Go code.

// guard is the value of the bytes around a destination, which a kernel
// must not change.
const guard = 0xa5

// guarded returns n bytes of value guard with 16 more on each side, and the
// n bytes in the middle.
func guarded(n int) (all, mid []byte) {
	all = bytes.Repeat([]byte{guard}, n+32)
	return all, all[16 : 16+n : 16+n]
}

func checkGuard(t *testing.T, all []byte, what string) {
	t.Helper()
	n := len(all) - 32
	for i, b := range all {
		if (i < 16 || i >= 16+n) && b != guard {
			t.Fatalf("%s: wrote outside the destination (byte %d)", what, i-16)
		}
	}
}

// randomPixels returns n pixels, often with repeated or close values.
func randomPixels(rng *rand.Rand, n int) []uint32 {
	p := make([]uint32, n)
	for i := range p {
		switch rng.Intn(4) {
		case 0:
			if i > 0 {
				p[i] = p[i-1] ^ uint32(rng.Intn(4))<<(8*rng.Intn(4))
				continue
			}
		case 1:
			p[i] = uint32(rng.Intn(4)) * 0x01010101 * uint32(rng.Intn(64))
			continue
		}
		p[i] = rng.Uint32()
	}
	return p
}

func u32Bytes(p []uint32) []byte {
	b := make([]byte, 4*len(p))
	for i, c := range p {
		binary.LittleEndian.PutUint32(b[4*i:], c)
	}
	return b
}

func TestEmitKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	kernels := []struct {
		name     string
		got, ref func([]uint32, []byte)
	}{
		{"emitRGBA", emitRGBA, emitRGBAGo},
		{"emitAddGreen", emitAddGreen, emitAddGreenGo},
	}
	for iter := 0; iter < 20000; iter++ {
		n := rng.Intn(80)
		if iter%100 == 0 {
			n = rng.Intn(5000)
		}
		src := randomPixels(rng, n)
		// The destination may hold fewer or more pixels than src, or a
		// partial pixel.
		dn := 4 * n
		if iter%3 == 0 {
			dn = max(0, dn+rng.Intn(17)-8)
		}
		for _, k := range kernels {
			all, dst := guarded(dn)
			want := bytes.Repeat([]byte{guard}, dn)
			k.ref(src, want)
			k.got(src, dst)
			if !bytes.Equal(dst, want) {
				t.Fatalf("%s: %d pixels into %d bytes: output differs", k.name, n, dn)
			}
			checkGuard(t, all, k.name)
		}
	}
}

func TestAddGreenKernel(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 20000; iter++ {
		n := rng.Intn(80)
		if iter%100 == 0 {
			n = rng.Intn(5000)
		}
		in := randomPixels(rng, n)
		want := make([]uint32, n)
		addGreenGo(in, want)
		out := make([]uint32, n+8)
		for i := range out {
			out[i] = 0xa5a5a5a5
		}
		addGreen(in, out[4:4+n])
		if !slices.Equal(out[4:4+n], want) {
			t.Fatalf("%d pixels: output differs", n)
		}
		for i, c := range out {
			if (i < 4 || i >= 4+n) && c != 0xa5a5a5a5 {
				t.Fatalf("%d pixels: wrote outside the destination", n)
			}
		}
		addGreen(in, in) // in place
		if !slices.Equal(in, want) {
			t.Fatalf("%d pixels: in-place output differs", n)
		}
	}
}

func TestLookupBytesKernel(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for iter := 0; iter < 5000; iter++ {
		var pal [256]byte
		rng.Read(pal[:])
		n := rng.Intn(100)
		if iter%50 == 0 {
			n = rng.Intn(10000)
		}
		in := make([]byte, n)
		if iter == 0 {
			n = 256 * 4
			in = make([]byte, n)
			for i := range in {
				in[i] = byte(i) // every index, in each lane position
			}
		} else {
			rng.Read(in)
		}
		want := make([]byte, n)
		lookupBytesGo(&pal, in, want)
		all, out := guarded(n)
		lookupBytes(&pal, in, out)
		if !bytes.Equal(out, want) {
			t.Fatalf("%d bytes: output differs", n)
		}
		checkGuard(t, all, "lookupBytes")
	}
}

// crossColorCheck runs crossColorRow on rows of width pixels (src) with
// the tile multipliers mults and compares with the Go reference, out of
// place and in place.
func crossColorCheck(t *testing.T, mults, src []uint32, width, bits int) {
	t.Helper()
	tiles := subSampleSize(width, bits)
	for y := 0; y*width < len(src); y++ {
		row := src[y*width : (y+1)*width]
		m := mults[y*tiles : (y+1)*tiles]
		want := make([]uint32, width)
		crossColorRowGo(m, row, want, bits)
		got := make([]uint32, width+2)
		got[0], got[width+1] = 0xa5a5a5a5, 0xa5a5a5a5
		crossColorRow(m, row, got[1:width+1], bits)
		if !slices.Equal(got[1:width+1], want) {
			for x := range want {
				if got[1+x] != want[x] {
					t.Fatalf("width %d bits %d: pixel %d (%08x, mults %06x): got %08x, want %08x", width, bits, x, row[x], m[x>>bits], got[1+x], want[x])
				}
			}
		}
		if got[0] != 0xa5a5a5a5 || got[width+1] != 0xa5a5a5a5 {
			t.Fatalf("width %d bits %d: wrote outside the row", width, bits)
		}
		inPlace := slices.Clone(row)
		crossColorRow(m, inPlace, inPlace, bits)
		if !slices.Equal(inPlace, want) {
			t.Fatalf("width %d bits %d: in-place output differs", width, bits)
		}
	}
}

// TestCrossColorExhaustive covers every value of each multiplier with
// every value of the channels it combines: (g2r, G, R), (g2b, G, B) and
// (r2b, R', B) with R' = R (g2r = 0), the other channels and multipliers
// random, in tiles of 4 pixels.
func TestCrossColorExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const bits = 2
	const width = 1024 // 256 tiles
	n := 1 << 16
	if testing.Short() {
		n = 1 << 12
	}
	for _, kind := range []int{0, 1, 2} {
		for m0 := 0; m0 < 256; m0++ {
			mults := make([]uint32, n/4)
			src := make([]uint32, n)
			for i := range src {
				a, b := uint32(i&0xff), uint32(i>>8&0xff) // two channels
				if testing.Short() {
					a, b = uint32(rng.Intn(256)), uint32(rng.Intn(256))
				}
				r := rng.Uint32()
				m := rng.Uint32() & 0xffffff
				switch kind {
				case 0: // g2r with G and R
					src[i] = r&0xff0000ff | b<<16 | a<<8
					m = m&^0xff | uint32(m0)
				case 1: // g2b with G and B
					src[i] = r&0xffff0000 | a<<8 | b
					m = m&^0xff00 | uint32(m0)<<8
				default: // r2b with R and B, g2r = 0
					src[i] = r&0xff00ff00 | a<<16 | b
					m = m&^0xff00ff | uint32(m0)<<16
				}
				mults[i>>bits] = m // the last pixel of a tile wins
			}
			crossColorCheck(t, mults, src, width, bits)
		}
	}
}

func TestCrossColorRow(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for iter := 0; iter < 20000; iter++ {
		bits := 2 + rng.Intn(8)
		width := 1 + rng.Intn(100)
		if iter%50 == 0 {
			width = 1 + rng.Intn(3000)
		}
		height := 1 + rng.Intn(3)
		mults := make([]uint32, subSampleSize(width, bits)*height)
		for i := range mults {
			mults[i] = rng.Uint32()
		}
		crossColorCheck(t, mults, randomPixels(rng, width*height), width, bits)
	}
}

// predictCheck runs predictRow on one row (in, with the previous output
// row up, extended by out[0]) and compares with the Go reference, out of
// place and in place.
func predictCheck(t *testing.T, modes, in, up []uint32, bits int, what string) {
	t.Helper()
	width := len(in)
	want := make([]uint32, width)
	want[0] = addPixels(in[0], up[0])
	upW := append(slices.Clone(up[:width]), want[0])
	predictRowGo(modes, in, want, upW, bits)

	got := make([]uint32, width+2)
	got[0], got[width+1] = 0xa5a5a5a5, 0xa5a5a5a5
	out := got[1 : width+1 : width+1]
	out[0] = want[0]
	predictRow(modes, in, out, upW, bits)
	if !slices.Equal(out, want) {
		for x := range want {
			if out[x] != want[x] {
				t.Fatalf("%s: width %d bits %d: pixel %d (mode %d): got %08x, want %08x", what, width, bits, x, modes[x>>bits]>>8&0xf, out[x], want[x])
			}
		}
	}
	if got[0] != 0xa5a5a5a5 || got[width+1] != 0xa5a5a5a5 {
		t.Fatalf("%s: width %d bits %d: wrote outside the row", what, width, bits)
	}
	// In place, with the row in the buffer after the previous row, as
	// inversePredictor does for a predictor after another transform.
	buf := append(slices.Clone(up[:width]), in...)
	buf[width] = want[0]
	predictRow(modes, buf[width:], buf[width:], buf[:width+1], bits)
	if !slices.Equal(buf[width:], want) {
		t.Fatalf("%s: width %d bits %d: in-place output differs", what, width, bits)
	}
}

// TestPredictRow checks every predictor mode, in runs of tiles of every
// size, against the Go reference.
func TestPredictRow(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	iters := 40000
	if testing.Short() {
		iters = 4000
	}
	for iter := 0; iter < iters; iter++ {
		bits := 2 + rng.Intn(8)
		width := 1 + rng.Intn(80)
		if iter%40 == 0 {
			width = 1 + rng.Intn(2000)
		}
		modes := make([]uint32, subSampleSize(width, bits))
		single := uint32(rng.Intn(16))
		for i := range modes {
			mode := single
			if iter%2 == 0 {
				mode = uint32(rng.Intn(16))
			}
			// Bits other than 8-11 must be ignored.
			modes[i] = rng.Uint32()&^0xf00 | mode<<8
		}
		in := randomPixels(rng, width)
		up := randomPixels(rng, width)
		if iter%3 == 0 {
			// Close values exercise the ties of mode 11 and the clamps.
			for i := range up {
				up[i] = in[i] ^ uint32(rng.Intn(3))<<(8*rng.Intn(4))
			}
		}
		predictCheck(t, modes, in, up, bits, fmt.Sprintf("iter %d", iter))
	}
}

// TestPredictSelectTies checks mode 11 on pixels whose sums of channel
// differences are equal or off by one.
func TestPredictSelectTies(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for iter := 0; iter < 20000; iter++ {
		width := 2 + rng.Intn(40)
		modes := []uint32{11 << 8}
		const bits = 9
		in := make([]uint32, width)
		up := make([]uint32, width)
		base := rng.Uint32()
		for i := range up {
			up[i] = base
			for c := 0; c < 4; c++ {
				up[i] += uint32(rng.Intn(3)) << (8 * c)
			}
			// Residuals that keep the output close to the top row.
			in[i] = uint32(rng.Intn(2)) << (8 * rng.Intn(4))
		}
		predictCheck(t, modes, in, up, bits, "ties")
	}
}

// bundledCheck expands the packed row src with bits (1-3) and a palette
// of random colors into an 8-bit and an RGBA destination of dn pixels, and
// compares with the Go references.
func bundledCheck(t *testing.T, pal []uint32, bits int, src []byte, dn int) {
	t.Helper()
	tr := &transform{kind: colorIndexingTransform, bits: bits, data: pal}
	var a alphaTables
	a.init(tr)
	want := bytes.Repeat([]byte{guard}, dn)
	expandBundledRowGo(&a, bits, src, want)
	all, got := guarded(dn)
	expandBundledRow(&a, bits, src, got)
	if !bytes.Equal(got, want) {
		t.Fatalf("8-bit, bits %d, %d packed bytes into %d: output differs", bits, len(src), dn)
	}
	checkGuard(t, all, "expandBundledRow")

	d := getDecoder()
	defer putDecoder(d)
	d.palReady, d.bundleReady = false, false
	rgba := d.paletteRGBA(tr)
	tab := d.bundleTable(tr)
	for _, n := range []int{4 * dn, 4*dn - 1 - min(dn, 2)} {
		n = max(n, 0)
		want := bytes.Repeat([]byte{guard}, n)
		expandBundledRGBAGo(tab, bits, src, want)
		all, got := guarded(n)
		expandBundledRGBA(rgba, tab, bits, src, got)
		if !bytes.Equal(got, want) {
			t.Fatalf("RGBA, bits %d, %d packed bytes into %d bytes: output differs", bits, len(src), n)
		}
		checkGuard(t, all, "expandBundledRGBA")
	}
}

func TestExpandBundled(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for iter := 0; iter < 10000; iter++ {
		bits := 1 + rng.Intn(3)
		pal := make([]uint32, 1<<(8>>bits))
		for i := range pal {
			pal[i] = rng.Uint32()
		}
		width := 1 + rng.Intn(200)
		if iter%50 == 0 {
			width = 1 + rng.Intn(4000)
		}
		src := make([]byte, subSampleSize(width, bits))
		if iter == 0 {
			for i := range src {
				src[i] = byte(i) // every packed value
			}
		} else {
			rng.Read(src)
		}
		dn := width
		if iter%4 == 0 {
			dn = rng.Intn(width + 1) // fewer output pixels than packed
		}
		bundledCheck(t, pal, bits, src, dn)
	}
}

func FuzzExpandBundled(f *testing.F) {
	f.Add([]byte("\x00\x01\x02\x03\x10\x20\x40\x80\xff\xfe\xaa\x55\x0f\xf0\x33\xcc\x99"), uint8(1), uint16(40))
	f.Fuzz(func(t *testing.T, src []byte, bits uint8, dn uint16) {
		b := 1 + int(bits)%3
		pal := fuzzPixels(src, 1<<(8>>b))
		bundledCheck(t, pal, b, src, min(int(dn), len(src)<<b+5))
	})
}

// fuzzPixels reads pixels from data.
func fuzzPixels(data []byte, n int) []uint32 {
	p := make([]uint32, n)
	for i := range p {
		if len(data) >= 4 {
			p[i] = binary.LittleEndian.Uint32(data)
			data = data[4:]
		} else {
			p[i] = uint32(i) * 0x9e3779b9
		}
	}
	return p
}

func FuzzEmitKernels(f *testing.F) {
	f.Add([]byte("\x01\x02\x03\x04\x05\x06\x07\x08\xff\xfe\xfd\xfc"), 3)
	f.Fuzz(func(t *testing.T, data []byte, dn int) {
		n := len(data) / 4
		src := fuzzPixels(data, n)
		dn = min(max(dn, 0), 4*n+64)
		for _, k := range []struct {
			got, ref func([]uint32, []byte)
		}{{emitRGBA, emitRGBAGo}, {emitAddGreen, emitAddGreenGo}} {
			want := bytes.Repeat([]byte{guard}, dn)
			k.ref(src, want)
			all, dst := guarded(dn)
			k.got(src, dst)
			if !bytes.Equal(dst, want) {
				t.Fatal("output differs")
			}
			checkGuard(t, all, "emit")
		}
		want := make([]uint32, n)
		addGreenGo(src, want)
		addGreen(src, src)
		if !slices.Equal(src, want) {
			t.Fatal("addGreen differs")
		}
	})
}

func FuzzLookupBytes(f *testing.F) {
	f.Add([]byte("palette"), []byte("0123456789abcdefghij"))
	f.Fuzz(func(t *testing.T, p, in []byte) {
		var pal [256]byte
		copy(pal[:], p)
		want := make([]byte, len(in))
		lookupBytesGo(&pal, in, want)
		all, out := guarded(len(in))
		lookupBytes(&pal, in, out)
		if !bytes.Equal(out, want) {
			t.Fatal("output differs")
		}
		checkGuard(t, all, "lookupBytes")
	})
}

func FuzzCrossColorRow(f *testing.F) {
	f.Add([]byte("\x10\x20\x30\x40\x50\x60\x70\x80\x90"), uint8(2), uint16(5))
	f.Fuzz(func(t *testing.T, data []byte, bits uint8, width uint16) {
		b := 2 + int(bits)%8
		w := 1 + int(width)%600
		mults := fuzzPixels(data, subSampleSize(w, b))
		src := fuzzPixels(data[min(len(data), 3):], w)
		crossColorCheck(t, mults, src, w, b)
	})
}

func FuzzPredictRow(f *testing.F) {
	f.Add([]byte("\x0b\x01\x02\x07\x05\x03\x0c\x0d\x0a\x09\x08"), uint8(2), uint16(37))
	f.Fuzz(func(t *testing.T, data []byte, bits uint8, width uint16) {
		b := 2 + int(bits)%8
		w := 1 + int(width)%600
		modes := make([]uint32, subSampleSize(w, b))
		for i := range modes {
			if i < len(data) {
				modes[i] = uint32(data[i]) << 8
			}
		}
		in := fuzzPixels(data, w)
		up := fuzzPixels(data[min(len(data), 5):], w)
		predictCheck(t, modes, in, up, b, "fuzz")
	})
}

// BenchmarkKernels times each kernel against its Go reference on rows of
// 3000 pixels.
func BenchmarkKernels(b *testing.B) {
	const w = 3000
	rng := rand.New(rand.NewSource(1))
	src := randomPixels(rng, w)
	up := append(randomPixels(rng, w), 0)
	tmp := make([]uint32, w)
	dst := make([]byte, 4*w)
	var pal [256]byte
	rng.Read(pal[:])
	idx := make([]byte, 16*w)
	rng.Read(idx)
	idxOut := make([]byte, len(idx))
	type impl struct {
		name string
		f    func()
	}
	bench := func(name string, bytes int, impls ...impl) {
		for _, im := range impls {
			b.Run(name+"/"+im.name, func(b *testing.B) {
				b.SetBytes(int64(bytes))
				for i := 0; i < b.N; i++ {
					im.f()
				}
			})
		}
	}
	bench("emitRGBA", 4*w, impl{"go", func() { emitRGBAGo(src, dst) }}, impl{"asm", func() { emitRGBA(src, dst) }})
	bench("emitAddGreen", 4*w, impl{"go", func() { emitAddGreenGo(src, dst) }}, impl{"asm", func() { emitAddGreen(src, dst) }})
	bench("addGreen", 4*w, impl{"go", func() { addGreenGo(src, tmp) }}, impl{"asm", func() { addGreen(src, tmp) }})
	bench("lookupBytes", len(idx), impl{"go", func() { lookupBytesGo(&pal, idx, idxOut) }}, impl{"asm", func() { lookupBytes(&pal, idx, idxOut) }})
	for _, bits := range []int{2, 4, 5} {
		mults := randomPixels(rng, subSampleSize(w, bits))
		bench(fmt.Sprintf("crossColor/bits%d", bits), 4*w,
			impl{"go", func() { crossColorRowGo(mults, src, tmp, bits) }},
			impl{"asm", func() { crossColorRow(mults, src, tmp, bits) }})
	}
	for bits := 1; bits <= 3; bits++ {
		tr := &transform{kind: colorIndexingTransform, bits: bits, data: randomPixels(rng, 1<<(8>>bits))}
		var a alphaTables
		a.init(tr)
		d := getDecoder()
		d.palReady, d.bundleReady = false, false
		rgba, tab := d.paletteRGBA(tr), d.bundleTable(tr)
		packed := idx[:w>>bits]
		bench(fmt.Sprintf("expandBundled8/bits%d", bits), w,
			impl{"go", func() { expandBundledRowGo(&a, bits, packed, idxOut[:w]) }},
			impl{"asm", func() { expandBundledRow(&a, bits, packed, idxOut[:w]) }})
		bench(fmt.Sprintf("expandBundledRGBA/bits%d", bits), 4*w,
			impl{"go", func() { expandBundledRGBAGo(tab, bits, packed, dst) }},
			impl{"asm", func() { expandBundledRGBA(rgba, tab, bits, packed, dst) }})
		putDecoder(d)
	}
	// Small residuals, as in real images.
	res := make([]uint32, w)
	for i := range res {
		res[i] = rng.Uint32() & 0x0f0f0f0f
	}
	for mode := uint32(0); mode < 14; mode++ {
		const bits = 5 // 32-pixel tiles, as in photos
		modes := make([]uint32, subSampleSize(w, bits))
		for i := range modes {
			modes[i] = mode << 8
		}
		bench(fmt.Sprintf("predict/mode%d", mode), 4*w,
			impl{"go", func() { predictRowGo(modes, res, tmp, up, bits) }},
			impl{"asm", func() { predictRow(modes, res, tmp, up, bits) }})
	}
}
