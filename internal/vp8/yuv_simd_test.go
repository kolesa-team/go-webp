package vp8

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"testing"
)

// Differential tests of the row converters used by the decoder
// (upsamplePair, sampleRow: NEON on arm64) against the portable reference
// (upsamplePairGo, sampleRowGo). On other platforms both sides run the Go
// code, which still checks the reference against yuvPix.

// fillRow fills b with test samples: mode 0 uniform random, 1 extremes
// (0, 1, 254, 255), 2 a smooth ramp with noise, 3 constant.
func fillRow(r *rand.Rand, b []byte, mode int) {
	c := byte(r.IntN(256))
	for i := range b {
		switch mode {
		case 0:
			b[i] = byte(r.IntN(256))
		case 1:
			b[i] = [4]byte{0, 1, 254, 255}[r.IntN(4)]
		case 2:
			b[i] = byte(i*3 + r.IntN(5))
		default:
			b[i] = c
		}
	}
}

const guardLen = 80

// guarded returns an n-byte slice inside a buffer whose surrounding
// guardLen bytes hold a pattern, and a func reporting whether the pattern
// survived.
func guarded(n int) ([]byte, func() bool) {
	buf := make([]byte, n+2*guardLen)
	for i := range buf {
		buf[i] = 0xa5
	}
	s := buf[guardLen : guardLen+n : guardLen+n]
	return s, func() bool {
		for _, v := range buf[:guardLen] {
			if v != 0xa5 {
				return false
			}
		}
		for _, v := range buf[guardLen+n:] {
			if v != 0xa5 {
				return false
			}
		}
		return true
	}
}

type pairCase struct {
	w                      int
	topY, botY             []byte
	topU, topV, curU, curV []byte
	topA, botA             []byte
}

func newPairCase(r *rand.Rand, w int, alpha, topOnly bool, mode int) pairCase {
	uvW := (w + 1) >> 1
	c := pairCase{w: w}
	mk := func(n int) []byte {
		b := make([]byte, n)
		fillRow(r, b, mode)
		return b
	}
	c.topY = mk(w)
	c.topU, c.topV, c.curU, c.curV = mk(uvW), mk(uvW), mk(uvW), mk(uvW)
	if !topOnly {
		c.botY = mk(w)
	}
	if alpha {
		c.topA = mk(w)
		if !topOnly {
			c.botA = mk(w)
		}
	}
	return c
}

// check runs the dispatched converter and the reference on c and reports
// any difference or write outside the destination rows.
func (c pairCase) check(t testing.TB) {
	t.Helper()
	w := c.w
	wantT, wantB := make([]byte, 4*w), make([]byte, 4*w)
	uvT, uvB := make([]uint32, w), make([]uint32, w)
	var refB []byte
	if c.botY != nil {
		refB = wantB
	}
	upsamplePairGo(c.topY, c.botY, c.topU, c.topV, c.curU, c.curV, wantT, refB, c.topA, c.botA, uvT, uvB, w)

	gotT, okT := guarded(4 * w)
	gotB, okB := guarded(4 * w)
	var dstB []byte
	if c.botY != nil {
		dstB = gotB
	}
	upsamplePair(c.topY, c.botY, c.topU, c.topV, c.curU, c.curV, gotT, dstB, c.topA, c.botA, make([]uint32, w), make([]uint32, w), w)
	if !okT() || !okB() {
		t.Fatalf("w=%d: write outside the destination rows", w)
	}
	if i := firstByteDiff(gotT, wantT); i >= 0 {
		t.Fatalf("w=%d alpha=%v: top row pixel %d = %x, want %x", w, c.topA != nil, i/4, gotT[i&^3:i&^3+4], wantT[i&^3:i&^3+4])
	}
	if c.botY != nil {
		if i := firstByteDiff(gotB, wantB); i >= 0 {
			t.Fatalf("w=%d alpha=%v: bottom row pixel %d = %x, want %x", w, c.botA != nil, i/4, gotB[i&^3:i&^3+4], wantB[i&^3:i&^3+4])
		}
	}
}

func firstByteDiff(a, b []byte) int {
	if bytes.Equal(a, b) {
		return -1
	}
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return len(a)
}

func testWidths() []int {
	var ws []int
	for w := 1; w <= 64; w++ {
		ws = append(ws, w)
	}
	return append(ws, 65, 66, 79, 80, 81, 100, 127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025, 3839, 3840, 4097)
}

func TestUpsamplePairMatchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	trials := 24
	if testing.Short() {
		trials = 8
	}
	for _, w := range testWidths() {
		for i := 0; i < trials; i++ {
			newPairCase(r, w, i&1 != 0, i%3 == 0, (i/2)%4).check(t)
		}
	}
}

type sampleCase struct {
	w              int
	ys, us, vs, as []byte
}

func newSampleCase(r *rand.Rand, w int, alpha bool, mode int) sampleCase {
	uvW := (w + 1) >> 1
	c := sampleCase{w: w, ys: make([]byte, w), us: make([]byte, uvW), vs: make([]byte, uvW)}
	fillRow(r, c.ys, mode)
	fillRow(r, c.us, mode)
	fillRow(r, c.vs, mode)
	if alpha {
		c.as = make([]byte, w)
		fillRow(r, c.as, mode)
	}
	return c
}

func (c sampleCase) check(t testing.TB) {
	t.Helper()
	want := make([]byte, 4*c.w)
	sampleRowGo(c.ys, c.us, c.vs, c.as, want, c.w)
	got, ok := guarded(4 * c.w)
	sampleRow(c.ys, c.us, c.vs, c.as, got, c.w)
	if !ok() {
		t.Fatalf("w=%d: write outside the destination row", c.w)
	}
	if i := firstByteDiff(got, want); i >= 0 {
		t.Fatalf("w=%d alpha=%v: pixel %d = %x, want %x", c.w, c.as != nil, i/4, got[i&^3:i&^3+4], want[i&^3:i&^3+4])
	}
}

func TestSampleRowMatchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, w := range testWidths() {
		for i := 0; i < 16; i++ {
			newSampleCase(r, w, i&1 != 0, (i/2)%4).check(t)
		}
	}
}

// TestYUVToRGBExhaustive converts all 2^24 (y, u, v) triples with the
// point-sampling converter and checks them against yuvPix (libwebp's
// VP8YuvToRgb). Row u holds, for every v, all 256 luma values; the second
// pass shifts the rows by 8 pixels so that every triple also lands in the
// other half of a 16-pixel NEON block.
func TestYUVToRGBExhaustive(t *testing.T) {
	const w = 256 * 256
	ys := make([]byte, w+8)
	us := make([]byte, (w+8)/2)
	vs := make([]byte, (w+8)/2)
	dst := make([]byte, 4*(w+8))
	for _, shift := range []int{0, 8} {
		for x := 0; x < w; x++ {
			ys[shift+x] = byte(x)
			vs[(shift+x)/2] = byte(x >> 8)
		}
		for u := 0; u < 256; u++ {
			for i := range us {
				us[i] = byte(u)
			}
			sampleRow(ys, us, vs, nil, dst, w+8)
			for x := 0; x < w; x++ {
				got := binary.LittleEndian.Uint32(dst[4*(shift+x):])
				want := yuvPix(yuvTab, byte(x), byte(u), byte(x>>8)) | 0xff000000
				if got != want {
					t.Fatalf("y=%d u=%d v=%d: got %08x, want %08x", x&255, u, x>>8, got, want)
				}
			}
		}
	}
}

// TestYUVGuardPages runs the converters on rows that end right before (or
// start right after) an inaccessible page: an out-of-bounds access faults.
func TestYUVGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, w := range []int{16, 17, 18, 31, 32, 33, 34, 47, 48, 100, 1023, 1024} {
		for _, atEnd := range []bool{true, false} {
			uvW := (w + 1) >> 1
			g := func(src []byte) []byte {
				b := guardedBytes(t, len(src), atEnd)
				copy(b, src)
				return b
			}
			c := newPairCase(r, w, true, false, 0)
			want := make([]byte, 4*w)
			wantB := make([]byte, 4*w)
			upsamplePairGo(c.topY, c.botY, c.topU, c.topV, c.curU, c.curV, want, wantB, c.topA, c.botA, make([]uint32, w), make([]uint32, w), w)
			gc := pairCase{w: w, topY: g(c.topY), botY: g(c.botY), topU: g(c.topU), topV: g(c.topV), curU: g(c.curU), curV: g(c.curV), topA: g(c.topA), botA: g(c.botA)}
			gotT, gotB := guardedBytes(t, 4*w, atEnd), guardedBytes(t, 4*w, atEnd)
			upsamplePair(gc.topY, gc.botY, gc.topU, gc.topV, gc.curU, gc.curV, gotT, gotB, gc.topA, gc.botA, make([]uint32, w), make([]uint32, w), w)
			if !bytes.Equal(gotT, want) || !bytes.Equal(gotB, wantB) {
				t.Fatalf("upsamplePair w=%d atEnd=%v differs", w, atEnd)
			}

			s := newSampleCase(r, w, true, 0)
			want = make([]byte, 4*w)
			sampleRowGo(s.ys, s.us, s.vs, s.as, want, w)
			got := guardedBytes(t, 4*w, atEnd)
			sampleRow(g(s.ys), g(s.us[:uvW]), g(s.vs[:uvW]), g(s.as), got, w)
			if !bytes.Equal(got, want) {
				t.Fatalf("sampleRow w=%d atEnd=%v differs", w, atEnd)
			}
		}
	}
}

// FuzzUpsamplePair compares the dispatched fancy upsampler with the Go
// reference on fuzzed rows.
func FuzzUpsamplePair(f *testing.F) {
	f.Add([]byte{1, 2, 3, 250, 0, 255}, uint16(17), uint8(0))
	f.Add([]byte{255, 0}, uint16(64), uint8(3))
	f.Add([]byte{9}, uint16(1), uint8(1))
	f.Fuzz(func(t *testing.T, seed []byte, w16 uint16, flags uint8) {
		w := 1 + int(w16)%600
		if len(seed) == 0 {
			seed = []byte{0}
		}
		k := 0
		mk := func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = seed[k%len(seed)] + byte(k/len(seed))*byte(flags|1)
				k++
			}
			return b
		}
		uvW := (w + 1) >> 1
		c := pairCase{w: w, topY: mk(w), topU: mk(uvW), topV: mk(uvW), curU: mk(uvW), curV: mk(uvW)}
		if flags&1 == 0 {
			c.botY = mk(w)
		}
		if flags&2 != 0 {
			c.topA = mk(w)
			if c.botY != nil {
				c.botA = mk(w)
			}
		}
		c.check(t)
	})
}

// FuzzSampleRow compares the dispatched point-sampling converter with the
// Go reference on fuzzed rows.
func FuzzSampleRow(f *testing.F) {
	f.Add([]byte{1, 2, 3, 250, 0, 255}, uint16(17), uint8(0))
	f.Add([]byte{255, 0}, uint16(64), uint8(2))
	f.Fuzz(func(t *testing.T, seed []byte, w16 uint16, flags uint8) {
		w := 1 + int(w16)%600
		if len(seed) == 0 {
			seed = []byte{0}
		}
		k := 0
		mk := func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = seed[k%len(seed)] + byte(k/len(seed))*byte(flags|1)
				k++
			}
			return b
		}
		uvW := (w + 1) >> 1
		c := sampleCase{w: w, ys: mk(w), us: mk(uvW), vs: mk(uvW)}
		if flags&2 != 0 {
			c.as = mk(w)
		}
		c.check(t)
	})
}

// Kernel benchmarks: the dispatched converters (NEON on arm64) and the Go
// reference on the same 1024-pixel rows.

func benchPairCase() pairCase {
	return newPairCase(rand.New(rand.NewPCG(7, 8)), 1024, false, false, 0)
}

func BenchmarkUpsamplePairKernel(b *testing.B) {
	c := benchPairCase()
	dt, db := make([]byte, 4*c.w), make([]byte, 4*c.w)
	uvT, uvB := make([]uint32, c.w), make([]uint32, c.w)
	b.SetBytes(2 * int64(c.w))
	for b.Loop() {
		upsamplePair(c.topY, c.botY, c.topU, c.topV, c.curU, c.curV, dt, db, nil, nil, uvT, uvB, c.w)
	}
}

func BenchmarkUpsamplePairGo(b *testing.B) {
	c := benchPairCase()
	dt, db := make([]byte, 4*c.w), make([]byte, 4*c.w)
	uvT, uvB := make([]uint32, c.w), make([]uint32, c.w)
	b.SetBytes(2 * int64(c.w))
	for b.Loop() {
		upsamplePairGo(c.topY, c.botY, c.topU, c.topV, c.curU, c.curV, dt, db, nil, nil, uvT, uvB, c.w)
	}
}

func BenchmarkSampleRowKernel(b *testing.B) {
	c := newSampleCase(rand.New(rand.NewPCG(7, 8)), 1024, true, 0)
	dst := make([]byte, 4*c.w)
	b.SetBytes(int64(c.w))
	for b.Loop() {
		sampleRow(c.ys, c.us, c.vs, c.as, dst, c.w)
	}
}

func BenchmarkSampleRowGo(b *testing.B) {
	c := newSampleCase(rand.New(rand.NewPCG(7, 8)), 1024, true, 0)
	dst := make([]byte, 4*c.w)
	b.SetBytes(int64(c.w))
	for b.Loop() {
		sampleRowGo(c.ys, c.us, c.vs, c.as, dst, c.w)
	}
}
