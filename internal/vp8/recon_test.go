package vp8

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"
)

// Differential tests of the macroblock reconstruction entry points (reconI4,
// reconI16, reconUV: the NEON kernels on arm64, the Go reference elsewhere
// and with -tags purego) against the Go reference of dsp.go. Every test
// compares the whole work buffer after the call, so stray writes anywhere in
// it are caught too.

// reconCase is one random macroblock: work buffer, coefficients, non-zero
// bits and modes.
type reconCase struct {
	b      [yuvSize]byte
	coeffs [384]int16
	bits   uint32
	uvBits uint32
	modes  [16]uint8
	mode   int
	uvMode int
}

// fillPixels fills p with one of several pixel distributions, including the
// extremes that exercise clipping.
func fillPixels(r *rand.Rand, p []byte) {
	switch r.IntN(6) {
	case 0, 1: // uniform
		for i := range p {
			p[i] = byte(r.Uint32())
		}
	case 2: // extremes only
		for i := range p {
			p[i] = []byte{0, 1, 127, 128, 254, 255}[r.IntN(6)]
		}
	case 3: // smooth around a level
		base := r.IntN(256)
		for i := range p {
			p[i] = clip8(base + r.IntN(9) - 4)
		}
	case 4: // two levels
		lo, hi := byte(r.IntN(256)), byte(r.IntN(256))
		for i := range p {
			if r.IntN(2) == 0 {
				p[i] = lo
			} else {
				p[i] = hi
			}
		}
	default: // constant
		v := byte(r.IntN(256))
		for i := range p {
			p[i] = v
		}
	}
}

// randCoeff returns one coefficient from a distribution chosen by kind.
func randCoeff(r *rand.Rand, kind int) int16 {
	switch kind {
	case 0: // full int16 range
		return int16(r.Uint32())
	case 1: // what a valid encoder produces
		return int16(r.IntN(4096) - 2048)
	case 2: // small
		return int16(r.IntN(129) - 64)
	case 3: // extremes
		return []int16{-32768, 32767, -32767, 0, 1, -1, 2047, -2048, 4095, -4096}[r.IntN(10)]
	default: // sparse
		if r.IntN(4) != 0 {
			return 0
		}
		return int16(r.IntN(2048) - 1024)
	}
}

// fillBlock fills a 4x4 coefficient block so that its transform code is
// honest (code 0: zero, 1: DC only, 2: in[0], in[1], in[4] only, 3: any),
// or, with garbage set, with arbitrary content whatever the code.
func fillBlock(r *rand.Rand, blk []int16, code uint32, garbage bool) {
	kind := r.IntN(5)
	for i := range blk[:16] {
		blk[i] = 0
		if garbage || code == 3 || i == 0 && code >= 1 || (i == 1 || i == 4) && code == 2 {
			blk[i] = randCoeff(r, kind)
		}
	}
}

func randReconCase(r *rand.Rand, c *reconCase) {
	fillPixels(r, c.b[:])
	if r.IntN(8) == 0 {
		// Different statistics for the edges than for the block interiors.
		fillPixels(r, c.b[:const8x8Top])
	}
	garbage := r.IntN(8) == 0
	c.bits = r.Uint32()
	switch r.IntN(6) {
	case 0:
		c.bits = 0
	case 1:
		c.bits &= 0x55555555 // DC only
	case 2:
		c.bits |= 0xaaaaaaaa // AC3 or full everywhere
	}
	for n := 0; n < 16; n++ {
		fillBlock(r, c.coeffs[n*16:], c.bits>>(30-2*n)&3, garbage)
	}
	c.uvBits = r.Uint32() & 0xffff
	switch r.IntN(4) {
	case 0:
		c.uvBits = 0
	case 1:
		c.uvBits &= 0x5555
	}
	for n := 0; n < 8; n++ {
		code := c.uvBits >> (8*(n/4) + 6 - 2*(n%4)) & 3
		fillBlock(r, c.coeffs[256+n*16:], code, garbage)
	}
	for i := range c.modes {
		c.modes[i] = uint8(r.IntN(10))
		if r.IntN(64) == 0 {
			c.modes[i] = uint8(r.Uint32()) // out of range: default case
		}
	}
	c.mode = r.IntN(7)
	c.uvMode = r.IntN(7)
	if r.IntN(64) == 0 {
		c.mode = r.IntN(1000) - 500
		c.uvMode = r.IntN(1000) - 500
	}
}

// const8x8Top bounds the top context rows of the work buffer (luma row -1
// starts at yOff-bps).
const const8x8Top = yOff

func diffBuffers(t *testing.T, what string, got, want *[yuvSize]byte) bool {
	t.Helper()
	if bytes.Equal(got[:], want[:]) {
		return true
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s: first difference at b[%d] (row %d, col %d from yOff): got %d, want %d",
				what, i, (i-yOff+bps*2)/bps-2, (i-yOff+bps*2)%bps, got[i], want[i])
			break
		}
	}
	return false
}

func checkI4(t *testing.T, c *reconCase) bool {
	t.Helper()
	got, want := c.b, c.b
	coeffs := c.coeffs
	reconI4(&got, &coeffs, c.bits, &c.modes)
	reconI4Ref(&want, &c.coeffs, c.bits, &c.modes)
	if coeffs != c.coeffs {
		t.Errorf("reconI4 modified the coefficients")
		return false
	}
	return diffBuffers(t, fmt.Sprintf("reconI4 modes=%v bits=%#x", c.modes, c.bits), &got, &want)
}

func checkI16(t *testing.T, c *reconCase) bool {
	t.Helper()
	got, want := c.b, c.b
	coeffs := c.coeffs
	reconI16(&got, &coeffs, c.bits, c.mode)
	reconI16Ref(&want, &c.coeffs, c.bits, c.mode)
	if coeffs != c.coeffs {
		t.Errorf("reconI16 modified the coefficients")
		return false
	}
	return diffBuffers(t, fmt.Sprintf("reconI16 mode=%d bits=%#x", c.mode, c.bits), &got, &want)
}

func checkUV(t *testing.T, c *reconCase) bool {
	t.Helper()
	got, want := c.b, c.b
	coeffs := c.coeffs
	reconUV(&got, &coeffs, c.uvBits, c.uvMode)
	reconUVRef(&want, &c.coeffs, c.uvBits, c.uvMode)
	if coeffs != c.coeffs {
		t.Errorf("reconUV modified the coefficients")
		return false
	}
	return diffBuffers(t, fmt.Sprintf("reconUV mode=%d bits=%#x", c.uvMode, c.uvBits), &got, &want)
}

func reconIters(t *testing.T) int {
	if testing.Short() {
		return 20000
	}
	return 250000
}

func TestReconRandom(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var c reconCase
	n := reconIters(t)
	for i := 0; i < n; i++ {
		randReconCase(r, &c)
		if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
			t.Fatalf("iteration %d", i)
		}
	}
}

// TestReconI4Modes checks every 4x4 mode in every sub-block position with
// random edges, including the top-right samples (replicated below the top
// row as reconstructMB does, or arbitrary).
func TestReconI4Modes(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	var c reconCase
	n := reconIters(t) / 10
	for mode := 0; mode < 10; mode++ {
		for i := 0; i < n; i++ {
			randReconCase(r, &c)
			for k := range c.modes {
				c.modes[k] = uint8(mode)
				if r.IntN(4) == 0 {
					c.modes[k] = uint8(r.IntN(10))
				}
			}
			if r.IntN(2) == 0 {
				tr := yOff - bps + 16
				for k := 1; k <= 3; k++ {
					copy(c.b[tr+4*k*bps:][:4], c.b[tr:][:4])
				}
			}
			if r.IntN(4) == 0 {
				c.bits = 0 // prediction only
			}
			if !checkI4(t, &c) {
				t.Fatalf("mode %d iteration %d", mode, i)
			}
		}
	}
}

// TestReconTrueMotionExhaustive checks the TM clipping for every (top, left,
// top-left) sample triple in the 16x16 and chroma predictors.
func TestReconTrueMotionExhaustive(t *testing.T) {
	var c reconCase
	for x := 0; x < 256; x++ {
		if testing.Short() && x%17 != 0 {
			continue
		}
		for blk := 0; blk < 16; blk++ { // top values 16*blk + i
			for half := 0; half < 16; half++ { // left values 16*half + j
				for i := 0; i < 16; i++ {
					c.b[yOff-bps+i] = byte(16*blk + i)
					c.b[yOff+i*bps-1] = byte(16*half + i)
				}
				for i := 0; i < 8; i++ {
					c.b[uOff-bps+i] = byte(16*blk + i)
					c.b[vOff-bps+i] = byte(16*blk + 8 + i)
					c.b[uOff+i*bps-1] = byte(16*half + i)
					c.b[vOff+i*bps-1] = byte(16*half + 8 + i)
				}
				c.b[yOff-bps-1], c.b[uOff-bps-1], c.b[vOff-bps-1] = byte(x), byte(x), byte(x)
				c.bits, c.uvBits, c.mode, c.uvMode = 0, 0, bTMPred, bTMPred
				if !checkI16(t, &c) || !checkUV(t, &c) {
					t.Fatalf("x=%d blk=%d half=%d", x, blk, half)
				}
			}
		}
	}
}

// TestReconTransformFullRange runs the inverse DCT on every value of every
// coefficient position (all other coefficients zero), and on random pairs
// of extreme coefficients, against several prediction levels.
func TestReconTransformFullRange(t *testing.T) {
	var c reconCase
	step := 1
	if testing.Short() {
		step = 61
	}
	levels := []byte{0, 1, 128, 254, 255}
	for pos := 0; pos < 16; pos++ {
		for v0 := -32768; v0 < 32768; v0 += 16 * step {
			c.coeffs = [384]int16{}
			for n := 0; n < 16; n++ {
				c.coeffs[n*16+pos] = int16(v0 + n*step)
			}
			lvl := levels[(v0>>4)&0xffff%len(levels)]
			for i := range c.b {
				c.b[i] = lvl
			}
			c.bits = 0xffffffff
			c.mode = bVEPred
			if !checkI16(t, &c) {
				t.Fatalf("pos %d v0 %d", pos, v0)
			}
		}
	}
	r := rand.New(rand.NewPCG(5, 6))
	n := reconIters(t) / 5
	for i := 0; i < n; i++ {
		c.coeffs = [384]int16{}
		for k := 0; k < 384; k++ {
			if r.IntN(3) == 0 {
				c.coeffs[k] = randCoeff(r, 3)
			}
		}
		fillPixels(r, c.b[:])
		c.bits, c.uvBits = 0xffffffff, 0xffff
		c.mode, c.uvMode = r.IntN(7), r.IntN(7)
		if !checkI16(t, &c) || !checkUV(t, &c) {
			t.Fatalf("iteration %d", i)
		}
	}
}

// reconBlockRef is reconBlock with the reference kernels.
func reconBlockRef(b *[yuvSize]byte, mbX, mbY int, blk *mbBlock) {
	if blk.isI4x4 {
		reconI4Ref(b, &blk.coeffs, blk.nonZeroY, &blk.imodes)
	} else {
		reconI16Ref(b, &blk.coeffs, blk.nonZeroY, checkMode(mbX, mbY, blk.imodes[0]))
	}
	reconUVRef(b, &blk.coeffs, blk.nonZeroUV, checkMode(mbX, mbY, blk.uvmode))
}

// TestReconDecoderStates reconstructs every macroblock of the test images
// with both implementations, starting from the decoder's own work buffer
// state, so the edge handling (initRowEdges, top-right replication) is the
// real one.
func TestReconDecoderStates(t *testing.T) {
	names := []string{"m4_q75.webp", "webp-logo-lossy.webp", "100x150_lossy.webp",
		"grad_250x199_o9.webp", "mix_250x199_o4.webp", "noise_129x65_o7.webp", "mix_64x48_p1.webp"}
	for _, name := range names {
		vp8 := loadVP8(t, name)
		d := new(decoder)
		if err := d.parseHeaders(vp8); err != nil {
			t.Fatal(err)
		}
		d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
		d.outW = d.width
		d.enterCritical()
		d.initFrame()
		blk := &d.blk
		n := 0
		for mbY := 0; mbY < d.mbH; mbY++ {
			tokenBR := &d.parts[mbY&d.numPartsMinusOne]
			d.startParseRow()
			d.initRowEdges(mbY)
			for mbX := 0; mbX < d.mbW; mbX++ {
				if err := d.parseMB(mbX, blk, tokenBR); err != nil {
					t.Fatal(err)
				}
				// The steps of reconstructMB, with both kernel sets run on
				// the same work buffer state.
				d.loadMBEdges(mbX, mbY, blk.isI4x4)
				want := d.yuvB
				reconBlockRef(&want, mbX, mbY, blk)
				reconBlock(&d.yuvB, mbX, mbY, blk)
				if !diffBuffers(t, fmt.Sprintf("%s mb (%d,%d) i4=%v", name, mbX, mbY, blk.isI4x4), &d.yuvB, &want) {
					t.FailNow()
				}
				d.storeMB(mbX, mbY)
				n++
			}
		}
		t.Logf("%s: %d macroblocks", name, n)
	}
}

// reconState is the input of reconBlock for one macroblock of a real image.
type reconState struct {
	b        [yuvSize]byte
	blk      mbBlock
	mbX, mbY int
}

// harvestRecon records the reconBlock inputs of every macroblock of the
// image.
func harvestRecon(tb testing.TB, name string) []reconState {
	vp8 := loadVP8(tb, name)
	d := new(decoder)
	if err := d.parseHeaders(vp8); err != nil {
		tb.Fatal(err)
	}
	d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
	d.outW = d.width
	d.enterCritical()
	d.initFrame()
	var states []reconState
	for mbY := 0; mbY < d.mbH; mbY++ {
		tokenBR := &d.parts[mbY&d.numPartsMinusOne]
		d.startParseRow()
		d.initRowEdges(mbY)
		for mbX := 0; mbX < d.mbW; mbX++ {
			if err := d.parseMB(mbX, &d.blk, tokenBR); err != nil {
				tb.Fatal(err)
			}
			d.loadMBEdges(mbX, mbY, d.blk.isI4x4)
			states = append(states, reconState{b: d.yuvB, blk: d.blk, mbX: mbX, mbY: mbY})
			reconBlock(&d.yuvB, mbX, mbY, &d.blk)
			d.storeMB(mbX, mbY)
		}
	}
	return states
}

// BenchmarkRecon replays the macroblocks of an image through the bound
// kernels ("kernel") and the Go reference ("ref"), in ns per macroblock:
// "mb" is reconBlock, the others a single kernel on the macroblocks that use
// it. The work buffer is reset (an 832-byte copy) before each call.
func BenchmarkRecon(b *testing.B) {
	for _, img := range []string{benchM4, benchLogo} {
		states := harvestRecon(b, img)
		var work [yuvSize]byte
		run := func(name string, filter func(*mbBlock) bool, f func(s *reconState)) {
			b.Run(img+"/"+name, func(b *testing.B) {
				var sel []*reconState
				for i := range states {
					if filter(&states[i].blk) {
						sel = append(sel, &states[i])
					}
				}
				if len(sel) == 0 {
					b.Skip("no macroblock of this kind")
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					s := sel[i%len(sel)]
					work = s.b
					f(s)
				}
			})
		}
		all := func(*mbBlock) bool { return true }
		i4 := func(blk *mbBlock) bool { return blk.isI4x4 }
		i16 := func(blk *mbBlock) bool { return !blk.isI4x4 }
		run("mb/kernel", all, func(s *reconState) { reconBlock(&work, s.mbX, s.mbY, &s.blk) })
		run("mb/ref", all, func(s *reconState) { reconBlockRef(&work, s.mbX, s.mbY, &s.blk) })
		run("i4/kernel", i4, func(s *reconState) { reconI4(&work, &s.blk.coeffs, s.blk.nonZeroY, &s.blk.imodes) })
		run("i4/ref", i4, func(s *reconState) { reconI4Ref(&work, &s.blk.coeffs, s.blk.nonZeroY, &s.blk.imodes) })
		run("i16/kernel", i16, func(s *reconState) {
			reconI16(&work, &s.blk.coeffs, s.blk.nonZeroY, checkMode(s.mbX, s.mbY, s.blk.imodes[0]))
		})
		run("i16/ref", i16, func(s *reconState) {
			reconI16Ref(&work, &s.blk.coeffs, s.blk.nonZeroY, checkMode(s.mbX, s.mbY, s.blk.imodes[0]))
		})
		run("uv/kernel", all, func(s *reconState) {
			reconUV(&work, &s.blk.coeffs, s.blk.nonZeroUV, checkMode(s.mbX, s.mbY, s.blk.uvmode))
		})
		run("uv/ref", all, func(s *reconState) {
			reconUVRef(&work, &s.blk.coeffs, s.blk.nonZeroUV, checkMode(s.mbX, s.mbY, s.blk.uvmode))
		})
		run("copy", all, func(s *reconState) {})
	}
}

// FuzzRecon compares the reconstruction entry points with the reference on
// arbitrary work buffers, coefficients, non-zero bits and modes. The input
// bytes seed the buffers (repeated as needed); the first bytes pick the
// bits and modes.
func FuzzRecon(f *testing.F) {
	f.Add([]byte{0})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x80, 0x7f})
	f.Add(bytes.Repeat([]byte{0x00, 0x80, 0xff, 0x7f, 0x01}, 300))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		at := func(i int) byte { return data[i%len(data)] }
		var c reconCase
		c.bits = uint32(at(0)) | uint32(at(1))<<8 | uint32(at(2))<<16 | uint32(at(3))<<24
		c.uvBits = uint32(at(4)) | uint32(at(5))<<8
		c.mode = int(int8(at(6)))
		c.uvMode = int(int8(at(7)))
		for i := range c.modes {
			c.modes[i] = at(8 + i)
		}
		for i := range c.b {
			c.b[i] = at(24 + i)
		}
		for i := range c.coeffs {
			c.coeffs[i] = int16(uint16(at(24+yuvSize+2*i)) | uint16(at(25+yuvSize+2*i))<<8)
		}
		if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
			t.FailNow()
		}
	})
}
