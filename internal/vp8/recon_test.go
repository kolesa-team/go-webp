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
	forEachReconKernel(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(1, 2))
		var c reconCase
		n := reconIters(t)
		for i := 0; i < n; i++ {
			randReconCase(r, &c)
			if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
				t.Fatalf("iteration %d", i)
			}
		}
	})
}

// TestReconBoundedCoefficients is TestReconRandom with coefficients mostly
// within the range of the 16-bit transforms of the x86 kernels, where they
// do not fall back to the reference, including the bound itself.
func TestReconBoundedCoefficients(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(41, 42))
		var c reconCase
		n := reconIters(t)
		for i := 0; i < n; i++ {
			randReconCase(r, &c)
			for k := range c.coeffs {
				if c.coeffs[k] != 0 || r.IntN(8) == 0 {
					switch r.IntN(8) {
					case 0:
						c.coeffs[k] = []int16{4095, -4095, 4094, -4094, 4096, -4096}[r.IntN(6)]
					case 1, 2:
						c.coeffs[k] = int16(r.IntN(8191) - 4095)
					default:
						c.coeffs[k] = int16(r.IntN(4096) - 2048)
					}
				}
			}
			if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
				t.Fatalf("iteration %d", i)
			}
		}
	})
}

// TestReconI4Modes checks every 4x4 mode in every sub-block position with
// random edges, including the top-right samples (replicated below the top
// row as reconstructMB does, or arbitrary).
func TestReconI4Modes(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
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
	})
}

// TestReconTrueMotionExhaustive checks the TM clipping for every (top, left,
// top-left) sample triple in the 16x16 and chroma predictors.
func TestReconTrueMotionExhaustive(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
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
	})
}

// TestReconTransformFullRange runs the inverse DCT on every value of every
// coefficient position (all other coefficients zero), and on random pairs
// of extreme coefficients, against several prediction levels.
func TestReconTransformFullRange(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
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
	})
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
	forEachReconLevel(t, testReconDecoderStates)
}

func testReconDecoderStates(t *testing.T) {
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
// kernels (at each SIMD level on amd64: "go", "sse41", "avx2", "avx512";
// "kernel" elsewhere) and the Go reference ("ref"), in ns per macroblock:
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
		eachReconLevel(false, func(level string) {
			run("mb/"+level, all, func(s *reconState) { reconBlock(&work, s.mbX, s.mbY, &s.blk) })
		})
		run("mb/ref", all, func(s *reconState) { reconBlockRef(&work, s.mbX, s.mbY, &s.blk) })
		eachReconLevel(false, func(level string) {
			run("i4/"+level, i4, func(s *reconState) { reconI4(&work, &s.blk.coeffs, s.blk.nonZeroY, &s.blk.imodes) })
		})
		run("i4/ref", i4, func(s *reconState) { reconI4Ref(&work, &s.blk.coeffs, s.blk.nonZeroY, &s.blk.imodes) })
		eachReconLevel(false, func(level string) {
			run("i16/"+level, i16, func(s *reconState) {
				reconI16(&work, &s.blk.coeffs, s.blk.nonZeroY, checkMode(s.mbX, s.mbY, s.blk.imodes[0]))
			})
		})
		run("i16/ref", i16, func(s *reconState) {
			reconI16Ref(&work, &s.blk.coeffs, s.blk.nonZeroY, checkMode(s.mbX, s.mbY, s.blk.imodes[0]))
		})
		eachReconLevel(false, func(level string) {
			run("uv/"+level, all, func(s *reconState) {
				reconUV(&work, &s.blk.coeffs, s.blk.nonZeroUV, checkMode(s.mbX, s.mbY, s.blk.uvmode))
			})
		})
		run("uv/ref", all, func(s *reconState) {
			reconUVRef(&work, &s.blk.coeffs, s.blk.nonZeroUV, checkMode(s.mbX, s.mbY, s.blk.uvmode))
		})
		run("copy", all, func(s *reconState) {})
	}
}

// BenchmarkReconGlue measures the per-macroblock edge setup and row cache
// copy (kernel and reference) on a 1024-pixel wide cache.
func BenchmarkReconGlue(b *testing.B) {
	var buf [yuvSize]byte
	var top topSamples
	const ys, uvs = 1024, 512
	cy, cu, cv := make([]byte, 16*ys), make([]byte, 8*uvs), make([]byte, 8*uvs)
	flags := uint32(edgeRotate | edgeTop | edgeTopRight | edgeI4)
	eachReconLevel(false, func(level string) {
		b.Run("load/"+level, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				loadEdges(&buf, &top, 0x01020304, flags)
			}
		})
	})
	b.Run("load/ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			loadEdgesRef(&buf, &top, 0x01020304, flags)
		}
	})
	eachReconLevel(false, func(level string) {
		b.Run("store/"+level, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				o := i & 63 * 16
				storeMBRows(&buf, &top, cy[o:], cu[o/2:], cv[o/2:], ys, uvs)
			}
		})
	})
	b.Run("store/ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			o := i & 63 * 16
			storeMBRowsRef(&buf, &top, cy[o:], cu[o/2:], cv[o/2:], ys, uvs)
		}
	})
}

// FuzzRecon compares the reconstruction entry points with the reference on
// arbitrary work buffers, coefficients, non-zero bits and modes. The input
// bytes seed the buffers (repeated as needed); the first bytes pick the
// bits and modes, and byte 23 whether the coefficients are cut to 13 bits.
func FuzzRecon(f *testing.F) {
	f.Add([]byte{0})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x80, 0x7f})
	f.Add(bytes.Repeat([]byte{0x00, 0x80, 0xff, 0x7f, 0x01}, 300))
	f.Add(append(bytes.Repeat([]byte{0xff, 0x01, 0x30}, 8)[:23], 1, 0x10, 0xf0, 0x7f, 0x80))
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
			if at(23)&1 != 0 {
				// Within [-4096, 4095], mostly the range the x86 kernels
				// transform without falling back to the reference.
				c.coeffs[i] = c.coeffs[i] << 3 >> 3
			}
		}
		eachReconLevel(false, func(level string) {
			if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
				t.Fatalf("level %s", level)
			}
		})
	})
}

// checkEdges compares loadEdges with loadEdgesRef.
func checkEdges(t *testing.T, b *[yuvSize]byte, top *topSamples, tr, flags uint32) bool {
	t.Helper()
	got, want := *b, *b
	top0 := *top
	loadEdges(&got, top, tr, flags)
	loadEdgesRef(&want, top, tr, flags)
	if *top != top0 {
		t.Errorf("loadEdges modified top")
		return false
	}
	return diffBuffers(t, fmt.Sprintf("loadEdges flags=%#x tr=%#x", flags, tr), &got, &want)
}

// checkStoreRows compares storeMBRows with storeMBRowsRef on a cache of the
// given geometry, with guard bytes around the cache rows.
func checkStoreRows(t *testing.T, r *rand.Rand, b *[yuvSize]byte, withTop bool, ys, uvs int) bool {
	t.Helper()
	ny, nuv := max(15*ys+16, 0), max(7*uvs+8, 0)
	mk := func(n int) []byte {
		s := make([]byte, n+64)
		for i := range s {
			s[i] = byte(r.Uint32())
		}
		return s
	}
	cy, cu, cv := mk(ny), mk(nuv), mk(nuv)
	cy2, cu2, cv2 := bytes.Clone(cy), bytes.Clone(cu), bytes.Clone(cv)
	var top topSamples
	for i := range top.y {
		top.y[i] = byte(r.Uint32())
	}
	top2 := top
	var tp, tp2 *topSamples
	if withTop {
		tp, tp2 = &top, &top2
	}
	b0 := *b
	storeMBRows(b, tp, cy[32:32+ny], cu[32:32+nuv], cv[32:32+nuv], ys, uvs)
	storeMBRowsRef(&b0, tp2, cy2[32:32+ny], cu2[32:32+nuv], cv2[32:32+nuv], ys, uvs)
	if !bytes.Equal(cy, cy2) || !bytes.Equal(cu, cu2) || !bytes.Equal(cv, cv2) || top != top2 || *b != b0 {
		t.Errorf("storeMBRows ys=%d uvs=%d top=%v differs from the reference", ys, uvs, withTop)
		return false
	}
	return true
}

func TestReconEdgesAndStore(t *testing.T) {
	forEachReconLevel(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(9, 10))
		var b [yuvSize]byte
		var top topSamples
		n := reconIters(t) / 10
		for i := 0; i < n; i++ {
			fillPixels(r, b[:])
			fillPixels(r, top.y[:])
			fillPixels(r, top.u[:])
			fillPixels(r, top.v[:])
			if !checkEdges(t, &b, &top, r.Uint32(), uint32(r.IntN(16))) {
				t.Fatalf("iteration %d", i)
			}
			if i%8 == 0 {
				ys, uvs := 16+r.IntN(200), 8+r.IntN(100)
				if r.IntN(4) == 0 {
					ys, uvs = 16, 8 // a single macroblock column
				}
				if !checkStoreRows(t, r, &b, r.IntN(2) == 0, ys, uvs) {
					t.Fatalf("iteration %d", i)
				}
			}
		}
		// Geometries the kernel does not take go to the reference (which
		// may panic on overlapping rows; only the absence of a crash or of
		// a mismatch matters here).
		for _, g := range [][2]int{{0, 0}, {1, 1}, {15, 7}} {
			func() {
				defer func() { _ = recover() }()
				checkStoreRows(t, r, &b, true, g[0], g[1])
			}()
		}
	})
}

// FuzzReconGlue compares loadEdges and storeMBRows with the reference on
// arbitrary buffers, flags and cache geometries.
func FuzzReconGlue(f *testing.F) {
	f.Add([]byte{0})
	f.Add([]byte{0x0f, 0x10, 0x08, 1, 2, 3, 4, 5, 6, 7, 8})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 3 {
			return
		}
		at := func(i int) byte { return data[i%len(data)] }
		var b [yuvSize]byte
		var top topSamples
		for i := range b {
			b[i] = at(8 + i)
		}
		for i := range top.y {
			top.y[i] = at(3 + 7*i)
		}
		for i := range top.u {
			top.u[i], top.v[i] = at(5+3*i), at(6+5*i)
		}
		tr := uint32(at(3)) | uint32(at(4))<<8 | uint32(at(5))<<16 | uint32(at(6))<<24
		eachReconLevel(false, func(level string) {
			if !checkEdges(t, &b, &top, tr, uint32(at(0))) {
				t.Fatalf("level %s", level)
			}
			r := rand.New(rand.NewPCG(uint64(at(1)), uint64(at(2))))
			if !checkStoreRows(t, r, &b, at(0)&0x10 != 0, 16+int(at(1)), 8+int(at(2))) {
				t.Fatalf("level %s", level)
			}
		})
	})
}

// checkWHT compares transformWHT with transformWHTRef, including the
// coefficients it must leave alone.
func checkWHT(t *testing.T, in *[16]int16, out *[384]int16) bool {
	t.Helper()
	got, want := *out, *out
	in0 := *in
	transformWHT(in, &got)
	transformWHTRef(in, &want)
	if *in != in0 {
		t.Errorf("transformWHT modified its input")
		return false
	}
	if got != want {
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("transformWHT(%v): out[%d] = %d, want %d", *in, i, got[i], want[i])
				break
			}
		}
		return false
	}
	return true
}

func TestTransformWHT(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(13, 14))
		var in [16]int16
		var out [384]int16
		for i := range out {
			out[i] = int16(r.Uint32())
		}
		step := 1
		if testing.Short() {
			step = 7
		}
		// Every value at every position (the others zero or extreme).
		for pos := 0; pos < 16; pos++ {
			for v := -32768; v < 32768; v += step {
				in = [16]int16{}
				if v&1 != 0 {
					for k := range in {
						in[k] = randCoeff(r, 3)
					}
				}
				in[pos] = int16(v)
				if !checkWHT(t, &in, &out) {
					t.FailNow()
				}
			}
		}
		for i := 0; i < reconIters(t); i++ {
			kind := r.IntN(5)
			for k := range in {
				in[k] = randCoeff(r, kind)
			}
			if !checkWHT(t, &in, &out) {
				t.FailNow()
			}
		}
	})
}

func FuzzTransformWHT(f *testing.F) {
	f.Add([]byte{0x00, 0x80, 0xff, 0x7f})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		var in [16]int16
		var out [384]int16
		for i := range in {
			in[i] = int16(uint16(data[(2*i)%len(data)]) | uint16(data[(2*i+1)%len(data)])<<8)
		}
		for i := range out {
			out[i] = int16(i * 37)
		}
		eachReconLevel(false, func(level string) {
			if !checkWHT(t, &in, &out) {
				t.Fatalf("level %s", level)
			}
		})
	})
}

func BenchmarkTransformWHT(b *testing.B) {
	in := [16]int16{100, -20, 3, 0, 7, 1, -1, 0, 0, 0, 2, 0, -5, 0, 0, 1}
	var out [384]int16
	eachReconLevel(false, func(level string) {
		b.Run(level, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				transformWHT(&in, &out)
			}
		})
	})
	b.Run("ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			transformWHTRef(&in, &out)
		}
	})
}
