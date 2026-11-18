package vp8

import (
	"bytes"
	"math"
	"math/rand/v2"
	"testing"
)

// Exhaustive and boundary tests of the reconstruction entry points against
// the Go reference, for the parts of the kernels whose arithmetic differs
// from the reference: the 4x4 kernel's masked inverse DCT with its
// saturating add of the positive and negative residual parts, its TM
// predictor built from saturating byte operations, its avg3/avg2 built from
// halving adds, and the row cache copy on unaligned cache rows.

// sameI4, sameI16 and sameUV are the cheap forms of checkI4, checkI16 and
// checkUV for the exhaustive loops; on a mismatch the caller reruns the
// check* function for the report.
func sameI4(c *reconCase) bool {
	got, want := c.b, c.b
	reconI4(&got, &c.coeffs, c.bits, &c.modes)
	reconI4Ref(&want, &c.coeffs, c.bits, &c.modes)
	return got == want
}

func sameI16(c *reconCase) bool {
	got, want := c.b, c.b
	reconI16(&got, &c.coeffs, c.bits, c.mode)
	reconI16Ref(&want, &c.coeffs, c.bits, c.mode)
	return got == want
}

func sameUV(c *reconCase) bool {
	got, want := c.b, c.b
	reconUV(&got, &c.coeffs, c.uvBits, c.uvMode)
	reconUVRef(&want, &c.coeffs, c.uvBits, c.uvMode)
	return got == want
}

// TestReconTransformCodesFullRange runs every int16 value at every
// coefficient position through every transform code of the three kernels
// (none, DC, AC3, full; the chroma DC and full paths), with the masked-out
// positions either zero or extreme, over prediction levels that hit both
// clipping bounds.
func TestReconTransformCodesFullRange(t *testing.T) {
	forEachReconKernel(t, testReconTransformCodesFullRange)
}

func testReconTransformCodesFullRange(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	var c reconCase
	step := 1
	if testing.Short() {
		step = 61
	}
	levels := []byte{0, 1, 127, 128, 254, 255}
	lumaBits := []uint32{0x55555555, 0xaaaaaaaa, 0xffffffff, 0}
	uvBits := []uint32{0x5555, 0xffff, 0xaaaa, 0x1540}
	iter := 0
	for pos := 0; pos < 16; pos++ {
		for v0 := -32768; v0 < 32768; v0 += 24 * step {
			iter++
			for n := 0; n < 24; n++ {
				blk := c.coeffs[n*16 : n*16+16]
				for k := range blk {
					blk[k] = 0
					if iter&1 != 0 {
						blk[k] = randCoeff(r, 3)
					}
				}
				blk[pos] = int16(v0 + n*step)
			}
			if iter&2 != 0 {
				fillPixels(r, c.b[:])
			} else {
				lvl := levels[iter/4%len(levels)]
				for i := range c.b {
					c.b[i] = lvl
				}
			}
			lumaBits[3] = r.Uint32()
			for _, bits := range lumaBits {
				c.bits = bits
				for k := range c.modes {
					c.modes[k] = uint8(r.IntN(10))
				}
				c.mode = r.IntN(7)
				if !sameI4(&c) || !sameI16(&c) {
					checkI4(t, &c)
					checkI16(t, &c)
					t.Fatalf("pos %d v0 %d bits %#x", pos, v0, bits)
				}
			}
			for _, bits := range uvBits {
				c.uvBits = bits
				c.uvMode = r.IntN(7)
				if !sameUV(&c) {
					checkUV(t, &c)
					t.Fatalf("pos %d v0 %d uvBits %#x", pos, v0, bits)
				}
			}
		}
	}
}

// TestReconTransformSaturation drives the inverse DCT outputs past the int16
// range (where the kernels saturate before the final clip): every sign
// pattern of extreme coefficients in a block (+32767 or -32768 at each of
// the 16 positions; for the AC3 code, of in[0], in[1], in[4]), on the
// prediction levels at both clipping bounds. The same patterns at the
// magnitudes around the bound of the x86 kernels' 16-bit transform (4095,
// beyond which they fall back to the reference) test it where its
// intermediate sums come closest to overflowing.
func TestReconTransformSaturation(t *testing.T) {
	forEachReconKernel(t, func(t *testing.T) {
		for _, m := range [][2]int16{{32767, -32768}, {4096, -4096}, {4095, -4095}, {4094, -4094}, {2048, -2048}} {
			testReconTransformSaturation(t, m[0], m[1])
		}
	})
}

func testReconTransformSaturation(t *testing.T, pos, neg int16) {
	var c reconCase
	step := 1
	if testing.Short() {
		step = 37
	}
	levels := []byte{0, 255, 1, 254, 128}
	for p := 0; p < 1<<16; p += step {
		for n := 0; n < 24; n++ {
			q := p ^ n*0x1111 // a different pattern in each block
			for k := 0; k < 16; k++ {
				c.coeffs[n*16+k] = pos
				if q>>k&1 != 0 {
					c.coeffs[n*16+k] = neg
				}
			}
		}
		lvl := levels[p%len(levels)]
		for i := range c.b {
			c.b[i] = lvl
		}
		c.mode, c.uvMode = bVEPred, bVEPred
		for k := range c.modes {
			c.modes[k] = bVEPred
		}
		for _, bits := range []uint32{0xffffffff, 0xaaaaaaaa, 0x55555555} {
			c.bits, c.uvBits = bits, bits&0xffff
			if !sameI4(&c) || !sameI16(&c) || !sameUV(&c) {
				checkI4(t, &c)
				checkI16(t, &c)
				checkUV(t, &c)
				t.Fatalf("magnitude %d/%d pattern %#x bits %#x", pos, neg, p, bits)
			}
		}
	}
}

// TestReconI4TrueMotionExhaustive checks the 4x4 TM predictor (saturating
// byte arithmetic in the kernel) for every (top, left, top-left) triple:
// each call covers four top and four left samples of sub-block 0.
func TestReconI4TrueMotionExhaustive(t *testing.T) {
	forEachReconKernel(t, testReconI4TrueMotionExhaustive)
}

func testReconI4TrueMotionExhaustive(t *testing.T) {
	var c reconCase
	for k := range c.modes {
		c.modes[k] = bTMPred
	}
	for x := 0; x < 256; x++ {
		if testing.Short() && x%17 != 0 {
			continue
		}
		for tq := 0; tq < 64; tq++ {
			for lq := 0; lq < 64; lq++ {
				for i := 0; i < 4; i++ {
					c.b[yOff-bps+i] = byte(4*tq + i)
					c.b[yOff+i*bps-1] = byte(4*lq + i)
				}
				c.b[yOff-bps-1] = byte(x)
				c.bits = 0
				if !sameI4(&c) {
					checkI4(t, &c)
					t.Fatalf("x=%d tq=%d lq=%d", x, tq, lq)
				}
			}
		}
	}
}

// TestReconI4AverageExhaustive checks avg3 (UHADD + URHADD in the kernel)
// for every sample triple through VE, and avg2 (URHADD) for every pair
// through VL: each call places one triple or pair at the start of the top
// window of each of the four sub-blocks of the first row.
func TestReconI4AverageExhaustive(t *testing.T) {
	forEachReconKernel(t, testReconI4AverageExhaustive)
}

func testReconI4AverageExhaustive(t *testing.T) {
	var c reconCase
	const s0 = yOff - bps - 1 // top-left sample of sub-block 0
	for k := range c.modes {
		c.modes[k] = bVEPred
	}
	for a := 0; a < 256; a++ {
		if testing.Short() && a%17 != 0 {
			continue
		}
		for b := 0; b < 256; b++ {
			for cq := 0; cq < 64; cq++ {
				for k := 0; k < 4; k++ {
					c.b[s0+4*k] = byte(a)
					c.b[s0+4*k+1] = byte(b)
					c.b[s0+4*k+2] = byte(4*cq + k)
				}
				c.bits = 0
				if !sameI4(&c) {
					checkI4(t, &c)
					t.Fatalf("avg3 a=%d b=%d cq=%d", a, b, cq)
				}
			}
		}
	}
	for k := range c.modes {
		c.modes[k] = bVLPred
	}
	for a := 0; a < 256; a++ {
		for bq := 0; bq < 64; bq++ {
			for k := 0; k < 4; k++ {
				c.b[s0+4*k+1] = byte(a)
				c.b[s0+4*k+2] = byte(4*bq + k)
			}
			c.bits = 0
			if !sameI4(&c) {
				checkI4(t, &c)
				t.Fatalf("avg2 a=%d bq=%d", a, bq)
			}
		}
	}
}

// TestReconModeBoundaries runs every mode value 0..255 (and int extremes
// for the 16x16 and chroma modes) on random buffers: out-of-range values
// must select the reference's default predictor.
func TestReconModeBoundaries(t *testing.T) {
	forEachReconLevel(t, testReconModeBoundaries)
}

func testReconModeBoundaries(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	var c reconCase
	modes := []int{math.MinInt, math.MinInt32, -256, -1, math.MaxInt32, math.MaxInt}
	for m := 0; m < 256; m++ {
		modes = append(modes, m)
	}
	for _, m := range modes {
		randReconCase(r, &c)
		for k := range c.modes {
			c.modes[k] = uint8(m)
		}
		c.mode, c.uvMode = m, m
		if !checkI4(t, &c) || !checkI16(t, &c) || !checkUV(t, &c) {
			t.Fatalf("mode %d", m)
		}
	}
}

// TestStoreMBRowsUnaligned copies macroblocks into cache rows that start at
// every offset 0..15 from a 16-byte boundary, for every luma stride 16..80
// and chroma stride 8..40, with guard bytes around each row cache.
func TestStoreMBRowsUnaligned(t *testing.T) {
	forEachReconLevel(t, testStoreMBRowsUnaligned)
}

func testStoreMBRowsUnaligned(t *testing.T) {
	r := rand.New(rand.NewPCG(25, 26))
	var b [yuvSize]byte
	backing := func(n int) []byte {
		s := make([]byte, n+64)
		for i := range s {
			s[i] = byte(r.Uint32())
		}
		return s
	}
	for ys := 16; ys <= 80; ys++ {
		for off := 0; off < 16; off++ {
			uvs := 8 + (ys-16)/2
			if ys&1 != 0 {
				uvs = 8 + r.IntN(33)
			}
			ny, nuv := 15*ys+16, 7*uvs+8
			oy, ou, ov := 16+off, 16+(off+5)%16, 16+(off+11)%16
			cy, cu, cv := backing(ny), backing(nuv), backing(nuv)
			cy2, cu2, cv2 := bytes.Clone(cy), bytes.Clone(cu), bytes.Clone(cv)
			fillPixels(r, b[:])
			var top, top2 topSamples
			b0 := b
			storeMBRows(&b, &top, cy[oy:oy+ny], cu[ou:ou+nuv], cv[ov:ov+nuv], ys, uvs)
			storeMBRowsRef(&b0, &top2, cy2[oy:oy+ny], cu2[ou:ou+nuv], cv2[ov:ov+nuv], ys, uvs)
			if !bytes.Equal(cy, cy2) || !bytes.Equal(cu, cu2) || !bytes.Equal(cv, cv2) || top != top2 || b != b0 {
				t.Fatalf("storeMBRows ys=%d uvs=%d offsets %d/%d/%d differs from the reference", ys, uvs, oy, ou, ov)
			}
		}
	}
}
