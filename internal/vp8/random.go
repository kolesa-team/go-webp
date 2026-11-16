package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): utils/random_utils.[ch] and
// the dithering code of dec/frame_dec.c and dsp/dec.c.

const (
	randomTableSize  = 55
	randomDitherFix  = 8 // fixed-point precision for dithering
	ditherAmpBits    = 7
	ditherAmpCenter  = 1 << ditherAmpBits
	ditherDescale    = 4
	ditherRounder    = 1 << (ditherDescale - 1)
	minDitherAmp     = 4
	ditherAmpTabSize = 12
)

// kQuantToDitherAmp maps the chroma AC quantizer to a dithering amplitude.
var kQuantToDitherAmp = [ditherAmpTabSize]uint8{8, 7, 6, 4, 4, 2, 2, 2, 1, 1, 1, 1}

// kRandomTable is libwebp's 31-bit subtractive generator seed table.
var kRandomTable = [randomTableSize]uint32{
	0x0de15230, 0x03b31886, 0x775faccb, 0x1c88626a, 0x68385c55, 0x14b3b828,
	0x4a85fef8, 0x49ddb84b, 0x64fcf397, 0x5c550289, 0x4a290000, 0x0d7ec1da,
	0x5940b7ab, 0x5492577d, 0x4e19ca72, 0x38d38c69, 0x0c01ee65, 0x32a1755f,
	0x5437f652, 0x5abb2c32, 0x0faa57b1, 0x73f533e7, 0x685feeda, 0x7563cce2,
	0x6e990e83, 0x4730a7ed, 0x4fc0d9c6, 0x496b153c, 0x4f1403fa, 0x541afb0c,
	0x73990b32, 0x26d7cb1c, 0x6fcc3706, 0x2cbb77d8, 0x75762f2a, 0x6425ccdd,
	0x24b35461, 0x0a7d8715, 0x220414a8, 0x141ebf67, 0x56b41583, 0x73e502e3,
	0x44cab16f, 0x28264d42, 0x73baaefb, 0x0a50ebed, 0x1d6ab6fb, 0x0d3ad40b,
	0x35db3b68, 0x2b081e83, 0x77ce6b95, 0x5181e5f0, 0x78853bbc, 0x009f9494,
	0x27e5ed3c,
}

// vp8Random is libwebp's VP8Random.
type vp8Random struct {
	index1, index2 int
	tab            [randomTableSize]uint32
}

func (rg *vp8Random) init() {
	rg.tab = kRandomTable
	rg.index1 = 0
	rg.index2 = 31
}

// bits2 returns a centered random value of numBits bits scaled by amp.
func (rg *vp8Random) bits2(numBits, amp int) int {
	diff := int64(rg.tab[rg.index1]) - int64(rg.tab[rg.index2])
	if diff < 0 {
		diff += 1 << 31
	}
	rg.tab[rg.index1] = uint32(diff)
	if rg.index1++; rg.index1 == randomTableSize {
		rg.index1 = 0
	}
	if rg.index2++; rg.index2 == randomTableSize {
		rg.index2 = 0
	}
	// sign-extend, 0-center
	v := int(int32(uint32(diff)<<1) >> (32 - numBits))
	v = (v * amp) >> randomDitherFix // restrict range
	v += 1 << (numBits - 1)          // shift back to 0.5-center
	return v
}

// dither8x8 adds random noise of amplitude amp to an 8x8 block.
func (rg *vp8Random) dither8x8(buf []byte, o, stride, amp int) {
	var dither [64]uint8
	for i := range dither {
		dither[i] = uint8(rg.bits2(ditherAmpBits+1, amp))
	}
	for j := 0; j < 8; j++ {
		row := buf[o+j*stride : o+j*stride+8]
		for i := range row {
			delta0 := int(dither[j*8+i]) - ditherAmpCenter
			delta1 := (delta0 + ditherRounder) >> ditherDescale
			row[i] = clip1(int(row[i]) + delta1)
		}
	}
}

// initDithering mirrors libwebp's VP8InitDithering.
func (d *decoder) initDithering(strength int) {
	d.dither = false
	if strength <= 0 {
		return
	}
	const maxAmp = (1 << randomDitherFix) - 1
	f := strength * maxAmp / 100
	if strength > 100 {
		f = maxAmp
	}
	if f <= 0 {
		return
	}
	allAmp := 0
	for s := range d.dqm {
		dqm := &d.dqm[s]
		if dqm.uvQuant < ditherAmpTabSize {
			idx := max(dqm.uvQuant, 0)
			dqm.dither = (f * int(kQuantToDitherAmp[idx])) >> 3
		}
		allAmp |= dqm.dither
	}
	if allAmp != 0 {
		d.rng.init()
		d.dither = true
	}
}
