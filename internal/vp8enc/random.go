package vp8enc

// Algorithm adapted from libwebp (BSD-3-Clause): utils/random_utils.[ch]
// (VP8Random) and the dithered RGB->YUV conversion of
// enc/picture_csp_enc.c.

const (
	randomTableSize = 55
	randomDitherFix = 8
)

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
	amp            int
	tab            [randomTableSize]uint32
}

// newRandom mirrors VP8InitRandom(rg, dithering).
func newRandom(dithering float32) *vp8Random {
	rg := &vp8Random{index1: 0, index2: 31, tab: kRandomTable}
	switch {
	case dithering < 0:
		rg.amp = 0
	case dithering > 1:
		rg.amp = 1 << randomDitherFix
	default:
		rg.amp = int(uint32(float32(1<<randomDitherFix) * dithering))
	}
	return rg
}

// bits mirrors VP8RandomBits: a random rounding term of numBits bits,
// centered on 1<<(numBits-1).
func (rg *vp8Random) bits(numBits uint) int32 {
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
	v := int(int32(uint32(diff)<<1) >> (32 - numBits)) // sign-extend, 0-center
	v = (v * rg.amp) >> randomDitherFix                // restrict range
	v += 1 << (numBits - 1)                            // back to 0.5-center
	return int32(v)
}

// DitheringFor returns the RGB->YUV conversion dithering strength libwebp's
// WebPEncode uses for a lossy config: 0 unless Preprocessing bit 1 is set,
// otherwise an amplitude decreasing from 1 to 0.5 as quality grows.
func DitheringFor(cfg *Config) float32 {
	if cfg == nil || cfg.Preprocessing&2 == 0 {
		return 0
	}
	x := cfg.Quality / 100
	x2 := x * x
	return 1 + (0.5-1)*x2*x2
}
