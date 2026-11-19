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
	"math/rand/v2"
	"testing"
)

// refBitReader is a literal transcription of libwebp 1.6's VP8BitReader
// (utils/bit_reader_utils.[ch] and bit_reader_inl_utils.h with BITS = 56),
// used as the oracle for the bool decoder.
type refBitReader struct {
	value uint64
	rng   uint32 // range_ (range - 1)
	bits  int
	buf   []byte
	pos   int
	eof   bool
}

func (br *refBitReader) init(data []byte) {
	*br = refBitReader{rng: 255 - 1, bits: -8, buf: data}
	br.loadNewBytes()
}

func (br *refBitReader) loadNewBytes() {
	if br.pos+8 <= len(br.buf) { // buf_ < buf_max_
		var in uint64
		for i := 0; i < 8; i++ {
			in = in<<8 | uint64(br.buf[br.pos+i])
		}
		br.pos += 56 >> 3
		br.value = in>>(64-56) | br.value<<56
		br.bits += 56
		return
	}
	// VP8LoadFinalBytes
	switch {
	case br.pos < len(br.buf):
		br.bits += 8
		br.value = uint64(br.buf[br.pos]) | br.value<<8
		br.pos++
	case !br.eof:
		br.value <<= 8
		br.bits += 8
		br.eof = true
	default:
		br.bits = 0
	}
}

// bitsLog2Floor is BitsLog2Floor(n), n > 0.
func bitsLog2Floor(n uint32) int {
	l := 0
	for n > 1 {
		n >>= 1
		l++
	}
	return l
}

// getBit is VP8GetBit.
func (br *refBitReader) getBit(prob int) int {
	rng := br.rng
	if br.bits < 0 {
		br.loadNewBytes()
	}
	pos := br.bits
	split := (rng * uint32(prob)) >> 8
	value := uint32(br.value >> uint(pos))
	bit := 0
	if value > split {
		bit = 1
		rng -= split
		br.value -= uint64(split+1) << uint(pos)
	} else {
		rng = split + 1
	}
	shift := 7 ^ bitsLog2Floor(rng)
	rng <<= uint(shift)
	br.bits -= shift
	br.rng = rng - 1
	return bit
}

// getSigned is VP8GetSigned.
func (br *refBitReader) getSigned(v int) int {
	if br.bits < 0 {
		br.loadNewBytes()
	}
	pos := br.bits
	split := br.rng >> 1
	value := uint32(br.value >> uint(pos))
	mask := int32(split-value) >> 31
	br.bits--
	br.rng += uint32(mask)
	br.rng |= 1
	br.value -= uint64((split+1)&uint32(mask)) << uint(pos)
	return (v ^ int(mask)) - int(mask)
}

func sameState(br *bitReader, ref *refBitReader) bool {
	return br.value == ref.value && br.rng == ref.rng && br.nbits == ref.bits &&
		br.pos == ref.pos && br.eof == ref.eof
}

// bitStreams returns random, constant and short streams; streams that
// start with large bytes drive value far above the range (corrupt data),
// which exercises the unsigned 32-bit compare of VP8GetBit.
func bitStreams(rnd *rand.Rand) [][]byte {
	var out [][]byte
	for _, n := range []int{0, 1, 2, 7, 8, 9, 15, 16, 64, 300} {
		for _, fill := range []int{-1, 0x00, 0xff, 0x80, 0x7f} {
			s := make([]byte, n)
			for i := range s {
				if fill < 0 {
					s[i] = byte(rnd.Uint32())
				} else {
					s[i] = byte(fill)
				}
			}
			out = append(out, s)
		}
	}
	return out
}

// TestBoolDecoderMatchesLibwebp decodes many streams with random
// probabilities through every entry point of the bool decoder (getBit, the
// register-based decodeBit/decodeSigned of the token loop, getSigned) and
// compares the decoded values and the whole reader state with the
// transcription of libwebp after every symbol, including past the end of
// the data.
func TestBoolDecoderMatchesLibwebp(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 2))
	for si, s := range bitStreams(rnd) {
		for round := 0; round < 8; round++ {
			var br bitReader
			var ref refBitReader
			br.init(s)
			ref.init(s)
			if !sameState(&br, &ref) {
				t.Fatalf("stream %d: init state differs", si)
			}
			for i := 0; i < 8*len(s)+200; i++ {
				prob := uint8(rnd.Uint32())
				if round == 0 {
					prob = uint8(i * 37) // includes 0 and 255
				}
				var got, want int
				switch op := rnd.IntN(4); op {
				case 0:
					got, want = br.getBit(prob), ref.getBit(int(prob))
				case 1:
					value, rng, nbits := br.value, br.rng, br.nbits
					if nbits < 0 {
						value, nbits = br.refill(value, nbits)
					}
					var bit bool
					value, rng, nbits, bit = decodeBit(value, rng, nbits, prob)
					br.value, br.rng, br.nbits = value, rng, nbits
					got, want = b2i(bit), ref.getBit(int(prob))
				case 2:
					v := 1 + rnd.IntN(2048)
					value, rng, nbits := br.value, br.rng, br.nbits
					if nbits < 0 {
						value, nbits = br.refill(value, nbits)
					}
					value, rng, nbits, got = decodeSigned(value, rng, nbits, v)
					br.value, br.rng, br.nbits = value, rng, nbits
					want = ref.getSigned(v)
				case 3:
					v := 1 + rnd.IntN(2048)
					got, want = br.getSigned(v), ref.getSigned(v)
				}
				if got != want || !sameState(&br, &ref) {
					t.Fatalf("stream %d (len %d) round %d symbol %d prob %d: got %d state %+v, want %d state %+v",
						si, len(s), round, i, prob, got, br, want, ref)
				}
				if br.rng < 127 || br.rng > 255 {
					t.Fatalf("stream %d symbol %d: range %d outside [127, 255]", si, i, br.rng)
				}
			}
		}
	}
}

// TestDecodeBitExhaustive checks decodeBit against VP8GetBit for every
// reachable range and every probability, with the compared value below,
// at, just above and far above (>= 2^31 above) the split, at several bit
// positions.
func TestDecodeBitExhaustive(t *testing.T) {
	for rng := uint32(127); rng <= 255; rng++ {
		for prob := 0; prob < 256; prob++ {
			split := (rng * uint32(prob)) >> 8
			for _, v := range []uint64{0, uint64(split), uint64(split) + 1, uint64(rng), uint64(rng) + 1, 255, 1 << 31, uint64(split) + 1<<31, 1<<32 - 1, 1<<32 + uint64(split)} {
				for _, nbits := range []int{0, 1, 7, 8, 31, 56} {
					value := v<<uint(nbits) | (uint64(prob)*0x9e3779b97f4a7c15)&(1<<uint(nbits)-1)
					ref := refBitReader{value: value, rng: rng, bits: nbits}
					want := ref.getBit(prob)
					gv, gr, gn, gb := decodeBit(value, rng, nbits, uint8(prob))
					if b2i(gb) != want || gv != ref.value || gr != ref.rng || gn != ref.bits {
						t.Fatalf("rng %d prob %d value %#x nbits %d: got (%#x %d %d %v), want (%#x %d %d %d)",
							rng, prob, value, nbits, gv, gr, gn, gb, ref.value, ref.rng, ref.bits, want)
					}
				}
			}
		}
	}
}
