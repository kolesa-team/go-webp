package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): utils/bit_reader_utils.[ch].

import (
	"encoding/binary"
	"math/bits"
)

// bitReader is the VP8 boolean entropy decoder. It mirrors libwebp's
// VP8BitReader: value holds up to 56+8 pending bits, rng stores range-1 in
// [127, 255], and nbits counts the bits buffered beyond the 8 bits that are
// always needed for the next comparison.
type bitReader struct {
	value uint64
	rng   uint32
	nbits int
	buf   []byte
	pos   int
	eof   bool
}

// init prepares br to decode data.
func (br *bitReader) init(data []byte) {
	br.rng = 255 - 1
	br.value = 0
	br.nbits = -8 // to load the very first 8 bits
	br.eof = false
	br.buf = data
	br.pos = 0
	br.loadNewBytes()
}

// loadNewBytes refills the value buffer with 56 bits when possible.
func (br *bitReader) loadNewBytes() {
	if br.pos+8 <= len(br.buf) {
		v := binary.BigEndian.Uint64(br.buf[br.pos:]) >> 8
		br.pos += 7
		br.value = v | br.value<<56
		br.nbits += 56
		return
	}
	br.loadFinalBytes()
}

// loadFinalBytes reads the trailing bytes one at a time. Once the data is
// exhausted it feeds 8 zero bits once and flags eof.
func (br *bitReader) loadFinalBytes() {
	if br.pos < len(br.buf) {
		br.nbits += 8
		br.value = uint64(br.buf[br.pos]) | br.value<<8
		br.pos++
	} else if !br.eof {
		br.value <<= 8
		br.nbits += 8
		br.eof = true
	} else {
		br.nbits = 0 // avoid undefined shifts
	}
}

// decodeBit decodes one boolean whose probability of being 0 is prob/256
// (libwebp's VP8GetBit) on register copies of the reader state. Hot loops
// keep value/rng/nbits in locals and call it (it is inlined) to avoid a
// memory round trip on the bit decoding dependency chain. nbits must be >= 0
// (refill first).
//
// The new range r (not minus one) is split+1 or rng-split, both in [1, 255]
// because rng stays in [127, 255] (255 only after decodeSigned). It is
// renormalized into [128, 255] by a left shift of 7-floor(log2(r)) =
// clz32(r)-24, which is one CLZ instruction instead of a table lookup on the
// dependency chain (libwebp computes it the same way, with BitsLog2Floor).
func decodeBit(value uint64, rng uint32, nbits int, prob uint8) (uint64, uint32, int, bool) {
	split := (rng * uint32(prob)) >> 8
	pos := uint(nbits) & 63
	// Unsigned "value > split" as in libwebp's VP8GetBit: a 32-bit signed
	// difference would disagree on corrupt streams where value-split >= 2^31.
	bit := uint32(value>>pos) > split
	r := split + 1
	if bit {
		r = rng - split
		value -= uint64(split+1) << pos
	}
	// r is in [1, 255], so the shift is in [0, 7]; the mask only tells the
	// compiler that it is a valid 32-bit shift amount.
	sh := uint(bits.LeadingZeros32(r)-24) & 31
	return value, (r << sh) - 1, nbits - int(sh), bit
}

// decodeSigned returns v or -v using one bit of probability 1/2 (libwebp's
// VP8GetSigned) on register copies of the reader state. nbits must be >= 0.
// Unlike VP8GetBit, VP8GetSigned compares with a signed 32-bit difference,
// which is kept here for bit-exactness on corrupt streams. The range does
// not need renormalizing: it loses exactly one bit.
func decodeSigned(value uint64, rng uint32, nbits int, v int) (uint64, uint32, int, int) {
	pos := uint(nbits) & 63
	split := rng >> 1
	mask := int32(split-uint32(value>>pos)) >> 31 // -1 if value > split
	rng = (rng + uint32(mask)) | 1
	value -= uint64((split+1)&uint32(mask)) << pos
	return value, rng, nbits - 1, (v ^ int(mask)) - int(mask)
}

// refill stores the register state back, loads new bytes and returns the
// updated state (slow path of the register-based decoding).
func (br *bitReader) refill(value uint64, nbits int) (uint64, int) {
	br.value, br.nbits = value, nbits
	br.loadNewBytes()
	return br.value, br.nbits
}

// getBit decodes one boolean whose probability of being 0 is prob/256.
func (br *bitReader) getBit(prob uint8) int {
	if br.nbits < 0 {
		br.loadNewBytes()
	}
	var bit bool
	br.value, br.rng, br.nbits, bit = decodeBit(br.value, br.rng, br.nbits, prob)
	return b2i(bit)
}

// getSigned returns v or -v using one bit of probability 1/2 (libwebp's
// VP8GetSigned).
func (br *bitReader) getSigned(v int) int {
	if br.nbits < 0 {
		br.loadNewBytes()
	}
	br.value, br.rng, br.nbits, v = decodeSigned(br.value, br.rng, br.nbits, v)
	return v
}

// getValue reads an n-bit unsigned literal, most significant bit first.
func (br *bitReader) getValue(n int) int {
	v := 0
	for ; n > 0; n-- {
		v |= br.getBit(0x80) << uint(n-1)
	}
	return v
}

// getSignedValue reads an n-bit magnitude followed by a sign bit.
func (br *bitReader) getSignedValue(n int) int {
	v := br.getValue(n)
	if br.getBit(0x80) != 0 {
		return -v
	}
	return v
}

// get reads a single flag bit of probability 1/2.
func (br *bitReader) get() bool { return br.getBit(0x80) != 0 }
