package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): utils/bit_reader_utils.[ch].

import (
	"encoding/binary"
	"math/bits"
)

// bitReader is the VP8 boolean entropy decoder. It mirrors libwebp's
// VP8BitReader: value holds up to 56+8 pending bits, rng stores range-1 in
// [127, 254], and nbits counts the bits buffered beyond the 8 bits that are
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

// kNormShift[r] is the left shift that renormalizes a range r in [1, 255]
// back into [128, 255].
var kNormShift = func() (t [256]uint8) {
	for r := 1; r < 256; r++ {
		t[r] = uint8(bits.LeadingZeros32(uint32(r)) - 24)
	}
	return
}()

// kRenorm[r] packs libwebp's kVP8NewRange (the renormalized range minus
// one, low byte) and kVP8Log2Range (the shift, high byte) for a range r.
var kRenorm = func() (t [256]uint16) {
	for r := 1; r < 256; r++ {
		sh := int(kNormShift[r])
		t[r] = uint16((r<<sh)-1) | uint16(sh)<<8
	}
	return
}()

// fill makes sure at least one bit can be decoded with bitNL.
func (br *bitReader) fill() {
	if br.nbits < 0 {
		br.loadNewBytes()
	}
}

// bitNL decodes one boolean whose probability of being 0 is prob/256. The
// caller must have called fill first ("no load" variant, small enough to be
// inlined in hot loops).
func (br *bitReader) bitNL(prob uint8) bool {
	var bit bool
	br.value, br.rng, br.nbits, bit = decodeBit(br.value, br.rng, br.nbits, prob)
	return bit
}

// decodeBit is bitNL on register copies of the reader state. Hot loops keep
// value/rng/nbits in locals and call it (inlined) to avoid a memory round
// trip on the bit decoding dependency chain. nbits must be >= 0.
func decodeBit(value uint64, rng uint32, nbits int, prob uint8) (uint64, uint32, int, bool) {
	split := (rng * uint32(prob)) >> 8
	pos := uint(nbits) & 63
	m := uint32(int32(split-uint32(value>>pos)) >> 31) // all ones if bit
	rng = split + 1 + ((rng - 2*split - 1) & m)
	value -= uint64((split+1)&m) << pos
	e := kRenorm[uint8(rng)]
	return value, uint32(e & 0xff), nbits - int(e>>8), m != 0
}

// decodeSigned is getSigned on register copies of the reader state.
func decodeSigned(value uint64, rng uint32, nbits int, v int) (uint64, uint32, int, int) {
	pos := uint(nbits)
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
	br.fill()
	if br.bitNL(prob) {
		return 1
	}
	return 0
}

// getSigned returns v or -v using one bit of probability 1/2 (libwebp's
// VP8GetSigned).
func (br *bitReader) getSigned(v int) int {
	if br.nbits < 0 {
		br.loadNewBytes()
	}
	pos := uint(br.nbits)
	split := br.rng >> 1
	value := uint32(br.value >> pos)
	mask := int32(split-value) >> 31 // -1 if value > split
	br.nbits--
	br.rng = (br.rng + uint32(mask)) | 1
	br.value -= uint64((split+1)&uint32(mask)) << pos
	return (v ^ int(mask)) - int(mask)
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
