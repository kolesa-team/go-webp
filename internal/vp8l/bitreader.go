package vp8l

// Algorithm adapted from libwebp (BSD-3-Clause): src/utils/bit_reader_utils.c (VP8LBitReader,
// including its end-of-stream rules).

import "encoding/binary"

// bitReader is an LSB-first bit reader over an in-memory buffer.
//
// val holds the not yet consumed bits, the next bit in bit 0. nbits is the
// number of valid bits in val; bits above nbits are either zero or already
// equal to the upcoming stream bits, which lets refill OR a whole 64-bit word
// in without masking. Reading past the end of data yields zero bits; pos then
// runs past len(data) and eos reports the overrun.
//
// The hot decoding loops copy val/nbits/pos into locals and inline the refill
// logic; the methods here are used for headers and Huffman code reading.
type bitReader struct {
	data  []byte
	pos   int
	val   uint64
	nbits uint
}

func (b *bitReader) init(data []byte) {
	b.data = data
	b.pos = 0
	b.val = 0
	b.nbits = 0
	b.refill()
}

// refill guarantees at least 56 valid bits in val (zero padded at the end of
// data).
func (b *bitReader) refill() {
	if b.pos+8 <= len(b.data) {
		b.val |= binary.LittleEndian.Uint64(b.data[b.pos:]) << b.nbits
		b.pos += int((63 - b.nbits) >> 3)
		b.nbits |= 56
		return
	}
	b.val, b.nbits, b.pos = refillSlow(b.data, b.val, b.nbits, b.pos)
}

// refillSlow loads the stream byte by byte near the end of data.
func refillSlow(data []byte, val uint64, nbits uint, pos int) (uint64, uint, int) {
	for nbits <= 56 {
		if pos < len(data) {
			val |= uint64(data[pos]) << nbits
		}
		pos++
		nbits += 8
	}
	return val, nbits, pos
}

// read returns the next n (n <= 32) bits.
func (b *bitReader) read(n uint) uint32 {
	if b.nbits < n {
		b.refill()
	}
	v := uint32(b.val & (uint64(1)<<n - 1))
	b.val >>= n
	b.nbits -= n
	return v
}

// eos reports whether more bits were consumed than the stream contains.
func (b *bitReader) eos() bool {
	return isEOS(len(b.data), b.pos, b.nbits)
}

// eosLimit returns the number of bits that may be consumed from n bytes
// before the end of stream is reached. Like libwebp, whose 64-bit window
// is zero-filled for streams shorter than 8 bytes, at least 64 bits are
// allowed.
func eosLimit(n int) int {
	return max(n, 8) * 8
}

func isEOS(n, pos int, nbits uint) bool {
	return pos*8-int(nbits) > eosLimit(n)
}

// tailReader emulates libwebp's VP8LBitReader once its 64-bit window holds
// the last bytes of the stream, including its end-of-stream quirks: a
// prefetch at bit position 64 wraps around to the window start, and
// detecting the end of stream resets the position to 0. Only the 8-bit
// alpha path, which tolerates running out of data on the very last pixel,
// can observe these quirks.
type tailReader struct {
	val    uint64 // window: the last 8 bytes (or the whole stream if shorter)
	bitPos int
	eos    bool
}

func newTailReader(data []byte, consumed int) tailReader {
	var t tailReader
	start := max(len(data)-8, 0)
	for i, c := range data[start:] {
		t.val |= uint64(c) << (8 * uint(i))
	}
	t.bitPos = consumed - start*8
	return t
}

func (t *tailReader) prefetch() uint64 { return uint64(uint32(t.val >> (uint(t.bitPos) & 63))) }

func (t *tailReader) setEOS() {
	t.eos = true
	t.bitPos = 0
}

// shift mirrors ShiftBytes with no bytes left to load.
func (t *tailReader) shift() {
	if t.bitPos > 64 {
		t.setEOS()
	}
}

func (t *tailReader) fill() {
	if t.bitPos >= 32 {
		t.shift()
	}
}

func (t *tailReader) readBits(n uint) int {
	if t.eos {
		t.setEOS()
		return 0
	}
	v := int(t.prefetch() & (uint64(1)<<n - 1))
	t.bitPos += int(n)
	t.shift()
	return v
}

func (t *tailReader) readSym(table []huffCode) int {
	v := t.prefetch()
	idx := int(v & 0xff)
	e := table[idx]
	if e.bits > rootBits {
		t.bitPos += rootBits
		v = t.prefetch()
		idx += int(e.value) + int(v&(uint64(1)<<(e.bits-rootBits)-1))
		e = table[idx]
	}
	t.bitPos += int(e.bits)
	return int(e.value)
}

func (t *tailReader) copyValue(sym int) int {
	if sym < 4 {
		return sym + 1
	}
	eb := uint(sym-2) >> 1
	return (2+sym&1)<<eb + t.readBits(eb) + 1
}

func (t *tailReader) atEnd() bool { return t.eos || t.bitPos > 64 }
