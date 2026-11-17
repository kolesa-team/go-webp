package vp8lenc

import (
	"encoding/binary"
	"slices"
)

// bitWriter writes VP8L bits LSB-first using a 64-bit accumulator.
type bitWriter struct {
	buf   []byte
	acc   uint64
	nbits uint
}

// putBits writes the n low bits of v (n <= 32; v must fit in n bits).
func (w *bitWriter) putBits(v uint32, n uint) {
	w.acc |= uint64(v) << w.nbits
	w.nbits += n
	if w.nbits >= 32 {
		w.buf = binary.LittleEndian.AppendUint32(w.buf, uint32(w.acc))
		w.acc >>= 32
		w.nbits -= 32
	}
}

// reserve makes room for at least n more bytes in the buffer.
func (w *bitWriter) reserve(n int) {
	if n > 0 && cap(w.buf)-len(w.buf) < n {
		w.buf = slices.Grow(w.buf, n)
	}
}

// numBits returns the number of bits written so far.
func (w *bitWriter) numBits() int { return len(w.buf)*8 + int(w.nbits) }

// reset clears the writer, keeping its buffer capacity.
func (w *bitWriter) reset() {
	w.buf = w.buf[:0]
	w.acc = 0
	w.nbits = 0
}

// appendWriter appends all bits of o to w.
func (w *bitWriter) appendWriter(o *bitWriter) {
	if w.nbits == 0 {
		w.buf = append(w.buf, o.buf...)
	} else {
		for i := 0; i+4 <= len(o.buf); i += 4 {
			w.putBits(binary.LittleEndian.Uint32(o.buf[i:]), 32)
		}
		for i := len(o.buf) &^ 3; i < len(o.buf); i++ {
			w.putBits(uint32(o.buf[i]), 8)
		}
	}
	n := o.nbits
	acc := o.acc
	for n > 0 {
		k := n
		if k > 32 {
			k = 32
		}
		w.putBits(uint32(acc)&uint32((uint64(1)<<k)-1), k)
		acc >>= k
		n -= k
	}
}

// finish flushes the pending bits and returns the byte slice.
func (w *bitWriter) finish() []byte {
	for w.nbits > 0 {
		w.buf = append(w.buf, byte(w.acc))
		w.acc >>= 8
		if w.nbits >= 8 {
			w.nbits -= 8
		} else {
			w.nbits = 0
		}
	}
	w.acc = 0
	return w.buf
}
