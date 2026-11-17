package vp8enc

// Boolean (arithmetic) encoder of the VP8 bitstream.
// Algorithm adapted from libwebp (BSD-3-Clause).

// kNorm[r] is the renormalization shift for a range-1 value r < 127.
var kNorm = [128]uint8{
	7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	0,
}

// kNewRange[r] = ((r + 1) << kNorm[r]) - 1
var kNewRange = [128]uint8{
	127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
	127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
	247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
	183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
	243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
	151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
	181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
	211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
	241, 243, 245, 247, 249, 251, 253, 127,
}

type bitWriter struct {
	rng    int32 // range-1
	value  int32
	run    int // number of outstanding 0xff bytes
	nbBits int // number of pending bits
	buf    []byte
}

func (bw *bitWriter) init(capHint int) {
	bw.rng = 255 - 1
	bw.value = 0
	bw.run = 0
	bw.nbBits = -8
	if cap(bw.buf) < capHint {
		bytePool.Put(bw.buf)
		bw.buf = bytePool.Get(capHint)
	}
	bw.buf = bw.buf[:0]
}

// release returns the output buffer to its pool.
func (bw *bitWriter) release() {
	bytePool.Put(bw.buf)
	bw.buf = nil
}

// grow makes room for n more bytes when the output outgrows its estimate:
// the larger buffer comes from bytePool and the outgrown one goes back to
// it (append would leave it to the garbage collector).
func (bw *bitWriter) grow(n int) {
	nb := bytePool.Get(max(2*cap(bw.buf), len(bw.buf)+n, 1024))
	nb = append(nb[:0], bw.buf...)
	bytePool.Put(bw.buf)
	bw.buf = nb
}

func (bw *bitWriter) flush() {
	s := uint(8 + bw.nbBits)
	bits := bw.value >> s
	bw.value -= bits << s
	bw.nbBits -= 8
	if bits&0xff != 0xff {
		if len(bw.buf)+bw.run >= cap(bw.buf) {
			bw.grow(bw.run + 1)
		}
		if bits&0x100 != 0 { // overflow -> propagate carry over pending 0xff's
			if n := len(bw.buf); n > 0 {
				bw.buf[n-1]++
			}
		}
		if bw.run > 0 {
			v := byte(0xff)
			if bits&0x100 != 0 {
				v = 0
			}
			for ; bw.run > 0; bw.run-- {
				bw.buf = append(bw.buf, v)
			}
		}
		bw.buf = append(bw.buf, byte(bits))
	} else {
		bw.run++ // delay writing of bytes 0xff, pending eventual carry.
	}
}

// putBit codes 'bit' with probability 'prob' (of the bit being 0, /256).
func (bw *bitWriter) putBit(bit bool, prob int) bool {
	split := (bw.rng * int32(prob)) >> 8
	if bit {
		bw.value += split + 1
		bw.rng -= split + 1
	} else {
		bw.rng = split
	}
	if bw.rng < 127 {
		shift := kNorm[bw.rng&127]
		bw.rng = int32(kNewRange[bw.rng&127])
		bw.value <<= shift
		bw.nbBits += int(shift)
		if bw.nbBits > 0 {
			bw.flush()
		}
	}
	return bit
}

func (bw *bitWriter) putBitUniform(bit bool) bool {
	split := bw.rng >> 1
	if bit {
		bw.value += split + 1
		bw.rng -= split + 1
	} else {
		bw.rng = split
	}
	if bw.rng < 127 {
		bw.rng = int32(kNewRange[bw.rng&127])
		bw.value <<= 1
		bw.nbBits++
		if bw.nbBits > 0 {
			bw.flush()
		}
	}
	return bit
}

func (bw *bitWriter) putBits(value uint32, nbBits int) {
	for mask := uint32(1) << uint(nbBits-1); mask != 0; mask >>= 1 {
		bw.putBitUniform(value&mask != 0)
	}
}

func (bw *bitWriter) putSignedBits(value int, nbBits int) {
	if !bw.putBitUniform(value != 0) {
		return
	}
	if value < 0 {
		bw.putBits(uint32((-value)<<1)|1, nbBits+1)
	} else {
		bw.putBits(uint32(value<<1), nbBits+1)
	}
}

// finish flushes the pending bits and returns the coded bytes.
func (bw *bitWriter) finish() []byte {
	bw.putBits(0, 9-bw.nbBits)
	bw.nbBits = 0 // pad with zeroes
	bw.flush()
	return bw.buf
}

// pos returns the current bit position (approximate, for statistics).
func (bw *bitWriter) pos() uint64 {
	return uint64(len(bw.buf)+bw.run)*8 + uint64(8+bw.nbBits)
}
