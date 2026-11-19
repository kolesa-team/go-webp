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

package vp8enc

import (
	"bytes"
	"math/rand"
	"testing"
)

// refPutBit is the straightforward boolean coder step (libwebp's
// VP8PutBit) that emitTokens must reproduce exactly.
func refPutBit(bw *bitWriter, bit int, prob int) {
	split := (bw.rng * int32(prob)) >> 8
	if bit != 0 {
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
}

// randomTokens returns n tokens mixing constant-probability and slot
// tokens. With skewed probabilities and unlikely bits it produces long
// 0xff runs and carries.
func randomTokens(rng *rand.Rand, n int, probas *[numProbaSlots]uint8) []uint16 {
	toks := make([]uint16, n)
	for i := range toks {
		var p uint8
		var t uint16
		if rng.Intn(3) == 0 {
			p = uint8(rng.Intn(256))
			t = tokenFixedProba | uint16(p)
		} else {
			slot := rng.Intn(numProbaSlots)
			p = probas[slot]
			t = uint16(slot)
		}
		bit := rng.Intn(2)
		switch rng.Intn(4) {
		case 0: // the unlikely value
			bit = 0
			if p >= 128 {
				bit = 1
			}
		case 1: // the likely value
			bit = 1
			if p >= 128 {
				bit = 0
			}
		}
		toks[i] = t | uint16(bit)<<15
	}
	return toks
}

// TestEmitTokensExact checks emitTokens and estimateTokenSize against the
// reference coder on random token streams, split into random chunks so
// that the coder state is saved and restored between calls.
func TestEmitTokensExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 200; iter++ {
		var probas [numProbaSlots]uint8
		for i := range probas {
			switch rng.Intn(4) {
			case 0:
				probas[i] = uint8(rng.Intn(4)) // near 0, including 0
			case 1:
				probas[i] = uint8(252 + rng.Intn(4)) // near 255, including 255
			default:
				probas[i] = uint8(rng.Intn(256))
			}
		}
		toks := randomTokens(rng, 1+rng.Intn(20000), &probas)
		var want, got bitWriter
		want.init(0)
		got.init(0)
		var wantSize uint64
		for _, tk := range toks {
			bit := int(tk >> 15)
			p := int(probas[tk&0x3fff%numProbaSlots])
			if tk&tokenFixedProba != 0 {
				p = int(tk & 0xff)
			}
			refPutBit(&want, bit, p)
			wantSize += uint64(bitCost(bit, uint8(p)))
		}
		var gotSize uint64
		for rest := toks; len(rest) > 0; {
			n := min(len(rest), rng.Intn(300))
			emitTokens(&got, rest[:n], &probas)
			gotSize += estimateTokenSize(rest[:n], &probas)
			rest = rest[n:]
		}
		if gotSize != wantSize {
			t.Fatalf("iter %d: estimateTokenSize %d, want %d", iter, gotSize, wantSize)
		}
		if want.rng != got.rng || want.value != got.value || want.nbBits != got.nbBits || want.run != got.run {
			t.Fatalf("iter %d: coder state differs", iter)
		}
		w, g := want.finish(), got.finish()
		if !bytes.Equal(w, g) {
			t.Fatalf("iter %d: %d tokens: output differs (%d vs %d bytes)", iter, len(toks), len(g), len(w))
		}
	}
	// empty input leaves the state untouched
	var bw bitWriter
	bw.init(0)
	emitTokens(&bw, nil, new([numProbaSlots]uint8))
	if bw.rng != 254 || bw.value != 0 || bw.nbBits != -8 || len(bw.buf) != 0 {
		t.Fatal("empty emit changed the coder state")
	}
}

// TestBitWriterGrow checks that an output outgrowing its initial capacity
// (grown through the pool) is identical to one that never grows.
func TestBitWriterGrow(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	var probas [numProbaSlots]uint8
	for i := range probas {
		probas[i] = uint8(rng.Intn(256))
	}
	toks := randomTokens(rng, 300000, &probas)
	var small, big bitWriter
	small.init(1)
	big.init(1 << 20)
	defer small.release()
	defer big.release()
	bigCap := cap(big.buf)
	emitTokens(&small, toks, &probas)
	emitTokens(&big, toks, &probas)
	s, b := small.finish(), big.finish()
	if cap(big.buf) != bigCap {
		t.Fatal("the large buffer grew")
	}
	if len(s) < 10000 || !bytes.Equal(s, b) {
		t.Fatalf("grown output differs (%d vs %d bytes)", len(s), len(b))
	}
}

func BenchmarkEmitTokens(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var probas [numProbaSlots]uint8
	for i := range probas {
		probas[i] = uint8(rng.Intn(256))
	}
	toks := randomTokens(rng, 1<<16, &probas)
	var bw bitWriter
	b.SetBytes(int64(len(toks)))
	for b.Loop() {
		bw.init(1 << 16)
		emitTokens(&bw, toks, &probas)
	}
}

// TestTokenBufferPages checks the paged token buffer across page boundaries:
// indices, chunked iteration and page reuse after reset.
func TestTokenBufferPages(t *testing.T) {
	var tb tokenBuffer
	defer tb.release()
	var s uint32
	for round := 0; round < 2; round++ {
		tb.reset()
		n := 2*tokenPageSize + 123
		for i := 0; i < n; i++ {
			if i%3 == 0 {
				tb.addConstant(i&1, i&0xff)
			} else {
				tb.add(i&1, i&0x3ff, &s)
			}
		}
		if tb.len() != n {
			t.Fatalf("len %d, want %d", tb.len(), n)
		}
		if len(tb.pages) != 3 {
			t.Fatalf("%d pages, want 3", len(tb.pages))
		}
		for _, r := range [][2]int{{0, n}, {tokenPageSize - 5, tokenPageSize + 5}, {7, 7}, {n - 1, n}} {
			i := r[0]
			for i < r[1] {
				c := tb.chunk(i, r[1])
				if len(c) == 0 {
					t.Fatalf("empty chunk at %d", i)
				}
				for j, v := range c {
					k := i + j
					want := uint16((k&1)<<15 | k&0x3ff)
					if k%3 == 0 {
						want = uint16((k&1)<<15 | tokenFixedProba | k&0xff)
					}
					if v != want {
						t.Fatalf("token %d = %#x, want %#x", k, v, want)
					}
				}
				i += len(c)
			}
		}
	}
}
