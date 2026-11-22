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

package vp8lenc

import "slices"

// Huffman code construction and storage.
// Algorithm adapted from libwebp (BSD-3-Clause).

// huffCode holds code lengths and bit-reversed codes for an alphabet.
type huffCode struct {
	lengths []uint8
	codes   []uint16
}

type huffToken struct {
	code  uint8 // 0..15 literal length, 16..18 repeat codes
	extra uint8
}

// huffScratch holds reusable buffers for Huffman tree construction.
type huffScratch struct {
	keys    []uint64
	weight  []uint64
	parent  []int32
	depth   []uint8
	goodRLE []bool
	counts  []uint32
	tokens  []huffToken
}

// optimizeForRLE adjusts population counts so that the resulting code
// lengths compress better with the RLE code-length codes.
func optimizeForRLE(counts []uint32, good []bool) {
	length := len(counts)
	for ; length >= 0; length-- {
		if length == 0 {
			return
		}
		if counts[length-1] != 0 {
			break
		}
	}
	good = good[:length+1]
	for i := range good {
		good[i] = false
	}
	{
		symbol := counts[0]
		stride := 0
		for i := 0; i < length+1; i++ {
			if i == length || counts[i] != symbol {
				if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
					for k := 0; k < stride; k++ {
						good[i-k-1] = true
					}
				}
				stride = 1
				if i != length {
					symbol = counts[i]
				}
			} else {
				stride++
			}
		}
	}
	{
		stride := uint32(0)
		limit := counts[0]
		sum := uint32(0)
		for i := 0; i < length+1; i++ {
			if i == length || good[i] || (i != 0 && good[i-1]) || !collapsible(counts[i], limit) {
				if stride >= 4 || (stride >= 3 && sum == 0) {
					count := (sum + stride/2) / stride
					if count < 1 {
						count = 1
					}
					if sum == 0 {
						count = 0
					}
					for k := uint32(0); k < stride; k++ {
						counts[i-int(k)-1] = count
					}
				}
				stride = 0
				sum = 0
				if i < length-3 {
					limit = (counts[i] + counts[i+1] + counts[i+2] + counts[i+3] + 2) / 4
				} else if i < length {
					limit = counts[i]
				} else {
					limit = 0
				}
			}
			stride++
			if i != length {
				sum += counts[i]
				if stride >= 4 {
					limit = (sum + stride/2) / stride
				}
			}
		}
	}
}

func collapsible(a, b uint32) bool {
	d := int64(a) - int64(b)
	return d < 4 && d > -4
}

// generateOptimalTree computes depth-limited Huffman code lengths for hist.
func (s *huffScratch) generateOptimalTree(hist []uint32, limit int, depths []uint8) {
	for i := range depths {
		depths[i] = 0
	}
	n := 0
	for _, c := range hist {
		if c != 0 {
			n++
		}
	}
	if n == 0 {
		return
	}
	if n == 1 {
		for i, c := range hist {
			if c != 0 {
				depths[i] = 1
			}
		}
		return
	}
	if cap(s.keys) < n {
		s.keys = make([]uint64, n)
	}
	if cap(s.weight) < 2*n {
		s.weight = make([]uint64, 2*n)
		s.parent = make([]int32, 2*n)
		s.depth = make([]uint8, 2*n)
	}
	keys := s.keys[:n]
	weight := s.weight[:2*n]
	parent := s.parent[:2*n]
	depth := s.depth[:2*n]
	for countMin := uint64(1); ; countMin *= 2 {
		k := 0
		for i, c := range hist {
			if c != 0 {
				w := uint64(c)
				if w < countMin {
					w = countMin
				}
				keys[k] = w<<16 | uint64(i)
				k++
			}
		}
		slices.Sort(keys)
		for i, key := range keys {
			weight[i] = key >> 16
		}
		// Two-queue Huffman construction: leaves [0,n), internal nodes [n,2n-1).
		li, ii, next := 0, n, n
		for next < 2*n-1 {
			var a, b int
			if li < n && (ii >= next || weight[li] <= weight[ii]) {
				a = li
				li++
			} else {
				a = ii
				ii++
			}
			if li < n && (ii >= next || weight[li] <= weight[ii]) {
				b = li
				li++
			} else {
				b = ii
				ii++
			}
			weight[next] = weight[a] + weight[b]
			parent[a] = int32(next)
			parent[b] = int32(next)
			next++
		}
		root := 2*n - 2
		depth[root] = 0
		maxDepth := uint8(0)
		for i := root - 1; i >= 0; i-- {
			d := depth[parent[i]] + 1
			depth[i] = d
			if i < n && d > maxDepth {
				maxDepth = d
			}
		}
		if int(maxDepth) <= limit {
			for i, key := range keys {
				depths[key&0xffff] = depth[i]
			}
			return
		}
	}
}

// convertBitDepthsToSymbols assigns canonical, bit-reversed codes.
func convertBitDepthsToSymbols(c *huffCode) {
	var depthCount [16]uint32
	var nextCode [16]uint32
	for _, d := range c.lengths {
		depthCount[d]++
	}
	depthCount[0] = 0
	code := uint32(0)
	for i := 1; i < 16; i++ {
		code = (code + depthCount[i-1]) << 1
		nextCode[i] = code
	}
	for i, d := range c.lengths {
		if d == 0 {
			c.codes[i] = 0
			continue
		}
		v := nextCode[d]
		nextCode[d]++
		c.codes[i] = uint16(reverseBits(v, uint(d)))
	}
}

func reverseBits(v uint32, n uint) uint32 {
	var r uint32
	for i := uint(0); i < n; i++ {
		r = r<<1 | (v & 1)
		v >>= 1
	}
	return r
}

// createHuffmanTree builds a length-limited code for hist (hist is modified).
func (s *huffScratch) createHuffmanTree(hist []uint32, limit int, c *huffCode) {
	if cap(s.goodRLE) < len(hist)+1 {
		s.goodRLE = make([]bool, len(hist)+1)
	}
	optimizeForRLE(hist, s.goodRLE[:len(hist)+1])
	s.generateOptimalTree(hist, limit, c.lengths)
	convertBitDepthsToSymbols(c)
}

func codeRepeatedValues(reps int, toks []huffToken, value, prev uint8) []huffToken {
	if value != prev {
		toks = append(toks, huffToken{value, 0})
		reps--
	}
	for reps >= 1 {
		if reps < 3 {
			for i := 0; i < reps; i++ {
				toks = append(toks, huffToken{value, 0})
			}
			break
		} else if reps < 7 {
			toks = append(toks, huffToken{16, uint8(reps - 3)})
			break
		} else {
			toks = append(toks, huffToken{16, 3})
			reps -= 6
		}
	}
	return toks
}

func codeRepeatedZeros(reps int, toks []huffToken) []huffToken {
	for reps >= 1 {
		if reps < 3 {
			for i := 0; i < reps; i++ {
				toks = append(toks, huffToken{0, 0})
			}
			break
		} else if reps < 11 {
			toks = append(toks, huffToken{17, uint8(reps - 3)})
			break
		} else if reps < 139 {
			toks = append(toks, huffToken{18, uint8(reps - 11)})
			break
		} else {
			toks = append(toks, huffToken{18, 0x7f})
			reps -= 138
		}
	}
	return toks
}

// compressedTokens RLE-encodes code lengths.
func compressedTokens(lengths []uint8, toks []huffToken) []huffToken {
	toks = toks[:0]
	prev := uint8(8)
	n := len(lengths)
	for i := 0; i < n; {
		v := lengths[i]
		k := i + 1
		for k < n && lengths[k] == v {
			k++
		}
		runs := k - i
		if v == 0 {
			toks = codeRepeatedZeros(runs, toks)
		} else {
			toks = codeRepeatedValues(runs, toks, v, prev)
			prev = v
		}
		i += runs
	}
	return toks
}

var codeLengthOrder = [codeLengthCodes]uint8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// storeHuffmanCode writes code c (simple or normal form) to bw.
func (s *huffScratch) storeHuffmanCode(bw *bitWriter, c *huffCode) {
	count := 0
	var symbols [2]int
	for i, l := range c.lengths {
		if l != 0 {
			if count < 2 {
				symbols[count] = i
			}
			count++
			if count > 2 {
				break
			}
		}
	}
	if count == 0 {
		bw.putBits(0x01, 4)
		return
	}
	if count <= 2 && symbols[0] < 256 && symbols[1] < 256 {
		bw.putBits(1, 1)
		bw.putBits(uint32(count-1), 1)
		if symbols[0] <= 1 {
			bw.putBits(0, 1)
			bw.putBits(uint32(symbols[0]), 1)
		} else {
			bw.putBits(1, 1)
			bw.putBits(uint32(symbols[0]), 8)
		}
		if count == 2 {
			bw.putBits(uint32(symbols[1]), 8)
		}
		return
	}
	s.storeFullHuffmanCode(bw, c)
}

func (s *huffScratch) storeFullHuffmanCode(bw *bitWriter, c *huffCode) {
	var clLengths [codeLengthCodes]uint8
	var clCodes [codeLengthCodes]uint16
	var clHist [codeLengthCodes]uint32
	cl := huffCode{lengths: clLengths[:], codes: clCodes[:]}

	bw.putBits(0, 1)
	s.tokens = compressedTokens(c.lengths, s.tokens)
	toks := s.tokens
	for _, t := range toks {
		clHist[t.code]++
	}
	s.createHuffmanTree(clHist[:], 7, &cl)

	// Store the code length code lengths.
	codesToStore := codeLengthCodes
	for ; codesToStore > 4; codesToStore-- {
		if clLengths[codeLengthOrder[codesToStore-1]] != 0 {
			break
		}
	}
	bw.putBits(uint32(codesToStore-4), 4)
	for i := 0; i < codesToStore; i++ {
		bw.putBits(uint32(clLengths[codeLengthOrder[i]]), 3)
	}
	clearIfOneSymbol(&cl)

	trailingZeroBits := 0
	trimmed := len(toks)
	for i := len(toks) - 1; i >= 0; i-- {
		ix := toks[i].code
		if ix == 0 || ix == 17 || ix == 18 {
			trimmed--
			trailingZeroBits += int(clLengths[ix])
			if ix == 17 {
				trailingZeroBits += 3
			} else if ix == 18 {
				trailingZeroBits += 7
			}
		} else {
			break
		}
	}
	writeTrimmed := trimmed > 1 && trailingZeroBits > 12
	length := len(toks)
	if writeTrimmed {
		length = trimmed
		bw.putBits(1, 1)
		if trimmed == 2 {
			bw.putBits(0, 3+2)
		} else {
			nb := log2Floor(uint32(trimmed - 2))
			nbitpairs := nb/2 + 1
			bw.putBits(uint32(nbitpairs-1), 3)
			bw.putBits(uint32(trimmed-2), uint(nbitpairs*2))
		}
	} else {
		bw.putBits(0, 1)
	}
	for _, t := range toks[:length] {
		ix := t.code
		bw.putBits(uint32(clCodes[ix]), uint(clLengths[ix]))
		switch ix {
		case 16:
			bw.putBits(uint32(t.extra), 2)
		case 17:
			bw.putBits(uint32(t.extra), 3)
		case 18:
			bw.putBits(uint32(t.extra), 7)
		}
	}
}

// clearIfOneSymbol zeroes the code when it has a single used symbol (the
// decoder reads zero bits for such codes).
func clearIfOneSymbol(c *huffCode) {
	count := 0
	for _, l := range c.lengths {
		if l != 0 {
			count++
			if count > 1 {
				return
			}
		}
	}
	for i := range c.lengths {
		c.lengths[i] = 0
		c.codes[i] = 0
	}
}
