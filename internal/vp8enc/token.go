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

// Token buffer: records the coefficient coding events before the final
// probabilities are known, then emits them.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	tokenFixedProba = 1 << 14 // token uses a constant probability
	tokenBit        = 1 << 15 // bit value
	numProbaSlots   = numTypes * numBands * numCtx * numProbas
)

type statsArray = [numBands][numCtx][numProbas]uint32

func tokenID(t, b, ctx int) int {
	return numProbas * (ctx + numCtx*(b+numBands*t))
}

// Tokens are stored in fixed-size pages, like libwebp's VP8TBuffer: the
// buffer grows one page at a time without ever copying, its footprint
// follows the actual number of tokens, and pages are recycled through
// tokenPagePool across encodes (1.5-2.5 B/px of pages for photos).
const (
	tokenPageShift = 13
	tokenPageSize  = 1 << tokenPageShift
	tokenPageMask  = tokenPageSize - 1
)

type tokenPage = [tokenPageSize]uint16

// tokenBuffer is a paged list of tokens. Tokens are addressed by a global
// index; index i lives in pages[i>>tokenPageShift][i&tokenPageMask].
type tokenBuffer struct {
	pages []*tokenPage
	cur   []uint16 // pages[n][:k], the page being filled (nil before the first token)
	n     int      // index of the current page
}

// reset empties the buffer, keeping its pages for reuse.
func (tb *tokenBuffer) reset() {
	tb.n = 0
	tb.cur = nil
	if len(tb.pages) > 0 {
		tb.cur = tb.pages[0][:0]
	}
}

// release returns all pages to the pool.
func (tb *tokenBuffer) release() {
	for i, pg := range tb.pages {
		tokenPagePool.Put(pg[:])
		tb.pages[i] = nil
	}
	tb.pages = tb.pages[:0]
	tb.cur = nil
	tb.n = 0
}

// len returns the number of tokens recorded.
func (tb *tokenBuffer) len() int { return tb.n<<tokenPageShift + len(tb.cur) }

// nextPage moves to the next page once the current one is full.
func (tb *tokenBuffer) nextPage() {
	if cap(tb.cur) != 0 {
		tb.n++
	}
	if tb.n == len(tb.pages) {
		// Pages are written before they are read: no clearing needed.
		tb.pages = append(tb.pages, (*tokenPage)(tokenPagePool.Get(tokenPageSize)))
	}
	tb.cur = tb.pages[tb.n][:0]
}

// chunk returns the tokens from index start up to end or the end of start's
// page, whichever comes first.
func (tb *tokenBuffer) chunk(start, end int) []uint16 {
	off := start & tokenPageMask
	return tb.pages[start>>tokenPageShift][off : off+min(end-start, tokenPageSize-off)]
}

// appendTokens appends already recorded tokens (the wavefront pass moves
// each row's tokens into the frame's buffer this way, in raster order).
func (tb *tokenBuffer) appendTokens(src []uint16) {
	for len(src) > 0 {
		if len(tb.cur) == cap(tb.cur) {
			tb.nextPage()
		}
		n := copy(tb.cur[len(tb.cur):cap(tb.cur)], src)
		tb.cur = tb.cur[:len(tb.cur)+n]
		src = src[n:]
	}
}

func (tb *tokenBuffer) add(bit int, idx int, s *uint32) int {
	if len(tb.cur) == cap(tb.cur) {
		tb.nextPage()
	}
	tb.cur = append(tb.cur, uint16(bit<<15|idx))
	return recordStats(bit, s)
}

func (tb *tokenBuffer) addConstant(bit int, proba int) {
	if len(tb.cur) == cap(tb.cur) {
		tb.nextPage()
	}
	tb.cur = append(tb.cur, uint16(bit<<15|tokenFixedProba|proba))
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// recordCoeffTokens records the tokens for one residual and returns whether
// it had non-zero coefficients.
func (tb *tokenBuffer) recordCoeffTokens(ctx int, res *residual, allStats *[numTypes]statsArray) int {
	coeffs := res.coeffs
	ctype := res.ctype
	last := res.last
	n := res.first
	stats := &allStats[ctype]
	baseID := tokenID(ctype, n, ctx)
	// should be stats[encBands[n]], but it's equivalent for n=0 or 1
	s := &stats[n][ctx]
	if tb.add(b2i(last >= 0), baseID+0, &s[0]) == 0 {
		return 0
	}
	for n < 16 {
		c := int(coeffs[n])
		n++
		sign := 0
		v := c
		if c < 0 {
			sign = 1
			v = -c
		}
		if tb.add(b2i(v != 0), baseID+1, &s[1]) == 0 {
			b := int(encBands[n])
			baseID = tokenID(ctype, b, 0)
			s = &stats[b][0]
			continue
		}
		if tb.add(b2i(v > 1), baseID+2, &s[2]) == 0 {
			b := int(encBands[n])
			baseID = tokenID(ctype, b, 1)
			s = &stats[b][1]
		} else {
			if tb.add(b2i(v > 4), baseID+3, &s[3]) == 0 {
				if tb.add(b2i(v != 2), baseID+4, &s[4]) != 0 {
					tb.add(b2i(v == 4), baseID+5, &s[5])
				}
			} else if tb.add(b2i(v > 10), baseID+6, &s[6]) == 0 {
				if tb.add(b2i(v > 6), baseID+7, &s[7]) == 0 {
					tb.addConstant(b2i(v == 6), 159)
				} else {
					tb.addConstant(b2i(v >= 9), 165)
					tb.addConstant(b2i(v&1 == 0), 145)
				}
			} else {
				var mask int
				var tab []uint8
				residue := v - 3
				if residue < (8 << 1) { // Cat3 (3b)
					tb.add(0, baseID+8, &s[8])
					tb.add(0, baseID+9, &s[9])
					residue -= 8 << 0
					mask = 1 << 2
					tab = vp8Cat3
				} else if residue < (8 << 2) { // Cat4 (4b)
					tb.add(0, baseID+8, &s[8])
					tb.add(1, baseID+9, &s[9])
					residue -= 8 << 1
					mask = 1 << 3
					tab = vp8Cat4
				} else if residue < (8 << 3) { // Cat5 (5b)
					tb.add(1, baseID+8, &s[8])
					tb.add(0, baseID+10, &s[10])
					residue -= 8 << 2
					mask = 1 << 4
					tab = vp8Cat5
				} else { // Cat6 (11b)
					tb.add(1, baseID+8, &s[8])
					tb.add(1, baseID+10, &s[10])
					residue -= 8 << 3
					mask = 1 << 10
					tab = vp8Cat6
				}
				for i := 0; mask != 0; i++ {
					tb.addConstant(b2i(residue&mask != 0), int(tab[i]))
					mask >>= 1
				}
			}
			b := int(encBands[n])
			baseID = tokenID(ctype, b, 2)
			s = &stats[b][2]
		}
		tb.addConstant(sign, 128)
		if n == 16 || tb.add(b2i(n <= last), baseID+0, &s[0]) == 0 {
			return 1 // EOB
		}
	}
	return 1
}

// emitTokenRange codes tokens [start, end) with the final probabilities.
func (tb *tokenBuffer) emitTokenRange(bw *bitWriter, start, end int, probas *[numProbaSlots]uint8) {
	for start < end {
		c := tb.chunk(start, end)
		emitTokens(bw, c, probas)
		start += len(c)
	}
}

// estimateSize returns the cost (1/256 bits) of all recorded tokens.
func (tb *tokenBuffer) estimateSize(probas *[numProbaSlots]uint8) uint64 {
	var size uint64
	for i, end := 0, tb.len(); i < end; {
		c := tb.chunk(i, end)
		size += estimateTokenSize(c, probas)
		i += len(c)
	}
	return size
}

// recordStatsRange accumulates the statistics of tokens [start, end).
func (tb *tokenBuffer) recordStatsRange(start, end int, stats *[numTypes]statsArray) {
	for start < end {
		c := tb.chunk(start, end)
		recordTokenStats(c, stats)
		start += len(c)
	}
}

// tokenProba returns the probability a token is coded with: its constant
// probability, or its slot in probas. For a constant token t&0x3fff is the
// probability itself (< 256), so the table load is always in range, and
// the two are selected with a mask rather than a branch (constant and slot
// tokens alternate unpredictably).
func tokenProba(t uint16, probas *[numProbaSlots]uint8) uint8 {
	p := probas[int(t&0x3fff)%numProbaSlots]
	m := -uint8(t >> 14 & 1) // 0xff for a constant-probability token
	return p&^m | uint8(t)&m
}

// kNorm256 and kNewRange256 extend kNorm and kNewRange to every value of
// range-1: values >= 127 need no renormalization (shift 0, same range).
var kNorm256, kNewRange256 = func() (norm, newRange [256]uint8) {
	for r := range 256 {
		if r < 127 {
			norm[r], newRange[r] = kNorm[r], kNewRange[r]
		} else {
			newRange[r] = uint8(r)
		}
	}
	return
}()

// emitTokens codes the tokens with the final probabilities. It is libwebp's
// VP8PutBit for every token, with the coder state kept in locals and the
// bit update and renormalization done without branches; only the byte
// output (flush) goes through the bitWriter. TestEmitTokensExact checks it
// against the straightforward coder.
func emitTokens(bw *bitWriter, toks []uint16, probas *[numProbaSlots]uint8) {
	rng, value, nbBits := bw.rng, bw.value, bw.nbBits
	for _, t := range toks {
		split := (rng * int32(tokenProba(t, probas))) >> 8
		mask := -int32(t >> 15) // bit ? -1 : 0
		value += (split + 1) & mask
		// bit ? rng - (split + 1) : split; rng stays in [0, 254].
		rng = split ^ ((split ^ (rng - split - 1)) & mask)
		shift := kNorm256[uint8(rng)]
		rng = int32(kNewRange256[uint8(rng)])
		value <<= shift & 7
		nbBits += int(shift)
		if nbBits > 0 {
			bw.rng, bw.value, bw.nbBits = rng, value, nbBits
			bw.flush()
			value, nbBits = bw.value, bw.nbBits
		}
	}
	bw.rng, bw.value, bw.nbBits = rng, value, nbBits
}

// estimateTokenSize returns the cost (1/256 bits) of the tokens.
func estimateTokenSize(toks []uint16, probas *[numProbaSlots]uint8) uint64 {
	var size uint64
	for _, t := range toks {
		// bitCost(bit, p) is entropyCost[p] for a 0 and entropyCost[255-p]
		// (= p^0xff) for a 1.
		size += uint64(entropyCost[tokenProba(t, probas)^uint8(-(t>>15))])
	}
	return size
}

// recordTokenStats accumulates the statistics of already recorded tokens.
func recordTokenStats(toks []uint16, stats *[numTypes]statsArray) {
	for _, t := range toks {
		if t&tokenFixedProba == 0 {
			idx := int(t & 0x3fff)
			p := idx % numProbas
			idx /= numProbas
			c := idx % numCtx
			idx /= numCtx
			b := idx % numBands
			ty := (idx / numBands) & 3
			recordStats(int(t>>15), &stats[ty][b][c][p])
		}
	}
}

func flattenProbas(dst *[numProbaSlots]uint8, src *[numTypes][numBands][numCtx][numProbas]uint8) {
	i := 0
	for t := range src {
		for b := range src[t] {
			for c := range src[t][b] {
				i += copy(dst[i:], src[t][b][c][:])
			}
		}
	}
}
