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

package vp8l

// Algorithm adapted from libwebp (BSD-3-Clause): two-level Huffman lookup
// tables (src/utils/huffman_utils.c) and the HTreeGroup fast paths
// (src/dec/vp8l_dec.c).

import "github.com/kolesa-team/go-webp/v2/internal/status"

const (
	numLiteralCodes  = 256
	numLengthCodes   = 24
	numDistanceCodes = 40
	maxCacheBits     = 11
	maxAlphabetSize  = numLiteralCodes + numLengthCodes + 1<<maxCacheBits

	rootBits      = 8 // root table bits of the main Huffman tables
	lengthsBits   = 7 // root table bits of the code-length code table
	maxCodeLength = 15

	packedBits    = 6 // HUFFMAN_PACKED_BITS
	packedSize    = 1 << packedBits
	specialMarker = 0x100 // packedCode.bits marker for non-literal green codes
)

// Indices of the five Huffman trees of a group.
const (
	treeGreen = iota
	treeRed
	treeBlue
	treeAlpha
	treeDist
)

// huffCode is one Huffman lookup table entry. In a root table, an entry with
// bits > rootBits links to a second-level table: value is the offset of that
// table relative to the root entry index and bits-rootBits its index width.
type huffCode struct {
	bits  uint8
	value uint16
}

// packedCode is a packed-table entry: either a full ARGB literal decoded with
// one lookup (bits < specialMarker), or a non-literal green code
// (bits = specialMarker + code length, value = green symbol).
type packedCode struct {
	bits  uint32
	value uint32
}

// htreeGroup is the set of five Huffman codes used for one entropy tile.
type htreeGroup struct {
	trees [5][]huffCode
	// literalARB holds the constant alpha/red/blue of a trivial literal (and
	// green too when isTrivialCode).
	literalARB       uint32
	isTrivialLiteral bool // red, blue and alpha trees have a single symbol
	isTrivialCode    bool // every pixel of the tile is literalARB
	usePacked        bool // packed holds a 6-bit one-lookup table
	packed           []packedCode
}

// readSym decodes one symbol from table t using the prefetched bits val.
// It returns the symbol and the number of bits used.
func readSym(t []huffCode, val uint64) (int, uint) {
	idx := int(val & 0xff)
	e := t[idx]
	if e.bits <= rootBits {
		return int(e.value), uint(e.bits)
	}
	idx += int(e.value) + int((val>>rootBits)&(uint64(1)<<(e.bits-rootBits)-1))
	e2 := t[idx]
	return int(e2.value), uint(e2.bits) + rootBits
}

// maxTableSize returns an upper bound of the table size (root bits 8) of a
// complete code over alphabetSize symbols (values from zlib's "enough").
func maxTableSize(alphabetSize int) int {
	switch {
	case alphabetSize <= numDistanceCodes:
		return 410
	case alphabetSize <= 256:
		return 630
	}
	// Green alphabet: 280 + color cache size.
	switch alphabetSize - numLiteralCodes - numLengthCodes {
	case 0:
		return 654
	case 1 << 1:
		return 656
	case 1 << 2:
		return 658
	case 1 << 3:
		return 662
	case 1 << 4:
		return 670
	case 1 << 5:
		return 686
	case 1 << 6:
		return 718
	case 1 << 7:
		return 782
	case 1 << 8:
		return 910
	case 1 << 9:
		return 1166
	case 1 << 10:
		return 1678
	}
	return 2702
}

// buildTable builds a two-level lookup table for the given code lengths
// into table. It returns the table size, 0 if the code is invalid, or -1 if
// table is too small (the caller then retries with a bigger one).
// sorted is scratch space of at least len(lengths) entries.
func buildTable(table []huffCode, root int, lengths []uint8, sorted []uint16) int {
	var count [maxCodeLength + 1]int
	for _, l := range lengths {
		count[l&15]++ // lengths are always <= 15
	}
	numSymbols := len(lengths) - count[0]
	if numSymbols == 0 {
		return 0
	}
	var offset [maxCodeLength + 1]int
	for l := 1; l < maxCodeLength; l++ {
		if count[l] > 1<<l {
			return 0
		}
		offset[l+1] = offset[l] + count[l]
	}
	for s, l := range lengths {
		if l != 0 {
			sorted[offset[l]] = uint16(s)
			offset[l]++
		}
	}
	tableSize := 1 << root
	if len(table) < tableSize {
		return -1
	}
	// Special case: a code with only one symbol uses zero bits.
	if numSymbols == 1 {
		c := huffCode{bits: 0, value: sorted[0]}
		t := table[:tableSize]
		for i := range t {
			t[i] = c
		}
		return tableSize
	}

	totalSize := tableSize
	mask := totalSize - 1
	key := 0
	numNodes := 1
	numOpen := 1
	sym := 0
	tbl := 0
	low := -1
	// Fill the root table.
	for l, step := 1, 2; l <= root; l, step = l+1, step<<1 {
		numOpen <<= 1
		numNodes += numOpen
		numOpen -= count[l]
		if numOpen < 0 {
			return 0
		}
		for ; count[l] > 0; count[l]-- {
			replicate(table[key:], step, tableSize, huffCode{bits: uint8(l), value: sorted[sym]})
			sym++
			key = nextKey(key, l)
		}
	}
	// Fill second-level tables and link them from the root table.
	for l, step := root+1, 2; l <= maxCodeLength; l, step = l+1, step<<1 {
		numOpen <<= 1
		numNodes += numOpen
		numOpen -= count[l]
		if numOpen < 0 {
			return 0
		}
		for ; count[l] > 0; count[l]-- {
			if key&mask != low {
				tbl += tableSize
				tableBits := nextTableBits(&count, l, root)
				tableSize = 1 << tableBits
				totalSize += tableSize
				if totalSize > len(table) {
					return -1
				}
				low = key & mask
				table[low] = huffCode{bits: uint8(tableBits + root), value: uint16(tbl - low)}
			}
			replicate(table[tbl+key>>root:], step, tableSize, huffCode{bits: uint8(l - root), value: sorted[sym]})
			sym++
			key = nextKey(key, l)
		}
	}
	// The tree must be complete.
	if numNodes != 2*numSymbols-1 {
		return 0
	}
	return totalSize
}

// replicate stores c at t[0], t[step], ... below end.
func replicate(t []huffCode, step, end int, c huffCode) {
	for end > 0 {
		end -= step
		t[end] = c
	}
}

// nextKey returns the bit-reversed increment of key, for a code of length l.
func nextKey(key, l int) int {
	step := 1 << (l - 1)
	for key&step != 0 {
		step >>= 1
	}
	if step == 0 {
		return key
	}
	return key&(step-1) + step
}

// nextTableBits returns the width of the next second-level table.
func nextTableBits(count *[maxCodeLength + 1]int, l, root int) int {
	left := 1 << (l - root)
	for l < maxCodeLength {
		left -= count[l]
		if left <= 0 {
			break
		}
		l++
		left <<= 1
	}
	return l - root
}

var codeLengthCodeOrder = [19]uint8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// readHuffmanCode reads one Huffman code over alphabetSize symbols and
// appends its lookup table to d.tables (single-symbol tables are shared).
// It returns the table offset; every table spans at least 1<<rootBits
// entries from there. The code lengths stay in d.codeLengths[:alphabetSize].
func (d *decoder) readHuffmanCode(alphabetSize int) (int, error) {
	br := &d.br
	lengths := d.codeLengths[:alphabetSize]
	clear(lengths)
	if br.read(1) == 1 {
		// Simple code: one or two symbols.
		numSymbols := br.read(1) + 1
		firstBits := br.read(1)
		s := int(br.read(1 + 7*uint(firstBits)))
		if s < alphabetSize {
			lengths[s] = 1
		}
		if numSymbols == 2 {
			s = int(br.read(8))
			if s < alphabetSize {
				lengths[s] = 1
			}
		}
	} else {
		var clc [len(codeLengthCodeOrder)]uint8
		n := int(br.read(4)) + 4
		for i := 0; i < n; i++ {
			clc[codeLengthCodeOrder[i]] = uint8(br.read(3))
		}
		if err := d.readCodeLengths(&clc, lengths); err != nil {
			return 0, err
		}
	}
	if br.eos() {
		return 0, status.NotEnoughData("vp8l: truncated huffman code")
	}
	off := len(d.tables)
	need := maxTableSize(alphabetSize)
	for {
		if cap(d.tables)-off < need {
			nt := make([]huffCode, off, 2*cap(d.tables)+need)
			copy(nt, d.tables)
			d.tables = nt
		}
		size := buildTable(d.tables[off:off+need], rootBits, lengths, d.sorted[:])
		if size == 0 {
			return 0, status.Bitstream("vp8l: invalid huffman code")
		}
		if size > 0 {
			if first := d.tables[off : off+1][0]; first.bits == 0 {
				// Single-symbol code: share one table per symbol.
				sym := first.value
				if o := d.single[sym]; o > 0 {
					d.tables = d.tables[:off]
					return int(o - 1), nil
				}
				d.single[sym] = int32(off + 1)
				d.singleSet = append(d.singleSet, sym)
			}
			d.tables = d.tables[:off+size]
			return off, nil
		}
		// Table bound too small (only possible for incomplete codes).
		need *= 2
		if need > 1<<20 {
			return 0, status.Bitstream("vp8l: invalid huffman code")
		}
	}
}

// readCodeLengths reads the code lengths of a normal Huffman code.
func (d *decoder) readCodeLengths(clc *[19]uint8, lengths []uint8) error {
	table := d.lenTable[:]
	if buildTable(table, lengthsBits, clc[:], d.sorted[:]) <= 0 {
		return status.Bitstream("vp8l: invalid code length code")
	}
	br := &d.br
	numSymbols := len(lengths)
	maxSymbol := numSymbols
	if br.read(1) == 1 {
		n := 2 + 2*uint(br.read(3))
		maxSymbol = 2 + int(br.read(n))
		if maxSymbol > numSymbols {
			return status.Bitstream("vp8l: invalid max symbol")
		}
	}
	prev := uint8(8)
	for sym := 0; sym < numSymbols; {
		if maxSymbol == 0 {
			break
		}
		maxSymbol--
		if br.nbits < 32 {
			br.refill()
		}
		e := table[br.val&(1<<lengthsBits-1)]
		br.val >>= e.bits
		br.nbits -= uint(e.bits)
		cl := e.value
		if cl < 16 {
			lengths[sym] = uint8(cl)
			sym++
			if cl != 0 {
				prev = uint8(cl)
			}
			continue
		}
		var repeat int
		var v uint8
		switch cl {
		case 16:
			repeat = int(br.read(2)) + 3
			v = prev
		case 17:
			repeat = int(br.read(3)) + 3
		default:
			repeat = int(br.read(7)) + 11
		}
		if sym+repeat > numSymbols {
			return status.Bitstream("vp8l: invalid code length repeat")
		}
		for end := sym + repeat; sym < end; sym++ {
			lengths[sym] = v
		}
	}
	return nil
}

// readHuffmanCodes reads the meta Huffman codes of an image of the given
// size into md.
func (d *decoder) readHuffmanCodes(md *metadata, xsize, ysize, cacheBits int, allowRecursion bool) error {
	br := &d.br
	numGroups := 1
	var mapping []int32
	md.huffBits = 0
	md.huffMask = -1
	md.huffXSize = 0
	md.huffImage = nil
	if allowRecursion && br.read(1) == 1 {
		bits := int(br.read(3)) + 2
		hx, hy := subSampleSize(xsize, bits), subSampleSize(ysize, bits)
		img, err := d.decodeSubImage(hx, hy)
		if err != nil {
			return err
		}
		maxGroup := 0
		for i, p := range img {
			g := (p >> 8) & 0xffff
			img[i] = g
			if int(g) >= maxGroup {
				maxGroup = int(g) + 1
			}
		}
		numGroups = maxGroup
		if maxGroup > 1000 || maxGroup > len(img) {
			// Remap the used group indices to [0, n) so that bogus huge
			// indices do not cause huge allocations.
			mapping = make([]int32, maxGroup)
			for i := range mapping {
				mapping[i] = -1
			}
			n := int32(0)
			for i, g := range img {
				m := &mapping[g]
				if *m == -1 {
					*m = n
					n++
				}
				img[i] = uint32(*m)
			}
			numGroups = int(n)
		}
		md.huffBits = bits
		md.huffMask = 1<<bits - 1
		md.huffXSize = hx
		md.huffImage = img
	}

	// Tables of the previous (sub-)image are no longer needed.
	d.tables = d.tables[:0]
	d.packed = d.packed[:0]
	d.resetSingle(0)
	if cap(d.groups) < numGroups {
		d.groups = make([]htreeGroup, numGroups)
	}
	groups := d.groups[:numGroups]
	offs := d.treeOffs[:0]

	cacheSize := 0
	if cacheBits > 0 {
		cacheSize = 1 << cacheBits
	}
	alphabet := [5]int{numLiteralCodes + numLengthCodes + cacheSize, 256, 256, 256, numDistanceCodes}
	numStored := numGroups
	if mapping != nil {
		numStored = len(mapping)
	}
	for i := 0; i < numStored; i++ {
		gi := i
		if mapping != nil {
			gi = int(mapping[i])
		}
		mark := len(d.tables)
		isTrivialLiteral := true
		totalBits := 0
		maxBits := 0
		var o [5]int
		for j := 0; j < 5; j++ {
			off, err := d.readHuffmanCode(alphabet[j])
			if err != nil {
				return err
			}
			o[j] = off
			rb := int(d.tables[off].bits)
			if j >= treeRed && j <= treeAlpha && isTrivialLiteral {
				isTrivialLiteral = rb == 0
			}
			totalBits += rb
			if j <= treeAlpha {
				m := uint8(0)
				for _, l := range d.codeLengths[:alphabet[j]] {
					m = max(m, l)
				}
				maxBits += int(m)
			}
		}
		if gi < 0 {
			// Unused group: validated, then dropped.
			d.tables = d.tables[:mark]
			d.resetSingle(mark)
			continue
		}
		offs = append(offs, o)
		g := &groups[gi]
		*g = htreeGroup{isTrivialLiteral: isTrivialLiteral}
		if isTrivialLiteral {
			red := uint32(d.tables[o[treeRed]].value)
			blue := uint32(d.tables[o[treeBlue]].value)
			alpha := uint32(d.tables[o[treeAlpha]].value)
			g.literalARB = alpha<<24 | red<<16 | blue
			if green := uint32(d.tables[o[treeGreen]].value); totalBits == 0 && green < numLiteralCodes {
				g.isTrivialCode = true
				g.literalARB |= green << 8
			}
		}
		g.usePacked = !g.isTrivialCode && maxBits < packedBits
	}
	d.treeOffs = offs

	// d.tables is final now: slice the trees out of it.
	numPacked := 0
	for i := range groups {
		if groups[i].usePacked {
			numPacked++
		}
	}
	if cap(d.packed) < numPacked*packedSize {
		d.packed = make([]packedCode, numPacked*packedSize)
	}
	d.packed = d.packed[:numPacked*packedSize]
	k, kp := 0, 0
	for i := 0; i < numStored; i++ {
		gi := i
		if mapping != nil {
			gi = int(mapping[i])
			if gi < 0 {
				continue
			}
		}
		o := offs[k]
		k++
		g := &groups[gi]
		for j := 0; j < 5; j++ {
			g.trees[j] = d.tables[o[j]:]
		}
		if g.usePacked {
			g.packed = d.packed[kp : kp+packedSize : kp+packedSize]
			kp += packedSize
			buildPacked(g)
		}
	}
	md.groups = groups
	return nil
}

// resetSingle forgets shared single-symbol tables at offsets >= from.
func (d *decoder) resetSingle(from int) {
	keep := d.singleSet[:0]
	for _, sym := range d.singleSet {
		if int(d.single[sym]-1) >= from {
			d.single[sym] = 0
		} else {
			keep = append(keep, sym)
		}
	}
	d.singleSet = keep
}

// buildPacked fills g.packed: for every 6-bit input, either the complete
// ARGB literal or the green non-literal code.
func buildPacked(g *htreeGroup) {
	for code := 0; code < packedSize; code++ {
		bits := code
		h := g.trees[treeGreen][bits&0xff]
		p := &g.packed[code]
		if int(h.value) >= numLiteralCodes {
			p.bits = uint32(h.bits) + specialMarker
			p.value = uint32(h.value)
			continue
		}
		p.bits = uint32(h.bits)
		p.value = uint32(h.value) << 8
		bits >>= h.bits
		h = g.trees[treeRed][bits&0xff]
		p.bits += uint32(h.bits)
		p.value |= uint32(h.value) << 16
		bits >>= h.bits
		h = g.trees[treeBlue][bits&0xff]
		p.bits += uint32(h.bits)
		p.value |= uint32(h.value)
		bits >>= h.bits
		h = g.trees[treeAlpha][bits&0xff]
		p.bits += uint32(h.bits)
		p.value |= uint32(h.value) << 24
	}
}
