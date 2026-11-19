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

// Color indexing (palette) transform.
// Algorithm adapted from libwebp (BSD-3-Clause).

const maxPaletteSize = 256

// collectPalette returns the sorted unique colors of argb, or nil when there
// are more than 256 of them.
func collectPalette(argb []uint32, dst []uint32) []uint32 {
	var s colorSet
	s.add(argb)
	return s.palette(dst)
}

const (
	colorSetBits = 11
	colorSetSize = 1 << colorSetBits
)

// colorSet collects the distinct colors of an image (in any number of
// add calls) until there are more than maxPaletteSize of them. The zero
// value is an empty set.
type colorSet struct {
	table   [colorSetSize]uint32
	inUse   [colorSetSize]bool
	n       int
	last    uint32 // the last color added (valid when n > 0)
	tooMany bool
}

// add adds the colors of argb.
func (s *colorSet) add(argb []uint32) {
	if s.tooMany || len(argb) == 0 {
		return
	}
	last := s.last
	if s.n == 0 {
		last = ^argb[0]
	}
	for _, p := range argb {
		if p == last {
			continue
		}
		last = p
		key := (p * 0x1e35a7bd) >> (32 - colorSetBits)
		for {
			if !s.inUse[key] {
				s.inUse[key] = true
				s.table[key] = p
				s.n++
				if s.n > maxPaletteSize {
					s.tooMany = true
					return
				}
				break
			} else if s.table[key] == p {
				break
			}
			key = (key + 1) & (colorSetSize - 1)
		}
	}
	s.last = last
}

// palette returns the sorted colors of the set in dst, or nil when there
// are more than maxPaletteSize of them.
func (s *colorSet) palette(dst []uint32) []uint32 {
	if s.tooMany {
		return nil
	}
	dst = dst[:0]
	for i := range s.table {
		if s.inUse[i] {
			dst = append(dst, s.table[i])
		}
	}
	slices.Sort(dst)
	return dst
}

func paletteComponentDistance(v uint32) uint32 {
	if v <= 128 {
		return v
	}
	return 256 - v
}

func paletteColorDistance(c1, c2 uint32) uint32 {
	d := subPixels(c1, c2)
	s := paletteComponentDistance(d&0xff) + paletteComponentDistance((d>>8)&0xff) +
		paletteComponentDistance((d>>16)&0xff)
	s *= 9
	s += paletteComponentDistance(d >> 24)
	return s
}

// paletteHasNonMonotonousDeltas reports whether the sorted palette has
// deltas that change sign in some channel.
func paletteHasNonMonotonousDeltas(p []uint32) bool {
	predict := uint32(0)
	signFound := uint8(0)
	for _, c := range p {
		diff := subPixels(c, predict)
		rd := (diff >> 16) & 0xff
		gd := (diff >> 8) & 0xff
		bd := diff & 0xff
		if rd != 0 {
			if rd < 0x80 {
				signFound |= 1
			} else {
				signFound |= 2
			}
		}
		if gd != 0 {
			if gd < 0x80 {
				signFound |= 8
			} else {
				signFound |= 16
			}
		}
		if bd != 0 {
			if bd < 0x80 {
				signFound |= 64
			} else {
				signFound |= 128
			}
		}
		predict = c
	}
	return signFound&(signFound<<1) != 0
}

// paletteSortMinimizeDeltas reorders the palette greedily so that
// consecutive entries are close (cheaper delta-coded palette).
func paletteSortMinimizeDeltas(sorted []uint32, out []uint32) []uint32 {
	out = append(out[:0], sorted...)
	if !paletteHasNonMonotonousDeltas(sorted) {
		return out
	}
	n := len(out)
	if n > 17 && out[0] == 0 {
		n--
		out[n], out[0] = out[0], out[n]
	}
	predict := uint32(0)
	for i := 0; i < n; i++ {
		bestIx := i
		bestScore := ^uint32(0)
		for k := i; k < n; k++ {
			s := paletteColorDistance(out[k], predict)
			if s < bestScore {
				bestScore = s
				bestIx = k
			}
		}
		out[bestIx], out[i] = out[i], out[bestIx]
		predict = out[i]
	}
	return out
}

// paletteXBits returns the pixel bundling shift for a palette size.
func paletteXBits(n int) int {
	switch {
	case n <= 2:
		return 3
	case n <= 4:
		return 2
	case n <= 16:
		return 1
	}
	return 0
}

// applyPalette maps argb (width x height) to bundled palette indices in dst
// (packed width x height). dst may be argb itself: each row is read entirely
// (into idxRow) before its bundled row, which starts no later, is written.
// It returns initRefsCap of the bundled image (computed on the way).
func applyPalette(argb []uint32, width, height int, palette []uint32, dst []uint32, idxRow []uint8) (refsCap int) {
	xbits := paletteXBits(len(palette))
	pw := subSampleSize(width, xbits)
	// Small open-addressing lookup table color -> index.
	const tabBits = 10
	var keys [1 << tabBits]uint32
	var vals [1 << tabBits]int16
	for i := range vals {
		vals[i] = -1
	}
	for i, c := range palette {
		k := (c * 0x1e35a7bd) >> (32 - tabBits)
		for vals[k] >= 0 {
			k = (k + 1) & (1<<tabBits - 1)
		}
		keys[k] = c
		vals[k] = int16(i)
	}
	lookup := func(c uint32) uint8 {
		k := (c * 0x1e35a7bd) >> (32 - tabBits)
		for keys[k] != c || vals[k] < 0 {
			k = (k + 1) & (1<<tabBits - 1)
		}
		return uint8(vals[k])
	}
	idxRow = idxRow[:width]
	est := 1
	for y := 0; y < height; y++ {
		row := argb[y*width : (y+1)*width]
		prevC := ^row[0]
		var prevI uint8
		for x, c := range row {
			if c != prevC {
				prevC = c
				prevI = lookup(c)
			}
			idxRow[x] = prevI
		}
		out := dst[y*pw : (y+1)*pw]
		bundleColorMap(idxRow, xbits, out)
		if y == 0 {
			est += newPixels(out[1:], nil, out[0])
		} else {
			est += newPixels(out, dst[(y-1)*pw:y*pw], dst[y*pw-1])
		}
	}
	return refsCapFromEstimate(pw*height, est)
}

// bundleColorMap packs 1<<xbits palette indices per output pixel (green).
func bundleColorMap(row []uint8, xbits int, dst []uint32) {
	if xbits > 0 {
		bitDepth := uint(1) << uint(3-xbits)
		mask := (1 << uint(xbits)) - 1
		code := uint32(argbBlack)
		for x, v := range row {
			xsub := x & mask
			if xsub == 0 {
				code = argbBlack
			}
			code |= uint32(v) << (8 + bitDepth*uint(xsub))
			dst[x>>uint(xbits)] = code
		}
	} else {
		for x, v := range row {
			dst[x] = argbBlack | uint32(v)<<8
		}
	}
}
