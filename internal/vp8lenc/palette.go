package vp8lenc

import "slices"

// Color indexing (palette) transform.
// Algorithm adapted from libwebp (BSD-3-Clause).

const maxPaletteSize = 256

// collectPalette returns the sorted unique colors of argb, or nil when there
// are more than 256 of them.
func collectPalette(argb []uint32, dst []uint32) []uint32 {
	const tabBits = 11
	const tabSize = 1 << tabBits
	var table [tabSize]uint32
	var inUse [tabSize]bool
	n := 0
	last := ^argb[0]
	for _, p := range argb {
		if p == last {
			continue
		}
		last = p
		key := (p * 0x1e35a7bd) >> (32 - tabBits)
		for {
			if !inUse[key] {
				inUse[key] = true
				table[key] = p
				n++
				if n > maxPaletteSize {
					return nil
				}
				break
			} else if table[key] == p {
				break
			}
			key = (key + 1) & (tabSize - 1)
		}
	}
	dst = dst[:0]
	for i := range table {
		if inUse[i] {
			dst = append(dst, table[i])
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
// (packed width x height).
func applyPalette(argb []uint32, width, height int, palette []uint32, dst []uint32, idxRow []uint8) {
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
		bundleColorMap(idxRow, xbits, dst[y*pw:(y+1)*pw])
	}
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
