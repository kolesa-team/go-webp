//go:build arm64 && !purego

package vp8

// NEON loop filters (filter_arm64.s), one call per macroblock.

// filterMBComplexNEON filters the edges of one macroblock selected by flags
// (filterEdgeLeft, filterEdgeTop, filterEdgeInner) with the complex filter,
// luma then U and V. y, u and v point at the macroblock's top-left samples;
// rows -4..15 / -4..7 and columns -4..15 / -4..7 around them must be
// accessible for the filtered edges (see blockInBounds). thrMB = limit+4
// and thrIn = limit must be at most 254.
//
//go:noescape
func filterMBComplexNEON(y, u, v *byte, yStride, uvStride, thrMB, thrIn, ilevel, hevThresh, flags int)

// filterMBSimpleNEON filters the luma edges selected by flags with the
// simple filter; rows -2..15 and columns -2..15 must be accessible for the
// filtered edges.
//
//go:noescape
func filterMBSimpleNEON(y *byte, yStride, thrMB, thrIn, flags int)

// filterMB applies the in-loop filter to the macroblock at column mbX of
// the current cache row (libwebp's DoFilter); it matches filterMBGo bit for
// bit.
func (d *decoder) filterMB(mbX, mbY int, f fInfo) {
	limit := int(f.limit)
	if limit == 0 {
		return
	}
	flags := 0
	if mbX > 0 {
		flags |= filterEdgeLeft
	}
	if mbY > 0 {
		flags |= filterEdgeTop
	}
	if f.inner != 0 {
		flags |= filterEdgeInner
	}
	if flags == 0 {
		return
	}
	ys := d.cacheYStride
	yo := d.cacheYOff + mbX*16
	cy := d.cacheY
	// The kernels compare thresholds as bytes; VP8 limits are at most
	// 2*63+63 = 189, so the fallback is never taken by valid streams.
	if limit > 250 {
		d.filterMBGo(mbX, mbY, f)
		return
	}
	if d.filterType == 1 {
		if !blockInBounds(len(cy), yo, ys, 16, 2, flags) {
			d.filterMBGo(mbX, mbY, f)
			return
		}
		filterMBSimpleNEON(&cy[yo], ys, limit+4, limit, flags)
		return
	}
	uvs := d.cacheUVStride
	uvo := d.cacheUVOff + mbX*8
	cu, cv := d.cacheU, d.cacheV
	if !blockInBounds(len(cy), yo, ys, 16, 4, flags) || !blockInBounds(min(len(cu), len(cv)), uvo, uvs, 8, 4, flags) {
		d.filterMBGo(mbX, mbY, f)
		return
	}
	filterMBComplexNEON(&cy[yo], &cu[uvo], &cv[uvo], ys, uvs, limit+4, limit, int(f.ilevel), int(f.hevThresh), flags)
}
