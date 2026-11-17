//go:build arm64 && !purego

package vp8

import "encoding/binary"

// NEON row converters (yuv_arm64.s). The wrappers below check every bound
// and handle the pixels the kernels leave out; the kernels never access
// memory outside the slices they are given.

// upsampleRowsNEON converts output pixels [1, 2*uvW-1) of the top row and,
// if botY is not nil, of the bottom row with the fancy upsampler. It reads
// topY/botY/topA/botA[1 : 2*uvW-1] and topU/topV/curU/curV[:uvW], writes
// topDst/botDst[4 : 4*(2*uvW-1)] and needs uvW >= 9. topA and botA may be
// nil (opaque).
//
//go:noescape
func upsampleRowsNEON(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA *byte, uvW int)

// sampleRowNEON converts pixels [0, w) with point-sampled chroma, w even
// and >= 16. It reads y[:w], a[:w] (a may be nil: opaque), u/v[:w/2] and
// writes dst[:4*w].
//
//go:noescape
func sampleRowNEON(y, u, v, a, dst *byte, w int)

// upsamplePair converts a pair of luma rows with the fancy upsampler; see
// upsamplePairGo, which it matches bit for bit. uvT and uvB are only used
// for narrow rows.
func upsamplePair(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA []byte, uvT, uvB []uint32, w int) {
	uvW := (w + 1) >> 1
	if uvW < 9 {
		upsamplePairGo(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA, uvT, uvB, w)
		return
	}
	topY = topY[:w]
	topDst = topDst[:4*w]
	topU, topV, curU, curV = topU[:uvW], topV[:uvW], curU[:uvW], curV[:uvW]
	var pBotY, pBotDst, pTopA, pBotA *byte
	if topA != nil {
		topA = topA[:w]
		pTopA = &topA[0]
	}
	if botY != nil {
		botY = botY[:w]
		botDst = botDst[:4*w]
		pBotY, pBotDst = &botY[0], &botDst[0]
		if botA != nil {
			botA = botA[:w]
			pBotA = &botA[0]
		}
	}
	upsampleRowsNEON(&topY[0], pBotY, &topU[0], &topV[0], &curU[0], &curV[0], &topDst[0], pBotDst, pTopA, pBotA, uvW)

	// The first pixel and, for even widths, the last one take their chroma
	// from a single column: (3*near + far + 2) >> 2.
	edgePixel(topDst, 0, topY, topA, topU[0], topV[0], curU[0], curV[0])
	if w&1 == 0 {
		edgePixel(topDst, w-1, topY, topA, topU[uvW-1], topV[uvW-1], curU[uvW-1], curV[uvW-1])
	}
	if botY != nil {
		edgePixel(botDst, 0, botY, botA, curU[0], curV[0], topU[0], topV[0])
		if w&1 == 0 {
			edgePixel(botDst, w-1, botY, botA, curU[uvW-1], curV[uvW-1], topU[uvW-1], topV[uvW-1])
		}
	}
}

// edgePixel converts pixel x of a fancy-upsampled row whose chroma is
// (3*near + far + 2) >> 2 (upsampleUV's first and last pixels).
func edgePixel(dst []byte, x int, ys, as []byte, nu, nv, fu, fv byte) {
	u := (3*uint32(nu) + uint32(fu) + 2) >> 2
	v := (3*uint32(nv) + uint32(fv) + 2) >> 2
	a := uint32(0xff000000)
	if as != nil {
		a = uint32(as[x]) << 24
	}
	binary.LittleEndian.PutUint32(dst[4*x:], yuvPix(yuvTab, ys[x], uint8(u), uint8(v))|a)
}

// sampleRow converts one luma row with point-sampled chroma; see
// sampleRowGo, which it matches bit for bit.
func sampleRow(ys, us, vs, as, dst []byte, w int) {
	if w < 16 {
		sampleRowGo(ys, us, vs, as, dst, w)
		return
	}
	ys = ys[:w]
	dst = dst[:4*w]
	uvW := (w + 1) >> 1
	us, vs = us[:uvW], vs[:uvW]
	var pa *byte
	if as != nil {
		as = as[:w]
		pa = &as[0]
	}
	sampleRowNEON(&ys[0], &us[0], &vs[0], pa, &dst[0], w&^1)
	if w&1 != 0 {
		a := uint32(0xff000000)
		if as != nil {
			a = uint32(as[w-1]) << 24
		}
		binary.LittleEndian.PutUint32(dst[4*w-4:], yuvPix(yuvTab, ys[w-1], us[uvW-1], vs[uvW-1])|a)
	}
}
