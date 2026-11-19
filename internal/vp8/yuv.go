package vp8

import (
	"encoding/binary"
	"math/bits"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Algorithm adapted from libwebp (BSD-3-Clause): dsp/yuv.h (14-bit fixed
// point YUV->RGB), dsp/upsampling.c (fancy upsampler) and dsp/yuv.c
// (point-sampling row converter).

// yuvLevel selects the x86 SIMD level of the row converters
// (yuv_amd64.go); tests lower it to cover every level. Other builds ignore
// it.
var yuvLevel = cpuinfo.Max

// clipY is libwebp's VP8Clip8 for a 14-bit fixed point value v:
// clamp(v >> 6, 0, 255). Out-of-range values are rare, so the branch is
// cheaper than unconditional clamping.
func clipY(v int) uint32 {
	v >>= 6
	if uint(v) > 255 {
		v = int(^uint(v>>(bits.UintSize-1)) & 255) // 0 if v < 0, else 255
	}
	return uint32(v)
}

// yuvTables holds libwebp's VP8YuvToRgb terms per sample value:
// y = MultHi(y,19077), vr = MultHi(v,26149)-14234,
// ub = MultHi(u,33050)-17685, ug = -MultHi(u,6419),
// vg = -MultHi(v,13320)+8708.
type yuvTables struct {
	y, vr, ub, ug, vg [256]int32
}

// yuvTab is a pointer variable (not a constant address) so that hot loops
// keep it in a register instead of rematerializing table addresses.
var yuvTab = func() *yuvTables {
	t := new(yuvTables)
	for i := 0; i < 256; i++ {
		t.y[i] = int32((i * 19077) >> 8)
		t.vr[i] = int32((i*26149)>>8 - 14234)
		t.ub[i] = int32((i*33050)>>8 - 17685)
		t.ug[i] = int32(-((i * 6419) >> 8))
		t.vg[i] = int32(-((i * 13320) >> 8) + 8708)
	}
	return t
}()

// yuvPix converts one pixel to packed little-endian R,G,B,A=0 using
// libwebp's 14-bit fixed point VP8YuvToRgb.
func yuvPix(t *yuvTables, y, u, v uint8) uint32 {
	yy := int(t.y[y])
	return clipY(yy+int(t.vr[v])) |
		clipY(yy+int(t.ug[u]+t.vg[v]))<<8 |
		clipY(yy+int(t.ub[u]))<<16
}

// opaque2 is the alpha bytes of two packed RGBA pixels. It is a variable
// on purpose: with a constant the compiler splits the 64-bit pixel store
// into byte stores.
var opaque2 uint64 = 0xff000000ff000000

// upsamplePairGo converts a pair of luma rows sharing the chroma rows
// topU/V (above) and curU/V (below) with libwebp's fancy upsampler
// (UpsampleRgbaLinePair). botY may be nil when only the top row is needed.
// uvT and uvB are scratch rows of at least w entries. topA and botA are the
// alpha rows of the two output rows, or nil for opaque output.
//
// This is the portable reference; upsamplePair (yuv_arm64.go,
// yuv_amd64.go, yuv_generic.go) is the entry point used by the decoder.
func upsamplePairGo(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA []byte, uvT, uvB []uint32, w int) {
	if botY == nil {
		upsampleUV(topU, topV, curU, curV, uvT, nil, w)
		convertRow(topY, uvT, topA, topDst, w)
		return
	}
	upsampleUV(topU, topV, curU, curV, uvT, uvB, w)
	convertRow(topY, uvT, topA, topDst, w)
	convertRow(botY, uvB, botA, botDst, w)
}

// upsampleUV computes the full resolution chroma (packed u | v<<16) of the
// top and, if uvB is not nil, bottom luma rows lying between chroma rows
// top and cur. This is the chroma part of libwebp's UPSAMPLE_FUNC.
func upsampleUV(topU, topV, curU, curV []byte, uvT, uvB []uint32, w int) {
	uvW := (w + 1) >> 1
	topU, topV, curU, curV = topU[:uvW], topV[:uvW], curU[:uvW], curV[:uvW]
	uvT = uvT[:w]
	tl := uint32(topU[0]) | uint32(topV[0])<<16 // top-left sample
	l := uint32(curU[0]) | uint32(curV[0])<<16  // left sample
	last := (w - 1) >> 1
	uvT[0] = (3*tl + l + 0x00020002) >> 2
	if uvB != nil {
		uvB = uvB[:w]
		uvB[0] = (3*l + tl + 0x00020002) >> 2
		for x := 1; x <= last; x++ {
			t := uint32(topU[x]) | uint32(topV[x])<<16 // top sample
			c := uint32(curU[x]) | uint32(curV[x])<<16 // sample
			avg := tl + t + l + c + 0x00080008
			diag12 := (avg + 2*(t+l)) >> 3
			diag03 := (avg + 2*(tl+c)) >> 3
			ut := (*[2]uint32)(uvT[2*x-1:])
			ut[0] = (diag12 + tl) >> 1
			ut[1] = (diag03 + t) >> 1
			ub := (*[2]uint32)(uvB[2*x-1:])
			ub[0] = (diag03 + l) >> 1
			ub[1] = (diag12 + c) >> 1
			tl, l = t, c
		}
		if w&1 == 0 {
			uvT[w-1] = (3*tl + l + 0x00020002) >> 2
			uvB[w-1] = (3*l + tl + 0x00020002) >> 2
		}
		return
	}
	for x := 1; x <= last; x++ {
		t := uint32(topU[x]) | uint32(topV[x])<<16
		c := uint32(curU[x]) | uint32(curV[x])<<16
		avg := tl + t + l + c + 0x00080008
		ut := (*[2]uint32)(uvT[2*x-1:])
		ut[0] = (((avg + 2*(t+l)) >> 3) + tl) >> 1
		ut[1] = (((avg + 2*(tl+c)) >> 3) + t) >> 1
		tl, l = t, c
	}
	if w&1 == 0 {
		uvT[w-1] = (3*tl + l + 0x00020002) >> 2
	}
}

// convertRow converts a luma row and its packed full resolution chroma to
// RGBA. as is the alpha row, or nil for opaque output.
func convertRow(ys []byte, uv []uint32, as, dst []byte, w int) {
	ys = ys[:w]
	uv = uv[:w]
	dst = dst[:4*w]
	if as != nil {
		as = as[:w]
	}
	a2 := opaque2
	t := yuvTab
	x := 0
	for ; x+1 < w; x += 2 {
		if as != nil {
			a2 = uint64(as[x])<<24 | uint64(as[x+1])<<56
		}
		// Two pixels of yuvPix, written out so that it gets inlined.
		y2 := (*[2]byte)(ys[x:])
		c2 := (*[2]uint32)(uv[x:])
		u0, v0 := uint8(c2[0]), uint8(c2[0]>>16)
		u1, v1 := uint8(c2[1]), uint8(c2[1]>>16)
		y0 := int(t.y[y2[0]])
		y1 := int(t.y[y2[1]])
		pa := clipY(y0+int(t.vr[v0])) | clipY(y0+int(t.ug[u0]+t.vg[v0]))<<8 | clipY(y0+int(t.ub[u0]))<<16
		pb := clipY(y1+int(t.vr[v1])) | clipY(y1+int(t.ug[u1]+t.vg[v1]))<<8 | clipY(y1+int(t.ub[u1]))<<16
		binary.LittleEndian.PutUint64(dst[4*x:], uint64(pa)|uint64(pb)<<32|a2)
	}
	if x < w {
		c := uv[x]
		a := uint32(0xff000000)
		if as != nil {
			a = uint32(as[x]) << 24
		}
		binary.LittleEndian.PutUint32(dst[4*x:], yuvPix(t, ys[x], uint8(c), uint8(c>>16))|a)
	}
}

// edgePixel converts pixel x of a fancy-upsampled row whose chroma is
// (3*near + far + 2) >> 2 (upsampleUV's first and last pixels). The SIMD
// wrappers use it for the pixels their kernels leave out.
func edgePixel(dst []byte, x int, ys, as []byte, nu, nv, fu, fv byte) {
	u := (3*uint32(nu) + uint32(fu) + 2) >> 2
	v := (3*uint32(nv) + uint32(fv) + 2) >> 2
	a := uint32(0xff000000)
	if as != nil {
		a = uint32(as[x]) << 24
	}
	binary.LittleEndian.PutUint32(dst[4*x:], yuvPix(yuvTab, ys[x], uint8(u), uint8(v))|a)
}

// sampleRowGo converts one luma row using nearest (point-sampled) chroma.
// as is the alpha row, or nil for opaque output. This is the portable
// reference; sampleRow is the entry point used by the decoder.
func sampleRowGo(ys, us, vs, as, dst []byte, w int) {
	ys = ys[:w]
	dst = dst[:4*w]
	uvW := (w + 1) >> 1
	us, vs = us[:uvW], vs[:uvW]
	if as != nil {
		as = as[:w]
	}
	a2 := opaque2
	t := yuvTab
	for x := 0; x+1 < w; x += 2 {
		if as != nil {
			a2 = uint64(as[x])<<24 | uint64(as[x+1])<<56
		}
		u, v := us[x>>1], vs[x>>1]
		y2 := (*[2]byte)(ys[x:])
		pa := yuvPix(t, y2[0], u, v)
		pb := yuvPix(t, y2[1], u, v)
		binary.LittleEndian.PutUint64(dst[4*x:], uint64(pa)|uint64(pb)<<32|a2)
	}
	if w&1 != 0 {
		a := uint32(0xff000000)
		if as != nil {
			a = uint32(as[w-1]) << 24
		}
		binary.LittleEndian.PutUint32(dst[4*w-4:], yuvPix(t, ys[w-1], us[uvW-1], vs[uvW-1])|a)
	}
}
