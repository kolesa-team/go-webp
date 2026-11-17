package vp8enc

import (
	"math"
	"sync"
)

// Iterative "sharp" RGB -> YUV 4:2:0 conversion (libsharpyuv), for 8-bit
// input with the sRGB transfer function and libwebp's conversion matrix.
// Algorithm adapted from libwebp's sharpyuv (BSD-3-Clause).

const (
	sharpIterations   = 4
	sharpYBits        = 10 // 8-bit input + 2 bits of precision
	gammaToLinearBits = 16
	g2lTabBits        = 10
	l2gTabBits        = 9
)

var (
	sharpOnce   sync.Once
	sharpG2LTab [1<<g2lTabBits + 2]uint32
	sharpL2GTab [1<<l2gTabBits + 2]uint32
)

func initSharpTables() {
	sharpOnce.Do(func() {
		const a = 0.09929682680944
		const thresh = 0.018053968510807
		const finalScale = 1 << gammaToLinearBits
		const kGamma = 1. / 0.45
		{
			norm := 1. / (1 << g2lTabBits)
			aRec := 1. / (1. + a)
			for v := 0; v <= 1<<g2lTabBits; v++ {
				g := norm * float64(v)
				var value float64
				if g <= thresh*4.5 {
					value = g / 4.5
				} else {
					value = math.Pow(aRec*(g+a), kGamma)
				}
				sharpG2LTab[v] = uint32(value*finalScale + .5)
			}
			sharpG2LTab[1<<g2lTabBits+1] = sharpG2LTab[1<<g2lTabBits]
		}
		{
			scale := 1. / (1 << l2gTabBits)
			for v := 0; v <= 1<<l2gTabBits; v++ {
				g := scale * float64(v)
				var value float64
				if g <= thresh {
					value = 4.5 * g
				} else {
					value = (1.+a)*math.Pow(g, 1./kGamma) - a
				}
				sharpL2GTab[v] = uint32(finalScale*value + 0.5)
			}
			sharpL2GTab[1<<l2gTabBits+1] = sharpL2GTab[1<<l2gTabBits]
		}
	})
}

// sharpToLinear converts a 10-bit gamma value to 16-bit linear.
func sharpToLinear(v uint16) uint32 {
	if v > 1<<g2lTabBits {
		v = 1 << g2lTabBits
	}
	return sharpG2LTab[v]
}

// sharpFromLinear converts a 16-bit linear value to 10-bit gamma.
func sharpFromLinear(value uint32) uint16 {
	const shiftR = gammaToLinearBits - l2gTabBits // 7
	tabPos := value >> shiftR
	if tabPos > 1<<l2gTabBits {
		tabPos = 1 << l2gTabBits
	}
	x := value - (tabPos << shiftR)
	v0 := sharpL2GTab[tabPos] >> (gammaToLinearBits - sharpYBits)
	v1 := sharpL2GTab[tabPos+1] >> (gammaToLinearBits - sharpYBits)
	v2 := (v1 - v0) * x
	return uint16(v0 + ((v2 + (1 << (shiftR - 1))) >> shiftR))
}

func rgbToGray(r, g, b int64) int {
	return int((13933*r + 46871*g + 4732*b + yuvHalf) >> yuvFix)
}

func scaleDown(a, b, c, d uint16) int {
	A := sharpToLinear(a)
	B := sharpToLinear(b)
	C := sharpToLinear(c)
	D := sharpToLinear(d)
	return int(sharpFromLinear((A + B + C + D + 2) >> 2))
}

func sharpUpdateW(src []uint16, dst []uint16, w int) {
	for i := 0; i < w; i++ {
		R := sharpToLinear(src[0*w+i])
		G := sharpToLinear(src[1*w+i])
		B := sharpToLinear(src[2*w+i])
		Y := rgbToGray(int64(R), int64(G), int64(B))
		dst[i] = sharpFromLinear(uint32(Y))
	}
}

func sharpUpdateChroma(src1, src2 []uint16, dst []int16, uvw int) {
	for i := 0; i < uvw; i++ {
		r := scaleDown(src1[0*uvw*2+2*i], src1[0*uvw*2+2*i+1], src2[0*uvw*2+2*i], src2[0*uvw*2+2*i+1])
		g := scaleDown(src1[1*uvw*2+2*i], src1[1*uvw*2+2*i+1], src2[1*uvw*2+2*i], src2[1*uvw*2+2*i+1])
		b := scaleDown(src1[2*uvw*2+2*i], src1[2*uvw*2+2*i+1], src2[2*uvw*2+2*i], src2[2*uvw*2+2*i+1])
		W := rgbToGray(int64(r), int64(g), int64(b))
		dst[0*uvw+i] = int16(r - W)
		dst[1*uvw+i] = int16(g - W)
		dst[2*uvw+i] = int16(b - W)
	}
}

func sharpStoreGray(rgb []uint16, y []uint16, w int) {
	for i := 0; i < w; i++ {
		y[i] = uint16(rgbToGray(int64(rgb[0*w+i]), int64(rgb[1*w+i]), int64(rgb[2*w+i])))
	}
}

func clipY10(v int) uint16 {
	const mx = 1<<sharpYBits - 1
	if v < 0 {
		return 0
	}
	if v > mx {
		return mx
	}
	return uint16(v)
}

func sharpFilter2(A, B, W0 int) uint16 {
	v0 := (A*3 + B + 2) >> 2
	return clipY10(v0 + W0)
}

func sharpImportRow(src []byte, picWidth int, dst []uint16) {
	w := (picWidth + 1) &^ 1
	for i := 0; i < picWidth; i++ {
		p := src[4*i : 4*i+3 : 4*i+3]
		dst[i+0*w] = uint16(p[0]) << 2
		dst[i+1*w] = uint16(p[1]) << 2
		dst[i+2*w] = uint16(p[2]) << 2
	}
	if picWidth&1 != 0 {
		dst[picWidth+0*w] = dst[picWidth+0*w-1]
		dst[picWidth+1*w] = dst[picWidth+1*w-1]
		dst[picWidth+2*w] = dst[picWidth+2*w-1]
	}
}

func sharpFilterRow(A, B []int16, n int, bestY []uint16, out []uint16) {
	for i := 0; i < n; i++ {
		a0, a1, b0, b1 := int(A[i]), int(A[i+1]), int(B[i]), int(B[i+1])
		v0 := (a0*9 + a1*3 + b0*3 + b1 + 8) >> 4
		v1 := (a1*9 + a0*3 + b1*3 + b0 + 8) >> 4
		out[2*i+0] = clipY10(int(bestY[2*i+0]) + v0)
		out[2*i+1] = clipY10(int(bestY[2*i+1]) + v1)
	}
}

func sharpInterpolateTwoRows(bestY []uint16, prevUV, curUV, nextUV []int16, w int, out1, out2 []uint16) {
	uvw := w >> 1
	n := (w - 1) >> 1
	for k := 0; k < 3; k++ {
		cu := curUV[k*uvw:]
		pu := prevUV[k*uvw:]
		nu := nextUV[k*uvw:]
		o1 := out1[k*w:]
		o2 := out2[k*w:]
		o1[0] = sharpFilter2(int(cu[0]), int(pu[0]), int(bestY[0]))
		o2[0] = sharpFilter2(int(cu[0]), int(nu[0]), int(bestY[w]))
		sharpFilterRow(cu, pu, n, bestY[1:], o1[1:])
		sharpFilterRow(cu, nu, n, bestY[w+1:], o2[1:])
		if w&1 == 0 {
			o1[w-1] = sharpFilter2(int(cu[uvw-1]), int(pu[uvw-1]), int(bestY[w-1]))
			o2[w-1] = sharpFilter2(int(cu[uvw-1]), int(nu[uvw-1]), int(bestY[w-1+w]))
		}
	}
}

func rgbToYUVComponent(r, g, b int, c0, c1, c2, c3 int) int {
	const sfix = 2
	const srounder = 1 << (yuvFix + sfix - 1)
	return (c0*r + c1*g + c2*b + c3 + srounder) >> (yuvFix + sfix)
}

func clip8i(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func sharpRGBToYUV(pix []byte, stride, width, height int, pic *Picture) {
	initSharpTables()
	w := (width + 1) &^ 1
	h := (height + 1) &^ 1
	uvw := w >> 1
	uvh := h >> 1
	// Scratch from the pools, not cleared: every element is written before
	// it is read (the setup loop below covers all w*h luma and uvw*3*uvh
	// chroma samples, the per-row buffers are filled before use). The
	// planes are separate buffers so that each stays within the
	// per-buffer share of the pool budget for large pictures.
	rows := uint16Pool.Get(w*3*2 + w*2)
	defer uint16Pool.Put(rows)
	tmp, bestRGBY := rows[:w*3*2:w*3*2], rows[w*3*2:]
	bestYBase := uint16Pool.Get(w * h)
	defer uint16Pool.Put(bestYBase)
	targetYBase := uint16Pool.Get(w * h)
	defer uint16Pool.Put(targetYBase)
	bestUVBase := int16Pool.Get(uvw * 3 * uvh)
	defer int16Pool.Put(bestUVBase)
	targetUVBase := int16Pool.Get(uvw * 3 * uvh)
	defer int16Pool.Put(targetUVBase)
	bestRGBUV := int16Pool.Get(uvw * 3)
	defer int16Pool.Put(bestRGBUV)
	diffYThreshold := uint64(3.0 * float64(w) * float64(h))

	src1 := tmp[0 : 3*w]
	src2 := tmp[3*w : 6*w]
	for j := 0; j < height; j += 2 {
		isLastRow := j == height-1
		sharpImportRow(pix[j*stride:], width, src1)
		if !isLastRow {
			sharpImportRow(pix[(j+1)*stride:], width, src2)
		} else {
			copy(src2, src1)
		}
		bestY := bestYBase[j*w:]
		targetY := targetYBase[j*w:]
		sharpStoreGray(src1, bestY[0:], w)
		sharpStoreGray(src2, bestY[w:], w)
		sharpUpdateW(src1, targetY, w)
		sharpUpdateW(src2, targetY[w:], w)
		uvo := (j >> 1) * 3 * uvw
		sharpUpdateChroma(src1, src2, targetUVBase[uvo:], uvw)
		copy(bestUVBase[uvo:uvo+3*uvw], targetUVBase[uvo:uvo+3*uvw])
	}

	prevDiffYSum := ^uint64(0)
	for iter := 0; iter < sharpIterations; iter++ {
		curUV := bestUVBase
		prevUV := bestUVBase
		diffYSum := uint64(0)
		for j := 0; j < h; j += 2 {
			uvo := (j >> 1) * 3 * uvw
			nextUV := curUV
			if j < h-2 {
				nextUV = curUV[3*uvw:]
			}
			bestY := bestYBase[j*w:]
			sharpInterpolateTwoRows(bestY, prevUV, curUV, nextUV, w, src1, src2)
			prevUV = curUV
			curUV = nextUV

			sharpUpdateW(src1, bestRGBY[0:], w)
			sharpUpdateW(src2, bestRGBY[w:], w)
			sharpUpdateChroma(src1, src2, bestRGBUV, uvw)

			targetY := targetYBase[j*w:]
			for i := 0; i < 2*w; i++ {
				diffY := int(targetY[i]) - int(bestRGBY[i])
				bestY[i] = clipY10(int(bestY[i]) + diffY)
				if diffY < 0 {
					diffYSum += uint64(-diffY)
				} else {
					diffYSum += uint64(diffY)
				}
			}
			targetUV := targetUVBase[uvo:]
			bestUV := bestUVBase[uvo:]
			for i := 0; i < 3*uvw; i++ {
				bestUV[i] += targetUV[i] - bestRGBUV[i]
			}
		}
		if iter > 0 {
			if diffYSum < diffYThreshold {
				break
			}
			if diffYSum > prevDiffYSum {
				break
			}
		}
		prevDiffYSum = diffYSum
	}

	// Convert to final YUV.
	const c3Y = (16 << 16) << 2
	const c3UV = (128 << 16) << 2
	for j := 0; j < height; j++ {
		bestY := bestYBase[j*w:]
		bestUV := bestUVBase[(j>>1)*3*uvw:]
		dst := pic.Y[j*pic.YStride : j*pic.YStride+width]
		for i := range dst {
			off := i >> 1
			W := int(bestY[i])
			r := int(bestUV[off+0*uvw]) + W
			g := int(bestUV[off+1*uvw]) + W
			b := int(bestUV[off+2*uvw]) + W
			dst[i] = clip8i(rgbToYUVComponent(r, g, b, 16839, 33059, 6420, c3Y))
		}
	}
	cw := (width + 1) >> 1
	for j := 0; j < (height+1)>>1; j++ {
		bestUV := bestUVBase[j*3*uvw:]
		du := pic.U[j*pic.UVStride : j*pic.UVStride+cw]
		dv := pic.V[j*pic.UVStride : j*pic.UVStride+cw]
		for i := range du {
			r := int(bestUV[i+0*uvw])
			g := int(bestUV[i+1*uvw])
			b := int(bestUV[i+2*uvw])
			du[i] = clip8i(rgbToYUVComponent(r, g, b, -9719, -19081, 28800, c3UV))
			dv[i] = clip8i(rgbToYUVComponent(r, g, b, 28800, -24116, -4684, c3UV))
		}
	}
}
