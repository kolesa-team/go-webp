package alpha

// Algorithm adapted from libwebp (BSD-3-Clause): src/utils/quant_levels_utils.c (QuantizeLevels) and
// src/utils/quant_levels_dec_utils.c (WebPDequantizeLevels).

import (
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// QuantizeLevels reduces the width x height plane in place to at most
// numLevels (2..256) distinct values with libwebp's k-means
// (QuantizeLevels in src/utils/quant_levels_utils.c). The encoder uses it
// when alpha_quality < 100, with numLevels = AlphaLevels(alpha_quality).
func QuantizeLevels(plane []byte, width, height, stride, numLevels int) error {
	if width <= 0 || height <= 0 || stride < width || numLevels < 2 || numLevels > 256 {
		return status.InvalidParam("alpha: invalid QuantizeLevels arguments")
	}
	if len(plane) < (height-1)*stride+width {
		return status.InvalidParam("alpha: plane too small")
	}
	const (
		maxIter        = 6
		errorThreshold = 1e-4
	)
	var freq [256]int
	var qLevel [256]int
	var invQLevel [256]float64
	minS, maxS := 255, 0
	numLevelsIn := 0
	for y := 0; y < height; y++ {
		for _, v := range plane[y*stride : y*stride+width] {
			freq[v]++
		}
	}
	for s, f := range freq {
		if f == 0 {
			continue
		}
		numLevelsIn++
		minS = min(minS, s)
		maxS = max(maxS, s)
	}
	if numLevelsIn <= numLevels {
		return nil
	}

	for i := 0; i < numLevels; i++ {
		invQLevel[i] = float64(minS) + float64(float64(maxS-minS)*float64(i))/float64(numLevels-1)
	}
	qLevel[minS] = 0
	qLevel[maxS] = numLevels - 1

	errThreshold := errorThreshold * float64(width*height)
	lastErr := 1e38
	for iter := 0; iter < maxIter; iter++ {
		var qSum, qCount [256]float64
		slot := 0
		for s := minS; s <= maxS; s++ {
			for slot < numLevels-1 && float64(2*s) > invQLevel[slot]+invQLevel[slot+1] {
				slot++
			}
			if freq[s] > 0 {
				qSum[slot] += float64(s * freq[s])
				qCount[slot] += float64(freq[s])
			}
			qLevel[s] = slot
		}
		if numLevels > 2 {
			for slot := 1; slot < numLevels-1; slot++ {
				if c := qCount[slot]; c > 0 {
					invQLevel[slot] = qSum[slot] / c
				}
			}
		}
		e := 0.0
		for s := minS; s <= maxS; s++ {
			d := float64(s) - invQLevel[qLevel[s]]
			// Explicit conversions forbid fused multiply-add, keeping the
			// rounding identical to libwebp's C code.
			e += float64(float64(float64(freq[s])*d) * d)
		}
		if lastErr-e < errThreshold {
			break
		}
		lastErr = e
	}

	var m [256]byte
	for s := minS; s <= maxS; s++ {
		m[s] = byte(int(invQLevel[qLevel[s]] + .5))
	}
	for y := 0; y < height; y++ {
		row := plane[y*stride : y*stride+width]
		for i, v := range row {
			row[i] = m[v]
		}
	}
	return nil
}

// Dequantization smoothing parameters (libwebp quant_levels_dec_utils.c).
const (
	fix     = 16 // fixed-point precision for averaging
	lfix    = 2  // extra precision for look-up table
	lutSize = (1 << (8 + lfix)) - 1
)

type smoother struct {
	scratch    []uint16 // (2*radius+2) rows of width: R circular rows + output row
	average    []uint16
	correction [1 + 2*lutSize]int16
}

var smootherPool = sync.Pool{New: func() any { return new(smoother) }}

// Dequantize smooths the width x height alpha plane in place to hide
// quantization banding, exactly like libwebp's WebPDequantizeLevels (the
// decoder's alpha_dithering_strength, 0..100). It should only be applied to
// planes whose ALPH header has PreProcessing == 1.
func Dequantize(plane []byte, width, height, stride, strength int) error {
	if strength < 0 || strength > 100 || width <= 0 || height <= 0 || stride < width {
		return status.InvalidParam("alpha: invalid Dequantize arguments")
	}
	if len(plane) < (height-1)*stride+width {
		return status.InvalidParam("alpha: plane too small")
	}
	radius := 4 * strength / 100
	if 2*radius+1 > width {
		radius = (width - 1) >> 1
	}
	if 2*radius+1 > height {
		radius = (height - 1) >> 1
	}
	if radius <= 0 {
		return nil
	}

	// Level statistics.
	var used [256]bool
	minV, maxV := 255, 0
	for y := 0; y < height; y++ {
		for _, v := range plane[y*stride : y*stride+width] {
			used[v] = true
		}
	}
	numLevels := 0
	for i, u := range used {
		if u {
			numLevels++
			minV = min(minV, i)
			maxV = max(maxV, i)
		}
	}
	if numLevels <= 2 {
		return nil
	}
	// Smallest distance between two consecutive used levels.
	minDist, last := maxV-minV, -1
	for i, u := range used {
		if u {
			if last >= 0 && i-last < minDist {
				minDist = i - last
			}
			last = i
		}
	}

	sm := smootherPool.Get().(*smoother)
	defer smootherPool.Put(sm)
	r := 2*radius + 1
	need := (r + 1) * width
	if cap(sm.scratch) < need {
		sm.scratch = make([]uint16, need)
	}
	if cap(sm.average) < width {
		sm.average = make([]uint16, width)
	}
	scratch := sm.scratch[:need]
	average := sm.average[:width]
	initCorrectionLUT(&sm.correction, minDist)
	correction := sm.correction[:]

	// Circular buffer of r rows [0, r*width) plus the output row "end".
	end := scratch[r*width : (r+1)*width]
	cur := 0
	top := (r - 1) * width
	clear(scratch[top : top+width])
	scale := uint32((1 << (fix + lfix)) / (r * r))
	src := 0
	dst := 0
	for row := -radius; row < height; row++ {
		// VFilter: vertical accumulation (all arithmetic modulo 2^16).
		{
			in := plane[src : src+width]
			c := scratch[cur : cur+width]
			t := scratch[top : top+width]
			out := end[:width]
			var sum uint16
			for x, v := range in {
				sum += uint16(v)
				nv := t[x] + sum
				out[x] = nv - c[x]
				c[x] = nv
			}
			top = cur
			cur += width
			if cur == r*width {
				cur = 0
			}
			if row >= 0 && row < height-1 {
				src += stride
			}
		}
		if row < radius {
			continue
		}
		// HFilter: horizontal accumulation with mirrored borders.
		{
			in := end[:width]
			out := average[:width]
			w := width
			x := 0
			for ; x <= radius; x++ {
				delta := in[x+radius-1] + in[radius-x]
				out[x] = uint16((uint32(delta) * scale) >> fix)
			}
			for ; x < w-radius; x++ {
				delta := in[x+radius] - in[x-radius-1]
				out[x] = uint16((uint32(delta) * scale) >> fix)
			}
			for ; x < w; x++ {
				delta := 2*in[w-1] - in[2*w-2-radius-x] - in[x-radius-1]
				out[x] = uint16((uint32(delta) * scale) >> fix)
			}
		}
		// ApplyFilter.
		{
			d := plane[dst : dst+width]
			avg := average[:width]
			for x, v8 := range d {
				v := int(v8)
				if v < maxV && v > minV {
					c := v + int(correction[lutSize+int(avg[x])-(v<<lfix)])
					if c&^0xff != 0 {
						if c < 0 {
							c = 0
						} else {
							c = 255
						}
					}
					d[x] = byte(c)
				}
			}
			dst += stride
		}
	}
	return nil
}

// initCorrectionLUT builds libwebp's correction curve: f(x) = x for
// x <= threshold2, 0 for x >= threshold1, linear in between, odd symmetric.
// Index lutSize is x = 0.
func initCorrectionLUT(lut *[1 + 2*lutSize]int16, minDist int) {
	threshold1 := minDist << lfix
	threshold2 := (3 * threshold1) >> 2
	maxThreshold := threshold2
	delta := threshold1 - threshold2
	for i := 1; i <= lutSize; i++ {
		var c int
		switch {
		case i <= threshold2:
			c = i
		case i < threshold1:
			c = maxThreshold * (threshold1 - i) / delta
		}
		c >>= lfix
		lut[lutSize+i] = int16(c)
		lut[lutSize-i] = int16(-c)
	}
	lut[lutSize] = 0
}
