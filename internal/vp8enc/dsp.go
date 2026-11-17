package vp8enc

// DSP primitives: transforms, intra predictors, distortion metrics and
// quantization. All blocks live in 32-byte-stride work buffers.
// Algorithm adapted from libwebp (BSD-3-Clause).

const (
	bps = 32 // stride of the work buffers

	yOff = 0
	uOff = 16
	vOff = 16 + 8

	yuvSize  = bps * 16
	predSize = 32*bps + 16*bps + 8*bps

	// Layout of the prediction buffer.
	i16DC16 = 0 * 16 * bps
	i16TM16 = i16DC16 + 16
	i16VE16 = 1 * 16 * bps
	i16HE16 = i16VE16 + 16
	c8DC8   = 2 * 16 * bps
	c8TM8   = c8DC8 + 1*16
	c8VE8   = 2*16*bps + 8*bps
	c8HE8   = c8VE8 + 1*16
	i4DC4   = 3*16*bps + 0
	i4TM4   = i4DC4 + 4
	i4VE4   = i4DC4 + 8
	i4HE4   = i4DC4 + 12
	i4RD4   = i4DC4 + 16
	i4VR4   = i4DC4 + 20
	i4LD4   = i4DC4 + 24
	i4VL4   = i4DC4 + 28
	i4HD4   = 3*16*bps + 4*bps
	i4HU4   = i4HD4 + 4
	i4TMP   = i4HD4 + 8
)

// Intra16 / chroma modes (libwebp numbering).
const (
	predDC = 0
	predTM = 1
	predVE = 2
	predHE = 3

	numPredModes = 4
	numBModes    = 10
)

var vp8I16ModeOffsets = [4]int{i16DC16, i16TM16, i16VE16, i16HE16}
var vp8UVModeOffsets = [4]int{c8DC8, c8TM8, c8VE8, c8HE8}
var vp8I4ModeOffsets = [numBModes]int{i4DC4, i4TM4, i4VE4, i4HE4, i4RD4, i4VR4, i4LD4, i4VL4, i4HD4, i4HU4}

var vp8Scan = [16]int{
	0 + 0*bps, 4 + 0*bps, 8 + 0*bps, 12 + 0*bps,
	0 + 4*bps, 4 + 4*bps, 8 + 4*bps, 12 + 4*bps,
	0 + 8*bps, 4 + 8*bps, 8 + 8*bps, 12 + 8*bps,
	0 + 12*bps, 4 + 12*bps, 8 + 12*bps, 12 + 12*bps,
}

var vp8ScanUV = [8]int{
	0 + 0*bps, 4 + 0*bps, 0 + 4*bps, 4 + 4*bps, // U
	8 + 0*bps, 12 + 0*bps, 8 + 4*bps, 12 + 4*bps, // V
}

var kZigzag = [16]uint8{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}

func clip8(v int32) uint8 {
	if uint32(v) <= 255 {
		return uint8(v)
	}
	if v < 0 {
		return 0
	}
	return 255
}

//------------------------------------------------------------------------------
// Transforms

// fTransform computes the forward DCT of src-ref (4x4, stride bps).
func fTransform(src, ref []byte, out *[16]int16) {
	var t [4][4]int32
	_ = src[3*bps+3]
	_ = ref[3*bps+3]
	for i := 0; i < 4; i++ {
		s := src[i*bps : i*bps+4 : i*bps+4]
		r := ref[i*bps : i*bps+4 : i*bps+4]
		d0 := int32(s[0]) - int32(r[0])
		d1 := int32(s[1]) - int32(r[1])
		d2 := int32(s[2]) - int32(r[2])
		d3 := int32(s[3]) - int32(r[3])
		a0 := d0 + d3
		a1 := d1 + d2
		a2 := d1 - d2
		a3 := d0 - d3
		ti := &t[i]
		ti[0] = (a0 + a1) * 8
		ti[1] = (a2*2217 + a3*5352 + 1812) >> 9
		ti[2] = (a0 - a1) * 8
		ti[3] = (a3*2217 - a2*5352 + 937) >> 9
	}
	for i := 0; i < 4; i++ {
		a0 := t[0][i] + t[3][i]
		a1 := t[1][i] + t[2][i]
		a2 := t[1][i] - t[2][i]
		a3 := t[0][i] - t[3][i]
		out[0+i] = int16((a0 + a1 + 7) >> 4)
		v := (a2*2217 + a3*5352 + 12000) >> 16
		if a3 != 0 {
			v++
		}
		out[4+i] = int16(v)
		out[8+i] = int16((a0 - a1 + 7) >> 4)
		out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}

// fTransformWHT computes the Walsh-Hadamard transform of the 16 DC
// coefficients of blocks[0..15].
func fTransformWHT(blocks *[16][16]int16, out *[16]int16) {
	var t [16]int32
	for i := 0; i < 4; i++ {
		b := blocks[i*4 : i*4+4 : i*4+4]
		in0, in1, in2, in3 := int32(b[0][0]), int32(b[1][0]), int32(b[2][0]), int32(b[3][0])
		a0 := in0 + in2
		a1 := in1 + in3
		a2 := in1 - in3
		a3 := in0 - in2
		t[0+i*4] = a0 + a1
		t[1+i*4] = a3 + a2
		t[2+i*4] = a3 - a2
		t[3+i*4] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0 := t[0+i] + t[8+i]
		a1 := t[4+i] + t[12+i]
		a2 := t[4+i] - t[12+i]
		a3 := t[0+i] - t[8+i]
		b0 := a0 + a1
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		out[0+i] = int16(b0 >> 1)
		out[4+i] = int16(b1 >> 1)
		out[8+i] = int16(b2 >> 1)
		out[12+i] = int16(b3 >> 1)
	}
}

// iTransformWHT is the inverse WHT; it writes the DC of blocks[0..15].
func iTransformWHT(in *[16]int16, blocks *[16][16]int16) {
	var t [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[0+i]) + int32(in[12+i])
		a1 := int32(in[4+i]) + int32(in[8+i])
		a2 := int32(in[4+i]) - int32(in[8+i])
		a3 := int32(in[0+i]) - int32(in[12+i])
		t[0+i] = a0 + a1
		t[8+i] = a0 - a1
		t[4+i] = a3 + a2
		t[12+i] = a3 - a2
	}
	for i := 0; i < 4; i++ {
		dc := t[0+i*4] + 3
		a0 := dc + t[3+i*4]
		a1 := t[1+i*4] + t[2+i*4]
		a2 := t[1+i*4] - t[2+i*4]
		a3 := dc - t[3+i*4]
		blocks[i*4+0][0] = int16((a0 + a1) >> 3)
		blocks[i*4+1][0] = int16((a3 + a2) >> 3)
		blocks[i*4+2][0] = int16((a0 - a1) >> 3)
		blocks[i*4+3][0] = int16((a3 - a2) >> 3)
	}
}

func mul1(a int32) int32 { return ((a * 20091) >> 16) + a }
func mul2(a int32) int32 { return (a * 35468) >> 16 }

// iTransform adds the inverse DCT of 'in' to ref and stores it into dst.
func iTransform(ref []byte, in *[16]int16, dst []byte) {
	_ = ref[3*bps+3]
	_ = dst[3*bps+3]
	if in[1]|in[2]|in[3]|in[4]|in[5]|in[6]|in[7]|in[8]|in[9]|in[10]|in[11]|in[12]|in[13]|in[14]|in[15] == 0 {
		// DC-only (or empty) block: same result as the full transform.
		dc := (int32(in[0]) + 4) >> 3
		for i := 0; i < 4; i++ {
			r := ref[i*bps : i*bps+4 : i*bps+4]
			o := dst[i*bps : i*bps+4 : i*bps+4]
			if dc == 0 {
				copy(o, r)
				continue
			}
			o[0] = clip8(int32(r[0]) + dc)
			o[1] = clip8(int32(r[1]) + dc)
			o[2] = clip8(int32(r[2]) + dc)
			o[3] = clip8(int32(r[3]) + dc)
		}
		return
	}
	var c [4][4]int32
	for i := 0; i < 4; i++ { // vertical pass
		i0, i4, i8, i12 := int32(in[i]), int32(in[4+i]), int32(in[8+i]), int32(in[12+i])
		a := i0 + i8
		b := i0 - i8
		cc := mul2(i4) - mul1(i12)
		d := mul1(i4) + mul2(i12)
		ci := &c[i]
		ci[0] = a + d
		ci[1] = b + cc
		ci[2] = b - cc
		ci[3] = a - d
	}
	_ = ref[3*bps+3]
	_ = dst[3*bps+3]
	for i := 0; i < 4; i++ { // horizontal pass
		dc := c[0][i] + 4
		a := dc + c[2][i]
		b := dc - c[2][i]
		cc := mul2(c[1][i]) - mul1(c[3][i])
		d := mul1(c[1][i]) + mul2(c[3][i])
		r := ref[i*bps : i*bps+4 : i*bps+4]
		o := dst[i*bps : i*bps+4 : i*bps+4]
		o[0] = clip8(int32(r[0]) + ((a + d) >> 3))
		o[1] = clip8(int32(r[1]) + ((b + cc) >> 3))
		o[2] = clip8(int32(r[2]) + ((b - cc) >> 3))
		o[3] = clip8(int32(r[3]) + ((a - d) >> 3))
	}
}

//------------------------------------------------------------------------------
// Intra predictions

func fill(dst []byte, v byte, size int) {
	for j := 0; j < size; j++ {
		row := dst[j*bps : j*bps+size]
		for i := range row {
			row[i] = v
		}
	}
}

func verticalPred(dst []byte, top []byte, size int) {
	if top == nil {
		fill(dst, 127, size)
		return
	}
	for j := 0; j < size; j++ {
		copy(dst[j*bps:j*bps+size], top[:size])
	}
}

// left[0] is the top-left sample, left[1..size] the left column.
func horizontalPred(dst []byte, left []byte, size int) {
	if left == nil {
		fill(dst, 129, size)
		return
	}
	for j := 0; j < size; j++ {
		row := dst[j*bps : j*bps+size]
		v := left[1+j]
		for i := range row {
			row[i] = v
		}
	}
}

func trueMotion(dst []byte, left, top []byte, size int) {
	if left != nil {
		if top != nil {
			tl := int32(left[0])
			top = top[:size]
			for y := 0; y < size; y++ {
				l := int32(left[1+y]) - tl
				row := dst[y*bps : y*bps+size]
				for x := range row {
					row[x] = clip8(l + int32(top[x]))
				}
			}
		} else {
			horizontalPred(dst, left, size)
		}
	} else {
		if top != nil {
			verticalPred(dst, top, size)
		} else {
			fill(dst, 129, size)
		}
	}
}

func dcMode(dst []byte, left, top []byte, size int, round int, shift uint) {
	dc := 0
	if top != nil {
		for j := 0; j < size; j++ {
			dc += int(top[j])
		}
		if left != nil {
			for j := 0; j < size; j++ {
				dc += int(left[1+j])
			}
		} else {
			dc += dc
		}
		dc = (dc + round) >> shift
	} else if left != nil {
		for j := 0; j < size; j++ {
			dc += int(left[1+j])
		}
		dc += dc
		dc = (dc + round) >> shift
	} else {
		dc = 0x80
	}
	fill(dst, byte(dc), size)
}

// predLuma16 computes the 4 intra16 predictions into p.
func predLuma16(p []byte, left, top []byte) {
	dcMode(p[i16DC16:], left, top, 16, 16, 5)
	verticalPred(p[i16VE16:], top, 16)
	horizontalPred(p[i16HE16:], left, 16)
	trueMotion(p[i16TM16:], left, top, 16)
}

// predChroma8 computes the 4 chroma predictions (U and V side by side).
// uvTop holds 8 U then 8 V samples.
func predChroma8(p []byte, uLeft, vLeft, uvTop []byte) {
	var top []byte
	if uvTop != nil {
		top = uvTop[0:8]
	}
	dcMode(p[c8DC8:], uLeft, top, 8, 8, 4)
	verticalPred(p[c8VE8:], top, 8)
	horizontalPred(p[c8HE8:], uLeft, 8)
	trueMotion(p[c8TM8:], uLeft, top, 8)
	if uvTop != nil {
		top = uvTop[8:16]
	}
	dcMode(p[c8DC8+8:], vLeft, top, 8, 8, 4)
	verticalPred(p[c8VE8+8:], top, 8)
	horizontalPred(p[c8HE8+8:], vLeft, 8)
	trueMotion(p[c8TM8+8:], vLeft, top, 8)
}

func avg3(a, b, c uint32) byte { return byte((a + 2*b + c + 2) >> 2) }
func avg2(a, b uint32) byte    { return byte((a + b + 1) >> 1) }

func put4(dst []byte, a, b, c, d byte) {
	_ = dst[3]
	dst[0], dst[1], dst[2], dst[3] = a, b, c, d
}

// predLuma4 computes the 10 intra4 predictions. 'b' is the boundary array
// and 't' the index of the top sample: left samples are b[t-5..t-2]
// (bottom to top), top-left is b[t-1], top is b[t..t+3] and top-right is
// b[t+4..t+7].
func predLuma4(p []byte, b []byte, t int) {
	s := b[t-5 : t+8]
	_ = s[12]
	L := uint32(s[0])
	K := uint32(s[1])
	J := uint32(s[2])
	I := uint32(s[3])
	X := uint32(s[4])
	A := uint32(s[5])
	B := uint32(s[6])
	C := uint32(s[7])
	D := uint32(s[8])
	E := uint32(s[9])
	F := uint32(s[10])
	G := uint32(s[11])
	H := uint32(s[12])

	// DC4
	{
		dc := byte((4 + A + B + C + D + L + K + J + I) >> 3)
		d := p[i4DC4:]
		for y := 0; y < 4; y++ {
			put4(d[y*bps:], dc, dc, dc, dc)
		}
	}
	// TM4
	{
		d := p[i4TM4:]
		tl := int32(X)
		lefts := [4]int32{int32(I), int32(J), int32(K), int32(L)}
		for y := 0; y < 4; y++ {
			l := lefts[y] - tl
			put4(d[y*bps:], clip8(l+int32(A)), clip8(l+int32(B)), clip8(l+int32(C)), clip8(l+int32(D)))
		}
	}
	// VE4
	{
		d := p[i4VE4:]
		v0, v1, v2, v3 := avg3(X, A, B), avg3(A, B, C), avg3(B, C, D), avg3(C, D, E)
		for y := 0; y < 4; y++ {
			put4(d[y*bps:], v0, v1, v2, v3)
		}
	}
	// HE4
	{
		d := p[i4HE4:]
		a, bb, c, dd := avg3(X, I, J), avg3(I, J, K), avg3(J, K, L), avg3(K, L, L)
		put4(d[0*bps:], a, a, a, a)
		put4(d[1*bps:], bb, bb, bb, bb)
		put4(d[2*bps:], c, c, c, c)
		put4(d[3*bps:], dd, dd, dd, dd)
	}
	// RD4
	{
		d := p[i4RD4:]
		jkl := avg3(J, K, L)
		ijk := avg3(I, J, K)
		xij := avg3(X, I, J)
		axi := avg3(A, X, I)
		bax := avg3(B, A, X)
		cba := avg3(C, B, A)
		dcb := avg3(D, C, B)
		put4(d[0*bps:], axi, bax, cba, dcb)
		put4(d[1*bps:], xij, axi, bax, cba)
		put4(d[2*bps:], ijk, xij, axi, bax)
		put4(d[3*bps:], jkl, ijk, xij, axi)
	}
	// VR4
	{
		d := p[i4VR4:]
		xa := avg2(X, A)
		ab := avg2(A, B)
		bc := avg2(B, C)
		cd := avg2(C, D)
		kji := avg3(K, J, I)
		jix := avg3(J, I, X)
		ixa := avg3(I, X, A)
		xab := avg3(X, A, B)
		abc := avg3(A, B, C)
		bcd := avg3(B, C, D)
		put4(d[0*bps:], xa, ab, bc, cd)
		put4(d[1*bps:], ixa, xab, abc, bcd)
		put4(d[2*bps:], jix, xa, ab, bc)
		put4(d[3*bps:], kji, ixa, xab, abc)
	}
	// LD4
	{
		d := p[i4LD4:]
		abc := avg3(A, B, C)
		bcd := avg3(B, C, D)
		cde := avg3(C, D, E)
		def := avg3(D, E, F)
		efg := avg3(E, F, G)
		fgh := avg3(F, G, H)
		ghh := avg3(G, H, H)
		put4(d[0*bps:], abc, bcd, cde, def)
		put4(d[1*bps:], bcd, cde, def, efg)
		put4(d[2*bps:], cde, def, efg, fgh)
		put4(d[3*bps:], def, efg, fgh, ghh)
	}
	// VL4
	{
		d := p[i4VL4:]
		ab := avg2(A, B)
		bc := avg2(B, C)
		cd := avg2(C, D)
		de := avg2(D, E)
		abc := avg3(A, B, C)
		bcd := avg3(B, C, D)
		cde := avg3(C, D, E)
		def := avg3(D, E, F)
		efg := avg3(E, F, G)
		fgh := avg3(F, G, H)
		put4(d[0*bps:], ab, bc, cd, de)
		put4(d[1*bps:], abc, bcd, cde, def)
		put4(d[2*bps:], bc, cd, de, efg)
		put4(d[3*bps:], bcd, cde, def, fgh)
	}
	// HD4
	{
		d := p[i4HD4:]
		ix := avg2(I, X)
		ji := avg2(J, I)
		kj := avg2(K, J)
		lk := avg2(L, K)
		abc := avg3(A, B, C)
		xab := avg3(X, A, B)
		ixa := avg3(I, X, A)
		jix := avg3(J, I, X)
		kji := avg3(K, J, I)
		lkj := avg3(L, K, J)
		put4(d[0*bps:], ix, ixa, xab, abc)
		put4(d[1*bps:], ji, jix, ix, ixa)
		put4(d[2*bps:], kj, kji, ji, jix)
		put4(d[3*bps:], lk, lkj, kj, kji)
	}
	// HU4
	{
		d := p[i4HU4:]
		ij := avg2(I, J)
		jk := avg2(J, K)
		kl := avg2(K, L)
		ijk := avg3(I, J, K)
		jkl := avg3(J, K, L)
		kll := avg3(K, L, L)
		l := byte(L)
		put4(d[0*bps:], ij, ijk, jk, jkl)
		put4(d[1*bps:], jk, jkl, kl, kll)
		put4(d[2*bps:], kl, kll, l, l)
		put4(d[3*bps:], l, l, l, l)
	}
}

//------------------------------------------------------------------------------
// Metrics

func sseRow16(a, b []byte) int {
	_ = a[15]
	_ = b[15]
	s := 0
	for i := 0; i < 16; i++ {
		d := int(a[i]) - int(b[i])
		s += d * d
	}
	return s
}

func sseRow4(a, b []byte) int {
	_ = a[3]
	_ = b[3]
	d0 := int(a[0]) - int(b[0])
	d1 := int(a[1]) - int(b[1])
	d2 := int(a[2]) - int(b[2])
	d3 := int(a[3]) - int(b[3])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3
}

func sse16x16Ref(a, b []byte) int {
	s := 0
	for y := 0; y < 16; y++ {
		s += sseRow16(a[y*bps:], b[y*bps:])
	}
	return s
}

func sse16x8Ref(a, b []byte) int {
	s := 0
	for y := 0; y < 8; y++ {
		s += sseRow16(a[y*bps:], b[y*bps:])
	}
	return s
}

func sse4x4(a, b []byte) int {
	return sseRow4(a, b) + sseRow4(a[bps:], b[bps:]) + sseRow4(a[2*bps:], b[2*bps:]) + sseRow4(a[3*bps:], b[3*bps:])
}

func iabs(v int32) int32 {
	m := v >> 31
	return (v ^ m) - m
}

func ttRow(in []byte) (int32, int32, int32, int32) {
	_ = in[3]
	a0 := int32(in[0]) + int32(in[2])
	a1 := int32(in[1]) + int32(in[3])
	a2 := int32(in[1]) - int32(in[3])
	a3 := int32(in[0]) - int32(in[2])
	return a0 + a1, a3 + a2, a3 - a2, a0 - a1
}

// tTransform returns the weighted sum of the absolute Hadamard coefficients.
func tTransform(in []byte, w *[16]uint16) int32 {
	var t [16]int32
	t[0], t[1], t[2], t[3] = ttRow(in[0:])
	t[4], t[5], t[6], t[7] = ttRow(in[bps:])
	t[8], t[9], t[10], t[11] = ttRow(in[2*bps:])
	t[12], t[13], t[14], t[15] = ttRow(in[3*bps:])
	var sum int32
	for i := 0; i < 4; i++ {
		a0 := t[0+i] + t[8+i]
		a1 := t[4+i] + t[12+i]
		a2 := t[4+i] - t[12+i]
		a3 := t[0+i] - t[8+i]
		b0 := a0 + a1
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		sum += int32(w[0+i])*iabs(b0) + int32(w[4+i])*iabs(b1) + int32(w[8+i])*iabs(b2) + int32(w[12+i])*iabs(b3)
	}
	return sum
}

//------------------------------------------------------------------------------
// Quantization

const (
	qfix     = 17
	maxLevel = 2047
)

func biasOf(b uint32) uint32 { return b << (qfix - 8) }

func quantDiv(n, iq, b uint32) int32 { return int32((n*iq + b) >> qfix) }

// matrix holds the quantization parameters of one coefficient type.
type matrix struct {
	q       [16]uint32 // quantizer steps
	iq      [16]uint32 // reciprocals, fixed point
	bias    [16]uint32 // rounding bias
	zthresh [16]uint32 // value below which a coefficient is zeroed
	sharpen [16]uint32 // frequency boosters for slight sharpening

	// 16-bit copies of q, iq and sharpen for the vector quantizer, valid
	// when vecOK is set (see prepareVec).
	q16, iq16, sharpen16 [16]uint16
	vecOK                bool
}

// prepareVec fills the 16-bit copies of the quantizer parameters. A vector
// quantizer working on 16-bit lanes (|coeff| <= 32768, then
// coeff+sharpen and (coeff+sharpen)*iq computed exactly in 16 and 32 bits)
// matches quantizeBlock bit for bit when iq < 2^16 and sharpen < 2^15; only
// the low 16 bits of q matter, since the dequantized value is truncated to
// int16. The encoder's quantizers (q >= 4, so iq <= 32768, and sharpen <=
// 12) always qualify; vecOK records it for any other matrix.
func (m *matrix) prepareVec() {
	m.vecOK = true
	for i := range m.q {
		m.q16[i] = uint16(m.q[i])
		m.iq16[i] = uint16(m.iq[i])
		m.sharpen16[i] = uint16(m.sharpen[i])
		if m.iq[i] > 0xffff || m.sharpen[i] > 0x7fff {
			m.vecOK = false
		}
	}
}

// kInvZigzag maps a natural-order coefficient index to its zigzag position.
var kInvZigzag = [16]uint8{0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15}

// quantizeBlock quantizes in[] (natural order) into out[] (zigzag order),
// replacing in[] with the dequantized values. Returns true if any level is
// non-zero. Branchless: QUANTDIV is zero exactly when coeff <= zthresh.
func quantizeBlock(in *[16]int16, out *[16]int16, m *matrix) bool {
	var nz int32
	for j := 0; j < 16; j++ {
		v := int32(in[j])
		s := v >> 31
		coeff := uint32((v^s)-s) + m.sharpen[j]
		level := min(int32((coeff*m.iq[j]+m.bias[j])>>qfix), maxLevel)
		level = (level ^ s) - s
		in[j] = int16(level * int32(m.q[j]))
		out[kInvZigzag[j]&15] = int16(level)
		nz |= level
	}
	return nz != 0
}

// isFlat returns true if the number of non-zero AC levels is <= thresh.
func isFlat(levels []([16]int16), thresh int) bool {
	score := 0
	for b := range levels {
		l := &levels[b]
		for i := 1; i < 16; i++ {
			v := uint16(l[i])
			score += int((v | -v) >> 15)
		}
		if score > thresh {
			return false
		}
	}
	return true
}

func isFlatSource16(src []byte) bool {
	v := src[0]
	for y := 0; y < 16; y++ {
		row := src[y*bps : y*bps+16]
		for _, s := range row {
			if s != v {
				return false
			}
		}
	}
	return true
}

func copy4x4(src, dst []byte) {
	for y := 0; y < 4; y++ {
		copy(dst[y*bps:y*bps+4], src[y*bps:y*bps+4])
	}
}

func copy16x8(src, dst []byte) {
	for y := 0; y < 8; y++ {
		copy(dst[y*bps:y*bps+16], src[y*bps:y*bps+16])
	}
}

//------------------------------------------------------------------------------
// Histogram (analysis)

const maxCoeffThresh = 31

// collectHistogram returns the (maxValue, lastNonZero) of the histogram of
// DCT coefficients of ref-pred over the blocks of t (tabY16 or tabUV).
func collectHistogram(ref, pred []byte, t *pairTable) (int, int) {
	var distribution [maxCoeffThresh + 1]int
	var coeffs [16][16]int16
	fTransformPairs(ref, pred, t, coeffs[:])
	for _, out := range coeffs[:2*t.n] {
		for k := 0; k < 16; k++ {
			v := int32(out[k])
			s := v >> 31
			distribution[min((v^s-s)>>3, maxCoeffThresh)&maxCoeffThresh]++
		}
	}
	maxValue, lastNonZero := 0, 1
	for k, value := range distribution {
		if value > 0 {
			if value > maxValue {
				maxValue = value
			}
			lastNonZero = k
		}
	}
	return maxValue, lastNonZero
}
