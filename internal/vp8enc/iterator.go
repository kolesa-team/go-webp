package vp8enc

// Macroblock iterator: walks the macroblocks, keeping the neighbouring
// samples, prediction modes and non-zero contexts up to date.
// Algorithm adapted from libwebp (BSD-3-Clause).

// mbInfo holds the coding decisions of one macroblock.
type mbInfo struct {
	typ     uint8 // 0=i4x4, 1=i16x16
	uvMode  uint8
	segment uint8
	alpha   uint8 // quantization-susceptibility
	skip    bool
}

// vp8TopLeftI4 gives, for each 4x4 sub-block, the position of its top
// sample in the boundary array.
var vp8TopLeftI4 = [16]int{
	17, 21, 25, 29,
	13, 17, 21, 25,
	9, 13, 17, 21,
	5, 9, 13, 17,
}

type iterator struct {
	enc  *encoder
	x, y int

	yuvIn, yuvOut, yuvOut2, yuvP []byte

	predsIdx   int // index of the current macroblock's modes in enc.preds
	mbIdx      int // index of the current macroblock in enc.mbInfo
	countDown  int
	i4         int  // current intra4x4 sub-block
	i4Top      int  // index of the top sample in i4Boundary
	doTrellis  bool // perform extra level optimisation
	analysis   bool // boundary samples come from the source (analysis pass)
	topNz      [9]int
	leftNz     [9]int // leftNz[8] is independent
	leftDerr   [2][2]int8
	i4Boundary [40]byte

	// yLeft[0] is the top-left sample, yLeft[1..16] the left column.
	yLeft        [17]byte
	uLeft, vLeft [9]byte
	tmp32        [32]byte // source top samples (analysis pass)

	// scratch
	tmp16     [16][16]int16
	tmpUV     [8][16]int16
	tmpLevels [1][16]int16
	i4Levels  [16][16]int16
	srcTT     [16]int32 // texture transform of the source luma blocks
	rdTmp     modeScore
	rdUV      modeScore
	rd        modeScore

	yuvMem [3*yuvSize + predSize]byte
}

// init resets the iterator and the encoder's top context for a pass over
// the frame.
func (it *iterator) init(e *encoder) {
	it.reset(e)
	e.initTop()
}

// reset resets the iterator only (helper iterators of the analysis pass,
// which does not use the encoder's top context).
func (it *iterator) reset(e *encoder) {
	it.enc = e
	it.yuvIn = it.yuvMem[0:yuvSize:yuvSize]
	it.yuvOut = it.yuvMem[yuvSize : 2*yuvSize : 2*yuvSize]
	it.yuvOut2 = it.yuvMem[2*yuvSize : 3*yuvSize : 3*yuvSize]
	it.yuvP = it.yuvMem[3*yuvSize:]
	it.analysis = false
	it.doTrellis = false
	it.setRow(0)
	it.countDown = e.mbW * e.mbH
}

func (e *encoder) initTop() {
	for i := range e.yTop {
		e.yTop[i] = 127
	}
	for i := range e.uvTop {
		e.uvTop[i] = 127
	}
	for i := range e.nz {
		e.nz[i] = 0
	}
	for i := range e.topDerr {
		e.topDerr[i] = [2][2]int8{}
	}
}

func (it *iterator) initLeft() {
	v := byte(127)
	if it.y > 0 {
		v = 129
	}
	it.yLeft[0], it.uLeft[0], it.vLeft[0] = v, v, v
	for i := 1; i < 17; i++ {
		it.yLeft[i] = 129
	}
	for i := 1; i < 9; i++ {
		it.uLeft[i] = 129
		it.vLeft[i] = 129
	}
	it.leftNz[8] = 0
	it.leftDerr = [2][2]int8{}
}

func (it *iterator) setRow(y int) {
	e := it.enc
	it.x = 0
	it.y = y
	it.predsIdx = e.predsW + 1 + y*4*e.predsW
	it.mbIdx = y * e.mbW
	it.initLeft()
}

func (it *iterator) setCountDown(n int) { it.countDown = n }

func (it *iterator) mb() *mbInfo { return &it.enc.mbInfo[it.mbIdx] }

// next advances to the next macroblock; returns false when done.
func (it *iterator) next() bool {
	it.x++
	if it.x == it.enc.mbW {
		it.setRow(it.y + 1)
	} else {
		it.predsIdx += 4
		it.mbIdx++
	}
	it.countDown--
	return it.countDown > 0
}

func importBlock(src []byte, srcStride int, dst []byte, w, h, size int) {
	for i := 0; i < h; i++ {
		d := dst[i*bps : i*bps+size]
		copy(d[:w], src[i*srcStride:i*srcStride+w])
		if w < size {
			v := d[w-1]
			for k := w; k < size; k++ {
				d[k] = v
			}
		}
	}
	for i := h; i < size; i++ {
		copy(dst[i*bps:i*bps+size], dst[(i-1)*bps:(i-1)*bps+size])
	}
}

func importLine(src []byte, srcStride int, dst []byte, l, total int) {
	i := 0
	for ; i < l; i++ {
		dst[i] = src[i*srcStride]
	}
	for ; i < total; i++ {
		dst[i] = dst[l-1]
	}
}

// importMB loads the source samples of the current macroblock, replicating
// the picture boundary. In analysis mode the prediction boundary is also
// taken from the source.
func (it *iterator) importMB() {
	e := it.enc
	pic := e.pic
	x, y := it.x, it.y
	yo := y*16*pic.YStride + x*16
	uvo := y*8*pic.UVStride + x*8
	w := min(pic.Width-x*16, 16)
	h := min(pic.Height-y*16, 16)
	uvw := (w + 1) >> 1
	uvh := (h + 1) >> 1

	importBlock(pic.Y[yo:], pic.YStride, it.yuvIn[yOff:], w, h, 16)
	importBlock(pic.U[uvo:], pic.UVStride, it.yuvIn[uOff:], uvw, uvh, 8)
	importBlock(pic.V[uvo:], pic.UVStride, it.yuvIn[vOff:], uvw, uvh, 8)
	if !it.analysis {
		return
	}
	if x == 0 {
		it.initLeft()
	} else {
		if y == 0 {
			it.yLeft[0], it.uLeft[0], it.vLeft[0] = 127, 127, 127
		} else {
			it.yLeft[0] = pic.Y[yo-1-pic.YStride]
			it.uLeft[0] = pic.U[uvo-1-pic.UVStride]
			it.vLeft[0] = pic.V[uvo-1-pic.UVStride]
		}
		importLine(pic.Y[yo-1:], pic.YStride, it.yLeft[1:], h, 16)
		importLine(pic.U[uvo-1:], pic.UVStride, it.uLeft[1:], uvh, 8)
		importLine(pic.V[uvo-1:], pic.UVStride, it.vLeft[1:], uvh, 8)
	}
	if y == 0 {
		for i := range it.tmp32 {
			it.tmp32[i] = 127
		}
	} else {
		importLine(pic.Y[yo-pic.YStride:], 1, it.tmp32[0:16], w, 16)
		importLine(pic.U[uvo-pic.UVStride:], 1, it.tmp32[16:24], uvw, 8)
		importLine(pic.V[uvo-pic.UVStride:], 1, it.tmp32[24:32], uvw, 8)
	}
}

// top returns the luma and chroma top samples (nil when unavailable).
func (it *iterator) top() ([]byte, []byte) {
	if it.y == 0 {
		return nil, nil
	}
	if it.analysis {
		return it.tmp32[0:16], it.tmp32[16:32]
	}
	o := it.x * 16
	return it.enc.yTop[o:], it.enc.uvTop[o : o+16]
}

func (it *iterator) makeLuma16Preds() {
	var left []byte
	if it.x != 0 {
		left = it.yLeft[:]
	}
	top, _ := it.top()
	predLuma16(it.yuvP, left, top)
}

func (it *iterator) makeChroma8Preds() {
	var uLeft, vLeft []byte
	if it.x != 0 {
		uLeft, vLeft = it.uLeft[:], it.vLeft[:]
	}
	_, uvTop := it.top()
	predChroma8(it.yuvP, uLeft, vLeft, uvTop)
}

func (it *iterator) makeIntra4Preds() {
	predLuma4(it.yuvP, it.i4Boundary[:], it.i4Top)
}

// saveBoundary saves the reconstructed samples as the prediction boundary
// of the next macroblocks.
func (it *iterator) saveBoundary() {
	e := it.enc
	x, y := it.x, it.y
	ysrc := it.yuvOut[yOff:]
	uvsrc := it.yuvOut[uOff:]
	o := x * 16
	if x < e.mbW-1 { // left
		for i := 0; i < 16; i++ {
			it.yLeft[1+i] = ysrc[15+i*bps]
		}
		for i := 0; i < 8; i++ {
			it.uLeft[1+i] = uvsrc[7+i*bps]
			it.vLeft[1+i] = uvsrc[15+i*bps]
		}
		// top-left (before 'top'!)
		it.yLeft[0] = e.yTop[o+15]
		it.uLeft[0] = e.uvTop[o+7]
		it.vLeft[0] = e.uvTop[o+8+7]
	}
	if y < e.mbH-1 { // top
		copy(e.yTop[o:o+16], ysrc[15*bps:15*bps+16])
		copy(e.uvTop[o:o+16], uvsrc[7*bps:7*bps+16])
	}
}

// Non-zero context bits:
//
//	 0  1  2  3  Y
//	 4  5  6  7
//	 8  9 10 11
//	12 13 14 15
//	16 17        U
//	18 19
//	20 21        V
//	22 23
//	24           DC-intra16
func bit(nz uint32, n uint) int { return int((nz >> n) & 1) }

func (it *iterator) nzToBytes() {
	nz := it.enc.nz
	tnz := nz[it.x+1]
	lnz := nz[it.x]
	t := &it.topNz
	l := &it.leftNz
	t[0], t[1], t[2], t[3] = bit(tnz, 12), bit(tnz, 13), bit(tnz, 14), bit(tnz, 15)
	t[4], t[5] = bit(tnz, 18), bit(tnz, 19)
	t[6], t[7] = bit(tnz, 22), bit(tnz, 23)
	t[8] = bit(tnz, 24)
	l[0], l[1], l[2], l[3] = bit(lnz, 3), bit(lnz, 7), bit(lnz, 11), bit(lnz, 15)
	l[4], l[5] = bit(lnz, 17), bit(lnz, 19)
	l[6], l[7] = bit(lnz, 21), bit(lnz, 23)
	// left-DC is special, iterated separately
}

func (it *iterator) bytesToNz() {
	t := &it.topNz
	l := &it.leftNz
	nz := uint32(t[0]<<12 | t[1]<<13 | t[2]<<14 | t[3]<<15)
	nz |= uint32(t[4]<<18 | t[5]<<19 | t[6]<<22 | t[7]<<23)
	nz |= uint32(t[8] << 24) // we propagate the _top_ bit, esp. for intra4
	nz |= uint32(l[0]<<3 | l[1]<<7 | l[2]<<11)
	nz |= uint32(l[4]<<17 | l[6]<<21)
	it.enc.nz[it.x+1] = nz
}

func (it *iterator) setIntra16Mode(mode int) {
	e := it.enc
	p := it.predsIdx
	m := uint8(mode)
	for y := 0; y < 4; y++ {
		row := e.preds[p : p+4]
		row[0], row[1], row[2], row[3] = m, m, m, m
		p += e.predsW
	}
	it.mb().typ = 1
}

func (it *iterator) setIntra4Mode(modes *[16]uint8) {
	e := it.enc
	p := it.predsIdx
	for y := 0; y < 4; y++ {
		copy(e.preds[p:p+4], modes[y*4:y*4+4])
		p += e.predsW
	}
	it.mb().typ = 0
}

// startI4 prepares the boundary samples for the intra4x4 iteration.
func (it *iterator) startI4() {
	e := it.enc
	it.i4 = 0
	it.i4Top = vp8TopLeftI4[0]
	b := &it.i4Boundary
	for i := 0; i < 17; i++ { // left, bottom to top, then top-left
		b[i] = it.yLeft[16-i]
	}
	top, _ := it.top()
	if top == nil {
		for i := 0; i < 20; i++ {
			b[17+i] = 127
		}
	} else {
		copy(b[17:33], top[:16])
		if it.x < e.mbW-1 {
			copy(b[33:37], top[16:20])
		} else { // replicate the last valid pixel four times
			v := b[17+15]
			b[33], b[34], b[35], b[36] = v, v, v, v
		}
	}
	it.nzToBytes()
}

// rotateI4 updates the boundary with the reconstructed sub-block and moves
// to the next one. Returns false when all 16 sub-blocks are done.
func (it *iterator) rotateI4(yuvOut []byte) bool {
	blk := yuvOut[vp8Scan[it.i4]:]
	t := it.i4Top
	b := &it.i4Boundary
	for i := 0; i <= 3; i++ {
		b[t-4+i] = blk[i+3*bps] // store future top samples
	}
	if it.i4&3 != 3 { // if not on the right sub-blocks #3, #7, #11, #15
		for i := 0; i <= 2; i++ { // store future left samples
			b[t+i] = blk[3+(2-i)*bps]
		}
	} else { // else replicate top-right samples, as says the specs.
		for i := 0; i <= 3; i++ {
			b[t+i] = b[t+i+4]
		}
	}
	it.i4++
	if it.i4 == 16 {
		return false
	}
	it.i4Top = vp8TopLeftI4[it.i4]
	return true
}
