package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): dec/tree_dec.c
// (ParseIntraMode), dec/vp8_dec.c (GetCoeffs, ParseResiduals) and
// dec/frame_dec.c (ReconstructRow).

// kZigzag maps the coefficient scan order to raster positions.
var kZigzag = [16]uint8{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}

// Extra bits probabilities for DCT_CAT3..DCT_CAT6, zero terminated.
var (
	kCat3    = []uint8{173, 148, 140}
	kCat4    = []uint8{176, 155, 140, 135}
	kCat5    = []uint8{180, 157, 141, 134, 130}
	kCat6    = []uint8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129}
	kCat3456 = [4][]uint8{kCat3, kCat4, kCat5, kCat6}
)

// kYModesIntra4 is the intra 4x4 mode decoding tree: positive entries are
// the next node, other entries are negated modes.
var kYModesIntra4 = [18]int8{
	-bDCPred, 1,
	-bTMPred, 2,
	-bVEPred, 3,
	4, 6,
	-bHEPred, 5,
	-bRDPred, -bVRPred,
	-bLDPred, 7,
	-bVLPred, 8,
	-bHDPred, -bHUPred,
}

// kScan is the offset of each luma 4x4 sub-block inside the work buffer.
var kScan = [16]int{
	0 + 0*bps, 4 + 0*bps, 8 + 0*bps, 12 + 0*bps,
	0 + 4*bps, 4 + 4*bps, 8 + 4*bps, 12 + 4*bps,
	0 + 8*bps, 4 + 8*bps, 8 + 8*bps, 12 + 8*bps,
	0 + 12*bps, 4 + 12*bps, 8 + 12*bps, 12 + 12*bps,
}

// parseIntraMode reads the segment, skip flag and prediction modes of the
// macroblock at column mbX from the first partition.
func (d *decoder) parseIntraMode(mbX int, blk *mbBlock) {
	br := &d.br
	top := (*[4]uint8)(d.intraT[4*mbX:])
	left := &d.intraL
	if d.updateMap {
		if br.getBit(d.segProba[0]) == 0 {
			blk.segment = uint8(br.getBit(d.segProba[1]))
		} else {
			blk.segment = uint8(br.getBit(d.segProba[2]) + 2)
		}
	} else {
		blk.segment = 0 // default for intra
	}
	blk.skip = false
	if d.useSkipProba {
		blk.skip = br.getBit(d.skipP) != 0
	}
	blk.isI4x4 = br.getBit(145) == 0
	if !blk.isI4x4 {
		var ymode uint8
		if br.getBit(156) != 0 {
			if br.getBit(128) != 0 {
				ymode = bTMPred
			} else {
				ymode = bHEPred
			}
		} else {
			if br.getBit(163) != 0 {
				ymode = bVEPred
			} else {
				ymode = bDCPred
			}
		}
		blk.imodes[0] = ymode
		*top = [4]uint8{ymode, ymode, ymode, ymode}
		*left = [4]uint8{ymode, ymode, ymode, ymode}
	} else {
		modes := &blk.imodes
		value, rng, nbits := br.value, br.rng, br.nbits
		var bit bool
		for y := 0; y < 4; y++ {
			ymode := left[y]
			for x := 0; x < 4; x++ {
				prob := &kBModesProba[top[x]][ymode]
				// Generic tree-parsing
				i := 0
				for {
					if nbits < 0 {
						value, nbits = br.refill(value, nbits)
					}
					value, rng, nbits, bit = decodeBit(value, rng, nbits, prob[i])
					i = int(kYModesIntra4[2*i+b2i(bit)])
					if i <= 0 {
						break
					}
				}
				ymode = uint8(-i)
				top[x] = ymode
			}
			modes[4*y+0], modes[4*y+1], modes[4*y+2], modes[4*y+3] = top[0], top[1], top[2], top[3]
			left[y] = ymode
		}
		br.value, br.rng, br.nbits = value, rng, nbits
	}
	switch {
	case br.getBit(142) == 0:
		blk.uvmode = bDCPred
	case br.getBit(114) == 0:
		blk.uvmode = bVEPred
	case br.getBit(183) != 0:
		blk.uvmode = bTMPred
	default:
		blk.uvmode = bHEPred
	}
}

// getLargeValue decodes a coefficient magnitude >= 2.
func getLargeValue(br *bitReader, p *[numProbas]uint8) int {
	if br.getBit(p[3]) == 0 {
		if br.getBit(p[4]) == 0 {
			return 2
		}
		return 3 + br.getBit(p[5])
	}
	if br.getBit(p[6]) == 0 {
		if br.getBit(p[7]) == 0 {
			return 5 + br.getBit(159)
		}
		v := 7 + 2*br.getBit(165)
		return v + br.getBit(145)
	}
	bit1 := br.getBit(p[8])
	bit0 := br.getBit(p[9+bit1])
	cat := 2*bit1 + bit0
	v := 0
	for _, pr := range kCat3456[cat] {
		v += v + br.getBit(pr)
	}
	return v + 3 + (8 << uint(cat))
}

// getCoeffs decodes the tokens of one 4x4 block starting at coefficient n
// and returns the index after the last non-zero coefficient (16 at most).
func getCoeffs(br *bitReader, prob *[16 + 1][numCtx][numProbas]uint8, ctx int, dq *[2]int32, n int, out *[16]int16) int {
	value, rng, nbits := br.value, br.rng, br.nbits
	var bit bool
	p := &prob[n&15][ctx]
	for ; n < 16; n++ {
		if nbits < 0 {
			value, nbits = br.refill(value, nbits)
		}
		if value, rng, nbits, bit = decodeBit(value, rng, nbits, p[0]); !bit {
			break // previous coeff was last non-zero coefficient
		}
		for {
			if nbits < 0 {
				value, nbits = br.refill(value, nbits)
			}
			if value, rng, nbits, bit = decodeBit(value, rng, nbits, p[1]); bit {
				break
			}
			// zero coefficient
			n++
			if n == 16 {
				br.value, br.rng, br.nbits = value, rng, nbits
				return 16
			}
			p = &prob[n][0]
		}
		// non-zero coefficient
		var v int
		next := &prob[n+1]
		if nbits < 0 {
			value, nbits = br.refill(value, nbits)
		}
		if value, rng, nbits, bit = decodeBit(value, rng, nbits, p[2]); !bit {
			v = 1
			p = &next[1]
		} else {
			br.value, br.rng, br.nbits = value, rng, nbits
			v = getLargeValue(br, p)
			value, rng, nbits = br.value, br.rng, br.nbits
			p = &next[2]
		}
		if nbits < 0 {
			value, nbits = br.refill(value, nbits)
		}
		value, rng, nbits, v = decodeSigned(value, rng, nbits, v)
		q := dq[1]
		if n == 0 {
			q = dq[0]
		}
		out[kZigzag[n&15]] = int16(int32(v) * q)
	}
	br.value, br.rng, br.nbits = value, rng, nbits
	return n
}

// nzCodeBits appends the 2-bit transform selector of a block.
func nzCodeBits(nzCoeffs uint32, nz int, dcNZ bool) uint32 {
	nzCoeffs <<= 2
	switch {
	case nz > 3:
		nzCoeffs |= 3
	case nz > 1:
		nzCoeffs |= 2
	case dcNZ:
		nzCoeffs |= 1
	}
	return nzCoeffs
}

// parseResiduals decodes all coefficients of the macroblock and reports
// whether every coefficient is zero.
func (d *decoder) parseResiduals(mbX int, blk *mbBlock, br *bitReader) bool {
	q := &d.dqm[blk.segment]
	dst := &blk.coeffs
	clear(dst[:])
	var acProba *[16 + 1][numCtx][numProbas]uint8
	first := 0
	if !blk.isI4x4 { // parse DC
		var dc [16]int16
		ctx := int(d.nzDCT[mbX] + d.nzDCL)
		nz := getCoeffs(br, &d.proba[1], ctx, &q.y2, 0, &dc)
		v := uint8(0)
		if nz > 0 {
			v = 1
		}
		d.nzDCT[mbX], d.nzDCL = v, v
		if nz > 1 { // more than just the DC -> perform the full transform
			transformWHT(&dc, dst)
		} else { // only DC is non-zero -> inlined simplified transform
			dc0 := int16((int(dc[0]) + 3) >> 3)
			for i := 0; i < 16*16; i += 16 {
				dst[i] = dc0
			}
		}
		first = 1
		acProba = &d.proba[0]
	} else {
		acProba = &d.proba[3]
	}

	tnz := uint32(d.nzT[mbX] & 0x0f)
	lnz := uint32(d.nzL & 0x0f)
	var nonZeroY, nonZeroUV uint32
	for y := 0; y < 4; y++ {
		l := lnz & 1
		var nzCoeffs uint32
		for x := 0; x < 4; x++ {
			ctx := int(l + (tnz & 1))
			blkc := (*[16]int16)(dst[(y*4+x)*16:])
			nz := getCoeffs(br, acProba, ctx, &q.y1, first, blkc)
			l = 0
			if nz > first {
				l = 1
			}
			tnz = (tnz >> 1) | (l << 7)
			nzCoeffs = nzCodeBits(nzCoeffs, nz, blkc[0] != 0)
		}
		tnz >>= 4
		lnz = (lnz >> 1) | (l << 7)
		nonZeroY = (nonZeroY << 8) | nzCoeffs
	}
	outTNZ := tnz
	outLNZ := lnz >> 4

	for ch := 0; ch < 4; ch += 2 {
		var nzCoeffs uint32
		tnz = uint32(d.nzT[mbX]) >> uint(4+ch)
		lnz = uint32(d.nzL) >> uint(4+ch)
		for y := 0; y < 2; y++ {
			l := lnz & 1
			for x := 0; x < 2; x++ {
				ctx := int(l + (tnz & 1))
				blkc := (*[16]int16)(dst[16*16+(ch*2+y*2+x)*16:])
				nz := getCoeffs(br, &d.proba[2], ctx, &q.uv, 0, blkc)
				l = 0
				if nz > 0 {
					l = 1
				}
				tnz = (tnz >> 1) | (l << 3)
				nzCoeffs = nzCodeBits(nzCoeffs, nz, blkc[0] != 0)
			}
			tnz >>= 2
			lnz = (lnz >> 1) | (l << 5)
		}
		// Note: we don't really need the per-4x4 uv bits
		nonZeroUV |= nzCoeffs << uint(4*ch)
		outTNZ |= (tnz << 4) << uint(ch)
		outLNZ |= (lnz & 0xf0) << uint(ch)
	}
	d.nzT[mbX] = uint8(outTNZ)
	d.nzL = uint8(outLNZ)

	blk.nonZeroY = nonZeroY
	blk.nonZeroUV = nonZeroUV
	blk.dither = 0
	if nonZeroUV&0xaaaa == 0 {
		blk.dither = uint8(q.dither)
	}
	return nonZeroY|nonZeroUV == 0
}

// checkMode substitutes the DC predictor variants used at frame edges.
func checkMode(mbX, mbY int, mode uint8) int {
	if mode == bDCPred {
		if mbX == 0 {
			if mbY == 0 {
				return bDCPredNoTopLeft
			}
			return bDCPredNoLeft
		}
		if mbY == 0 {
			return bDCPredNoTop
		}
		return bDCPred
	}
	return int(mode)
}

// initRowEdges prepares the work buffer borders at the start of a row.
func (d *decoder) initRowEdges(mbY int) {
	b := d.yuvB[:]
	for j := 0; j < 16; j++ {
		b[yOff+j*bps-1] = 129
	}
	for j := 0; j < 8; j++ {
		b[uOff+j*bps-1] = 129
		b[vOff+j*bps-1] = 129
	}
	if mbY > 0 {
		b[yOff-1-bps], b[uOff-1-bps], b[vOff-1-bps] = 129, 129, 129
	} else {
		for i := yOff - bps - 1; i < yOff-bps-1+16+4+1; i++ {
			b[i] = 127
		}
		for i := uOff - bps - 1; i < uOff-bps-1+8+1; i++ {
			b[i] = 127
		}
		for i := vOff - bps - 1; i < vOff-bps-1+8+1; i++ {
			b[i] = 127
		}
	}
}

// reconstructMB predicts the macroblock, adds the residuals and copies the
// result into the row cache.
func (d *decoder) reconstructMB(mbX, mbY int, blk *mbBlock) {
	d.loadMBEdges(mbX, mbY, blk.isI4x4)
	reconBlock(&d.yuvB, mbX, mbY, blk)
	d.storeMB(mbX, mbY)
}

// reconBlock runs the prediction and residual kernels of the macroblock on
// the work buffer, whose edges loadMBEdges has set up.
func reconBlock(b *[yuvSize]byte, mbX, mbY int, blk *mbBlock) {
	if blk.isI4x4 {
		reconI4(b, &blk.coeffs, blk.nonZeroY, &blk.imodes)
	} else {
		reconI16(b, &blk.coeffs, blk.nonZeroY, checkMode(mbX, mbY, blk.imodes[0]))
	}
	reconUV(b, &blk.coeffs, blk.nonZeroUV, checkMode(mbX, mbY, blk.uvmode))
}

// loadMBEdges sets up the left, top and (for intra 4x4) top-right samples
// of the macroblock in the work buffer.
func (d *decoder) loadMBEdges(mbX, mbY int, isI4x4 bool) {
	b := d.yuvB[:]
	if mbX > 0 {
		// Rotate in the left samples from the previous block.
		for j := -1; j < 16; j++ {
			o := yOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+12:])
		}
		for j := -1; j < 8; j++ {
			o := uOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+4:])
			o = vOff + j*bps
			*(*[4]byte)(b[o-4:]) = *(*[4]byte)(b[o+4:])
		}
	}
	top := &d.yuvT[mbX]
	if mbY > 0 {
		*(*[16]byte)(b[yOff-bps:]) = top.y
		*(*[8]byte)(b[uOff-bps:]) = top.u
		*(*[8]byte)(b[vOff-bps:]) = top.v
	}
	if isI4x4 {
		tr := yOff - bps + 16
		if mbY > 0 {
			if mbX >= d.mbW-1 { // on rightmost edge
				v := top.y[15]
				b[tr], b[tr+1], b[tr+2], b[tr+3] = v, v, v, v
			} else {
				*(*[4]byte)(b[tr:]) = *(*[4]byte)(d.yuvT[mbX+1].y[:4])
			}
		}
		// replicate the top-right pixels below
		for k := 1; k <= 3; k++ {
			*(*[4]byte)(b[tr+4*k*bps:]) = *(*[4]byte)(b[tr:])
		}
	}
}

// storeMB saves the bottom samples of the reconstructed macroblock for the
// next row and copies it into the row cache.
func (d *decoder) storeMB(mbX, mbY int) {
	b := d.yuvB[:]
	top := &d.yuvT[mbX]
	// stash away top samples for next block
	if mbY < d.mbH-1 {
		top.y = *(*[16]byte)(b[yOff+15*bps:])
		top.u = *(*[8]byte)(b[uOff+7*bps:])
		top.v = *(*[8]byte)(b[vOff+7*bps:])
	}
	// Transfer reconstructed samples to the row cache.
	ys := d.cacheYStride
	yo := d.cacheYOff + mbX*16
	for j := 0; j < 16; j++ {
		*(*[16]byte)(d.cacheY[yo+j*ys:]) = *(*[16]byte)(b[yOff+j*bps:])
	}
	uvs := d.cacheUVStride
	uvo := d.cacheUVOff + mbX*8
	for j := 0; j < 8; j++ {
		*(*[8]byte)(d.cacheU[uvo+j*uvs:]) = *(*[8]byte)(b[uOff+j*bps:])
		*(*[8]byte)(d.cacheV[uvo+j*uvs:]) = *(*[8]byte)(b[vOff+j*bps:])
	}
}
