package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): dec/vp8_dec.c, dec/quant_dec.c,
// dec/tree_dec.c (header, segment, filter, partition, quantizer and
// probability parsing).

import "github.com/kolesa-team/go-webp/v2/internal/status"

const (
	numMBSegments    = 4
	numRefLFDeltas   = 4
	numModeLFDeltas  = 4
	maxNumPartitions = 8
	numTypes         = 4
	numBands         = 8
	numCtx           = 3
	numProbas        = 11
	numBModes        = 10

	// frameHeaderSize is the size of the key frame tag plus start code and
	// dimensions.
	frameHeaderSize = 10
)

// kBands maps a coefficient index to its probability band. The 17th entry
// is a sentinel used when looking one coefficient ahead.
var kBands = [16 + 1]uint8{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0}

// frameInfo is the result of validating the uncompressed key frame header.
type frameInfo struct {
	width, height int
	partLen       int // size of the first partition
}

// parseFrameTag validates the 10-byte key frame header like libwebp's
// VP8GetInfo; as in WebPDecode, every failure except a short buffer is
// reported as a bitstream error.
func parseFrameTag(data []byte) (frameInfo, error) {
	var fi frameInfo
	if len(data) < frameHeaderSize {
		return fi, status.NotEnoughData("truncated VP8 frame header")
	}
	b := data[:frameHeaderSize]
	tag := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
	if b[3] != 0x9d || b[4] != 0x01 || b[5] != 0x2a {
		return fi, status.Bitstream("bad VP8 start code")
	}
	if tag&1 != 0 {
		return fi, status.Bitstream("not a key frame")
	}
	if (tag>>1)&7 > 3 {
		return fi, status.Bitstream("incorrect keyframe parameters")
	}
	if (tag>>4)&1 == 0 {
		return fi, status.Bitstream("frame not displayable")
	}
	fi.partLen = int(tag >> 5)
	if fi.partLen >= len(data) {
		return fi, status.Bitstream("bad partition length")
	}
	fi.width = int(uint16(b[7])<<8|uint16(b[6])) & 0x3fff
	fi.height = int(uint16(b[9])<<8|uint16(b[8])) & 0x3fff
	if fi.width == 0 || fi.height == 0 {
		return fi, status.Bitstream("invalid VP8 dimensions")
	}
	return fi, nil
}

// DecodeHeader parses the VP8 frame header of a VP8 chunk payload (starting
// at the 3-byte frame tag) and returns the frame size.
func DecodeHeader(data []byte) (width, height int, err error) {
	fi, err := parseFrameTag(data)
	if err != nil {
		return 0, 0, err
	}
	return fi.width, fi.height, nil
}

// parseHeaders parses the frame header and the first partition's global
// parameters, and sets up the token partitions.
func (d *decoder) parseHeaders(data []byte) error {
	fi, err := parseFrameTag(data)
	if err != nil {
		return err
	}
	d.width, d.height = fi.width, fi.height
	d.mbW = (d.width + 15) >> 4
	d.mbH = (d.height + 15) >> 4

	// Reset the segment header and probabilities (key frame).
	d.useSegment = false
	d.updateMap = false
	d.absoluteDelta = true
	d.quantizer = [numMBSegments]int{}
	d.filterStrength = [numMBSegments]int{}
	d.segProba = [3]uint8{255, 255, 255}
	d.refLFDelta = [numRefLFDeltas]int{}
	d.modeLFDelta = [numModeLFDeltas]int{}

	buf := data[frameHeaderSize:]
	if fi.partLen > len(buf) {
		return status.NotEnoughData("bad partition length")
	}
	br := &d.br
	br.init(buf[:fi.partLen])
	buf = buf[fi.partLen:]

	br.get() // color space
	br.get() // clamping type
	if !d.parseSegmentHeader() {
		return status.Bitstream("cannot parse segment header")
	}
	if !d.parseFilterHeader() {
		return status.Bitstream("cannot parse filter header")
	}
	if err := d.parsePartitions(buf); err != nil {
		return err
	}
	d.parseQuant()
	br.get() // ignore the value of update_proba
	d.parseProba()
	return nil
}

func (d *decoder) parseSegmentHeader() bool {
	br := &d.br
	d.useSegment = br.get()
	if d.useSegment {
		d.updateMap = br.get()
		if br.get() { // update data
			d.absoluteDelta = br.get()
			for s := range d.quantizer {
				d.quantizer[s] = 0
				if br.get() {
					d.quantizer[s] = br.getSignedValue(7)
				}
			}
			for s := range d.filterStrength {
				d.filterStrength[s] = 0
				if br.get() {
					d.filterStrength[s] = br.getSignedValue(6)
				}
			}
		}
		if d.updateMap {
			for s := range d.segProba {
				d.segProba[s] = 255
				if br.get() {
					d.segProba[s] = uint8(br.getValue(8))
				}
			}
		}
	} else {
		d.updateMap = false
	}
	return !br.eof
}

func (d *decoder) parseFilterHeader() bool {
	br := &d.br
	d.simple = br.get()
	d.level = br.getValue(6)
	d.sharpness = br.getValue(3)
	d.useLFDelta = br.get()
	if d.useLFDelta {
		if br.get() { // update lf-delta?
			for i := range d.refLFDelta {
				if br.get() {
					d.refLFDelta[i] = br.getSignedValue(6)
				}
			}
			for i := range d.modeLFDelta {
				if br.get() {
					d.modeLFDelta[i] = br.getSignedValue(6)
				}
			}
		}
	}
	switch {
	case d.level == 0:
		d.filterType = 0
	case d.simple:
		d.filterType = 1
	default:
		d.filterType = 2
	}
	return !br.eof
}

func (d *decoder) parsePartitions(buf []byte) error {
	d.numPartsMinusOne = (1 << uint(d.br.getValue(2))) - 1
	last := d.numPartsMinusOne
	if len(buf) < 3*last {
		return status.NotEnoughData("cannot parse partitions")
	}
	sz := buf
	part := buf[3*last:]
	for p := 0; p < last; p++ {
		psize := int(sz[0]) | int(sz[1])<<8 | int(sz[2])<<16
		if psize > len(part) {
			psize = len(part)
		}
		d.parts[p].init(part[:psize])
		part = part[psize:]
		sz = sz[3:]
	}
	d.parts[last].init(part)
	if len(part) == 0 {
		return status.NotEnoughData("cannot parse partitions")
	}
	return nil
}

func clipQ(v, m int) int {
	return min(max(v, 0), m)
}

func (d *decoder) parseQuant() {
	br := &d.br
	baseQ0 := br.getValue(7)
	opt := func() int {
		if br.get() {
			return br.getSignedValue(4)
		}
		return 0
	}
	dqy1DC := opt()
	dqy2DC := opt()
	dqy2AC := opt()
	dquvDC := opt()
	dquvAC := opt()
	for i := 0; i < numMBSegments; i++ {
		var q int
		if d.useSegment {
			q = d.quantizer[i]
			if !d.absoluteDelta {
				q += baseQ0
			}
		} else {
			if i > 0 {
				d.dqm[i] = d.dqm[0]
				continue
			}
			q = baseQ0
		}
		m := &d.dqm[i]
		m.y1[0] = int32(kDcTable[clipQ(q+dqy1DC, 127)])
		m.y1[1] = int32(kAcTable[clipQ(q, 127)])
		m.y2[0] = int32(kDcTable[clipQ(q+dqy2DC, 127)]) * 2
		// For all x in [0..284], x*155/100 is bitwise equal to (x*101581) >> 16.
		m.y2[1] = (int32(kAcTable[clipQ(q+dqy2AC, 127)]) * 101581) >> 16
		if m.y2[1] < 8 {
			m.y2[1] = 8
		}
		m.uv[0] = int32(kDcTable[clipQ(q+dquvDC, 117)])
		m.uv[1] = int32(kAcTable[clipQ(q+dquvAC, 127)])
		m.uvQuant = q + dquvAC
		m.dither = 0
	}
}

func (d *decoder) parseProba() {
	br := &d.br
	for t := 0; t < numTypes; t++ {
		var bands [numBands][numCtx][numProbas]uint8
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					v := coeffsProba0[t][b][c][p]
					if br.getBit(coeffsUpdateProba[t][b][c][p]) != 0 {
						v = uint8(br.getValue(8))
					}
					bands[b][c][p] = v
				}
			}
		}
		for n := 0; n < 17; n++ {
			d.proba[t][n] = bands[kBands[n]]
		}
	}
	d.useSkipProba = br.get()
	if d.useSkipProba {
		d.skipP = uint8(br.getValue(8))
	}
}

// precomputeFilterStrengths mirrors libwebp's PrecomputeFilterStrengths.
func (d *decoder) precomputeFilterStrengths() {
	if d.filterType == 0 {
		return
	}
	for s := 0; s < numMBSegments; s++ {
		baseLevel := d.level
		if d.useSegment {
			baseLevel = d.filterStrength[s]
			if !d.absoluteDelta {
				baseLevel += d.level
			}
		}
		for i4x4 := 0; i4x4 <= 1; i4x4++ {
			info := &d.fstrengths[s][i4x4]
			level := baseLevel
			if d.useLFDelta {
				level += d.refLFDelta[0]
				if i4x4 != 0 {
					level += d.modeLFDelta[0]
				}
			}
			level = min(max(level, 0), 63)
			if level > 0 {
				ilevel := level
				if d.sharpness > 0 {
					if d.sharpness > 4 {
						ilevel >>= 2
					} else {
						ilevel >>= 1
					}
					if ilevel > 9-d.sharpness {
						ilevel = 9 - d.sharpness
					}
				}
				if ilevel < 1 {
					ilevel = 1
				}
				info.ilevel = uint8(ilevel)
				info.limit = uint8(2*level + ilevel)
				switch {
				case level >= 40:
					info.hevThresh = 2
				case level >= 15:
					info.hevThresh = 1
				default:
					info.hevThresh = 0
				}
			} else {
				info.limit = 0 // no filtering
			}
			info.inner = uint8(i4x4)
		}
	}
}
