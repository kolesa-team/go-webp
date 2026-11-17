package vp8enc

import "math/bits"

// Batched DSP entry points. The mode search works on whole macroblocks
// (16 luma or 8 chroma 4x4 blocks per prediction mode) and on all 10
// intra4 modes of one sub-block at a time, so the kernels below take a
// list of block pairs instead of a single 4x4 block: an assembly
// implementation then pays the call overhead once per macroblock or
// sub-block, not once per 4x4 transform.
//
// The *Ref functions are the reference semantics, written on top of the
// single-block Go code of dsp.go; they are always compiled so that the
// tests can compare an assembly implementation against them. The entry
// points without the suffix (fTransformPairs, quantizeBlocks, ...) are
// bound to assembly in dsp_arm64.go and to these references in
// dsp_generic.go.

// pairTable lists pairs of horizontally adjacent 4x4 blocks. Pair k is made
// of blocks 2k and 2k+1; off[k][0] and off[k][1] are the byte offsets of
// the left block of the pair from two base pointers (the right block is 4
// bytes further). Rows are bps bytes apart.
type pairTable struct {
	off  [8][2]uint16
	n    int    // number of pairs
	need [2]int // minimum length of the slices the offsets index
}

func newPairTable(offs ...[2]uint16) pairTable {
	var t pairTable
	t.n = len(offs)
	for k, o := range offs {
		t.off[k] = o
		for i := 0; i < 2; i++ {
			t.need[i] = max(t.need[i], int(o[i])+3*bps+8)
		}
	}
	return t
}

// predOffI4 is the offset of the intra4 prediction of mode m from i4DC4 in
// the prediction buffer (modes 0-7 on the first 4 rows, 8-9 below).
func predOffI4(m int) uint16 { return uint16((m&7)*4 + (m>>3)*4*bps) }

var (
	// tabY16: the 16 luma blocks in raster (vp8Scan) order; both offsets
	// are the block position (source and prediction, or prediction and
	// reconstruction, share the layout).
	tabY16 = newPairTable(
		[2]uint16{0, 0}, [2]uint16{8, 8},
		[2]uint16{4 * bps, 4 * bps}, [2]uint16{4*bps + 8, 4*bps + 8},
		[2]uint16{8 * bps, 8 * bps}, [2]uint16{8*bps + 8, 8*bps + 8},
		[2]uint16{12 * bps, 12 * bps}, [2]uint16{12*bps + 8, 12*bps + 8})
	// tabUV: the 8 chroma blocks in vp8ScanUV order (U then V).
	tabUV = newPairTable(
		[2]uint16{0, 0}, [2]uint16{4 * bps, 4 * bps},
		[2]uint16{8, 8}, [2]uint16{4*bps + 8, 4*bps + 8})
	// tabI4Src: the source sub-block (duplicated side by side, see
	// iterator.i4Src) against the 10 intra4 predictions, which start at
	// i4DC4.
	tabI4Src = newPairTable(
		[2]uint16{0, predOffI4(0)}, [2]uint16{0, predOffI4(2)}, [2]uint16{0, predOffI4(4)},
		[2]uint16{0, predOffI4(6)}, [2]uint16{0, predOffI4(8)})
	// tabI4Rec: the 10 intra4 predictions and their reconstructions, which
	// use the same layout (iterator.i4Rec).
	tabI4Rec = newPairTable(
		[2]uint16{predOffI4(0), predOffI4(0)}, [2]uint16{predOffI4(2), predOffI4(2)},
		[2]uint16{predOffI4(4), predOffI4(4)}, [2]uint16{predOffI4(6), predOffI4(6)},
		[2]uint16{predOffI4(8), predOffI4(8)})
)

// fTransformPairsRef computes, for each pair k of t, out[2k+b] =
// fTransform(src[off[k][0]+4b:], ref[off[k][1]+4b:]), b = 0, 1.
func fTransformPairsRef(src, ref []byte, t *pairTable, out [][16]int16) {
	for k := 0; k < t.n; k++ {
		s, r := int(t.off[k][0]), int(t.off[k][1])
		fTransform(src[s:], ref[r:], &out[2*k])
		fTransform(src[s+4:], ref[r+4:], &out[2*k+1])
	}
}

// iTransformPairsRef computes, for each pair k of t, the reconstruction
// iTransform(ref[off[k][0]+4b:], in[2k+b], dst[off[k][1]+4b:]), b = 0, 1.
func iTransformPairsRef(ref []byte, in [][16]int16, dst []byte, t *pairTable) {
	for k := 0; k < t.n; k++ {
		r, d := int(t.off[k][0]), int(t.off[k][1])
		iTransform(ref[r:], &in[2*k], dst[d:])
		iTransform(ref[r+4:], &in[2*k+1], dst[d+4:])
	}
}

// levelInfoOf summarizes a block of quantized levels (zigzag order) for the
// rate and flatness estimates: bits 0-7 hold the index of the last
// non-zero level plus one (0 for an empty block), bits 8-15 the number of
// non-zero levels after the first (the AC levels).
func levelInfoOf(c *[16]int16) uint16 {
	m := uint32(0)
	for i := 15; i >= 0; i-- {
		v := uint32(uint16(c[i]))
		m = m<<1 | (v|-v)>>31
	}
	return uint16(bits.Len32(m)) | uint16(bits.OnesCount32(m>>1))<<8
}

// infoLast returns the index of the last non-zero level, or -1.
func infoLast(info uint16) int { return int(info&0xff) - 1 }

// infoNzAC returns the number of non-zero AC levels.
func infoNzAC(info uint16) int { return int(info >> 8) }

// quantizeBlocksRef quantizes each block of in into the matching block of
// out with quantizeBlock, and stores levelInfoOf of each block into info
// unless info is nil. Bit n of the result is set if block n has a non-zero
// level.
func quantizeBlocksRef(in, out [][16]int16, m *matrix, info []uint16) uint32 {
	if len(out) < len(in) || len(in) > 32 || (info != nil && len(info) < len(in)) {
		panic("vp8enc: quantizeBlocks: bad block count")
	}
	nz := uint32(0)
	for n := range in {
		if quantizeBlock(&in[n], &out[n], m) {
			nz |= 1 << uint(n)
		}
		if info != nil {
			info[n] = levelInfoOf(&out[n])
		}
	}
	return nz
}

// sse4x4PairsRef computes out[2k+b] = sse4x4(a[off[k][0]+4b:],
// b[off[k][1]+4b:]).
func sse4x4PairsRef(a, b []byte, t *pairTable, out []int32) {
	for k := 0; k < t.n; k++ {
		x, y := int(t.off[k][0]), int(t.off[k][1])
		out[2*k] = int32(sse4x4(a[x:], b[y:]))
		out[2*k+1] = int32(sse4x4(a[x+4:], b[y+4:]))
	}
}

// tTransformPairsRef computes out[2k+b] = tTransform(a[off[k][0]+4b:],
// &kWeightY); the second offset of the table is not used.
func tTransformPairsRef(a []byte, t *pairTable, out []int32) {
	for k := 0; k < t.n; k++ {
		x := int(t.off[k][0])
		out[2*k] = tTransform(a[x:], &kWeightY)
		out[2*k+1] = tTransform(a[x+4:], &kWeightY)
	}
}
