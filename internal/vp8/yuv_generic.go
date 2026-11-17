//go:build !arm64 || purego

package vp8

// upsamplePair converts a pair of luma rows with the fancy upsampler; see
// upsamplePairGo.
func upsamplePair(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA []byte, uvT, uvB []uint32, w int) {
	upsamplePairGo(topY, botY, topU, topV, curU, curV, topDst, botDst, topA, botA, uvT, uvB, w)
}

// sampleRow converts one luma row with point-sampled chroma; see
// sampleRowGo.
func sampleRow(ys, us, vs, as, dst []byte, w int) {
	sampleRowGo(ys, us, vs, as, dst, w)
}
