//go:build !arm64 || purego

package resample

// archState holds no kernel state without assembly; kind is always 0 (the
// Go kernels).
type archState struct{ kind uint8 }

func archTableSize(dstWidth, numChannels int) int { return 0 }

func (r *Rescaler) initArch(tab []uint32) {}

func (r *Rescaler) shrinkRow(src []byte, acc []uint32) { r.shrinkRowRef(src, acc) }

func (r *Rescaler) expandRow(src []byte, acc []uint32) { r.expandRowRef(src, acc) }

func exportExpand(dst []byte, frow []uint32, fy uint32) { exportExpandGo(dst, frow, fy) }

func exportExpandMix(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	exportExpandMixGo(dst, frow, irow, a, b, fy)
}

func exportShrink(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	exportShrinkGo(dst, irow, frow, yscale, fxy)
}

func exportShrinkZero(dst []byte, irow []uint32, fxy uint32) { exportShrinkZeroGo(dst, irow, fxy) }

func premultiplyRow(row []byte) { multRGBA(row, &premulScale) }

func unpremultiplyRow(row []byte) { multRGBA(row, &unpremulScale) }
