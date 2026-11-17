//go:build !arm64 || purego

package alpha

// Without the NEON kernels of unfilter_arm64.s, the unfilters are their
// pure-Go references.

func unfilterHorizontal(pred byte, in, out []byte) { unfilterHorizontalGo(pred, in, out) }
func unfilterVertical(prev, in, out []byte)        { unfilterVerticalGo(prev, in, out) }
