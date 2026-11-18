//go:build (!arm64 && !amd64) || purego

package alpha

// Without the assembly kernels of unfilter_arm64.s and unfilter_amd64.s,
// the unfilters are their pure-Go references.

func unfilterHorizontal(pred byte, in, out []byte) { unfilterHorizontalGo(pred, in, out) }
func unfilterVertical(prev, in, out []byte)        { unfilterVerticalGo(prev, in, out) }
