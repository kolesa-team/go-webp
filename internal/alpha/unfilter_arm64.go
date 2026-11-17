//go:build arm64 && !purego

package alpha

// NEON kernels of the horizontal and vertical unfilters
// (unfilter_arm64.s). The wrappers check the lengths and give the assembly
// whole blocks of 16 bytes; the last len(in)%16 bytes are done in Go.

//go:noescape
func unfilterHorizontalAsm(in, out *byte, n, pred int)

//go:noescape
func unfilterVerticalAsm(prev, in, out *byte, n int)

// unfilterHorizontal sets out[i] = pred + in[0] + ... + in[i] (mod 256).
// in may alias out.
func unfilterHorizontal(pred byte, in, out []byte) {
	out = out[:len(in)]
	n := len(in) &^ 15
	if n > 0 {
		unfilterHorizontalAsm(&in[0], &out[0], n, int(pred))
		pred = out[n-1]
	}
	for i := n; i < len(in); i++ {
		pred += in[i]
		out[i] = pred
	}
}

// unfilterVertical sets out[i] = prev[i] + in[i] (mod 256) for i < len(in).
// in may alias out.
func unfilterVertical(prev, in, out []byte) {
	out, prev = out[:len(in)], prev[:len(in)]
	n := len(in) &^ 15
	if n > 0 {
		unfilterVerticalAsm(&prev[0], &in[0], &out[0], n)
	}
	for i := n; i < len(in); i++ {
		out[i] = prev[i] + in[i]
	}
}
