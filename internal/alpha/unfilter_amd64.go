//go:build amd64 && !purego

package alpha

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// SSE4.1 and AVX2 kernels of the horizontal and vertical unfilters
// (unfilter_amd64.s), selected by x86Level. The wrappers check the lengths
// and give the assembly whole blocks of 16 bytes; the last len(in)%16 bytes
// are done in Go. (The gradient unfilter stays in Go: its serial chain per
// byte is no shorter in vector registers, see libwebp's
// GradientPredictInverse_SSE2.)

//go:noescape
func unfilterHorizontalSSE41(in, out *byte, n, pred int)

//go:noescape
func unfilterHorizontalAVX2(in, out *byte, n, pred int)

//go:noescape
func unfilterVerticalSSE41(prev, in, out *byte, n int)

//go:noescape
func unfilterVerticalAVX2(prev, in, out *byte, n int)

// unfilterHorizontal sets out[i] = pred + in[0] + ... + in[i] (mod 256).
// in may alias out.
func unfilterHorizontal(pred byte, in, out []byte) {
	out = out[:len(in)]
	n := len(in) &^ 15
	if n == 0 || x86Level < cpuinfo.SSE41 {
		unfilterHorizontalGo(pred, in, out)
		return
	}
	if x86Level >= cpuinfo.AVX2 {
		unfilterHorizontalAVX2(&in[0], &out[0], n, int(pred))
	} else {
		unfilterHorizontalSSE41(&in[0], &out[0], n, int(pred))
	}
	unfilterHorizontalGo(out[n-1], in[n:], out[n:])
}

// unfilterVertical sets out[i] = prev[i] + in[i] (mod 256) for i < len(in).
// in may alias out.
func unfilterVertical(prev, in, out []byte) {
	out, prev = out[:len(in)], prev[:len(in)]
	n := len(in) &^ 15
	if n == 0 || x86Level < cpuinfo.SSE41 {
		unfilterVerticalGo(prev, in, out)
		return
	}
	if x86Level >= cpuinfo.AVX2 {
		unfilterVerticalAVX2(&prev[0], &in[0], &out[0], n)
	} else {
		unfilterVerticalSSE41(&prev[0], &in[0], &out[0], n)
	}
	unfilterVerticalGo(prev[n:], in[n:], out[n:])
}
