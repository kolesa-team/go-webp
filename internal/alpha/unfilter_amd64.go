// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
