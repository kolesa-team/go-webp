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

//go:build (!arm64 && !amd64) || purego

package vp8l

// Without the assembly kernels of dsp_arm64.s and dsp_*amd64.s, the per-row
// kernels are their pure-Go references.

const haveAsm = false

func emitRGBA(src []uint32, dst []byte)                { emitRGBAGo(src, dst) }
func emitAddGreen(src []uint32, dst []byte)            { emitAddGreenGo(src, dst) }
func addGreen(in, out []uint32)                        { addGreenGo(in, out) }
func lookupBytes(pal *[256]byte, in, out []byte)       { lookupBytesGo(pal, in, out) }
func lookupRGBA(pal *[256]uint32, src, dst []byte)     { lookupRGBAGo(pal, src, dst) }
func crossColorRow(mults, src, dst []uint32, bits int) { crossColorRowGo(mults, src, dst, bits) }
func predictRow(modes, in, out, up []uint32, bits int) { predictRowGo(modes, in, out, up, bits) }

func expandBundledRow(a *alphaTables, bits int, src, dst []byte) {
	expandBundledRowGo(a, bits, src, dst)
}

func expandBundledRGBA(pal *[256]uint32, tab []uint64, bits int, src, dst []byte) {
	expandBundledRGBAGo(tab, bits, src, dst)
}
