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

//go:build amd64

package cpuinfo

import "golang.org/x/sys/cpu"

func init() {
	c := &cpu.X86
	X86.HasSSE41 = c.HasSSE2 && c.HasSSE3 && c.HasSSSE3 && c.HasSSE41
	// x/sys/cpu reports AVX/AVX2/AVX-512 only when the OS saves the
	// corresponding register state (XGETBV), so no further check is needed.
	X86.HasAVX2 = X86.HasSSE41 && c.HasAVX && c.HasAVX2 && c.HasBMI1 && c.HasBMI2 && c.HasFMA
	X86.HasAVX512 = X86.HasAVX2 && c.HasAVX512F && c.HasAVX512BW && c.HasAVX512CD && c.HasAVX512DQ && c.HasAVX512VL
	X86.HasAVX512VBMI = X86.HasAVX512 && c.HasAVX512VBMI
	X86.HasAVX512VBMI2 = X86.HasAVX512 && c.HasAVX512VBMI2
	X86.HasAVX512VNNI = X86.HasAVX512 && c.HasAVX512VNNI
	X86.HasAVX512GFNI = X86.HasAVX512 && c.HasAVX512GFNI
	switch {
	case X86.HasAVX512:
		Max = AVX512
	case X86.HasAVX2:
		Max = AVX2
	case X86.HasSSE41:
		Max = SSE41
	}
}
