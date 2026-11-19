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

// Package cpuinfo reports which x86 SIMD extensions the assembly kernels of
// the codec may use. On amd64 it is filled from golang.org/x/sys/cpu (which
// also checks that the operating system saves the vector registers); on
// every other architecture all fields are false and Max is Go.
//
// Kernels pick an implementation by Level. Each package keeps its own
// selected level in a variable initialized from Max, so that tests can run
// every level the machine supports (see Levels) against the pure Go code.
package cpuinfo

// Level is an x86-64 SIMD level, from the pure Go code up.
type Level int

const (
	// Go is the pure Go code (no assembly).
	Go Level = iota
	// SSE41 is SSE2 through SSE4.1 (SSSE3 included).
	SSE41
	// AVX2 is x86-64-v3: AVX, AVX2, BMI1, BMI2, FMA, with the OS saving
	// the ymm registers.
	AVX2
	// AVX512 is x86-64-v4: AVX2 plus AVX-512 F, BW, CD, DQ and VL, with
	// the OS saving the zmm and mask registers. Optional extensions
	// (VBMI, VBMI2, VNNI, GFNI) are reported separately in X86.
	AVX512
)

func (l Level) String() string {
	switch l {
	case Go:
		return "go"
	case SSE41:
		return "sse41"
	case AVX2:
		return "avx2"
	case AVX512:
		return "avx512"
	}
	return "unknown"
}

// X86 holds the extensions available beyond the levels; all false on
// other architectures.
var X86 struct {
	HasSSE41       bool // SSE2, SSE3, SSSE3, SSE4.1
	HasAVX2        bool // the AVX2 level (x86-64-v3)
	HasAVX512      bool // the AVX512 level (x86-64-v4)
	HasAVX512VBMI  bool // VPERMB, VPERMT2B, VPMULTISHIFTQB (needs HasAVX512)
	HasAVX512VBMI2 bool // VPEXPANDB/VPCOMPRESSB, VPSHLDV... (needs HasAVX512)
	HasAVX512VNNI  bool // VPDPBUSD, VPDPWSSD (needs HasAVX512)
	HasAVX512GFNI  bool // GF2P8AFFINEQB... (needs HasAVX512)
}

// Max is the highest level this machine supports.
var Max = Go

// Levels returns the levels this machine supports, Go first.
func Levels() []Level {
	ls := []Level{Go}
	for l := SSE41; l <= Max; l++ {
		ls = append(ls, l)
	}
	return ls
}
