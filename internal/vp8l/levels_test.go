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

package vp8l

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// simdLevel is one configuration of the amd64 kernels (dsp_amd64.go).
type simdLevel struct {
	name  string
	level cpuinfo.Level
	vbmi  bool
}

// simdLevels returns the kernel configurations of this machine: every level
// of cpuinfo.Levels, the AVX-512 one with VBMI when the machine has it, and
// then also AVX-512 without VBMI. On other architectures it is the Go level
// only, where the wrappers ignore it.
func simdLevels() []simdLevel {
	var ls []simdLevel
	for _, l := range cpuinfo.Levels() {
		ls = append(ls, simdLevel{l.String(), l, l == cpuinfo.AVX512 && cpuinfo.X86.HasAVX512VBMI})
	}
	if cpuinfo.X86.HasAVX512VBMI {
		ls = append(ls, simdLevel{"avx512-novbmi", cpuinfo.AVX512, false})
	}
	return ls
}

// setLevel selects the kernel configuration l and returns a function that
// restores the previous one.
func setLevel(l simdLevel) (restore func()) {
	level, vbmi := x86Level, x86VBMI
	x86Level, x86VBMI = l.level, l.vbmi
	return func() { x86Level, x86VBMI = level, vbmi }
}

// forEachLevel runs f as a subtest at every kernel configuration.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	for _, l := range simdLevels() {
		restore := setLevel(l)
		t.Run(l.name, f)
		restore()
	}
}

// atEachLevel runs f at every kernel configuration (fuzz targets).
func atEachLevel(f func()) {
	for _, l := range simdLevels() {
		func() {
			defer setLevel(l)()
			f()
		}()
	}
}
