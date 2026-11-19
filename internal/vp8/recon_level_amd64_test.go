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

package vp8

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// eachReconLevel runs f with reconLevel set to every SIMD level of the
// machine (cpuinfo.Levels, the Go reference first) and restores it.
// kernelsOnly skips the levels that add no kernel of their own (Go, whose
// entry points are the reference, and AVX512, which runs the AVX2 kernels)
// when a kernel level exists, for the exhaustive tests.
func eachReconLevel(kernelsOnly bool, f func(name string)) {
	saved := reconLevel
	defer func() { reconLevel = saved }()
	for _, l := range cpuinfo.Levels() {
		if kernelsOnly && cpuinfo.Max > cpuinfo.Go && (l == cpuinfo.Go || l == cpuinfo.AVX512) {
			continue
		}
		reconLevel = l
		f(l.String())
	}
}

// withReconLevel runs f with reconLevel set to l and restores it, so that
// whole-decoder tests can switch the reconstruction kernels together with
// the row converters and the loop filters.
func withReconLevel(l cpuinfo.Level, f func()) {
	saved := reconLevel
	defer func() { reconLevel = saved }()
	reconLevel = l
	f()
}
