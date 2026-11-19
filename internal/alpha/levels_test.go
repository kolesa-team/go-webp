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

package alpha

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// setLevel selects the SIMD level l of the amd64 kernels
// (unfilter_amd64.go) and returns a function that restores the previous
// one.
func setLevel(l cpuinfo.Level) (restore func()) {
	prev := x86Level
	x86Level = l
	return func() { x86Level = prev }
}

// forEachLevel runs f as a subtest at every SIMD level of this machine
// (cpuinfo.Levels; only the Go level on other architectures, where the
// wrappers ignore it).
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	for _, l := range cpuinfo.Levels() {
		restore := setLevel(l)
		t.Run(l.String(), f)
		restore()
	}
}

// atEachLevel runs f at every SIMD level (fuzz targets).
func atEachLevel(f func()) {
	for _, l := range cpuinfo.Levels() {
		func() {
			defer setLevel(l)()
			f()
		}()
	}
}
