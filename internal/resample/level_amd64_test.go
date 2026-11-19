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

package resample

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// forEachLevel runs f as a subtest at every SIMD level this machine
// supports (Go first), so that the selected kernels are compared with the
// Go reference at each of them. Rescalers must be created inside f: they
// keep the level they were initialized with.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	saved := x86Level
	defer func() { x86Level = saved }()
	for _, l := range levels() {
		x86Level = l
		t.Run(l.String(), f)
	}
}

// eachLevel is forEachLevel without subtests, for fuzz targets.
func eachLevel(f func()) {
	saved := x86Level
	defer func() { x86Level = saved }()
	for _, l := range levels() {
		x86Level = l
		f()
	}
}

// levels returns the levels the tests run at: every level this machine
// supports, or under the race detector only Go and the best one. The race
// detector does not see the memory accesses of assembly, the wrappers are
// the same Go code at every level, and the plain test run covers the
// others (with all four levels the package's race run took twice as long
// on an AVX-512 machine).
func levels() []cpuinfo.Level {
	ls := cpuinfo.Levels()
	if raceEnabled && len(ls) > 2 {
		ls = []cpuinfo.Level{ls[0], ls[len(ls)-1]}
	}
	return ls
}

// benchLevels returns the names of the levels to benchmark the selected
// kernels at, and a function selecting the i-th one (restore with
// restoreLevel).
func benchLevels() ([]string, func(i int)) {
	ls := cpuinfo.Levels()[1:]
	names := make([]string, len(ls))
	for i, l := range ls {
		names[i] = l.String()
	}
	return names, func(i int) { x86Level = ls[i] }
}

// restoreLevel resets the kernel level to the machine's best.
func restoreLevel() { x86Level = cpuinfo.Max }
