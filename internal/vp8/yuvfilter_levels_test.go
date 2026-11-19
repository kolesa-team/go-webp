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

package vp8

import (
	"fmt"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// withLevels runs fn as a subtest once per SIMD level this machine
// supports (cpuinfo.Levels), with *lvl set to that level, and restores
// *lvl afterwards. lvl is the selector of the kernels under test
// (yuvLevel, filterLevel). On non-x86 machines only the "go" level exists
// and the selector is ignored, so fn runs once on the platform's kernels.
func withLevels(t *testing.T, lvl *cpuinfo.Level, fn func(t *testing.T)) {
	t.Helper()
	saved := *lvl
	defer func() { *lvl = saved }()
	for _, l := range cpuinfo.Levels() {
		*lvl = l
		t.Run(l.String(), fn)
	}
}

// forLevels calls fn once per supported level with *lvl set to it, for
// fuzz targets (which cannot start subtests); fn gets the level name for
// its failure messages.
func forLevels(lvl *cpuinfo.Level, fn func(level string)) {
	saved := *lvl
	defer func() { *lvl = saved }()
	for _, l := range cpuinfo.Levels() {
		*lvl = l
		fn(l.String())
	}
}

// benchLevels runs fn as a sub-benchmark per supported level.
func benchLevels(b *testing.B, lvl *cpuinfo.Level, fn func(b *testing.B)) {
	saved := *lvl
	defer func() { *lvl = saved }()
	for _, l := range cpuinfo.Levels() {
		*lvl = l
		b.Run(fmt.Sprint(l), fn)
	}
}
