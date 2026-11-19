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

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// The horizontal and vertical unfilters have a pure-Go reference
// (filters.go) and, on arm64 and amd64, assembly versions selected by the
// wrappers of unfilter_<arch>.go (unfilter_generic.go elsewhere and with
// the purego tag).

// x86Level is the SIMD level of the amd64 kernels (unfilter_amd64.go): below
// a kernel's level its next lower version runs, down to the Go reference.
// Tests set it to every level of cpuinfo.Levels. Unused on other
// architectures.
var x86Level = cpuinfo.Max
