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

package vp8lenc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// testLevels are the kernel levels the tests run: every level of this
// machine, the pure Go code first.
func testLevels() []cpuinfo.Level { return cpuinfo.Levels() }

// kernelLevels are the levels of testLevels with kernels of their own (the
// levels above maxKernelLevel run its kernels): the last one runs the
// kernels of the default level.
func kernelLevels() []cpuinfo.Level {
	var ls []cpuinfo.Level
	for _, l := range testLevels() {
		if l <= maxKernelLevel {
			ls = append(ls, l)
		}
	}
	return ls
}

// setLevel selects the kernels of level l and returns a function restoring
// the previous selection.
func setLevel(l cpuinfo.Level) (restore func()) {
	old := x86Level
	x86Level = l
	return func() { x86Level = old }
}

// setPrefetch sets the prefetch distance of the hash chain kernel.
func setPrefetch(pf int) (restore func()) {
	old := hcPrefetch
	hcPrefetch = pf
	return func() { hcPrefetch = old }
}

// setAddRefsMin sets the smallest parse of the addRefs kernel.
func setAddRefsMin(n int) (restore func()) {
	old := addRefsMin
	addRefsMin = n
	return func() { addRefsMin = old }
}

// setChunks sets the chunks of the whole-image kernels to n (pixChunk to n
// rounded up to a multiple of 8).
func setChunks(n int) (restore func()) {
	h, r, c, p := hcChunk, refsChunk, cacheChunk, pixChunk
	hcChunk, refsChunk, cacheChunk, pixChunk = n, n, n, (n+7)&^7
	return func() { hcChunk, refsChunk, cacheChunk, pixChunk = h, r, c, p }
}

// setLz77ReachMin sets the smallest range of the lz77Reach kernel.
func setLz77ReachMin(m int) (restore func()) {
	old := lz77ReachMin
	lz77ReachMin = m
	return func() { lz77ReachMin = old }
}
