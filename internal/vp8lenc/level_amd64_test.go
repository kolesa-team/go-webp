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

// setLz77ReachMin sets the smallest range of the lz77Reach kernel.
func setLz77ReachMin(m int) (restore func()) {
	old := lz77ReachMin
	lz77ReachMin = m
	return func() { lz77ReachMin = old }
}
