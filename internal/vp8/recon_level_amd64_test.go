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
