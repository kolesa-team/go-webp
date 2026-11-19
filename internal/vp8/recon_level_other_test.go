//go:build !amd64 || purego

package vp8

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// eachReconLevel runs f once: only amd64 selects the reconstruction kernels
// by SIMD level (see recon_level_amd64_test.go).
func eachReconLevel(kernelsOnly bool, f func(name string)) {
	f("kernel")
}

// withReconLevel runs f: the reconstruction kernels have no level here.
func withReconLevel(_ cpuinfo.Level, f func()) {
	f()
}
