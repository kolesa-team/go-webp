//go:build !amd64 || purego

package vp8

// eachReconLevel runs f once: only amd64 selects the reconstruction kernels
// by SIMD level (see recon_level_amd64_test.go).
func eachReconLevel(kernelsOnly bool, f func(name string)) {
	f("kernel")
}
