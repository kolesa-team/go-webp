package vp8

import "testing"

// forEachReconLevel runs f as a subtest at every reconstruction kernel
// level (see eachReconLevel).
func forEachReconLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	eachReconLevel(false, func(name string) { t.Run(name, f) })
}

// forEachReconKernel is forEachReconLevel for the exhaustive tests: only
// the levels with kernels of their own.
func forEachReconKernel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	eachReconLevel(true, func(name string) { t.Run(name, f) })
}
