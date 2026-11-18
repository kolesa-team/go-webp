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
