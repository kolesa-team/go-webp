//go:build amd64 && !purego

package pipeline

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// forEachLevel runs f as a subtest at every SIMD level this machine
// supports (Go first), so that the selected kernels are compared with the
// Go reference at each of them.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	saved := x86Level
	defer func() { x86Level = saved }()
	for _, l := range cpuinfo.Levels() {
		x86Level = l
		t.Run(l.String(), f)
	}
}

// eachLevel is forEachLevel without subtests, for fuzz targets.
func eachLevel(f func()) {
	saved := x86Level
	defer func() { x86Level = saved }()
	for _, l := range cpuinfo.Levels() {
		x86Level = l
		f()
	}
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
