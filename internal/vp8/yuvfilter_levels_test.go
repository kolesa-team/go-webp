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
