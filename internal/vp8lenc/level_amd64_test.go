//go:build amd64 && !purego

package vp8lenc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// testLevels are the kernel levels the tests run: every level of this
// machine, the pure Go code first.
func testLevels() []cpuinfo.Level { return cpuinfo.Levels() }

// setLevel selects the kernels of level l and returns a function restoring
// the previous selection.
func setLevel(l cpuinfo.Level) (restore func()) {
	old := x86Level
	x86Level = l
	return func() { x86Level = old }
}
