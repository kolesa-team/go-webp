//go:build !amd64 || purego

package vp8enc

import "testing"

// Without x86 kernel levels the DSP entry points have a single
// implementation (NEON on arm64, the Go reference elsewhere): the level
// helpers of dsp_levels_amd64_test.go run their function once.

func eachLevel(f func(level string)) { f("default") }

func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	f(t)
}

func benchLevels(b *testing.B, name string, f func()) {
	b.Run(name+"/vec", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			f()
		}
	})
}
