package vp8l

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// simdLevel is one configuration of the amd64 kernels (dsp_amd64.go).
type simdLevel struct {
	name  string
	level cpuinfo.Level
	vbmi  bool
}

// simdLevels returns the kernel configurations of this machine: every level
// of cpuinfo.Levels, the AVX-512 one with VBMI when the machine has it, and
// then also AVX-512 without VBMI. On other architectures it is the Go level
// only, where the wrappers ignore it.
func simdLevels() []simdLevel {
	var ls []simdLevel
	for _, l := range cpuinfo.Levels() {
		ls = append(ls, simdLevel{l.String(), l, l == cpuinfo.AVX512 && cpuinfo.X86.HasAVX512VBMI})
	}
	if cpuinfo.X86.HasAVX512VBMI {
		ls = append(ls, simdLevel{"avx512-novbmi", cpuinfo.AVX512, false})
	}
	return ls
}

// setLevel selects the kernel configuration l and returns a function that
// restores the previous one.
func setLevel(l simdLevel) (restore func()) {
	level, vbmi := x86Level, x86VBMI
	x86Level, x86VBMI = l.level, l.vbmi
	return func() { x86Level, x86VBMI = level, vbmi }
}

// forEachLevel runs f as a subtest at every kernel configuration.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	for _, l := range simdLevels() {
		restore := setLevel(l)
		t.Run(l.name, f)
		restore()
	}
}

// atEachLevel runs f at every kernel configuration (fuzz targets).
func atEachLevel(f func()) {
	for _, l := range simdLevels() {
		func() {
			defer setLevel(l)()
			f()
		}()
	}
}
