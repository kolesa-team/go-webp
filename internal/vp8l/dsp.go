package vp8l

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// The per-row kernels of the inverse transforms and the output conversion
// have a pure-Go reference (the ...Go functions, the specification) and,
// on arm64 and amd64, assembly versions selected by the wrappers of
// dsp_<arch>.go (dsp_generic.go elsewhere and with the purego tag).

var (
	// x86Level is the SIMD level of the amd64 kernels (dsp_amd64.go):
	// below a kernel's level its next lower version runs, down to the Go
	// reference. Tests set it to every level of cpuinfo.Levels. Unused on
	// other architectures.
	x86Level = cpuinfo.Max
	// x86VBMI enables the AVX-512 kernels that need AVX512-VBMI (VPERMI2B)
	// at x86Level AVX512. A variable for tests.
	x86VBMI = cpuinfo.X86.HasAVX512VBMI
)
