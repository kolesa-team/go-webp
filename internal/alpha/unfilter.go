package alpha

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// The horizontal and vertical unfilters have a pure-Go reference
// (filters.go) and, on arm64 and amd64, assembly versions selected by the
// wrappers of unfilter_<arch>.go (unfilter_generic.go elsewhere and with
// the purego tag).

// x86Level is the SIMD level of the amd64 kernels (unfilter_amd64.go): below
// a kernel's level its next lower version runs, down to the Go reference.
// Tests set it to every level of cpuinfo.Levels. Unused on other
// architectures.
var x86Level = cpuinfo.Max
