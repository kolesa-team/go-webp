//go:build arm64 && !purego

package resample

import (
	"math/rand"
	"testing"
)

// TestKernelCoverage checks that the geometries of TestImportKernels and
// FuzzImportKernels reach every assembly kernel.
func TestKernelCoverage(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	hits := map[uint8]int{}
	for i := 0; i < 20000; i++ {
		ch := []int{1, 4}[i%2]
		sw, dw := kernelWidths(rnd, i%5 == 0)
		rs, err := New(sw, 2, dw, 1, ch)
		if err != nil {
			t.Fatal(err)
		}
		hits[rs.arch.kind]++
	}
	for k := uint8(kindGo); k <= kindExpand4; k++ {
		if hits[k] < 100 {
			t.Errorf("kernel %d: %d hits", k, hits[k])
		}
	}
}

// TestKernelSelection checks that the assembly kernels are the ones the
// usual geometries get, so that the differential tests exercise them.
func TestKernelSelection(t *testing.T) {
	for _, c := range []struct {
		sw, dw, ch int
		kind       uint8
	}{
		{1024, 300, 4, kindShrink4},
		{3000, 320, 4, kindShrink4},
		{3000, 6, 4, kindShrink4},
		{3000, 5, 4, kindGo}, // spans of more than maxShrink4Span pixels
		{3, 1, 4, kindGo},    // narrower than one 16-byte load
		{512, 320, 1, kindShrink1T16},
		{1024, 320, 1, kindShrink1T16},
		{1500, 320, 1, kindShrink1T32},
		{3000, 320, 1, kindShrink1T64},
		{3000, 100, 1, kindShrink1},
		{5000, 2, 1, kindGo}, // spans of more than maxShrink1Span pixels
		{12, 3, 1, kindGo},   // narrower than one 16-byte window
		{3000, 320, 2, kindGo},
		{300, 1000, 4, kindExpand4},
		{1, 1000, 4, kindGo}, // one pixel wide
		{300, 1000, 1, kindGo},
	} {
		rs, err := New(c.sw, 10, c.dw, 5, c.ch)
		if err != nil {
			t.Fatal(err)
		}
		if rs.arch.kind != c.kind {
			t.Errorf("%d -> %d, %d channels: kernel %d, want %d", c.sw, c.dw, c.ch, rs.arch.kind, c.kind)
		}
	}
}
