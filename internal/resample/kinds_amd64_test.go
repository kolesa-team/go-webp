//go:build amd64 && !purego

package resample

import (
	"math/rand"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// TestKernelCoverage checks that the geometries of TestImportKernels and
// FuzzImportKernels reach every assembly kernel.
func TestKernelCoverage(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
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
	forEachLevel(t, func(t *testing.T) {
		w8, w16 := uint8(kindShrink4), uint8(kindShrink4)
		if x86Level >= cpuinfo.AVX2 {
			w8, w16 = kindShrink4W8, kindShrink4W16
		}
		for _, c := range []struct {
			sw, dw, ch int
			kind       uint8
		}{
			{1024, 300, 4, w8},
			{2560, 320, 4, w8},
			{2561, 320, 4, w16},
			{3000, 320, 4, w16},
			{5120, 320, 4, w16},
			{5121, 320, 4, kindShrink4},
			{3000, 6, 4, kindShrink4},
			{3000, 5, 4, kindGo},   // spans of more than maxShrink4Span pixels
			{6, 3, 4, kindShrink4}, // too narrow for one 32-byte window
			{3, 1, 4, kindGo},      // narrower than one 16-byte load
			{675, 320, 1, kindShrink1P},
			{338, 320, 1, kindShrink1P},
			{1500, 320, 1, kindShrink1P},
			{2560, 320, 1, kindShrink1P},
			{2561, 320, 1, kindShrink1S},
			{3000, 320, 1, kindShrink1S},
			{5120, 320, 1, kindShrink1S},
			{5121, 320, 1, kindShrink1},
			{3000, 100, 1, kindShrink1},
			{5000, 2, 1, kindShrink1},
			{100, 7, 1, kindShrink1}, // too narrow for one record
			{15, 8, 1, kindGo},       // narrower than one 16-byte window
			{12, 3, 1, kindGo},
			{3000, 320, 2, kindGo},
			{300, 1000, 4, kindExpand4},
			{1, 1000, 4, kindGo}, // one pixel wide
			{300, 1000, 1, kindGo},
		} {
			rs, err := New(c.sw, 10, c.dw, 5, c.ch)
			if err != nil {
				t.Fatal(err)
			}
			want := c.kind
			if x86Level < cpuinfo.SSE41 {
				want = kindGo
			}
			if rs.arch.kind != want {
				t.Errorf("%d -> %d, %d channels: kernel %d, want %d", c.sw, c.dw, c.ch, rs.arch.kind, want)
			}
		}
	})
}
