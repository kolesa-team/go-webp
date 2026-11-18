//go:build amd64 && !purego

package vp8

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// Tests of the amd64 reconstruction kernels called directly: besides
// matching the reference (which the differential tests check through the
// wrappers), each kernel must handle every macroblock whose coefficients
// are within the bound of its 16-bit transform itself, and decline (return
// false, leaving the macroblock to the reference) exactly when a
// coefficient it would transform is outside it.

type reconKernelSet struct {
	level cpuinfo.Level
	i4    func(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, modes *[16]uint8) bool
	i16   func(b *[yuvSize]byte, coeffs *[384]int16, bits uint32, mode int) bool
	uv    func(b *[yuvSize]byte, coeffs *[384]int16, codes uint32, mode int) bool
}

func reconKernelSets() []reconKernelSet {
	all := []reconKernelSet{
		{cpuinfo.SSE41, reconI4SSE41, reconI16SSE41, reconUVSSE41},
		{cpuinfo.AVX2, reconI4AVX2, reconI16AVX2, reconUVAVX2},
	}
	var sets []reconKernelSet
	for _, s := range all {
		if s.level <= cpuinfo.Max {
			sets = append(sets, s)
		}
	}
	return sets
}

// codeUses reports whether the transform of code uses coefficient k.
func codeUses(code uint32, k int) bool {
	switch code {
	case 1:
		return k == 0
	case 2:
		return k == 0 || k == 1 || k == 4
	case 3:
		return true
	}
	return false
}

// rowInRange reports whether the coefficients of blocks (with their codes)
// that a full transform would use are all within [-4095, 4095].
func rowInRange(coeffs *[384]int16, blocks []int, codes []uint32) bool {
	for i, n := range blocks {
		for k := 0; k < 16; k++ {
			if c := coeffs[n*16+k]; codeUses(codes[i], k) && (c > 4095 || c < -4095) {
				return false
			}
		}
	}
	return true
}

// lumaKernelOK is the result the 4x4 and 16x16 kernels must return: false
// iff a block row with an AC code (taking the full transform) uses an
// out-of-range coefficient.
func lumaKernelOK(coeffs *[384]int16, bits uint32) bool {
	for r := 0; r < 4; r++ {
		row := bits >> (24 - 8*r) & 0xff
		if row&0xaa == 0 {
			continue
		}
		var blocks []int
		var codes []uint32
		for j := 0; j < 4; j++ {
			blocks = append(blocks, 4*r+j)
			codes = append(codes, row>>(6-2*j)&3)
		}
		if !rowInRange(coeffs, blocks, codes) {
			return false
		}
	}
	return true
}

func chromaKernelOK(coeffs *[384]int16, codes uint32) bool {
	if codes&0xaaaa == 0 {
		return true
	}
	cu, cv := codes&3, codes>>8&3
	return rowInRange(coeffs, []int{16, 17, 20, 21}, []uint32{cu, cu, cv, cv}) &&
		rowInRange(coeffs, []int{18, 19, 22, 23}, []uint32{cu, cu, cv, cv})
}

// boundCoeff returns a coefficient that is mostly within the 16-bit bound,
// with the bound itself and values just past it now and then.
func boundCoeff(r *rand.Rand) int16 {
	switch r.IntN(16) {
	case 0:
		return []int16{4095, -4095, 4094, -4094}[r.IntN(4)]
	case 1:
		return []int16{4096, -4096, 4097, -32768, 32767}[r.IntN(5)]
	case 2, 3, 4:
		return int16(r.IntN(8191) - 4095)
	}
	return randCoeff(r, 1+r.IntN(4)) // kinds 1-4
}

func TestReconKernelsDirect(t *testing.T) {
	sets := reconKernelSets()
	if len(sets) == 0 {
		t.Skip("no SIMD kernels on this machine")
	}
	for _, s := range sets {
		t.Run(s.level.String(), func(t *testing.T) {
			r := rand.New(rand.NewPCG(31, 32))
			var c reconCase
			n := reconIters(t)
			taken := [3]int{}
			for i := 0; i < n; i++ {
				randReconCase(r, &c)
				if r.IntN(2) == 0 {
					for k := range c.coeffs {
						if c.coeffs[k] != 0 || r.IntN(8) == 0 {
							c.coeffs[k] = boundCoeff(r)
						}
					}
				}
				coeffs := c.coeffs

				got, want := c.b, c.b
				ok := s.i4(&got, &coeffs, c.bits, &c.modes)
				if exp := lumaKernelOK(&c.coeffs, c.bits); ok != exp {
					t.Fatalf("i4 iteration %d: returned %v, want %v (bits %#x)", i, ok, exp, c.bits)
				}
				if ok {
					taken[0]++
					reconI4Ref(&want, &c.coeffs, c.bits, &c.modes)
					if !diffBuffers(t, fmt.Sprintf("i4 iteration %d modes=%v bits=%#x", i, c.modes, c.bits), &got, &want) {
						t.FailNow()
					}
				}

				got, want = c.b, c.b
				ok = s.i16(&got, &coeffs, c.bits, c.mode)
				if exp := c.bits == 0 || lumaKernelOK(&c.coeffs, c.bits); ok != exp {
					t.Fatalf("i16 iteration %d: returned %v, want %v (bits %#x)", i, ok, exp, c.bits)
				}
				if ok {
					taken[1]++
					reconI16Ref(&want, &c.coeffs, c.bits, c.mode)
					if !diffBuffers(t, fmt.Sprintf("i16 iteration %d mode=%d bits=%#x", i, c.mode, c.bits), &got, &want) {
						t.FailNow()
					}
				}

				got, want = c.b, c.b
				codes := uvPlaneCodes(c.uvBits) | uvPlaneCodes(c.uvBits>>8)<<8
				ok = s.uv(&got, &coeffs, codes, c.uvMode)
				if exp := chromaKernelOK(&c.coeffs, codes); ok != exp {
					t.Fatalf("uv iteration %d: returned %v, want %v (codes %#x)", i, ok, exp, codes)
				}
				if ok {
					taken[2]++
					reconUVRef(&want, &c.coeffs, c.uvBits, c.uvMode)
					if !diffBuffers(t, fmt.Sprintf("uv iteration %d mode=%d bits=%#x", i, c.uvMode, c.uvBits), &got, &want) {
						t.FailNow()
					}
				}
				if coeffs != c.coeffs {
					t.Fatalf("iteration %d: the kernels modified the coefficients", i)
				}
			}
			t.Logf("kernel results used: i4 %d, i16 %d, uv %d of %d", taken[0], taken[1], taken[2], n)
		})
	}
}
