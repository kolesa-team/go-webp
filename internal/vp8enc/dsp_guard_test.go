//go:build darwin || linux

package vp8enc

import (
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// Guard-page tests of the batched DSP entry points: every buffer handed to
// a kernel is placed right before (or right after) an inaccessible page,
// so a load or store one byte outside the slice faults instead of reading
// the slack of a Go allocation. The results are compared with the Go
// reference on ordinary copies of the same data.

// guarded returns a slice of n bytes of fresh anonymous memory that ends
// exactly at an inaccessible page (atEnd), or starts skew bytes after one.
func guarded(tb testing.TB, n, skew int, atEnd bool) []byte {
	tb.Helper()
	ps := os.Getpagesize()
	pages := max((n+skew+ps-1)/ps, 1)
	mem, err := syscall.Mmap(-1, 0, (pages+2)*ps, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		tb.Fatalf("mmap: %v", err)
	}
	tb.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:ps], syscall.PROT_NONE); err != nil {
		tb.Fatalf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[(pages+1)*ps:], syscall.PROT_NONE); err != nil {
		tb.Fatalf("mprotect: %v", err)
	}
	data := mem[ps : (pages+1)*ps]
	if atEnd {
		return data[len(data)-n : len(data) : len(data)]
	}
	return data[skew : skew+n : skew+n]
}

func guardedBlocks(tb testing.TB, n int, atEnd bool) [][16]int16 {
	b := guarded(tb, 32*n, 0, atEnd)
	return unsafe.Slice((*[16]int16)(unsafe.Pointer(unsafe.SliceData(b))), n)
}

func guardedInt32(tb testing.TB, n int, atEnd bool) []int32 {
	b := guarded(tb, 4*n, 0, atEnd)
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(b))), n)
}

func guardedUint16(tb testing.TB, n int, atEnd bool) []uint16 {
	b := guarded(tb, 2*n, 0, atEnd)
	return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(b))), n)
}

// guardTables are the encoder's pair tables plus tables whose largest
// offsets sit at the far end of their buffers.
func guardTables() []*pairTable {
	t1 := newPairTable([2]uint16{0, 0})
	t2 := newPairTable([2]uint16{12*bps + 24, 3}, [2]uint16{5, 9*bps + 17})
	t3 := newPairTable([2]uint16{1, 2}, [2]uint16{3, 4}, [2]uint16{bps + 5, 2*bps + 6},
		[2]uint16{7, 8}, [2]uint16{9, 10}, [2]uint16{11, 12}, [2]uint16{13, 14}, [2]uint16{15, 16})
	return []*pairTable{&tabY16, &tabUV, &tabI4Src, &tabI4Rec, &tabI4Self, &t1, &t2, &t3}
}

func TestVecGuardPages(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(21))
		for _, atEnd := range []bool{true, false} {
			for ti, tab := range guardTables() {
				n0, n1, nb := tab.need[0], tab.need[1], 2*tab.n
				a := guarded(t, n0, 0, atEnd)
				b := guarded(t, n1, 0, atEnd)
				fillPixels(rng, a)
				fillPixels(rng, b)
				ra, rb := append([]byte(nil), a...), append([]byte(nil), b...)

				blocks := guardedBlocks(t, nb, atEnd)
				want := make([][16]int16, nb)
				fTransformPairs(a, b, tab, blocks)
				fTransformPairsRef(ra, rb, tab, want)
				for i := range want {
					if blocks[i] != want[i] {
						t.Fatalf("table %d atEnd=%v: fTransform block %d: %v, want %v", ti, atEnd, i, blocks[i], want[i])
					}
				}

				res := guardedInt32(t, nb, atEnd)
				wres := make([]int32, nb)
				tTransformPairs(a, tab, res)
				tTransformPairsRef(ra, tab, wres)
				for i := range wres {
					if res[i] != wres[i] {
						t.Fatalf("table %d atEnd=%v: tTransform %v, want %v", ti, atEnd, res, wres)
					}
				}
				sse4x4Pairs(a, b, tab, res)
				sse4x4PairsRef(ra, rb, tab, wres)
				for i := range wres {
					if res[i] != wres[i] {
						t.Fatalf("table %d atEnd=%v: sse4x4 %v, want %v", ti, atEnd, res, wres)
					}
				}

				// inverse transform: small coefficients (16-bit path) and
				// large ones (int32 path)
				for _, scale := range []int{2047, 32768} {
					in := guardedBlocks(t, nb, atEnd)
					for i := range in {
						for j := range in[i] {
							in[i][j] = int16(rng.Intn(2*scale) - scale)
						}
					}
					win := append([][16]int16(nil), in...)
					dst := guarded(t, n1, 0, atEnd)
					fillPixels(rng, dst)
					wdst := append([]byte(nil), dst...)
					iTransformPairs(a, in, dst, tab)
					iTransformPairsRef(ra, win, wdst, tab)
					if string(dst) != string(wdst) {
						t.Fatalf("table %d atEnd=%v scale %d: iTransform differs", ti, atEnd, scale)
					}
				}
			}

			// quantizer: in, out and info all against guard pages
			var m matrix
			m.q[0], m.q[1] = uint32(kDcTable[30]), uint32(kAcTable[30])
			expandMatrix(&m, 0)
			for _, n := range []int{1, 2, 8, 10, 16, 32} {
				in := guardedBlocks(t, n, atEnd)
				out := guardedBlocks(t, n, atEnd)
				info := guardedUint16(t, n, atEnd)
				for i := range in {
					fillCoeffs(rng, in[i][:])
				}
				win := append([][16]int16(nil), in...)
				wout := make([][16]int16, n)
				winfo := make([]uint16, n)
				nz := quantizeBlocks(in, out, &m, info)
				wnz := quantizeBlocksRef(win, wout, &m, winfo)
				if nz != wnz {
					t.Fatalf("quantize n=%d atEnd=%v: nz %#x, want %#x", n, atEnd, nz, wnz)
				}
				for i := 0; i < n; i++ {
					if in[i] != win[i] || out[i] != wout[i] || info[i] != winfo[i] {
						t.Fatalf("quantize n=%d atEnd=%v: block %d differs", n, atEnd, i)
					}
				}
				if n <= 16 {
					c0, mb := zeroBinStats(in)
					c0r, mbr := zeroBinStatsRef(win)
					if c0 != c0r || mb != mbr {
						t.Fatalf("zeroBinStats n=%d atEnd=%v: (%d, %d), want (%d, %d)", n, atEnd, c0, mb, c0r, mbr)
					}
				}
			}

			// residual cost: the block of levels against a guard page
			res, p := randomResiduals(rng, 64, []int{1, 3, 70, 2047})
			for i := range res {
				blk := guardedBlocks(t, 1, atEnd)
				blk[0] = *res[i].coeffs
				res[i].coeffs = &blk[0]
				for ctx := 0; ctx < numCtx; ctx++ {
					if got, want := p.residualCost(ctx, &res[i]), p.residualCostRef(ctx, &res[i]); got != want {
						t.Fatalf("residualCost atEnd=%v %v: %d, want %d", atEnd, blk[0], got, want)
					}
				}
			}
		}
	})
}

// TestVecGuardPagesSkewed runs the kernels that take plain pointers on
// buffers of exactly the size the wrappers require, ending at a guard page
// and starting 0..15 bytes after one (unaligned starts).
func TestVecGuardPagesSkewed(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(22))
		for skew := 0; skew < 16; skew++ {
			for _, atEnd := range []bool{true, false} {
				if atEnd && skew > 0 {
					continue
				}
				s := skew
				// sse16x16 / sse16x8
				for _, rows := range []int{16, 8} {
					n := (rows-1)*bps + 16
					a := guarded(t, n, s, atEnd)
					b := guarded(t, n, (s+5)&15, atEnd)
					fillPixels(rng, a)
					fillPixels(rng, b)
					var got, want int
					if rows == 16 {
						got, want = sse16x16(a, b), sse16x16Ref(a, b)
					} else {
						got, want = sse16x8(a, b), sse16x8Ref(a, b)
					}
					if got != want {
						t.Fatalf("sse16x%d skew %d atEnd=%v: %d, want %d", rows, s, atEnd, got, want)
					}
				}

				// intra16 and chroma predictors
				yLeft := guarded(t, 17, s, atEnd)
				yTop := guarded(t, 16, (s+3)&15, atEnd)
				uLeft := guarded(t, 9, (s+7)&15, atEnd)
				vLeft := guarded(t, 9, (s+11)&15, atEnd)
				uvTop := guarded(t, 16, (s+13)&15, atEnd)
				for _, e := range [][]byte{yLeft, yTop, uLeft, vLeft, uvTop} {
					fillPixels(rng, e)
				}
				p16 := guarded(t, i16HE16+15*bps+16, s, atEnd)
				w16 := make([]byte, len(p16))
				for i := range p16 {
					p16[i] = byte(i * 3)
					w16[i] = p16[i]
				}
				predLuma16(p16, yLeft, yTop)
				predLuma16Ref(w16, yLeft, yTop)
				if string(p16) != string(w16) {
					t.Fatalf("predLuma16 skew %d atEnd=%v differs", s, atEnd)
				}
				p8 := guarded(t, c8HE8+7*bps+16, s, atEnd)
				w8 := make([]byte, len(p8))
				for i := range p8 {
					p8[i] = byte(i * 5)
					w8[i] = p8[i]
				}
				predChroma8(p8, uLeft, vLeft, uvTop)
				predChroma8Ref(w8, uLeft, vLeft, uvTop)
				if string(p8) != string(w8) {
					t.Fatalf("predChroma8 skew %d atEnd=%v differs", s, atEnd)
				}

				// intra4: the boundary slice ends 11 bytes after the top sample
				// (the 16-byte load) or starts 5 bytes before it
				for _, top := range []int{5, 17, 29} {
					bnd := guarded(t, top+11, s, atEnd)
					fillPixels(rng, bnd)
					p4 := guarded(t, i4DC4+7*bps+8, s, atEnd)
					w4 := make([]byte, len(p4))
					for i := range p4 {
						p4[i] = byte(i * 7)
						w4[i] = p4[i]
					}
					predLuma4(p4, bnd, top)
					predLuma4Ref(w4, bnd, top)
					if string(p4) != string(w4) {
						t.Fatalf("predLuma4 top %d skew %d atEnd=%v differs", top, s, atEnd)
					}
				}
			}
		}
	})
}
