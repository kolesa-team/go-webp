package vp8

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"
)

// Differential tests of filterMB (NEON on arm64) against the portable
// reference filterMBGo on random macroblock caches.

// filterCache is a decoder whose row cache holds mbW macroblocks, laid out
// like initFrame does it.
func newFilterCache(filterType, mbW int) *decoder {
	d := &decoder{filterType: filterType, mbW: mbW}
	d.extraRows = kFilterExtraRows[filterType]
	d.cacheYStride = 16 * mbW
	d.cacheUVStride = 8 * mbW
	d.cacheYOff = d.extraRows * d.cacheYStride
	d.cacheUVOff = (d.extraRows / 2) * d.cacheUVStride
	d.cacheY = make([]byte, (d.extraRows+16)*d.cacheYStride)
	d.cacheU = make([]byte, (d.extraRows/2+8)*d.cacheUVStride)
	d.cacheV = make([]byte, (d.extraRows/2+8)*d.cacheUVStride)
	return d
}

// fillFilterPlane fills a plane with data that exercises the filter
// decisions: mode 0 uniform random, 1 flat with noise of amplitude amp,
// 2 a step between two levels plus noise, 3 extreme values, 4 a gradient
// plus noise.
func fillFilterPlane(r *rand.Rand, b []byte, stride, mode, amp int) {
	base := r.IntN(256)
	step := r.IntN(2*amp+1) - amp
	noise := func() int { return r.IntN(amp+1) - amp/2 }
	for i := range b {
		x, y := i%stride, i/stride
		var v int
		switch mode {
		case 0:
			v = r.IntN(256)
		case 1:
			v = base + noise()
		case 2:
			v = base + noise()
			if (x/4)&1 != 0 || (y/4)&1 != 0 {
				v += step
			}
		case 3:
			v = [6]int{0, 1, 127, 128, 254, 255}[r.IntN(6)]
		default:
			v = base + (x+y)*step/8 + noise()
		}
		b[i] = byte(min(max(v, 0), 255))
	}
}

func cloneCache(d *decoder) *decoder {
	c := *d
	c.cacheY = bytes.Clone(d.cacheY)
	c.cacheU = bytes.Clone(d.cacheU)
	c.cacheV = bytes.Clone(d.cacheV)
	return &c
}

// checkFilterMB filters macroblock (mbX, mbY) of a copy of d with both
// implementations and compares the caches.
func checkFilterMB(t testing.TB, d *decoder, mbX, mbY int, f fInfo) {
	t.Helper()
	ref, got := cloneCache(d), cloneCache(d)
	ref.filterMBGo(mbX, mbY, f)
	got.filterMB(mbX, mbY, f)
	for _, p := range []struct {
		name     string
		a, b     []byte
		stride   int
		baseRows int
	}{
		{"Y", got.cacheY, ref.cacheY, d.cacheYStride, d.extraRows},
		{"U", got.cacheU, ref.cacheU, d.cacheUVStride, d.extraRows / 2},
		{"V", got.cacheV, ref.cacheV, d.cacheUVStride, d.extraRows / 2},
	} {
		if i := firstByteDiff(p.a, p.b); i >= 0 {
			t.Fatalf("type=%d mb=(%d,%d) %+v: plane %s differs at row %d col %d: got %d, want %d",
				d.filterType, mbX, mbY, f, p.name, i/p.stride-p.baseRows, i%p.stride, p.a[i], p.b[i])
		}
	}
}

// TestFilterMBMatchesGo sweeps the thresholds (limit 1..250, every
// interior limit 0..71 plus a few large ones, hev thresholds 0..3 and
// random) with random edge selections and pixel data.
func TestFilterMBMatchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	step := 1
	if testing.Short() {
		step = 7
	}
	ilevels := []int{}
	for i := 0; i <= 71; i++ {
		ilevels = append(ilevels, i)
	}
	ilevels = append(ilevels, 100, 160, 254, 255)
	for _, ft := range []int{1, 2} {
		d := newFilterCache(ft, 3)
		n := 0
		for limit := 1; limit <= 250; limit += step {
			for _, il := range ilevels {
				if ft == 1 && il > 0 {
					break // the simple filter ignores ilevel and hev
				}
				hev := []int{0, 1, 2, 3, r.IntN(256)}[r.IntN(5)]
				f := fInfo{limit: uint8(limit), ilevel: uint8(il), hevThresh: uint8(hev), inner: uint8(r.IntN(2))}
				mode := r.IntN(5)
				amp := 1 + r.IntN(2*limit+2*il+8)
				fillFilterPlane(r, d.cacheY, d.cacheYStride, mode, amp)
				fillFilterPlane(r, d.cacheU, d.cacheUVStride, mode, amp)
				fillFilterPlane(r, d.cacheV, d.cacheUVStride, mode, amp)
				checkFilterMB(t, d, r.IntN(3), r.IntN(2), f)
				n++
			}
			if ft == 1 {
				// More simple filter cases per limit, as there is no ilevel loop.
				for k := 0; k < 40; k++ {
					f := fInfo{limit: uint8(limit), inner: uint8(r.IntN(2))}
					mode := r.IntN(5)
					amp := 1 + r.IntN(2*limit+8)
					fillFilterPlane(r, d.cacheY, d.cacheYStride, mode, amp)
					checkFilterMB(t, d, r.IntN(3), r.IntN(2), f)
					n++
				}
			}
		}
		t.Logf("filter type %d: %d macroblocks", ft, n)
	}
}

// TestFilterMBRandom filters many random macroblocks with realistic
// strengths (precomputeFilterStrengths' ranges) in sequence on the same
// cache, as the decoder does.
func TestFilterMBRandom(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	n := 200000
	if testing.Short() {
		n = 20000
	}
	for _, ft := range []int{1, 2} {
		ref, got := newFilterCache(ft, 4), (*decoder)(nil)
		for i := 0; i < n/2; i++ {
			if i%64 == 0 {
				mode, amp := r.IntN(5), 1+r.IntN(80)
				fillFilterPlane(r, ref.cacheY, ref.cacheYStride, mode, amp)
				fillFilterPlane(r, ref.cacheU, ref.cacheUVStride, mode, amp)
				fillFilterPlane(r, ref.cacheV, ref.cacheUVStride, mode, amp)
				got = cloneCache(ref)
			}
			level := 1 + r.IntN(63)
			il := max(1, min(level>>r.IntN(3), 9-r.IntN(8)))
			if r.IntN(2) == 0 {
				il = level
			}
			hev := 0
			if level >= 40 {
				hev = 2
			} else if level >= 15 {
				hev = 1
			}
			f := fInfo{limit: uint8(2*level + il), ilevel: uint8(il), hevThresh: uint8(hev), inner: uint8(r.IntN(2))}
			mbX, mbY := r.IntN(4), r.IntN(2)
			ref.filterMBGo(mbX, mbY, f)
			got.filterMB(mbX, mbY, f)
			if !bytes.Equal(ref.cacheY, got.cacheY) || !bytes.Equal(ref.cacheU, got.cacheU) || !bytes.Equal(ref.cacheV, got.cacheV) {
				t.Fatalf("type=%d step %d mb=(%d,%d) %+v: caches differ", ft, i, mbX, mbY, f)
			}
		}
	}
}

// walkFilterPlane fills a plane with a 2D random walk whose steps are
// drawn from steps (with random signs) and an occasional jump to an
// extreme value, so that many lines sit right at a filter decision
// threshold or clip at 0 or 255.
func walkFilterPlane(r *rand.Rand, b []byte, stride int, steps []int) {
	for i := range b {
		prev := r.IntN(256)
		switch x, y := i%stride, i/stride; {
		case x > 0 && (y == 0 || r.IntN(2) == 0):
			prev = int(b[i-1])
		case y > 0:
			prev = int(b[i-stride])
		}
		s := steps[r.IntN(len(steps))]
		if r.IntN(2) == 0 {
			s = -s
		}
		v := prev + s
		if r.IntN(64) == 0 {
			v = []int{0, 127, 128, 255}[r.IntN(4)]
		}
		b[i] = byte(min(max(v, 0), 255))
	}
}

// TestFilterMBThresholdWalk compares filterMB with filterMBGo on caches
// with arbitrary strides and block offsets (any alignment), every limit
// 1..255 (251..255 take the Go fallback) and interior and hev thresholds
// at 0, small, VP8's maximum and 255, on pixels whose neighbour
// differences concentrate at those thresholds.
func TestFilterMBThresholdWalk(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 20))
	n := 100000
	if testing.Short() {
		n = 20000
	}
	for i := 0; i < n; i++ {
		ft, mbW := 1+r.IntN(2), 1+r.IntN(3)
		d := newFilterCache(ft, mbW)
		if i&1 != 0 {
			d.cacheYStride += r.IntN(17)
			d.cacheUVStride += r.IntN(17)
			d.cacheYOff = d.extraRows*d.cacheYStride + r.IntN(16)
			d.cacheUVOff = d.extraRows/2*d.cacheUVStride + r.IntN(16)
			d.cacheY = make([]byte, d.cacheYOff+16*d.cacheYStride+r.IntN(4))
			d.cacheU = make([]byte, d.cacheUVOff+8*d.cacheUVStride+r.IntN(4))
			d.cacheV = make([]byte, d.cacheUVOff+8*d.cacheUVStride+r.IntN(4))
		}
		limit := 1 + r.IntN(255)
		if r.IntN(2) == 0 {
			limit = min(2*(1+r.IntN(63))+r.IntN(64), 255)
		}
		il := []int{0, 1, 2, 63, 254, 255, r.IntN(64), r.IntN(256)}[r.IntN(8)]
		hev := []int{0, 1, 2, 255, r.IntN(256)}[r.IntN(5)]
		steps := []int{0, 0, 1, hev, hev + 1, il, il + 1, max(il-1, 0), limit / 2, limit/2 + 1, limit / 4, (limit + 4) / 2, (limit+4)/2 + 1, 127, 128, r.IntN(256)}
		walkFilterPlane(r, d.cacheY, d.cacheYStride, steps)
		walkFilterPlane(r, d.cacheU, d.cacheUVStride, steps)
		walkFilterPlane(r, d.cacheV, d.cacheUVStride, steps)
		f := fInfo{limit: uint8(limit), ilevel: uint8(il), hevThresh: uint8(hev), inner: uint8(r.IntN(2))}
		checkFilterMB(t, d, r.IntN(mbW), r.IntN(2), f)
	}
}

// TestFilterKernelGuardPages runs the NEON kernels (through filterMB) on
// planes cut to exactly the bytes a macroblock's edges may touch, placed
// against an inaccessible page at either end.
func TestFilterKernelGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	for _, ft := range []int{1, 2} {
		k := 4
		if ft == 1 {
			k = 2
		}
		for flags := 1; flags < 8; flags++ {
			for _, atEnd := range []bool{true, false} {
				// One macroblock column at mbX=1 of a 2-wide cache, with
				// exactly k rows above it.
				mbX, mbY := 0, 0
				if flags&filterEdgeLeft != 0 {
					mbX = 1
				}
				if flags&filterEdgeTop != 0 {
					mbY = 1
				}
				f := fInfo{limit: uint8(20 + r.IntN(40)), ilevel: uint8(1 + r.IntN(20)), hevThresh: uint8(r.IntN(3)), inner: uint8(flags >> 2)}
				src := newFilterCache(ft, 2)
				src.cacheYOff = k * src.cacheYStride
				src.cacheUVOff = k * src.cacheUVStride
				// Planes trimmed to [lo, hi) of the block at mbX.
				yo := src.cacheYOff + mbX*16
				uvo := src.cacheUVOff + mbX*8
				yLo, yHi := planeRange(yo, src.cacheYStride, 16, k, flags)
				uvLo, uvHi := planeRange(uvo, src.cacheUVStride, 8, k, flags)
				src.cacheY = make([]byte, yHi-yLo)
				src.cacheU = make([]byte, uvHi-uvLo)
				src.cacheV = make([]byte, uvHi-uvLo)
				src.cacheYOff -= yLo
				src.cacheUVOff -= uvLo
				fillFilterPlane(r, src.cacheY, src.cacheYStride, 1, 30)
				fillFilterPlane(r, src.cacheU, src.cacheUVStride, 1, 30)
				fillFilterPlane(r, src.cacheV, src.cacheUVStride, 1, 30)
				ref := cloneCache(src)
				ref.filterMBGo(mbX, mbY, f)

				got := *src
				got.cacheY = guardedBytes(t, len(src.cacheY), atEnd)
				copy(got.cacheY, src.cacheY)
				got.cacheU = guardedBytes(t, len(src.cacheU), atEnd)
				copy(got.cacheU, src.cacheU)
				got.cacheV = guardedBytes(t, len(src.cacheV), atEnd)
				copy(got.cacheV, src.cacheV)
				got.filterMB(mbX, mbY, f)
				if !bytes.Equal(got.cacheY, ref.cacheY) || !bytes.Equal(got.cacheU, ref.cacheU) || !bytes.Equal(got.cacheV, ref.cacheV) {
					t.Fatalf("type=%d flags=%d atEnd=%v: result differs", ft, flags, atEnd)
				}
			}
		}
	}
}

// planeRange is the byte range [lo, hi) that blockInBounds requires.
func planeRange(o, stride, n, k, flags int) (lo, hi int) {
	lo = o
	if flags&filterEdgeTop != 0 {
		lo -= k * stride
	}
	if flags&filterEdgeLeft != 0 {
		lo = min(lo, o-k)
	}
	return lo, o + (n-1)*stride + n
}

// FuzzFilterMB compares filterMB with filterMBGo on fuzzed caches and
// strengths.
func FuzzFilterMB(f *testing.F) {
	f.Add([]byte{128, 130, 125, 140, 90}, uint8(30), uint8(10), uint8(1), uint8(7))
	f.Add([]byte{0, 255}, uint8(189), uint8(63), uint8(2), uint8(15))
	f.Add([]byte{7}, uint8(1), uint8(0), uint8(0), uint8(0))
	f.Fuzz(func(t *testing.T, seed []byte, limit, ilevel, hev, sel uint8) {
		if len(seed) == 0 || limit == 0 {
			return
		}
		d := newFilterCache(1+int(sel>>4)&1, 2)
		k := 0
		for _, p := range [][]byte{d.cacheY, d.cacheU, d.cacheV} {
			for i := range p {
				p[i] = seed[k%len(seed)] + byte(k/len(seed))*(sel|1)
				k++
			}
		}
		fi := fInfo{limit: limit, ilevel: ilevel, hevThresh: hev, inner: sel & 1}
		checkFilterMB(t, d, int(sel>>1)&1, int(sel>>2)&1, fi)
	})
}

// Kernel benchmarks: one macroblock with all edges, the dispatched filter
// (NEON on arm64) and the Go reference.

func benchFilterMB(b *testing.B, ft int, fn func(d *decoder, mbX, mbY int, f fInfo)) {
	d := newFilterCache(ft, 4)
	r := rand.New(rand.NewPCG(17, 18))
	fillFilterPlane(r, d.cacheY, d.cacheYStride, 2, 12)
	fillFilterPlane(r, d.cacheU, d.cacheUVStride, 2, 12)
	fillFilterPlane(r, d.cacheV, d.cacheUVStride, 2, 12)
	f := fInfo{limit: 2*30 + 10, ilevel: 10, hevThresh: 1, inner: 1}
	for b.Loop() {
		for mbX := 0; mbX < 4; mbX++ {
			fn(d, mbX, 1, f)
		}
	}
}

func BenchmarkFilterMB(b *testing.B) {
	for _, ft := range []int{1, 2} {
		b.Run(fmt.Sprintf("type%d/kernel", ft), func(b *testing.B) {
			benchFilterMB(b, ft, (*decoder).filterMB)
		})
		b.Run(fmt.Sprintf("type%d/go", ft), func(b *testing.B) {
			benchFilterMB(b, ft, (*decoder).filterMBGo)
		})
	}
}
