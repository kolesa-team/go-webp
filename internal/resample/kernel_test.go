package resample

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"
)

// The row kernels the build selects (assembly on arm64) against the Go
// reference kernels, on identical rescaler state. Without assembly both
// sides are the Go kernels and the tests are trivially true.

// importRow runs the selected or the reference import kernel on src.
func importRow(rs *Rescaler, src []byte, acc []uint32, ref bool) {
	switch {
	case rs.xExpand && ref:
		rs.expandRowRef(src, acc)
	case rs.xExpand:
		rs.expandRow(src, acc)
	case ref:
		rs.shrinkRowRef(src, acc)
	default:
		rs.shrinkRow(src, acc)
	}
}

// checkImport compares both import kernels on one random row of a
// sw -> dw rescaler with ch channels, with and without accumulation. src,
// when not nil, is used as the source row (it must hold sw*ch bytes) and
// work as the work buffer.
func checkImport(t testing.TB, rnd *rand.Rand, sw, dw, ch int, src []byte, work []uint32) {
	t.Helper()
	var rs Rescaler
	if err := rs.Init(sw, 2, dw, 1, ch, work); err != nil {
		t.Fatal(err)
	}
	if src == nil {
		src = make([]byte, sw*ch)
	}
	copy(src, randomPixels(rnd, sw*ch))
	n := dw * ch
	acc0 := make([]uint32, n)
	for i := range acc0 {
		acc0[i] = rnd.Uint32()
	}
	cols := slices.Clone(rs.cols)
	for _, accumulate := range []bool{true, false} {
		var gotF, gotA, wantF, wantA []uint32
		for _, ref := range []bool{false, true} {
			var acc []uint32
			if accumulate {
				copy(rs.irow, acc0)
				acc = rs.irow
			}
			for i := range rs.frow {
				rs.frow[i] = 0xdeadbeef
			}
			importRow(&rs, src, acc, ref)
			if ref {
				wantF, wantA = slices.Clone(rs.frow), slices.Clone(rs.irow)
			} else {
				gotF, gotA = slices.Clone(rs.frow), slices.Clone(rs.irow)
			}
		}
		if !slices.Equal(gotF, wantF) || (accumulate && !slices.Equal(gotA, wantA)) {
			i := 0
			for i < n && gotF[i] == wantF[i] && (!accumulate || gotA[i] == wantA[i]) {
				i++
			}
			t.Fatalf("%d -> %d, %d channels, acc %v (kernel %d): differs at %d: frow %#x want %#x",
				sw, dw, ch, accumulate, rs.arch.kind, i, gotF[min(i, n-1)], wantF[min(i, n-1)])
		}
	}
	if !slices.Equal(cols, rs.cols) {
		t.Fatalf("%d -> %d, %d channels: column table overwritten", sw, dw, ch)
	}
}

// kernelWidths returns a source and destination width pair, biased
// towards the ratios where the kernels switch strategy.
func kernelWidths(rnd *rand.Rand, expand bool) (int, int) {
	sw := 1 + rnd.Intn(1+rnd.Intn(3000))
	if expand {
		return sw, sw + 1 + rnd.Intn(1+rnd.Intn(2000))
	}
	var k float64
	switch rnd.Intn(4) {
	case 0:
		k = 1 + 3*rnd.Float64()
	case 1:
		k = []float64{4, 8, 16, 17, 32, 511, 512, 2047, 2048}[rnd.Intn(9)] + rnd.Float64() - 0.5
	default:
		k = 1 + 40*rnd.Float64()
	}
	dw := max(1, min(sw, int(float64(sw)/max(1, k)+0.5)))
	return sw, dw
}

func TestImportKernels(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	n := 200000
	if testing.Short() {
		n = 20000
	}
	for i := 0; i < n; i++ {
		ch := []int{1, 4, 1, 4, 2, 3}[i%6]
		sw, dw := kernelWidths(rnd, i%5 == 0)
		checkImport(t, rnd, sw, dw, ch, nil, nil)
	}
}

func FuzzImportKernels(f *testing.F) {
	f.Add(uint16(1000), uint16(300), uint8(4), int64(1))
	f.Add(uint16(1000), uint16(300), uint8(1), int64(2))
	f.Add(uint16(30), uint16(700), uint8(4), int64(3))
	f.Add(uint16(4100), uint16(2), uint8(1), int64(4))
	f.Fuzz(func(t *testing.T, sw, dw uint16, ch uint8, seed int64) {
		w, ow, c := int(sw%5000)+1, int(dw%5000)+1, int(ch%4)+1
		checkImport(t, rand.New(rand.NewSource(seed)), w, ow, c, nil, nil)
	})
}

// exportCase runs ExportRow with the selected kernels and with the
// reference ones on a copy of rs, and compares the rows and the state.
func checkExport(t testing.TB, rs *Rescaler, dst []byte) {
	t.Helper()
	n := rs.dstWidth * rs.numChannels
	ref := cloneRescaler(rs)
	want := make([]byte, n)
	exportRowRef(ref, want)
	if !rs.ExportRow(dst) {
		t.Fatal("no pending row")
	}
	if !bytes.Equal(dst[:n], want) {
		i := 0
		for dst[i] == want[i] {
			i++
		}
		t.Fatalf("%dx%d -> %dx%d, %d channels, row %d: byte %d is %d, want %d",
			rs.srcWidth, rs.srcHeight, rs.dstWidth, rs.dstHeight, rs.numChannels, rs.dstY-1, i, dst[i], want[i])
	}
	if !slices.Equal(rs.irow, ref.irow) || rs.yAccum != ref.yAccum {
		t.Fatalf("%dx%d -> %dx%d, %d channels, row %d: state differs after export",
			rs.srcWidth, rs.srcHeight, rs.dstWidth, rs.dstHeight, rs.numChannels, rs.dstY-1)
	}
}

// cloneRescaler returns a deep copy of rs.
func cloneRescaler(rs *Rescaler) *Rescaler {
	c := *rs
	c.work = slices.Clone(rs.work)
	n := WorkSize(rs.dstWidth, rs.numChannels) / 2
	lo, hi := c.work[:n:n], c.work[n:2*n:2*n]
	if &rs.irow[0] == &rs.work[0] {
		c.irow, c.frow = lo, hi
	} else {
		c.irow, c.frow = hi, lo
	}
	m := 2*n + len(rs.cols)
	c.cols = c.work[2*n : m : m]
	return &c
}

// exportRowRef is ExportRow with the Go kernels.
func exportRowRef(r *Rescaler, dst []byte) {
	n := r.dstWidth * r.numChannels
	irow, frow := r.irow[:n], r.frow[:n]
	switch {
	case r.yExpand:
		if r.yAccum == 0 {
			exportExpandGo(dst, frow, r.fyScale)
		} else {
			b := frac(uint64(-r.yAccum), uint64(r.ySub))
			exportExpandMixGo(dst, frow, irow, uint32(one-uint64(b)), b, r.fyScale)
		}
	case r.fxyScale != 0:
		if yscale := r.fyScale * uint32(-r.yAccum); yscale != 0 {
			exportShrinkGo(dst, irow, frow, yscale, r.fxyScale)
		} else {
			exportShrinkZeroGo(dst, irow, r.fxyScale)
		}
	default:
		for i, v := range irow {
			dst[i] = byte(v)
			irow[i] = 0
		}
	}
	r.yAccum += r.yAdd
	r.dstY++
}

// checkRescale streams a random image through a rescaler, comparing every
// exported row with the reference kernels. dst, when not nil, receives the
// rows (it must hold dw*ch bytes).
func checkRescale(t testing.TB, rnd *rand.Rand, sw, sh, dw, dh, ch int, dst []byte) {
	t.Helper()
	rs, err := New(sw, sh, dw, dh, ch)
	if err != nil {
		t.Fatal(err)
	}
	if dst == nil {
		dst = make([]byte, dw*ch)
	}
	row := make([]byte, sw*ch)
	for !rs.InputDone() {
		copy(row, randomPixels(rnd, sw*ch))
		rs.ImportRow(row)
		for rs.HasPendingOutput() {
			checkExport(t, rs, dst)
		}
	}
	for rs.HasPendingOutput() {
		checkExport(t, rs, dst)
	}
}

func TestExportKernels(t *testing.T) {
	rnd := rand.New(rand.NewSource(12))
	n := 3000
	if testing.Short() {
		n = 300
	}
	for i := 0; i < n; i++ {
		sw, sh, dw, dh := randomGeometry(rnd)
		ch := []int{1, 4, 1, 4, 2, 3}[i%6]
		checkRescale(t, rnd, sw, sh, dw, dh, ch, nil)
	}
}

func FuzzExportKernels(f *testing.F) {
	f.Add(uint16(100), uint16(100), uint16(30), uint16(40), uint8(4), int64(1))
	f.Add(uint16(30), uint16(20), uint16(70), uint16(90), uint8(1), int64(2))
	f.Fuzz(func(t *testing.T, sw, sh, dw, dh uint16, ch uint8, seed int64) {
		w, h, ow, oh := int(sw%400)+1, int(sh%400)+1, int(dw%400)+1, int(dh%400)+1
		checkRescale(t, rand.New(rand.NewSource(seed)), w, h, ow, oh, int(ch%4)+1, nil)
	})
}
