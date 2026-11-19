//go:build amd64 && !purego && (darwin || linux)

package resample

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// The import kernels called directly with arbitrary column tables (any
// spans up to each kernel's limit, any weights, any scales), not only the
// tables a geometry produces, against the Go kernels. The source row is as
// short as each kernel's contract allows and ends right before an
// inaccessible page (or starts right after one); frow, acc and the records
// end right before one.

// contractWord returns a random word, often an extreme one.
func contractWord(rnd *rand.Rand) uint32 {
	switch rnd.Intn(6) {
	case 0:
		return 0xffffffff - uint32(rnd.Intn(3))
	case 1:
		return uint32(rnd.Intn(3))
	case 2:
		return 0x7fffffff + uint32(rnd.Intn(3))
	default:
		return rnd.Uint32()
	}
}

// contractSpan returns a span length in [1, limit], often 1 or limit.
func contractSpan(rnd *rand.Rand, limit int) int {
	switch rnd.Intn(4) {
	case 0:
		return 1
	case 1:
		return limit
	default:
		return 1 + rnd.Intn(limit)
	}
}

// contractRows holds guarded frow and acc rows of n cells (acc0 the
// accumulator's start values) and their Go counterparts.
type contractRows struct {
	frow, acc, wantF, wantA, acc0 []uint32
	free                          func()
}

func newContractRows(t *testing.T, rnd *rand.Rand, n int) *contractRows {
	frow, f1 := guardedWords(t, n, false)
	acc, f2 := guardedWords(t, n, false)
	r := &contractRows{frow: frow, acc: acc, wantF: make([]uint32, n), wantA: make([]uint32, n),
		acc0: make([]uint32, n), free: func() { f1(); f2() }}
	for i := range r.acc0 {
		r.acc0[i] = rnd.Uint32()
	}
	return r
}

// run runs kernel and ref (each given frow and acc, acc nil or not) and
// compares the results.
func (r *contractRows) run(t *testing.T, what string, kernel, ref func(frow, acc []uint32)) {
	t.Helper()
	for _, accumulate := range []bool{false, true} {
		var acc, wantA []uint32
		if accumulate {
			copy(r.acc, r.acc0)
			copy(r.wantA, r.acc0)
			acc, wantA = r.acc, r.wantA
		}
		for i := range r.frow {
			r.frow[i], r.wantF[i] = 0xdeadbeef, 0xdeadbeef
		}
		kernel(r.frow, acc)
		ref(r.wantF, wantA)
		if !slices.Equal(r.frow, r.wantF) || accumulate && !slices.Equal(r.acc, r.wantA) {
			i := 0
			for r.frow[i] == r.wantF[i] && (!accumulate || r.acc[i] == r.wantA[i]) {
				i++
			}
			t.Fatalf("%s, acc %v: cell %d of %d: frow %#x want %#x", what, accumulate, i, len(r.frow), r.frow[i], r.wantF[i])
		}
	}
}

// contractShrinkCols returns n column table entries with spans of at most
// limit pixels and arbitrary na, and the start of the last span and the
// sum of all spans.
func contractShrinkCols(rnd *rand.Rand, n, limit int) (cols []uint32, last, sum int) {
	cols = make([]uint32, 2*n)
	for x := 0; x < n; x++ {
		k := contractSpan(rnd, limit)
		cols[2*x], cols[2*x+1] = uint32(k), contractWord(rnd)
		last = sum
		sum += k
	}
	return cols, last, sum
}

func contractIters() int {
	if testing.Short() || raceEnabled {
		return 1500
	}
	return 5000
}

func TestShrinkKernelsContract(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
	type kernel func(frow, acc *uint32, src *byte, cols *uint32, n int, xSub, fx uint32)
	for _, k := range []struct {
		name            string
		ch, limit, room int // room: pixels read from every span start
		pairs           bool
		level           cpuinfo.Level
		f               kernel
	}{
		{"shrinkRow4SSE41", 4, maxShrink4Span, 4, false, cpuinfo.SSE41, shrinkRow4SSE41},
		{"shrink4W8AVX2", 4, maxShrink4W8Span, 8, true, cpuinfo.AVX2, shrink4W8AVX2},
		{"shrink4W16AVX2", 4, maxShrink4W16Span, 16, true, cpuinfo.AVX2, shrink4W16AVX2},
		{"shrinkRow1SSE41", 1, 700, 16, false, cpuinfo.SSE41, shrinkRow1SSE41},
		{"shrinkRow1SSE41/short", 1, 40, 16, false, cpuinfo.SSE41, shrinkRow1SSE41},
	} {
		t.Run(k.name, func(t *testing.T) {
			if cpuinfo.Max < k.level {
				t.Skip("level not supported")
			}
			rnd := rand.New(rand.NewSource(31))
			for it := 0; it < contractIters(); it++ {
				n := 1 + rnd.Intn(40)
				if k.pairs {
					n = 2 * (1 + rnd.Intn(20))
				}
				c, last, sum := contractShrinkCols(rnd, n, k.limit)
				cols, freeCols := guardedWords(t, len(c), false)
				copy(cols, c)
				sw := max(sum, last+k.room)
				src, freeSrc := guardedBytes(t, sw*k.ch, it%3 == 0)
				copy(src, randomPixels(rnd, sw*k.ch))
				xSub, fx := contractWord(rnd), contractWord(rnd)
				rows := newContractRows(t, rnd, k.ch*n)
				cnt := n
				if k.pairs {
					cnt = n / 2
				}
				rows.run(t, k.name, func(frow, acc []uint32) {
					var ap *uint32
					if acc != nil {
						ap = &acc[0]
					}
					k.f(&frow[0], ap, &src[0], &cols[0], cnt, xSub, fx)
				}, func(frow, acc []uint32) {
					if k.ch == 4 {
						shrinkRow4Go(frow, acc, src, cols, xSub, fx, [4]uint32{})
					} else {
						shrinkRow1Go(frow, acc, src, cols, xSub, fx, 0)
					}
				})
				rows.free()
				freeCols()
				freeSrc()
			}
		})
	}
}

func TestShrink1RecordKernelsContract(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
	type kernel func(frow, acc *uint32, src *byte, rec *byte, blocks int, xSub, fx uint32)
	for _, k := range []struct {
		name  string
		limit int
		size  int
		build func(rec []byte, cols []uint32, srcWidth int)
		level cpuinfo.Level
		f     kernel
	}{
		{"shrink1PSSE41", maxShrink1PSpan, shrink1PRecordSize, buildShrink1PRecords, cpuinfo.SSE41, shrink1PSSE41},
		{"shrink1PAVX2", maxShrink1PSpan, shrink1PRecordSize, buildShrink1PRecords, cpuinfo.AVX2, shrink1PAVX2},
		{"shrink1SSSE41", maxShrink1SSpan, shrink1SRecordSize, buildShrink1SRecords, cpuinfo.SSE41, shrink1SSSE41},
		{"shrink1SAVX2", maxShrink1SSpan, shrink1SRecordSize, buildShrink1SRecords, cpuinfo.AVX2, shrink1SAVX2},
	} {
		t.Run(k.name, func(t *testing.T) {
			if cpuinfo.Max < k.level {
				t.Skip("level not supported")
			}
			rnd := rand.New(rand.NewSource(32))
			for it := 0; it < contractIters(); it++ {
				blocks := 1 + rnd.Intn(6)
				n := 8 * blocks
				cols, _, sum := contractShrinkCols(rnd, n, k.limit)
				sw := max(sum, 16)
				src, freeSrc := guardedBytes(t, sw, it%3 == 0)
				copy(src, randomPixels(rnd, sw))
				rec, freeRec := guardedBytes(t, blocks*k.size, false)
				for i := range rec {
					rec[i] = byte(rnd.Intn(256))
				}
				k.build(rec, cols, sw)
				xSub, fx := contractWord(rnd), contractWord(rnd)
				rows := newContractRows(t, rnd, n)
				rows.run(t, k.name, func(frow, acc []uint32) {
					var ap *uint32
					if acc != nil {
						ap = &acc[0]
					}
					k.f(&frow[0], ap, &src[0], &rec[0], blocks, xSub, fx)
				}, func(frow, acc []uint32) {
					shrinkRow1Go(frow, acc, src, cols, xSub, fx, 0)
				})
				rows.free()
				freeRec()
				freeSrc()
			}
		})
	}
}

func TestExpandKernelsContract(t *testing.T) {
	if cpuinfo.Max < cpuinfo.SSE41 {
		t.Skip("no SSE4.1")
	}
	t.Run("expandRow4SSE41", func(t *testing.T) {
		rnd := rand.New(rand.NewSource(33))
		for it := 0; it < contractIters(); it++ {
			n := 1 + rnd.Intn(40)
			sw := 2 + rnd.Intn(1+rnd.Intn(64))
			cols, freeCols := guardedWords(t, 2*n, false)
			for x := 0; x < n; x++ {
				l := rnd.Intn(sw - 1)
				if rnd.Intn(3) == 0 {
					l = sw - 2
				}
				cols[2*x], cols[2*x+1] = uint32(l), contractWord(rnd)
			}
			src, freeSrc := guardedBytes(t, 4*sw, it%3 == 0)
			copy(src, randomPixels(rnd, 4*sw))
			xAdd := contractWord(rnd)
			rows := newContractRows(t, rnd, 4*n)
			rows.run(t, "expandRow4SSE41", func(frow, acc []uint32) {
				var ap *uint32
				if acc != nil {
					ap = &acc[0]
				}
				expandRow4SSE41(&frow[0], ap, &src[0], &cols[0], n, xAdd)
			}, func(frow, acc []uint32) {
				expandRow4Go(frow, acc, src, cols, xAdd)
			})
			rows.free()
			freeCols()
			freeSrc()
		}
	})
	type kernel func(frow, acc *uint32, src *byte, rec *byte, blocks int, xAdd uint32)
	for _, k := range []struct {
		name  string
		level cpuinfo.Level
		f     kernel
	}{
		{"expand1SSE41", cpuinfo.SSE41, expand1SSE41},
		{"expand1AVX2", cpuinfo.AVX2, expand1AVX2},
	} {
		t.Run(k.name, func(t *testing.T) {
			if cpuinfo.Max < k.level {
				t.Skip("level not supported")
			}
			rnd := rand.New(rand.NewSource(34))
			for it := 0; it < contractIters(); it++ {
				blocks := 1 + rnd.Intn(6)
				n := 8 * blocks
				// Left pixels that never decrease and advance by at most one
				// per destination pixel, as in any expansion.
				cols := make([]uint32, 2*n)
				l := rnd.Intn(8)
				for x := 0; x < n; x++ {
					if x > 0 && rnd.Intn(2) == 0 {
						l++
					}
					cols[2*x], cols[2*x+1] = uint32(l), contractWord(rnd)
				}
				sw := max(l+2, 16) + rnd.Intn(2)
				src, freeSrc := guardedBytes(t, sw, it%3 == 0)
				copy(src, randomPixels(rnd, sw))
				rec, freeRec := guardedBytes(t, blocks*expand1RecordSize, false)
				for i := range rec {
					rec[i] = byte(rnd.Intn(256))
				}
				buildExpand1Records(rec, cols, sw)
				xAdd := contractWord(rnd)
				rows := newContractRows(t, rnd, n)
				rows.run(t, k.name, func(frow, acc []uint32) {
					var ap *uint32
					if acc != nil {
						ap = &acc[0]
					}
					k.f(&frow[0], ap, &src[0], &rec[0], blocks, xAdd)
				}, func(frow, acc []uint32) {
					expandRow1Go(frow, acc, src, cols, xAdd)
				})
				rows.free()
				freeRec()
				freeSrc()
			}
		})
	}
}
