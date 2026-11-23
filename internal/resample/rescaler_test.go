// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package resample

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
)

func TestScaledDimensions(t *testing.T) {
	tests := []struct{ sw, sh, w, h, ew, eh int }{
		{100, 50, 40, 0, 40, 20},
		{100, 50, 0, 25, 50, 25},
		{100, 50, 33, 0, 33, 17}, // rounds up
		{3, 7, 0, 1, 1, 1},
		{100, 50, 7, 9, 7, 9},
	}
	for _, tt := range tests {
		w, h, ok := ScaledDimensions(tt.sw, tt.sh, tt.w, tt.h)
		if !ok {
			t.Errorf("ScaledDimensions(%d, %d, %d, %d) failed", tt.sw, tt.sh, tt.w, tt.h)
		}
		if w != tt.ew || h != tt.eh {
			t.Errorf("ScaledDimensions(%d, %d, %d, %d) = %dx%d, want %dx%d", tt.sw, tt.sh, tt.w, tt.h, w, h, tt.ew, tt.eh)
		}
	}
	for _, bad := range [][4]int{{10, 10, 0, 0}, {10, 10, -1, 5}, {0, 10, 5, 5}, {10, 10, 1 << 30, 5}} {
		_, _, ok := ScaledDimensions(bad[0], bad[1], bad[2], bad[3])
		if ok {
			t.Errorf("ScaledDimensions(%d, %d, %d, %d) succeeded, want failure", bad[0], bad[1], bad[2], bad[3])
		}
	}
}

func TestInitInvalid(t *testing.T) {
	for _, d := range [][5]int{{0, 1, 1, 1, 4}, {1, 0, 1, 1, 4}, {1, 1, 0, 1, 4}, {1, 1, 1, 0, 4}, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 5}} {
		_, err := New(d[0], d[1], d[2], d[3], d[4])
		if !errors.Is(err, ErrInvalid) {
			t.Errorf("New(%d, %d, %d, %d, %d) = %v, want %v", d[0], d[1], d[2], d[3], d[4], err, ErrInvalid)
		}
	}
}

func TestConstantImage(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		// Any rescale of a constant image is constant.
		r := rand.New(rand.NewSource(1))
		for i := 0; i < 200; i++ {
			sw, sh, dw, dh := 1+r.Intn(40), 1+r.Intn(40), 1+r.Intn(40), 1+r.Intn(40)
			src := make([]byte, sw*sh*4)
			for j := range src {
				src[j] = []byte{17, 200, 0, 255}[j%4]
			}
			out := rescaleAll(t, src, sw, sh, dw, dh, 4)
			for j, v := range out {
				if want := []byte{17, 200, 0, 255}[j%4]; want != v {
					t.Fatalf("%dx%d -> %dx%d: out[%d] = %d, want %d", sw, sh, dw, dh, j, v, want)
				}
			}
		}
	})
}

func TestReuseWork(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		src := make([]byte, 30*20)
		for i := range src {
			src[i] = byte(i)
		}
		want := rescaleAll(t, src, 30, 20, 7, 5, 1)
		var rs Rescaler
		work := make([]uint32, 100)
		if err := rs.Init(30, 20, 7, 5, 1, work); err != nil {
			t.Fatalf("Init(30, 20, 7, 5, 1) with a caller's work buffer: %v", err)
		}
		out := make([]byte, 7*5)
		for y := 0; y < 20; y++ {
			rs.ImportRow(src[y*30:])
			for rs.HasPendingOutput() {
				rs.ExportRow(out[rs.DstY()*7:])
			}
		}
		if !bytes.Equal(want, out) {
			t.Errorf("output with a caller's work buffer = %v, want %v", out, want)
		}
		if rs.ExportRow(out) {
			t.Error("ExportRow after the last row = true, want false")
		}
		if !rs.InputDone() {
			t.Error("InputDone() = false, want true")
		}
		if !rs.OutputDone() {
			t.Error("OutputDone() = false, want true")
		}
		if got := rs.DstWidth(); got != 7 {
			t.Errorf("DstWidth() = %d, want 7", got)
		}
		if got := rs.DstHeight(); got != 5 {
			t.Errorf("DstHeight() = %d, want 5", got)
		}
	})
}

func TestPremultiply(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		row := []byte{200, 100, 50, 255, 200, 100, 50, 0, 200, 100, 50, 128}
		PremultiplyRGBA(row)
		if want := []byte{200, 100, 50, 255, 0, 0, 0, 0, 100, 50, 25, 128}; !bytes.Equal(want, row) {
			t.Errorf("after PremultiplyRGBA: row = %v, want %v", row, want)
		}
		UnpremultiplyRGBA(row)
		if want := []byte{200, 100, 50, 255, 0, 0, 0, 0, 199, 100, 50, 128}; !bytes.Equal(want, row) {
			t.Errorf("after UnpremultiplyRGBA: row = %v, want %v", row, want)
		}
	})
}

func FuzzRescale(f *testing.F) {
	f.Add(uint16(10), uint16(10), uint16(3), uint16(4), uint8(4), int64(1))
	f.Fuzz(func(t *testing.T, sw, sh, dw, dh uint16, ch uint8, seed int64) {
		eachLevel(func() {
			w, h, ow, oh := int(sw%300)+1, int(sh%300)+1, int(dw%300)+1, int(dh%300)+1
			c := int(ch%4) + 1
			src := make([]byte, w*h*c)
			rand.New(rand.NewSource(seed)).Read(src)
			_ = rescaleAll(t, src, w, h, ow, oh, c)
		})
	})
}

func benchImage(w, h int) []byte {
	r := rand.New(rand.NewSource(2))
	b := make([]byte, w*h*4)
	r.Read(b)
	return b
}

func benchRescale(b *testing.B, sw, sh, dw, dh int) {
	src := benchImage(sw, sh)
	out := make([]byte, dw*4)
	var rs Rescaler
	work := make([]uint32, BufferSize(dw, 4))
	b.SetBytes(int64(sw * sh * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rs.Init(sw, sh, dw, dh, 4, work)
		for y := 0; y < sh; y++ {
			rs.ImportRow(src[y*sw*4:])
			for rs.ExportRow(out) {
			}
		}
	}
}

func BenchmarkShrink1024to300(b *testing.B) { benchRescale(b, 1024, 768, 300, 225) }
func BenchmarkExpand300to1024(b *testing.B) { benchRescale(b, 300, 225, 1024, 768) }

func BenchmarkPremultiply(b *testing.B) {
	row := benchImage(4096, 1)
	b.SetBytes(int64(len(row)))
	for i := 0; i < b.N; i++ {
		PremultiplyRGBA(row)
		row[3] = byte(i)
	}
}
