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

package vp8enc

import (
	"bytes"
	"image"
	"math/bits"
	"runtime"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
)

// TestLargeEncodeReusesBuffers checks that an encode above the former
// 2 MP pooling cutoff reuses all its scratch: once warm, an encode
// allocates little more than its output.
func TestLargeEncodeReusesBuffers(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	bufpool.Trim()
	img := synthImage(1600, 1400, 3, 1) // 2.24 MP, with transparent areas
	for _, sharp := range []bool{false, true} {
		cfg := DefaultConfig()
		cfg.Method = 2
		cfg.Preprocessing = 1 // segment smoothing scratch
		enc := func() int {
			pic := FromNRGBA(img, sharp, false)
			out, err := Encode(pic, &cfg)
			pic.Release()
			if err != nil {
				t.Fatal(err)
			}
			return len(out)
		}
		enc() // warm the pools
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		n := enc()
		runtime.ReadMemStats(&b)
		// The output, plus the encoder struct when sync.Pool dropped it.
		if alloc := b.TotalAlloc - a.TotalAlloc; alloc > uint64(n)+256<<10 {
			t.Errorf("sharp=%v: warm encode allocated %d bytes for a %d-byte output", sharp, alloc, n)
		}
	}
}

// TestPoolBudgetDoesNotChangeOutput encodes with pooling disabled, with a
// budget too small for most buffers and with the default budget.
func TestPoolBudgetDoesNotChangeOutput(t *testing.T) {
	img := synthImage(333, 251, 3, 5)
	if testing.Short() {
		img = synthImage(133, 71, 3, 5)
	}
	cfgs := []Config{DefaultConfig(), DefaultConfig(), DefaultConfig()}
	cfgs[1].Method = 0
	cfgs[2].Method, cfgs[2].ThreadLevel, cfgs[2].Pass, cfgs[2].TargetSize = 6, true, 3, 5000
	var want [][]byte
	for _, cfg := range cfgs {
		want = append(want, encodeNRGBA(t, img, cfg))
	}
	defer bufpool.SetLimit(bufpool.SetLimit(0))
	for _, limit := range []int64{0, 64 << 10, bufpool.DefaultLimit} {
		bufpool.SetLimit(limit)
		for round := 0; round < 2; round++ {
			for i, cfg := range cfgs {
				for _, sharp := range []bool{false, true} {
					pic := FromNRGBA(img, sharp, false)
					got, err := Encode(pic, &cfg)
					pic.Release()
					if err != nil {
						t.Fatal(err)
					}
					if !sharp && !bytes.Equal(got, want[i]) {
						t.Errorf("limit %d round %d config %d: output differs", limit, round, i)
					}
				}
			}
		}
		if r := bufpool.Retained(); r > limit {
			t.Errorf("limit %d: %d bytes retained", limit, r)
		}
	}
}

// poisonPool fills p with copies buffers of garbage for every size class up
// to maxN elements, each at the largest capacity of its class, so that any
// Get(n) with n <= maxN is served a buffer that was never cleared.
func poisonPool[T any](p *bufpool.Pool[T], maxN, copies int, garbage T) {
	for n := 1; n <= maxN; {
		next := n + 1
		if n >= 8 {
			step := 1 << (bits.Len(uint(n)) - 3) // a quarter of the power of two
			next = (n/step + 1) * step
		}
		for range copies {
			b := make([]T, next-1) // the largest size of n's class
			for i := range b {
				b[i] = garbage
			}
			p.Put(b)
		}
		n = next
	}
}

// TestPoisonedPoolsDoNotChangeOutput serves every pooled buffer filled with
// garbage and checks that the output matches the one obtained with pooling
// disabled (fresh, zeroed buffers): every buffer the encoder takes without
// clearing it must be fully written before it is read.
func TestPoisonedPoolsDoNotChangeOutput(t *testing.T) {
	type tc struct {
		img       *image.NRGBA
		sharp     bool
		dithering float32
		cfg       Config
	}
	sizes := [][2]int{{1, 1}, {1, 37}, {37, 1}, {17, 33}, {133, 71}}
	var cases []tc
	for i, sz := range sizes {
		img := synthImage(sz[0], sz[1], 3, int64(i)) // with transparent blocks
		for m := 0; m <= 6; m += 2 {
			cfg := DefaultConfig()
			cfg.Method = m
			cfg.Preprocessing = 1 // segment-map smoothing
			cfg.ThreadLevel = m == 0
			cfg.Partitions = 3
			cfg.LowMemory = true
			cases = append(cases,
				tc{img, false, 0, cfg},
				tc{img, true, 0, cfg},
				tc{img, false, 0.75, cfg})
		}
		cfg := DefaultConfig()
		cfg.Pass, cfg.TargetSize = 3, 400
		cases = append(cases, tc{img, false, 0, cfg})
	}
	encode := func(c tc) []byte {
		pic := FromNRGBADithered(c.img, c.sharp, false, c.dithering)
		defer pic.Release()
		out, err := Encode(pic, &c.cfg)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	defer bufpool.SetLimit(bufpool.SetLimit(0)) // pooling off: fresh buffers
	var want [][]byte
	for _, c := range cases {
		want = append(want, encode(c))
	}
	bufpool.SetLimit(bufpool.DefaultLimit)
	defer bufpool.Trim()
	poison := func() {
		bufpool.Trim()
		const maxN = 64 << 10 // above the largest request of the cases
		poisonPool(&bytePool, maxN, 3, 0xa5)
		poisonPool(&int32Pool, maxN, 3, -0x5a5a5a5b)
		poisonPool(&uint16Pool, maxN, 3, 0xa5a5)
		poisonPool(&int16Pool, maxN, 3, -0x5a5b)
		poisonPool(&mbInfoPool, 2048, 3, mbInfo{typ: 1, uvMode: 3, segment: 3, alpha: 0xa5, skip: true})
		for range 16 {
			pg := make([]uint16, tokenPageSize)
			for k := range pg {
				pg[k] = 0xa5a5
			}
			tokenPagePool.Put(pg)
		}
	}
	for i, c := range cases {
		poison()
		if got := encode(c); !bytes.Equal(got, want[i]) {
			w, h := c.img.Rect.Dx(), c.img.Rect.Dy()
			t.Errorf("%dx%d m%d sharp=%v dithering=%v pass=%d: output with poisoned pools differs",
				w, h, c.cfg.Method, c.sharp, c.dithering, c.cfg.Pass)
		}
	}

	// Pooled YCbCr pictures (odd sizes and origins, 4:2:0 and generic
	// chroma paths) against never-pooled snapshots.
	for _, r := range []image.Rectangle{image.Rect(0, 0, 1, 1), image.Rect(0, 0, 37, 1),
		image.Rect(0, 0, 1, 37), image.Rect(1, 3, 34, 24)} {
		for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio444} {
			ycc := image.NewYCbCr(image.Rect(0, 0, 40, 40), ratio)
			for i := range ycc.Y {
				ycc.Y[i] = uint8(i * 7)
			}
			for i := range ycc.Cb {
				ycc.Cb[i], ycc.Cr[i] = uint8(i*3), uint8(255-i*5)
			}
			sub := ycc.SubImage(r).(*image.YCbCr)
			want, err := Encode(SnapshotYCbCr(sub), nil)
			if err != nil {
				t.Fatal(err)
			}
			poison()
			pic := FromYCbCr(sub)
			got, err := Encode(pic, nil)
			pic.Release()
			if err != nil || !bytes.Equal(got, want) {
				t.Errorf("YCbCr %v %v: output with poisoned pools differs (err %v)", r, ratio, err)
			}
		}
	}
}

func TestPictureRelease(t *testing.T) {
	var nilPic *Picture
	nilPic.Release()
	(&Picture{}).Release()

	img := synthImage(37, 21, 0, 3)
	pic := FromNRGBA(img, false, true)
	pic.Release()
	if pic.Y != nil || pic.U != nil || pic.V != nil {
		t.Fatal("planes still set after Release")
	}
	pic.Release() // second call: no-op

	ycc := image.NewYCbCr(image.Rect(0, 0, 37, 21), image.YCbCrSubsampleRatio420)
	snap := SnapshotYCbCr(ycc)
	snap.Release() // no-op: snapshot storage is not pooled
	if snap.Y == nil {
		t.Fatal("Release cleared a snapshot")
	}
	if _, err := Encode(snap, nil); err != nil {
		t.Fatal(err)
	}
	pooled := FromYCbCr(ycc)
	data, err := Encode(pooled, nil)
	if err != nil {
		t.Fatal(err)
	}
	pooled.Release()
	if want, _ := Encode(snap, nil); !bytes.Equal(data, want) {
		t.Fatal("FromYCbCr and SnapshotYCbCr encode differently")
	}
}

func TestEncodeWithHeadroom(t *testing.T) {
	img := synthImage(65, 47, 1, 2)
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	want, err := Encode(pic, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, headroom := range []int{0, 1, 30} {
		out, err := EncodeWithHeadroom(pic, nil, headroom)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != headroom+len(want) || !bytes.Equal(out[headroom:], want) {
			t.Fatalf("headroom %d: payload differs", headroom)
		}
		if !bytes.Equal(out[:headroom], make([]byte, headroom)) {
			t.Fatalf("headroom %d: not zero", headroom)
		}
		if cap(out) <= len(out) {
			t.Fatalf("headroom %d: no spare byte for a RIFF padding byte", headroom)
		}
	}
	if _, err := EncodeWithHeadroom(pic, nil, -1); err == nil {
		t.Fatal("negative headroom accepted")
	}
	if _, err := EncodeWithHeadroom(nil, nil, 8); err == nil {
		t.Fatal("nil picture accepted")
	}
}
