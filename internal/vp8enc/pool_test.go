package vp8enc

import (
	"bytes"
	"image"
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
	}
	if _, err := EncodeWithHeadroom(pic, nil, -1); err == nil {
		t.Fatal("negative headroom accepted")
	}
	if _, err := EncodeWithHeadroom(nil, nil, 8); err == nil {
		t.Fatal("nil picture accepted")
	}
}
