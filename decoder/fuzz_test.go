package decoder

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// FuzzDecode feeds arbitrary files and options through the public decoder:
// it must never panic or hang, whatever the input.
func FuzzDecode(f *testing.F) {
	files, _ := filepath.Glob("../test_data/images/*.webp")
	files = append(files, "testdata/alpha_bomb_vp8_truncated.webp", "testdata/alpha_truncated_both.webp")
	for _, name := range files {
		data, err := os.ReadFile(name)
		if err != nil || len(data) > 100<<10 {
			continue
		}
		f.Add(data, uint8(0), uint16(0), uint16(0))
		f.Add(data, uint8(0xff), uint16(0x1234), uint16(0x0505))
	}
	f.Fuzz(func(t *testing.T, data []byte, flags uint8, crop, scale uint16) {
		o := &Options{
			BypassFiltering:        flags&1 != 0,
			NoFancyUpsampling:      flags&2 != 0,
			UseThreads:             flags&4 != 0,
			Flip:                   flags&8 != 0,
			DitheringStrength:      int(flags>>4) * 7,
			AlphaDitheringStrength: int(flags>>4) * 7,
		}
		if crop != 0 {
			o.Crop = image.Rectangle{
				Min: image.Pt(int(crop&0xf), int(crop>>4&0xf)),
				Max: image.Pt(int(crop>>8&0xf)+1, int(crop>>12)+1),
			}
		}
		if scale != 0 {
			o.Scale = image.Rectangle{Max: image.Pt(int(scale&0xff)+1, int(scale>>8)+1)}
		}
		d, err := NewDecoder(bytes.NewReader(data), o)
		if err != nil {
			return
		}
		feat := d.GetFeatures()
		if px := uint64(feat.Width) * uint64(feat.Height); px > 1<<22 {
			// Large declared frames: only decode a small window (to keep the
			// fuzzer fast) and check that nothing proportional to the full
			// RGBA frame is allocated. The alpha plane (1 byte per pixel)
			// and lossless transform sub-images (< 1 byte per pixel) are
			// legitimately sized by the declared dimensions.
			o.Crop = image.Rectangle{Max: image.Pt(16, 16)}
			o.Scale = image.Rectangle{}
			o.Flip = false
			d, err := NewDecoder(bytes.NewReader(data), o)
			if err != nil {
				return
			}
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			start := time.Now()
			_, _ = d.Decode()
			elapsed := time.Since(start)
			runtime.ReadMemStats(&m1)
			// Only the rows down to the window are decoded, so neither the
			// alpha plane nor the pixels below the window may cost time
			// or memory; the lossless transform and entropy sub-images
			// (at most 3/4 byte per pixel) are sized by the declared
			// dimensions.
			if a, limit := m1.TotalAlloc-m0.TotalAlloc, 64<<20+px; a > limit {
				t.Fatalf("%dx%d frame: decoding a 16x16 window allocated %d bytes (limit %d)", feat.Width, feat.Height, a, limit)
			}
			if elapsed > 5*time.Second {
				t.Fatalf("%dx%d frame: decoding a 16x16 window took %v", feat.Width, feat.Height, elapsed)
			}
			return
		}
		img, err := d.Decode()
		if err == nil && img.Bounds().Empty() {
			t.Fatal("empty image without error")
		}
	})
}
