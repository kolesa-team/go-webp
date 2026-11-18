package vp8lenc

import (
	"bytes"
	"encoding/binary"
	"image"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
)

// FuzzEncode checks that arbitrary small images and configurations never
// panic, always produce a well-formed header and, when lossless, decode to
// the source pixels (the RGB of fully transparent pixels may change without
// Exact). Flag bit 6 runs every ThreadLevel stage in parallel, even for
// these small images.
func FuzzEncode(f *testing.F) {
	f.Add([]byte{3, 7, 4, 75, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Add([]byte{0, 0, 0, 0, 255, 255, 255, 255})
	f.Add([]byte{16, 2, 6, 100, 9, 9, 9, 9, 0, 0, 0, 0})
	f.Add([]byte{39, 38, 4, 75, 64, 0, 0, 0, 0, 200, 10, 30, 255, 7, 99, 3, 128, 1, 2, 3, 255})
	f.Add([]byte{30, 0, 3, 40, 65, 0, 0, 0, 0, 5, 6, 7, 255, 8, 9, 10, 255, 11})
	f.Add([]byte{17, 25, 5, 90, 66, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 4 {
			return
		}
		w := int(data[0])%40 + 1
		h := int(data[1])%40 + 1
		method := int(data[2]) % 7
		quality := float32(data[3]) * 100 / 255
		flags := byte(0)
		if len(data) > 4 {
			flags = data[4]
		}
		pix := data[4:]
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		if len(pix) > 0 {
			// Few distinct source bytes give palettes; many give photos.
			for i := range img.Pix {
				img.Pix[i] = pix[i%len(pix)] ^ byte(i/len(pix))*(flags&1)
			}
		}
		cfg := &Config{
			Quality:      quality,
			Method:       method,
			Exact:        flags&2 != 0,
			NearLossless: []int{100, 100, 60, 0}[(flags>>2)&3],
			LowMemory:    flags&16 != 0,
			ThreadLevel:  flags&32 != 0,
		}
		if flags&64 != 0 {
			cfg.ThreadLevel = true
			defer forceParallel()()
		}
		out, err := Encode(img, cfg)
		if err != nil {
			t.Fatal(err)
		}
		v := binary.LittleEndian.Uint32(out[1:5])
		if out[0] != vp8lSignature || int(v&0x3fff)+1 != w || int((v>>14)&0x3fff)+1 != h {
			t.Fatal("bad header")
		}
		if cfg.NearLossless == 100 {
			got := decodeVP8L(t, out, w, h)
			for i := 0; i < len(got); i += 4 {
				src := img.Pix[i : i+4]
				if got[i+3] != src[3] || (src[3] != 0 || cfg.Exact) && !bytes.Equal(got[i:i+3], src[:3]) {
					t.Fatalf("%+v: pixel %d decodes to %v, want %v", *cfg, i/4, got[i:i+4], src)
				}
			}
		}
		plane := make([]byte, w*h)
		for i := range plane {
			plane[i] = img.Pix[4*i+3]
		}
		stream, err := EncodeAlpha(plane, w, h, w, method, quality)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]byte, w*h)
		if err := vp8l.DecodeAlpha(stream, w, h, got); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, plane) {
			t.Fatal("alpha plane does not decode to the source")
		}
		// Every kernel level (dsp_amd64.go) gives the same bytes.
		atEveryLevel(func(l cpuinfo.Level) {
			o, err := Encode(img, cfg)
			if err != nil || !bytes.Equal(o, out) {
				t.Fatalf("level %v %+v: output differs from the default level's (%v)", l, *cfg, err)
			}
			a, err := EncodeAlpha(plane, w, h, w, method, quality)
			if err != nil || !bytes.Equal(a, stream) {
				t.Fatalf("level %v: alpha stream differs from the default level's (%v)", l, err)
			}
		})
	})
}
