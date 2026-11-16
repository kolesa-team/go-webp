package vp8lenc

import (
	"encoding/binary"
	"image"
	"testing"
)

// FuzzEncode checks that arbitrary small images and configurations never
// panic and always produce a well-formed header.
func FuzzEncode(f *testing.F) {
	f.Add([]byte{3, 7, 4, 75, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Add([]byte{0, 0, 0, 0, 255, 255, 255, 255})
	f.Add([]byte{16, 2, 6, 100, 9, 9, 9, 9, 0, 0, 0, 0})
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
		out, err := Encode(img, cfg)
		if err != nil {
			t.Fatal(err)
		}
		v := binary.LittleEndian.Uint32(out[1:5])
		if out[0] != vp8lSignature || int(v&0x3fff)+1 != w || int((v>>14)&0x3fff)+1 != h {
			t.Fatal("bad header")
		}
		plane := make([]byte, w*h)
		for i := range plane {
			plane[i] = img.Pix[4*i+3]
		}
		if _, err := EncodeAlpha(plane, w, h, w, method, quality); err != nil {
			t.Fatal(err)
		}
	})
}
