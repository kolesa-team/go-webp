package vp8enc

import (
	"image"
	"testing"
)

// FuzzEncode encodes arbitrary small pictures with arbitrary (valid or not)
// configurations: the encoder must never panic and must produce a
// structurally valid frame for valid configurations.
func FuzzEncode(f *testing.F) {
	f.Add([]byte{3, 7, 75, 4, 4, 50, 60, 0, 1, 0, 0, 0, 1, 2, 3, 4, 5})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{40, 1, 100, 6, 1, 100, 100, 7, 0, 3, 3, 100, 255, 0, 255})
	f.Add([]byte{16, 16, 10, 2, 3, 20, 30, 2, 1, 1, 2, 50, 9, 9, 9, 200, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 12 {
			return
		}
		w := 1 + int(data[0])%48
		h := 1 + int(data[1])%48
		cfg := Config{
			Quality:         float32(data[2]) / 2.5,
			Method:          int(data[3]) % 8,
			Segments:        int(data[4]) % 6,
			SNSStrength:     int(data[5]) % 101,
			FilterStrength:  int(data[6]) % 101,
			FilterSharpness: int(data[7]) % 8,
			FilterType:      int(data[8]) % 2,
			Partitions:      int(data[9]) % 4,
			Preprocessing:   int(data[10]) % 4,
			PartitionLimit:  int(data[11]) % 101,
			Autofilter:      data[4]&0x80 != 0,
			EmulateJPEGSize: data[5]&0x80 != 0,
			ThreadLevel:     data[6]&0x80 != 0,
			Pass:            1 + int(data[7]>>4)%3,
		}
		if data[8]&0x80 != 0 {
			cfg.TargetSize = int(data[8]) * 7
		}
		if data[9]&0x80 != 0 {
			cfg.TargetPSNR = float32(data[9] & 63)
		}
		pix := data[12:]
		if len(pix) == 0 {
			pix = []byte{128}
		}
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = pix[i%len(pix)]
		}
		pic := FromNRGBA(img, data[10]&0x80 != 0, data[11]&0x80 != 0)
		defer pic.Release()
		out, err := Encode(pic, &cfg)
		if err != nil {
			if cfg.validate() == nil {
				t.Fatalf("valid config %+v failed: %v", cfg, err)
			}
			return
		}
		fr := parseFrame(t, out)
		if fr.width != w || fr.height != h || fr.numParts != 1<<cfg.Partitions {
			t.Fatalf("bad frame %+v for %dx%d", fr, w, h)
		}
		// raw planes with a large stride
		raw := &Picture{Width: w, Height: h, YStride: w + 3, UVStride: (w+1)/2 + 1}
		raw.Y = make([]byte, raw.YStride*h)
		raw.U = make([]byte, raw.UVStride*((h+1)/2))
		raw.V = make([]byte, raw.UVStride*((h+1)/2))
		for i := range raw.Y {
			raw.Y[i] = pix[(3*i)%len(pix)]
		}
		for i := range raw.U {
			raw.U[i] = pix[(5*i+1)%len(pix)]
			raw.V[i] = pix[(7*i+2)%len(pix)]
		}
		if _, err := Encode(raw, &cfg); err != nil {
			t.Fatalf("raw picture: %v", err)
		}
	})
}
