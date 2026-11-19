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

//go:build interop

package vp8enc

// Interoperability tests against libwebp's command line tools (dwebp, cwebp).
// Run with: go test -tags interop ./internal/vp8enc/

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func wrapRIFF(vp8 []byte) []byte {
	n := len(vp8)
	pad := n & 1
	out := make([]byte, 0, 20+n+pad)
	out = append(out, "RIFF"...)
	out = binary.LittleEndian.AppendUint32(out, uint32(4+8+n+pad))
	out = append(out, "WEBPVP8 "...)
	out = binary.LittleEndian.AppendUint32(out, uint32(n))
	out = append(out, vp8...)
	if pad != 0 {
		out = append(out, 0)
	}
	return out
}

func readPAM(t *testing.T, path string) *image.NRGBA {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r := bufio.NewReader(bytes.NewReader(data))
	w, h := 0, 0
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			t.Fatalf("bad PAM header: %v", err)
		}
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "WIDTH"):
			w, _ = strconv.Atoi(strings.Fields(line)[1])
		case strings.HasPrefix(line, "HEIGHT"):
			h, _ = strconv.Atoi(strings.Fields(line)[1])
		}
		if line == "ENDHDR" {
			break
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if _, err := io.ReadFull(r, img.Pix); err != nil {
		t.Fatal(err)
	}
	return img
}

func requireTool(t *testing.T, name string) {
	t.Helper()
	if _, err := exec.LookPath(name); err != nil {
		t.Skipf("%s not found", name)
	}
}

// dwebpDecode decodes a WebP file with libwebp and returns its RGBA output.
func dwebpDecode(t *testing.T, webpData []byte) *image.NRGBA {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "in.webp")
	out := filepath.Join(dir, "out.pam")
	if err := os.WriteFile(in, webpData, 0o644); err != nil {
		t.Fatal(err)
	}
	if o, err := exec.Command("dwebp", in, "-pam", "-o", out).CombinedOutput(); err != nil {
		t.Fatalf("dwebp failed: %v\n%s", err, o)
	}
	return readPAM(t, out)
}

// rgbPSNR compares the RGB channels of the opaque pixels.
func rgbPSNR(a, b *image.NRGBA) float64 {
	var sse float64
	n := 0
	w, h := a.Rect.Dx(), a.Rect.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pa := a.Pix[a.PixOffset(a.Rect.Min.X+x, a.Rect.Min.Y+y):]
			pb := b.Pix[b.PixOffset(b.Rect.Min.X+x, b.Rect.Min.Y+y):]
			if pa[3] != 255 {
				continue
			}
			for c := 0; c < 3; c++ {
				d := float64(pa[c]) - float64(pb[c])
				sse += d * d
			}
			n += 3
		}
	}
	if sse == 0 || n == 0 {
		return 99
	}
	return 10 * math.Log10(255*255*float64(n)/sse)
}

func writePNG(t *testing.T, img image.Image, path string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

// TestInteropDwebp encodes a corpus with many configurations, decodes it
// with dwebp and checks the quality against cwebp's at the same settings
// (and an absolute floor for natural images).
func TestInteropDwebp(t *testing.T) {
	requireTool(t, "dwebp")
	requireTool(t, "cwebp")
	type item struct {
		name    string
		img     *image.NRGBA
		natural bool
	}
	corpus := []item{
		{"logo", loadTestImage(t, "logo.jpg"), true},
		{"gradient-1x1", synthImage(1, 1, 0, 1), false},
		{"gradient-3x7", synthImage(3, 7, 0, 1), false},
		{"noise-15x17", synthImage(15, 17, 1, 1), false},
		{"edges-17x33", synthImage(17, 33, 2, 1), false},
		{"alpha-100x75", synthImage(100, 75, 3, 1), false},
		{"gradient-1000x1", synthImage(1000, 1, 0, 1), false},
		{"gradient-1x300", synthImage(1, 300, 0, 1), false},
		{"edges-257x129", synthImage(257, 129, 2, 3), false},
	}
	minPSNR := map[float32]float64{10: 28, 50: 33, 75: 35, 90: 36, 100: 37}
	dir := t.TempDir()
	for _, it := range corpus {
		pngPath := filepath.Join(dir, it.name+".png")
		writePNG(t, it.img, pngPath)
		for _, q := range []float32{10, 50, 75, 90, 100} {
			for _, m := range []int{0, 2, 4, 6} {
				cfg := DefaultConfig()
				cfg.Quality = q
				cfg.Method = m
				cfg.Partitions = m % 4
				cfg.Segments = 1 + m%4
				cfg.Autofilter = m == 6
				cfg.FilterType = m & 1
				name := fmt.Sprintf("%s/q%v/m%d", it.name, q, m)
				pic := FromNRGBA(it.img, false, true) // exact, like cwebp -exact
				data, err := Encode(pic, &cfg)
				pic.Release()
				if err != nil {
					t.Fatal(err)
				}
				dec := dwebpDecode(t, wrapRIFF(data))
				if dec.Rect.Dx() != it.img.Rect.Dx() || dec.Rect.Dy() != it.img.Rect.Dy() {
					t.Fatalf("%s: dims %v", name, dec.Rect)
				}
				p := rgbPSNR(it.img, dec)
				if it.natural && p < minPSNR[q] {
					t.Errorf("%s: PSNR %.2f dB < %.2f dB", name, p, minPSNR[q])
				}
				// reference: cwebp with the same main settings
				ref := filepath.Join(dir, "ref.webp")
				args := []string{"-quiet", "-q", fmt.Sprint(q), "-m", fmt.Sprint(m),
					"-segments", fmt.Sprint(cfg.Segments), "-exact", pngPath, "-o", ref}
				if cfg.FilterType == 0 {
					args = append(args, "-nostrong")
				}
				if cfg.Autofilter {
					args = append(args, "-af")
				}
				if o, err := exec.Command("cwebp", args...).CombinedOutput(); err != nil {
					t.Fatalf("cwebp: %v\n%s", err, o)
				}
				refData, err := os.ReadFile(ref)
				if err != nil {
					t.Fatal(err)
				}
				pRef := rgbPSNR(it.img, dwebpDecode(t, refData))
				if p < pRef-1 {
					t.Errorf("%s: PSNR %.2f dB, cwebp %.2f dB", name, p, pRef)
				}
			}
		}
	}
}

// TestInteropVsCwebp checks that sizes and PSNR are close to cwebp's at the
// same quality / method.
func TestInteropVsCwebp(t *testing.T) {
	requireTool(t, "dwebp")
	requireTool(t, "cwebp")
	for _, name := range []string{"source.jpg", "logo.jpg"} {
		img := loadTestImage(t, name)
		dir := t.TempDir()
		pngPath := filepath.Join(dir, "src.png")
		writePNG(t, img, pngPath)
		for _, q := range []float32{10, 50, 75, 90, 100} {
			for _, m := range []int{0, 4, 6} {
				cfg := DefaultConfig()
				cfg.Quality = q
				cfg.Method = m
				ours := wrapRIFF(encodeNRGBA(t, img, cfg))
				ref := filepath.Join(dir, "ref.webp")
				args := []string{"-quiet", "-q", fmt.Sprint(q), "-m", fmt.Sprint(m), pngPath, "-o", ref}
				if o, err := exec.Command("cwebp", args...).CombinedOutput(); err != nil {
					t.Fatalf("cwebp: %v\n%s", err, o)
				}
				refData, err := os.ReadFile(ref)
				if err != nil {
					t.Fatal(err)
				}
				pOurs := rgbPSNR(img, dwebpDecode(t, ours))
				pRef := rgbPSNR(img, dwebpDecode(t, refData))
				ratio := float64(len(ours)) / float64(len(refData))
				t.Logf("%s q=%v m=%d: size %d vs %d (%+.1f%%), PSNR %.2f vs %.2f", name, q, m,
					len(ours), len(refData), 100*(ratio-1), pOurs, pRef)
				if ratio > 1.10 {
					t.Errorf("%s q=%v m=%d: size %d is more than 10%% above cwebp's %d", name, q, m, len(ours), len(refData))
				}
				if pOurs < pRef-0.5 {
					t.Errorf("%s q=%v m=%d: PSNR %.2f more than 0.5 dB below cwebp's %.2f", name, q, m, pOurs, pRef)
				}
			}
		}
	}
}

// TestInteropOptions exercises less common options through dwebp.
func TestInteropOptions(t *testing.T) {
	requireTool(t, "dwebp")
	img := loadTestImage(t, "logo.jpg")
	cases := []Config{
		func() Config { c := DefaultConfig(); c.TargetSize = 15000; c.Pass = 6; return c }(),
		func() Config { c := DefaultConfig(); c.TargetPSNR = 35; c.Pass = 6; return c }(),
		func() Config { c := DefaultConfig(); c.Preprocessing = 3; return c }(),
		func() Config { c := DefaultConfig(); c.EmulateJPEGSize = true; return c }(),
		func() Config { c := DefaultConfig(); c.FilterStrength = 0; return c }(),
		func() Config { c := DefaultConfig(); c.Autofilter = true; c.Segments = 1; return c }(),
		func() Config { c := DefaultConfig(); c.SNSStrength = 100; c.FilterSharpness = 7; return c }(),
		func() Config { c := DefaultConfig(); c.PartitionLimit = 100; c.Method = 6; return c }(),
	}
	for i, cfg := range cases {
		dec := dwebpDecode(t, wrapRIFF(encodeNRGBA(t, img, cfg)))
		if p := rgbPSNR(img, dec); p < 28 {
			t.Errorf("case %d (%+v): PSNR %.2f", i, cfg, p)
		}
	}
	// sharp YUV route
	pic := FromNRGBA(img, true, false)
	data, err := Encode(pic, nil)
	pic.Release()
	if err != nil {
		t.Fatal(err)
	}
	if p := rgbPSNR(img, dwebpDecode(t, wrapRIFF(data))); p < 30 {
		t.Errorf("sharp yuv: PSNR %.2f", p)
	}
}

// TestInteropExtremeDims checks the maximal VP8 dimensions with dwebp.
func TestInteropExtremeDims(t *testing.T) {
	requireTool(t, "dwebp")
	for _, sz := range [][2]int{{16383, 1}, {1, 16383}, {16383, 17}} {
		img := synthImage(sz[0], sz[1], 0, 1)
		for _, m := range []int{0, 4, 6} {
			cfg := DefaultConfig()
			cfg.Method = m
			cfg.Partitions = 3
			dec := dwebpDecode(t, wrapRIFF(encodeNRGBA(t, img, cfg)))
			if dec.Rect.Dx() != sz[0] || dec.Rect.Dy() != sz[1] {
				t.Fatalf("%v: decoded dims %v", sz, dec.Rect)
			}
			if p := rgbPSNR(img, dec); p < 25 {
				t.Errorf("%v m%d: PSNR %.2f", sz, m, p)
			}
		}
	}
}
