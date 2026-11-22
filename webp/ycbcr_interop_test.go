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

package webp

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kolesa-team/go-webp/v2/decoder"
)

// dwebpYUV decodes file with "dwebp -yuv" and returns the raw Y, U, V (and
// A) planes of a w x h output.
func dwebpYUV(t *testing.T, dir string, file []byte, w, h int, alpha bool, flags ...string) (y, u, v, a []byte) {
	t.Helper()
	in := filepath.Join(dir, "in.webp")
	out := filepath.Join(dir, "out.yuv")
	if err := os.WriteFile(in, file, 0o644); err != nil {
		t.Fatal(err)
	}
	args := append([]string{in}, flags...)
	args = append(args, "-yuv", "-quiet", "-o", out)
	msg, err := exec.Command(tool(t, "DWEBP", "dwebp"), args...).CombinedOutput()
	if err != nil {
		t.Fatalf("dwebp %v: %v: %s", flags, err, msg)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	uw, uh := (w+1)/2, (h+1)/2
	n := w*h + 2*uw*uh
	if alpha {
		n += w * h
	}
	if len(data) != n {
		t.Fatalf("dwebp %v output size %d bytes, want %d", flags, len(data), n)
	}
	y, data = data[:w*h], data[w*h:]
	u, data = data[:uw*uh], data[uw*uh:]
	v, a = data[:uw*uh], data[uw*uh:]
	return
}

// TestInteropDecodeYCbCr compares the planes of DecodeYCbCr/DecodeNYCbCrA
// with libwebp's Y'CbCr(A) output ("dwebp -yuv").
func TestInteropDecodeYCbCr(t *testing.T) {
	dir := t.TempDir()
	opts := []struct {
		flags string
		opts  decoder.Options
	}{
		{"", decoder.Options{}},
		{"-nofilter", decoder.Options{BypassFiltering: true}},
		{"-dither 100", decoder.Options{DitheringStrength: 100}},
		{"-alpha_dither", decoder.Options{AlphaDitheringStrength: 100}},
		{"-mt", decoder.Options{UseThreads: true}},
		{"-flip", decoder.Options{Flip: true}},
		{"-crop 3 5 11 9", decoder.Options{Crop: image.Rectangle{Min: image.Pt(3, 5), Max: image.Pt(11, 9)}}},
		{"-crop 2 4 12 9 -flip", decoder.Options{Crop: image.Rectangle{Min: image.Pt(2, 4), Max: image.Pt(12, 9)}, Flip: true}},
	}
	encodings := []string{
		"-q 75",
		"-q 10 -m 6 -sns 100 -f 100",
		"-q 50 -alpha_q 30 -alpha_filter best",
		"-q 50 -alpha_q 60 -alpha_method 0",
		"-q 70 -sharp_yuv -exact",
	}
	var files []struct {
		name string
		data []byte
	}
	for _, size := range [][2]int{{64, 48}, {37, 21}} {
		for _, s := range synthImages(size[0], size[1]) {
			for _, enc := range encodings {
				files = append(files, struct {
					name string
					data []byte
				}{fmt.Sprintf("%s %dx%d [cwebp %s]", s.name, size[0], size[1], enc), cwebp(t, dir, s.img, strings.Fields(enc)...)})
			}
		}
	}
	for _, f := range []string{"m4_q75.webp", "webp-logo-lossy.webp"} {
		data, err := os.ReadFile("../test_data/images/" + f)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, struct {
			name string
			data []byte
		}{f, data})
	}
	for _, f := range files {
		feat, err := GetFeatures(bytes.NewReader(f.data))
		if err != nil {
			t.Fatalf("%s: GetFeatures: %v", f.name, err)
		}
		for _, o := range opts {
			name := f.name + " [dwebp " + o.flags + "]"
			w, h := feat.Width, feat.Height
			if o.opts.Crop.Max.X > 0 {
				w, h = o.opts.Crop.Max.X, o.opts.Crop.Max.Y
			}
			y, u, v, a := dwebpYUV(t, dir, f.data, w, h, feat.HasAlpha, strings.Fields(o.flags)...)
			oo := o.opts
			dec, err := decoder.NewDecoder(bytes.NewReader(f.data), &oo)
			if err != nil {
				t.Fatalf("%s: NewDecoder: %v", name, err)
			}
			na, err := dec.DecodeNYCbCrA()
			if err != nil {
				t.Fatalf("%s: DecodeNYCbCrA: %v", name, err)
			}
			if want := image.Rect(0, 0, w, h); na.Rect != want {
				t.Fatalf("%s: DecodeNYCbCrA Rect = %v, want %v", name, na.Rect, want)
			}
			if !bytes.Equal(y, na.Y) {
				t.Fatalf("%s: Y differs", name)
			}
			if !bytes.Equal(u, na.Cb) {
				t.Fatalf("%s: U differs", name)
			}
			if !bytes.Equal(v, na.Cr) {
				t.Fatalf("%s: V differs", name)
			}
			if feat.HasAlpha {
				if !bytes.Equal(a, na.A) {
					t.Fatalf("%s: A differs", name)
				}
				continue
			}
			ycc, err := dec.DecodeYCbCr()
			if err != nil {
				t.Fatalf("%s: DecodeYCbCr: %v", name, err)
			}
			if !(bytes.Equal(y, ycc.Y) && bytes.Equal(u, ycc.Cb) && bytes.Equal(v, ycc.Cr)) {
				t.Fatalf("%s: DecodeYCbCr planes differ from dwebp -yuv", name)
			}
		}
	}
}
