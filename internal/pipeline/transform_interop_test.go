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

package pipeline

// Interop tests comparing the crop / scale / flip sinks with libwebp's dwebp.
// Run with: go test -tags interop ./internal/pipeline/ -run Interop -v
// (needs cwebp and dwebp in PATH).
//
// INTEROP_DWEBP selects another dwebp-compatible binary, and
// INTEROP_EXACT_RESCALE=1 requires bit-exact lossless rescaling: use both with
// a libwebp build whose rescaler runs the portable C code (libwebp's NEON
// rescaler rounds differently by +-1 in rare cases, which un-premultiplying
// can amplify for low alpha values).

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// readPAM parses the RGBA PAM files written by "dwebp -pam".
func readPAM(t testing.TB, path string) *image.NRGBA {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("readPAM: %v", err)
	}
	br := bufio.NewReader(bytes.NewReader(data))
	w, h, off := 0, 0, 0
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("readPAM(%s): reading the header: %v", path, err)
		}
		off += len(line)
		f := strings.Fields(line)
		if len(f) == 2 && f[0] == "WIDTH" {
			w, _ = strconv.Atoi(f[1])
		}
		if len(f) == 2 && f[0] == "HEIGHT" {
			h, _ = strconv.Atoi(f[1])
		}
		if strings.TrimSpace(line) == "ENDHDR" {
			break
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if len(img.Pix) != len(data)-off {
		t.Fatalf("readPAM(%s): %d bytes of %dx%d pixels, want %d", path, len(data)-off, w, h, len(img.Pix))
	}
	copy(img.Pix, data[off:])
	return img
}

func dwebp(t testing.TB, dir, in string, args ...string) *image.NRGBA {
	t.Helper()
	out := filepath.Join(dir, fmt.Sprintf("out%d.pam", rand.Int()))
	bin := os.Getenv("INTEROP_DWEBP")
	if bin == "" {
		bin = "dwebp"
	}
	cmd := exec.Command(bin, append([]string{in, "-pam", "-o", out}, args...)...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, b)
	}
	return readPAM(t, out)
}

func diffStats(a, b *image.NRGBA) (maxd int, mean float64, n int) {
	if a.Rect != b.Rect {
		return 999, 999, 0
	}
	sum := 0
	for i := range a.Pix {
		d := int(a.Pix[i]) - int(b.Pix[i])
		if d < 0 {
			d = -d
		}
		if d > 0 {
			n++
		}
		sum += d
		maxd = max(maxd, d)
	}
	return maxd, float64(sum) / float64(len(a.Pix)), n
}

func corpus(t *testing.T, dir string) []string {
	t.Helper()
	files, _ := filepath.Glob("../../test_data/images/*.webp")
	var out []string
	for _, f := range files {
		if fi, err := os.Stat(f); err == nil && fi.Size() > 0 {
			out = append(out, f)
		}
	}
	for _, args := range [][]string{
		{"-q", "70"},
		{"-lossless"},
		{"-q", "80", "-resize", "333", "0"},
		{"-lossless", "-resize", "301", "0"},
	} {
		dst := filepath.Join(dir, fmt.Sprintf("enc%d.webp", len(out)))
		cmd := exec.Command("cwebp", append(append([]string{"-quiet"}, args...), "../../test_data/images/logo.jpg", "-o", dst)...)
		b, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%v: %v\n%s", cmd, err, b)
		}
		out = append(out, dst)
	}
	if extra := os.Getenv("INTEROP_CORPUS"); extra != "" {
		more, _ := filepath.Glob(filepath.Join(extra, "*.webp"))
		out = append(out, more...)
	}
	return out
}

func TestInteropTransforms(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not found")
	}
	dir := t.TempDir()
	r := rand.New(rand.NewSource(1))
	type stat struct {
		maxd   int
		sum    float64
		n      int
		diffPx int
	}
	stats := map[string]*stat{}
	add := func(k string, m int, mean float64, n int) {
		s := stats[k]
		if s == nil {
			s = &stat{}
			stats[k] = s
		}
		s.maxd = max(s.maxd, m)
		s.sum += mean
		s.n++
		s.diffPx += n
	}
	for _, file := range corpus(t, dir) {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		feat, err := container.GetFeatures(data)
		if err != nil || feat.HasAnimation {
			continue
		}
		lossless := feat.Format == utils.FormatLossless
		kind := "lossy"
		if lossless {
			kind = "lossless"
		}
		W, H := feat.Width, feat.Height
		full := dwebp(t, dir, file)
		fullNoFancy := dwebp(t, dir, file, "-nofancy")
		for i := 0; i < 6; i++ {
			x, y := r.Intn(W), r.Intn(H)
			cw, ch := 1+r.Intn(W-x), 1+r.Intn(H-y)
			crop := image.Rect(x, y, x+cw, y+ch)
			flip := i%2 == 1
			args := []string{"-crop", strconv.Itoa(x), strconv.Itoa(y), strconv.Itoa(cw), strconv.Itoa(ch)}
			if flip {
				args = append(args, "-flip")
			}
			p := Params{SrcW: W, SrcH: H, Crop: crop, Flip: flip}
			if !lossless {
				p.Crop = SnapCropForYUV(crop)
			}
			// crop (+flip): exact for lossless and for lossy without fancy upsampling.
			src := full
			if !lossless {
				src = fullNoFancy
				args = append(args, "-nofancy")
			}
			want := dwebp(t, dir, file, args...)
			s, img, err := NewSink(p)
			if err != nil {
				t.Fatalf("%s: NewSink(%+v): %v", file, p, err)
			}
			feed(t, s, src, 16, r, false)
			m, mean, n := diffStats(img(), want)
			add(kind+" crop", m, mean, n)
			if m != 0 {
				t.Fatalf("%s crop %v flip %v: max abs diff = %d, want 0", file, crop, flip, m)
			}

			if !lossless {
				// With fancy upsampling libwebp upsamples chroma inside the crop window only.
				want := dwebp(t, dir, file, args[:len(args)-1]...)
				s, img, err := NewSink(p)
				if err != nil {
					t.Fatalf("%s: NewSink(%+v): %v", file, p, err)
				}
				feed(t, s, full, 16, r, false)
				m, mean, n := diffStats(img(), want)
				add(kind+" crop (fancy upsampling)", m, mean, n)
			}

			// crop + resize (+flip).
			sw, sh := 1+r.Intn(2*cw), 1+r.Intn(2*ch)
			switch i % 3 {
			case 0: // shrink
				sw, sh = max(1, cw/(2+r.Intn(4))), max(1, ch/(2+r.Intn(4)))
			case 1: // expand
				sw, sh = cw+r.Intn(cw+1), ch+r.Intn(ch+1)
			}
			rargs := []string{"-resize", strconv.Itoa(sw), strconv.Itoa(sh)}
			if i%2 == 0 {
				rargs = append(rargs, args[:5]...)
				p.Crop = crop
				if !lossless {
					p.Crop = SnapCropForYUV(crop)
				}
			} else {
				p.Crop = image.Rectangle{}
			}
			if flip {
				rargs = append(rargs, "-flip")
			}
			p.ScaleW, p.ScaleH = sw, sh
			p.Premultiply = lossless
			want = dwebp(t, dir, file, rargs...)
			s, img, err = NewSink(p)
			if err != nil {
				t.Fatalf("%s: NewSink(%+v): %v", file, p, err)
			}
			feed(t, s, fullNoFancy, 16, r, false)
			m, mean, n = diffStats(img(), want)
			k := kind + " resize"
			add(k, m, mean, n)
			if os.Getenv("INTEROP_VERBOSE") != "" && !lossless {
				t.Logf("%s %dx%d crop %v -> %dx%d: max %d mean %.3f", filepath.Base(file), W, H, p.Crop, sw, sh, m, mean)
			}
			if lossless && os.Getenv("INTEROP_EXACT_RESCALE") == "1" && m != 0 {
				t.Fatalf("%s resize %v -> %dx%d: max abs diff = %d, want 0", file, p.Crop, sw, sh, m)
			}
			if lossless && !(mean < 0.01) {
				t.Fatalf("%s resize %v -> %dx%d: mean abs diff = %g, want < 0.01", file, p.Crop, sw, sh, mean)
			}
		}
	}
	for k, s := range stats {
		t.Logf("%-32s cases=%3d max_abs_diff=%3d mean_abs_diff=%.4f differing_samples=%d", k, s.n, s.maxd, s.sum/float64(s.n), s.diffPx)
	}
}

// dwebpYUV decodes file to raw Y, U, V (and A) planes with "dwebp -yuv".
func dwebpYUV(t testing.TB, dir, in string, w, h int, alpha bool, args ...string) (y, u, v, a []byte) {
	t.Helper()
	out := filepath.Join(dir, fmt.Sprintf("out%d.yuv", rand.Int()))
	b, err := exec.Command("dwebp", append([]string{in, "-yuv", "-o", out}, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("dwebp %s -yuv %v: %v\n%s", in, args, err, b)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("dwebpYUV: %v", err)
	}
	uw, uh := (w+1)/2, (h+1)/2
	want := w*h + 2*uw*uh
	if alpha {
		want += w * h
	}
	if want != len(data) {
		t.Fatalf("dwebp %s -yuv %v: %d bytes for %dx%d (alpha %v), want %d", in, args, len(data), w, h, alpha, want)
	}
	y, data = data[:w*h], data[w*h:]
	u, data = data[:uw*uh], data[uw*uh:]
	v, data = data[:uw*uh], data[uw*uh:]
	if alpha {
		a = data
	}
	return
}

// TestInteropYUVScaling checks the YUV-domain scaling path against dwebp's
// scaled RGBA output (loop filter disabled on both sides, since libwebp
// bypasses it for strong downscaling).
func TestInteropYUVScaling(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not found")
	}
	dir := t.TempDir()
	r := rand.New(rand.NewSource(2))
	worst, cases := 0, 0
	for _, file := range corpus(t, dir) {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		feat, err := container.GetFeatures(data)
		if err != nil || feat.HasAnimation || feat.Format != utils.FormatLossy {
			continue
		}
		W, H := feat.Width, feat.Height
		img, err := container.Parse(data)
		if err != nil {
			t.Fatalf("%s: container.Parse: %v", file, err)
		}
		hasAlpha := img.ALPH != nil
		yp, up, vp, ap := dwebpYUV(t, dir, file, W, H, hasAlpha, "-nofilter")
		for i := 0; i < 6; i++ {
			p := Params{SrcW: W, SrcH: H, Flip: i%3 == 2}
			args := []string{"-nofilter"}
			if i%2 == 1 {
				x, y := r.Intn(W), r.Intn(H)
				cw, ch := 1+r.Intn(W-x), 1+r.Intn(H-y)
				p.Crop = SnapCropForYUV(image.Rect(x, y, x+cw, y+ch))
				args = append(args, "-crop", strconv.Itoa(x), strconv.Itoa(y), strconv.Itoa(cw), strconv.Itoa(ch))
			}
			cw, ch := W, H
			if !p.Crop.Empty() {
				cw, ch = p.Crop.Dx(), p.Crop.Dy()
			}
			sw, sh := max(1, cw/(1+r.Intn(5))), max(1, ch/(1+r.Intn(5)))
			if i%3 == 1 {
				sw, sh = cw+r.Intn(cw+1), ch+r.Intn(ch+1)
			}
			p.ScaleW, p.ScaleH = sw, sh
			args = append(args, "-resize", strconv.Itoa(sw), strconv.Itoa(sh))
			if p.Flip {
				args = append(args, "-flip")
			}
			want := dwebp(t, dir, file, args...)
			s, out, err := NewSink(p)
			if err != nil {
				t.Fatalf("%s: NewSink(%+v): %v", file, p, err)
			}
			ys, ok := s.(YUVSink)
			if !ok {
				t.Fatalf("%s: NewSink(%+v) = %T, want a YUVSink", file, p, s)
			}
			uw := (W + 1) / 2
			for y := 0; y < s.NeedRows(); {
				n := min(H-y, []int{16, 1 + r.Intn(20), 2, 7}[i%4])
				var a []byte
				if hasAlpha {
					a = ap[y*W:]
				}
				if err := ys.CommitYUV(y, n, yp[y*W:], W, up[(y/2)*uw:], vp[(y/2)*uw:], uw, a, W); err != nil {
					t.Fatalf("%s %+v: CommitYUV(%d, %d): %v", file, p, y, n, err)
				}
				y += n
			}
			m, mean, _ := diffStats(out(), want)
			worst = max(worst, m)
			cases++
			if os.Getenv("INTEROP_EXACT_RESCALE") == "1" {
				if m != 0 {
					t.Fatalf("%s %+v (mean %.4f): max abs diff = %d, want 0", file, p, mean, m)
				}
			} else {
				// libwebp's NEON rescaler rounds +-1 differently in rare
				// cases; the YUV->RGB conversion amplifies that a little.
				if m > 4 {
					t.Fatalf("%s %+v (mean %.4f): max abs diff = %d, want <= 4", file, p, mean, m)
				}
				if !(mean < 0.001) {
					t.Fatalf("%s %+v: mean abs diff = %g, want < 0.001", file, p, mean)
				}
			}
		}
	}
	t.Logf("lossy YUV-domain resize: cases=%d max_abs_diff=%d", cases, worst)
}
