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

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/decoder"
)

// dwebpYUV decodes file with "dwebp -yuv" and returns the raw Y, U, V (and
// A) planes of a w x h output.
func dwebpYUV(t *testing.T, dir string, file []byte, w, h int, alpha bool, flags ...string) (y, u, v, a []byte) {
	t.Helper()
	in := filepath.Join(dir, "in.webp")
	out := filepath.Join(dir, "out.yuv")
	require.NoError(t, os.WriteFile(in, file, 0o644))
	args := append([]string{in}, flags...)
	args = append(args, "-yuv", "-quiet", "-o", out)
	msg, err := exec.Command(tool(t, "DWEBP", "dwebp"), args...).CombinedOutput()
	require.NoError(t, err, "dwebp %v: %s", flags, msg)
	data, err := os.ReadFile(out)
	require.NoError(t, err)
	uw, uh := (w+1)/2, (h+1)/2
	n := w*h + 2*uw*uh
	if alpha {
		n += w * h
	}
	require.Equal(t, n, len(data), "dwebp %v output size", flags)
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
		require.NoError(t, err)
		files = append(files, struct {
			name string
			data []byte
		}{f, data})
	}
	for _, f := range files {
		feat, err := GetFeatures(bytes.NewReader(f.data))
		require.NoError(t, err)
		for _, o := range opts {
			name := f.name + " [dwebp " + o.flags + "]"
			w, h := feat.Width, feat.Height
			if o.opts.Crop.Max.X > 0 {
				w, h = o.opts.Crop.Max.X, o.opts.Crop.Max.Y
			}
			y, u, v, a := dwebpYUV(t, dir, f.data, w, h, feat.HasAlpha, strings.Fields(o.flags)...)
			oo := o.opts
			dec, err := decoder.NewDecoder(bytes.NewReader(f.data), &oo)
			require.NoError(t, err)
			na, err := dec.DecodeNYCbCrA()
			require.NoError(t, err, name)
			require.Equal(t, image.Rect(0, 0, w, h), na.Rect, name)
			require.True(t, bytes.Equal(y, na.Y), "%s: Y differs", name)
			require.True(t, bytes.Equal(u, na.Cb), "%s: U differs", name)
			require.True(t, bytes.Equal(v, na.Cr), "%s: V differs", name)
			if feat.HasAlpha {
				require.True(t, bytes.Equal(a, na.A), "%s: A differs", name)
				continue
			}
			ycc, err := dec.DecodeYCbCr()
			require.NoError(t, err, name)
			require.True(t, bytes.Equal(y, ycc.Y) && bytes.Equal(u, ycc.Cb) && bytes.Equal(v, ycc.Cr), name)
		}
	}
}
