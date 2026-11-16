//go:build interop

// Interoperability tests against libwebp's command line tools (cwebp, dwebp).
// Run with: go test -tags interop ./webp/
// The tools must be in PATH (or set CWEBP / DWEBP).

package webp

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/encoder"
)

func tool(t *testing.T, env, name string) string {
	t.Helper()
	if p := os.Getenv(env); p != "" {
		return p
	}
	p, err := exec.LookPath(name)
	if err != nil {
		t.Skipf("%s not found", name)
	}
	return p
}

// readPAM parses the output of "dwebp -pam".
func readPAM(t *testing.T, data []byte) *image.NRGBA {
	t.Helper()
	end := bytes.Index(data, []byte("ENDHDR\n"))
	require.True(t, end > 0, "not a PAM file")
	var w, h int
	for _, line := range strings.Split(string(data[:end]), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && f[0] == "WIDTH" {
			w, _ = strconv.Atoi(f[1])
		}
		if len(f) == 2 && f[0] == "HEIGHT" {
			h, _ = strconv.Atoi(f[1])
		}
	}
	pix := data[end+7:]
	require.Equal(t, 4*w*h, len(pix))
	return &image.NRGBA{Pix: pix, Stride: 4 * w, Rect: image.Rect(0, 0, w, h)}
}

func dwebp(t *testing.T, dir string, file []byte, flags ...string) *image.NRGBA {
	t.Helper()
	in := filepath.Join(dir, "in.webp")
	out := filepath.Join(dir, "out.pam")
	require.NoError(t, os.WriteFile(in, file, 0o644))
	args := append([]string{in}, flags...)
	args = append(args, "-pam", "-quiet", "-o", out)
	cmd := exec.Command(tool(t, "DWEBP", "dwebp"), args...)
	msg, err := cmd.CombinedOutput()
	require.NoError(t, err, "dwebp %v: %s", flags, msg)
	data, err := os.ReadFile(out)
	require.NoError(t, err)
	return readPAM(t, data)
}

func cwebp(t *testing.T, dir string, img image.Image, flags ...string) []byte {
	t.Helper()
	in := filepath.Join(dir, "in.png")
	out := filepath.Join(dir, "out.webp")
	f, err := os.Create(in)
	require.NoError(t, err)
	require.NoError(t, png.Encode(f, img))
	require.NoError(t, f.Close())
	args := append([]string{"-quiet"}, flags...)
	args = append(args, in, "-o", out)
	msg, err := exec.Command(tool(t, "CWEBP", "cwebp"), args...).CombinedOutput()
	require.NoError(t, err, "cwebp %v: %s", flags, msg)
	data, err := os.ReadFile(out)
	require.NoError(t, err)
	return data
}

func decodeWith(t *testing.T, file []byte, o *decoder.Options) *image.NRGBA {
	t.Helper()
	img, err := Decode(bytes.NewReader(file), o)
	require.NoError(t, err)
	return img.(*image.NRGBA)
}

func requireSamePixels(t *testing.T, want, got *image.NRGBA, tolerance int, msg string) {
	t.Helper()
	require.Equal(t, want.Rect.Size(), got.Rect.Size(), msg)
	w, h := want.Rect.Dx(), want.Rect.Dy()
	for y := 0; y < h; y++ {
		a := want.Pix[y*want.Stride:][:4*w]
		b := got.Pix[y*got.Stride:][:4*w]
		if tolerance == 0 {
			if !bytes.Equal(a, b) {
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("%s: pixel (%d,%d) channel %d: libwebp %d, ours %d", msg, i/4, y, i%4, a[i], b[i])
					}
				}
			}
			continue
		}
		for i := range a {
			if d := int(a[i]) - int(b[i]); d > tolerance || d < -tolerance {
				t.Fatalf("%s: pixel (%d,%d) channel %d: libwebp %d, ours %d", msg, i/4, y, i%4, a[i], b[i])
			}
		}
	}
}

// TestInteropEncode checks that libwebp decodes our files to exactly the
// pixels our decoder produces (and, for lossless, to the source pixels).
func TestInteropEncode(t *testing.T) {
	dir := t.TempDir()
	type cfg struct {
		name string
		opts func() *encoder.Options
	}
	var cfgs []cfg
	for level := 0; level <= 9; level += 3 {
		level := level
		cfgs = append(cfgs, cfg{fmt.Sprintf("lossless z%d", level), func() *encoder.Options {
			o, _ := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, level)
			return o
		}})
	}
	for _, m := range []int{0, 2, 4, 6} {
		for _, q := range []float32{5, 75, 100} {
			m, q := m, q
			cfgs = append(cfgs, cfg{fmt.Sprintf("lossy m%d q%g", m, q), func() *encoder.Options {
				o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDefault, q)
				o.Method = m
				return o
			}})
		}
	}
	cfgs = append(cfgs, cfg{"lossy alpha raw", func() *encoder.Options {
		o, _ := encoder.NewLossyEncoderOptions(encoder.PresetPhoto, 60)
		o.AlphaCompression = 0
		return o
	}}, cfg{"lossy alpha best filter", func() *encoder.Options {
		o, _ := encoder.NewLossyEncoderOptions(encoder.PresetDrawing, 60)
		o.AlphaFiltering = 2
		o.Partitions = 3
		o.Segments = 1
		return o
	}})

	for _, size := range roundtripSizes {
		for _, s := range synthImages(size[0], size[1]) {
			for _, c := range cfgs {
				name := fmt.Sprintf("%dx%d %s %s", size[0], size[1], s.name, c.name)
				buf := &bytes.Buffer{}
				require.NoError(t, Encode(buf, s.img, c.opts()), name)
				ref := dwebp(t, dir, buf.Bytes())
				ours := decodeWith(t, buf.Bytes(), nil)
				requireSamePixels(t, ref, ours, 0, name)
			}
		}
	}
}

// TestInteropDecode decodes cwebp output with every decoder option and
// compares with dwebp.
func TestInteropDecode(t *testing.T) {
	dir := t.TempDir()
	type opt struct {
		flags string
		opts  decoder.Options
		tol   int // dwebp uses NEON/SSE rescaler code that may differ by a few units
	}
	opts := []opt{
		{"", decoder.Options{}, 0},
		{"-nofancy", decoder.Options{NoFancyUpsampling: true}, 0},
		{"-nofilter", decoder.Options{BypassFiltering: true}, 0},
		{"-dither 100", decoder.Options{DitheringStrength: 100}, 0},
		{"-alpha_dither", decoder.Options{AlphaDitheringStrength: 100}, 0},
		{"-mt", decoder.Options{UseThreads: true}, 0},
		{"-flip", decoder.Options{Flip: true}, 0},
		{"-crop 3 5 11 9", decoder.Options{Crop: image.Rectangle{Min: image.Pt(3, 5), Max: image.Pt(11, 9)}}, 0},
		{"-crop 2 4 11 9 -flip -nofancy", decoder.Options{Crop: image.Rectangle{Min: image.Pt(2, 4), Max: image.Pt(11, 9)}, Flip: true, NoFancyUpsampling: true}, 0},
		{"-resize 31 19", decoder.Options{Scale: image.Rectangle{Max: image.Pt(31, 19)}}, 4},
		{"-resize 150 101 -flip", decoder.Options{Scale: image.Rectangle{Max: image.Pt(150, 101)}, Flip: true}, 4},
		{"-crop 1 1 40 30 -resize 20 15", decoder.Options{Crop: image.Rectangle{Min: image.Pt(1, 1), Max: image.Pt(40, 30)}, Scale: image.Rectangle{Max: image.Pt(20, 15)}}, 4},
	}
	encodings := []string{
		"-q 75",
		"-q 10 -m 6 -sns 100 -f 100",
		"-q 90 -m 2 -segments 1 -partition_limit 50",
		"-q 50 -alpha_q 30 -alpha_filter best",
		"-q 50 -alpha_q 60 -alpha_method 0",
		"-q 70 -sharp_yuv -exact",
		"-lossless -z 0",
		"-lossless -z 6 -exact",
		"-lossless -z 9",
		"-near_lossless 40",
	}
	for _, s := range synthImages(64, 48) {
		for _, enc := range encodings {
			file := cwebp(t, dir, s.img, strings.Fields(enc)...)
			for _, o := range opts {
				name := fmt.Sprintf("%s [cwebp %s] [dwebp %s]", s.name, enc, o.flags)
				ref := dwebp(t, dir, file, strings.Fields(o.flags)...)
				o := o.opts
				ours := decodeWith(t, file, &o)
				tol := 0
				if o.Scale.Max.X > 0 {
					tol = 4
				}
				requireSamePixels(t, ref, ours, tol, name)
			}
		}
	}

	// Repository fixtures.
	for _, f := range []string{"m4_q75.webp", "webp-logo-lossy.webp", "webp-logo-lossless.webp"} {
		data, err := os.ReadFile("../test_data/images/" + f)
		require.NoError(t, err)
		for _, o := range opts {
			if strings.Contains(o.flags, "resize") {
				continue
			}
			ref := dwebp(t, dir, data, strings.Fields(o.flags)...)
			oo := o.opts
			requireSamePixels(t, ref, decodeWith(t, data, &oo), 0, f+" "+o.flags)
		}
	}
}

// TestInteropQuality compares our lossy encoder with cwebp at equal settings.
func TestInteropQuality(t *testing.T) {
	dir := t.TempDir()
	img := toNRGBA(loadJPEG(t, "source.jpg"))
	for _, q := range []float32{25, 75, 95} {
		for _, m := range []int{0, 4, 6} {
			ref := cwebp(t, dir, img, "-q", fmt.Sprint(q), "-m", fmt.Sprint(m))
			o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, q)
			require.NoError(t, err)
			o.Method = m
			buf := &bytes.Buffer{}
			require.NoError(t, Encode(buf, img, o))

			refPSNR := psnr(img, dwebp(t, dir, ref))
			ourPSNR := psnr(img, dwebp(t, dir, buf.Bytes()))
			ratio := float64(buf.Len()) / float64(len(ref))
			t.Logf("q%g m%d: libwebp %d bytes %.2f dB, ours %d bytes %.2f dB (size x%.3f)", q, m, len(ref), refPSNR, buf.Len(), ourPSNR, ratio)
			require.Greater(t, ourPSNR, refPSNR-0.5)
			require.Less(t, ratio, 1.10)
		}
	}
}

// TestInteropPhotoPreset checks PresetPhoto (dithered RGB->YUV conversion,
// Preprocessing bit 1) against cwebp -preset photo: the output size must
// stay close to libwebp's.
func TestInteropPhotoPreset(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"logo.jpg", "source.jpg"} {
		img := toNRGBA(loadJPEG(t, name))
		for _, q := range []float32{75, 95, 100} {
			ref := cwebp(t, dir, img, "-preset", "photo", "-q", fmt.Sprint(q))
			o, err := encoder.NewLossyEncoderOptions(encoder.PresetPhoto, q)
			require.NoError(t, err)
			buf := &bytes.Buffer{}
			require.NoError(t, Encode(buf, img, o))
			ratio := float64(buf.Len()) / float64(len(ref))
			t.Logf("%s photo q%g: libwebp %d bytes, ours %d bytes (x%.3f)", name, q, len(ref), buf.Len(), ratio)
			require.Less(t, ratio, 1.05, "%s q%g", name, q)
		}
	}
}

// TestInteropAlphaDitherCrop checks alpha dithering combined with cropping
// and scaling: libwebp smooths only the crop window of the alpha plane.
func TestInteropAlphaDitherCrop(t *testing.T) {
	dir := t.TempDir()
	const w, h = 129, 65
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := src.PixOffset(x, y)
			src.Pix[o+0] = uint8(2 * x)
			src.Pix[o+1] = uint8(3 * y)
			src.Pix[o+2] = uint8(x * y)
			src.Pix[o+3] = uint8((x*x + 3*y*y) / 40)
		}
	}
	file := cwebp(t, dir, src, "-q", "70", "-alpha_q", "50")
	for _, c := range [][4]int{{0, 0, 128, 64}, {w / 4, h / 4, w / 2, h / 2}, {3, 5, 100, 40}, {0, 0, w, h}} {
		want := dwebp(t, dir, file, "-alpha_dither", "-crop", fmt.Sprint(c[0]), fmt.Sprint(c[1]), fmt.Sprint(c[2]), fmt.Sprint(c[3]))
		got := decodeWith(t, file, &decoder.Options{
			AlphaDitheringStrength: 100,
			Crop:                   image.Rectangle{Min: image.Pt(c[0], c[1]), Max: image.Pt(c[2], c[3])},
		})
		requireSamePixels(t, want, got, 0, fmt.Sprintf("crop %v", c))

		want = dwebp(t, dir, file, "-alpha_dither", "-crop", fmt.Sprint(c[0]), fmt.Sprint(c[1]), fmt.Sprint(c[2]), fmt.Sprint(c[3]), "-resize", "40", "30")
		got = decodeWith(t, file, &decoder.Options{
			AlphaDitheringStrength: 100,
			Crop:                   image.Rectangle{Min: image.Pt(c[0], c[1]), Max: image.Pt(c[2], c[3])},
			Scale:                  image.Rectangle{Max: image.Pt(40, 30)},
		})
		requireSamePixels(t, want, got, 0, fmt.Sprintf("crop %v + scale", c))
	}
}
