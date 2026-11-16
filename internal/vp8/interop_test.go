//go:build interop

package vp8

// Interop tests compare the decoder with libwebp's dwebp on a corpus of
// lossy WebP files. They need dwebp in $PATH. The corpus directory is taken
// from $VP8_CORPUS (all *.webp files in it), plus the repository's
// test_data/images lossy samples.
//
//	VP8_CORPUS=/path/to/corpus go test -tags interop ./internal/vp8/

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

var statusRE = regexp.MustCompile(`Status: (\d+)`)

type interopMode struct {
	name  string
	flags []string
	opts  Options
}

var interopModes = []interopMode{
	{"default", nil, Options{}},
	{"nofancy", []string{"-nofancy"}, Options{NoFancyUpsampling: true}},
	{"nofilter", []string{"-nofilter"}, Options{BypassFiltering: true}},
	{"nofancy-nofilter", []string{"-nofancy", "-nofilter"}, Options{NoFancyUpsampling: true, BypassFiltering: true}},
	{"dither50", []string{"-dither", "50"}, Options{DitheringStrength: 50}},
	{"dither100", []string{"-dither", "100"}, Options{DitheringStrength: 100}},
	{"threads", nil, Options{UseThreads: true}},
	{"threads-nofancy-dither", []string{"-nofancy", "-dither", "70"}, Options{UseThreads: true, NoFancyUpsampling: true, DitheringStrength: 70}},
}

func interopFiles(t *testing.T) []string {
	files := []string{
		"../../test_data/images/m4_q75.webp",
		"../../test_data/images/webp-logo-lossy.webp",
		"../../test_data/images/100x150_lossy.webp",
	}
	if dir := os.Getenv("VP8_CORPUS"); dir != "" {
		m, err := filepath.Glob(filepath.Join(dir, "*.webp"))
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	return files
}

func TestInteropDwebp(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not found")
	}
	tmp := t.TempDir()
	for _, f := range interopFiles(t) {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		ri, err := extractVP8(data)
		if err != nil {
			continue // lossless or broken file: not ours
		}
		for _, m := range interopModes {
			out := filepath.Join(tmp, "out.pam")
			args := append(append([]string{}, m.flags...), "-pam", "-quiet", f, "-o", out)
			if msg, err := exec.Command("dwebp", args...).CombinedOutput(); err != nil {
				t.Fatalf("dwebp %v: %v %s", args, err, msg)
			}
			pam, err := os.ReadFile(out)
			if err != nil {
				t.Fatal(err)
			}
			want, err := readPAM(pam)
			if err != nil {
				t.Fatal(err)
			}
			var alpha []byte
			if ri.hasAlpha {
				alpha = alphaPlane(want)
			}
			opts := m.opts
			got, err := decodeImage(ri.vp8, alpha, &opts)
			if err != nil {
				t.Errorf("%s [%s]: %v", filepath.Base(f), m.name, err)
				continue
			}
			if d := firstDiff(got, want); d != "" {
				t.Errorf("%s [%s]: %s", filepath.Base(f), m.name, d)
			} else if !bytes.Equal(got.Pix, want.Pix) {
				t.Errorf("%s [%s]: mismatch", filepath.Base(f), m.name)
			}
		}
	}
}

// TestInteropTruncated checks that truncated streams fail (or succeed)
// exactly when dwebp does.
func TestInteropTruncated(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not found")
	}
	// Status codes of truncated partitions changed in libwebp 1.6.0 (1.3.2
	// reports SUSPENDED where 1.6 reports NOT_ENOUGH_DATA); older tools
	// are only checked for agreement on success/failure.
	checkStatus := dwebpAtLeast(t, 1, 6)
	tmp := t.TempDir()
	data, err := os.ReadFile("../../test_data/images/webp-logo-lossy.webp")
	if err != nil {
		t.Fatal(err)
	}
	ri, err := extractVP8(data)
	if err != nil {
		t.Fatal(err)
	}
	// Note: odd sizes below 10 are avoided: libwebp then also reads the RIFF
	// padding byte as part of the frame header.
	cuts := []int{5, 8, 10, 11, 12, 20, 50, 100, len(ri.vp8) / 3, len(ri.vp8) / 2, len(ri.vp8) - 100, len(ri.vp8) - 1}
	if pl := int(uint32(ri.vp8[0])|uint32(ri.vp8[1])<<8|uint32(ri.vp8[2])<<16) >> 5; pl+12 < len(ri.vp8) {
		cuts = append(cuts, pl+9, pl+10, pl+11, pl+12)
	}
	for _, cut := range cuts {
		if cut <= 0 || cut >= len(ri.vp8) {
			continue
		}
		payload := ri.vp8[:cut]
		riff := makeRIFF(payload)
		in := filepath.Join(tmp, "in.webp")
		if err := os.WriteFile(in, riff, 0o644); err != nil {
			t.Fatal(err)
		}
		msg, derr := exec.Command("dwebp", "-pam", in, "-o", filepath.Join(tmp, "o.pam")).CombinedOutput()
		_, gerr := decodeImage(payload, nil, nil)
		if (derr == nil) != (gerr == nil) {
			t.Errorf("cut %d: dwebp err=%v (%s), ours=%v", cut, derr, strings.TrimSpace(string(msg)), gerr)
			continue
		}
		if m := statusRE.FindSubmatch(msg); m != nil && checkStatus {
			want, _ := strconv.Atoi(string(m[1]))
			if got := int(status.CodeOf(gerr)); got != want {
				t.Errorf("cut %d: status %d (%v), dwebp %d", cut, got, gerr, want)
			}
		}
	}
}

// dwebpAtLeast reports whether dwebp's version is at least major.minor.
func dwebpAtLeast(t *testing.T, major, minor int) bool {
	out, err := exec.Command("dwebp", "-version").Output()
	if err != nil {
		t.Logf("dwebp -version: %v", err)
		return false
	}
	var ma, mi int
	if _, err := fmt.Sscanf(strings.TrimSpace(string(out)), "%d.%d", &ma, &mi); err != nil {
		t.Logf("dwebp -version %q: %v", out, err)
		return false
	}
	if ma != major {
		return ma > major
	}
	if mi < minor {
		t.Logf("dwebp %d.%d is older than %d.%d: status codes not compared", ma, mi, major, minor)
		return false
	}
	return true
}

func makeRIFF(vp8 []byte) []byte {
	pad := len(vp8) & 1
	n := 4 + 8 + len(vp8) + pad
	b := make([]byte, 0, 8+n)
	le := func(v int) []byte { return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)} }
	b = append(b, "RIFF"...)
	b = append(b, le(n)...)
	b = append(b, "WEBPVP8 "...)
	b = append(b, le(len(vp8))...)
	b = append(b, vp8...)
	if pad != 0 {
		b = append(b, 0)
	}
	return b
}

// TestInteropCrop compares DecodeRegion with dwebp -crop.
func TestInteropCrop(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not found")
	}
	tmp := t.TempDir()
	files := interopFiles(t)
	if len(files) > 40 {
		files = files[:40]
	}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		ri, err := extractVP8(data)
		if err != nil {
			continue
		}
		w, h, err := DecodeHeader(ri.vp8)
		if err != nil {
			t.Fatal(err)
		}
		rects := [][4]int{{0, 0, w, h}, {1, 1, w - 1, h - 1}, {w / 3, h / 4, w / 2, h / 2}, {w / 2, h / 2, w - w/2, h - h/2}, {3, 5, 1, 1}, {w - 1, h - 1, 1, 1}}
		for _, r := range rects {
			x, y, cw, ch := r[0], r[1], r[2], r[3]
			if cw <= 0 || ch <= 0 || x+cw > w || y+ch > h {
				continue
			}
			for _, m := range interopModes[:5] {
				out := filepath.Join(tmp, "out.pam")
				args := append(append([]string{}, m.flags...), "-crop", strconv.Itoa(x), strconv.Itoa(y), strconv.Itoa(cw), strconv.Itoa(ch), "-pam", "-quiet", f, "-o", out)
				if msg, err := exec.Command("dwebp", args...).CombinedOutput(); err != nil {
					t.Fatalf("dwebp %v: %v %s", args, err, msg)
				}
				pam, err := os.ReadFile(out)
				if err != nil {
					t.Fatal(err)
				}
				want, err := readPAM(pam)
				if err != nil {
					t.Fatal(err)
				}
				var alpha []byte
				if ri.hasAlpha {
					// dwebp only gives us the cropped alpha: rebuild a full plane.
					full, err := exec.Command("dwebp", "-pam", "-quiet", f, "-o", out).CombinedOutput()
					if err != nil {
						t.Fatal(err, string(full))
					}
					pam, _ := os.ReadFile(out)
					fimg, _ := readPAM(pam)
					alpha = alphaPlane(fimg)
				}
				got := image.NewNRGBA(image.Rect(0, 0, cw, ch))
				opts := m.opts
				if err := DecodeRegion(ri.vp8, alpha, &opts, image.Rect(x, y, x+cw, y+ch), pipeline.NewDirect(got)); err != nil {
					t.Errorf("%s crop %v [%s]: %v", filepath.Base(f), r, m.name, err)
					continue
				}
				if d := firstDiff(got, want); d != "" {
					t.Errorf("%s crop %v [%s]: %s", filepath.Base(f), r, m.name, d)
				}
			}
		}
	}
}
