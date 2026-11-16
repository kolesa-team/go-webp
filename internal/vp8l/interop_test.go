//go:build interop

package vp8l

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestInteropCorpus compares the decoder output against dwebp -pam for every
// *.webp file (with a matching *.pam reference) in $VP8L_CORPUS.
func TestInteropCorpus(t *testing.T) {
	dir := os.Getenv("VP8L_CORPUS")
	if dir == "" {
		t.Skip("VP8L_CORPUS not set")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.webp"))
	if len(files) == 0 {
		t.Fatal("empty corpus")
	}
	n := 0
	for _, f := range files {
		ref, err := os.ReadFile(strings.TrimSuffix(f, ".webp") + ".pam")
		if err != nil {
			continue
		}
		file, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		payload := findChunk(file, "VP8L")
		if payload == nil {
			continue
		}
		n++
		img, err := decodeVP8L(payload)
		if err != nil {
			t.Errorf("%s: %v", filepath.Base(f), err)
			continue
		}
		w, h, pix, err := parsePAM(ref)
		if err != nil {
			t.Fatal(err)
		}
		if w != img.Rect.Dx() || h != img.Rect.Dy() {
			t.Errorf("%s: size %dx%d, want %dx%d", filepath.Base(f), img.Rect.Dx(), img.Rect.Dy(), w, h)
			continue
		}
		if !bytes.Equal(img.Pix, pix) {
			for i := range pix {
				if pix[i] != img.Pix[i] {
					t.Errorf("%s: first mismatch at pixel (%d,%d) chan %d: got %d want %d", filepath.Base(f), (i/4)%w, (i/4)/w, i%4, img.Pix[i], pix[i])
					break
				}
			}
		}
	}
	t.Logf("checked %d files", n)
}

// TestInteropAlpha decodes the ALPH chunks (method 1) of the lossy files in
// $VP8L_ALPHA_CORPUS and compares the unfiltered result with the alpha
// channel of dwebp -pam.
func TestInteropAlpha(t *testing.T) {
	dir := os.Getenv("VP8L_ALPHA_CORPUS")
	if dir == "" {
		t.Skip("VP8L_ALPHA_CORPUS not set")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.webp"))
	n := 0
	for _, f := range files {
		ref, err := os.ReadFile(strings.TrimSuffix(f, ".webp") + ".pam")
		if err != nil {
			continue
		}
		file, _ := os.ReadFile(f)
		hdr, payload, w, h, ok := alphaFromFile(file)
		if !ok || hdr&3 != 1 {
			continue
		}
		n++
		dst := make([]byte, w*h)
		if err := DecodeAlpha(payload, w, h, dst); err != nil {
			t.Errorf("%s: %v", filepath.Base(f), err)
			continue
		}
		unfilterAlpha(dst, w, h, int(hdr>>2&3))
		_, _, pix, err := parsePAM(ref)
		if err != nil {
			t.Fatal(err)
		}
		for i := range dst {
			if dst[i] != pix[4*i+3] {
				t.Errorf("%s (filter %d): first mismatch at %d: got %d want %d", filepath.Base(f), hdr>>2&3, i, dst[i], pix[4*i+3])
				break
			}
		}
	}
	t.Logf("checked %d files", n)
}

// parsePAM parses a P7 RGB_ALPHA PAM file as written by dwebp -pam.
func parsePAM(b []byte) (w, h int, pix []byte, err error) {
	i := bytes.Index(b, []byte("ENDHDR\n"))
	if i < 0 || !bytes.HasPrefix(b, []byte("P7\n")) {
		return 0, 0, nil, fmt.Errorf("not a PAM file")
	}
	for _, line := range strings.Split(string(b[3:i]), "\n") {
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		v, _ := strconv.Atoi(f[1])
		switch f[0] {
		case "WIDTH":
			w = v
		case "HEIGHT":
			h = v
		case "DEPTH":
			if v != 4 {
				return 0, 0, nil, fmt.Errorf("depth %d", v)
			}
		}
	}
	pix = b[i+7:]
	if len(pix) != w*h*4 {
		return 0, 0, nil, fmt.Errorf("pam size mismatch")
	}
	return w, h, pix, nil
}
