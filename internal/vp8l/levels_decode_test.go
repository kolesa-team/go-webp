package vp8l

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

// End-to-end differential tests of the kernel configurations (simdLevels):
// whole pictures decoded at every SIMD level must give the output and the
// error of the pure-Go level, so that the wrappers are checked together
// with the transforms that call them (tile boundaries, mode 13 runs,
// in-place rows, the fused output step, the 8-bit alpha path). On other
// architectures there is one configuration, and the tests only decode.

// levelsCheck decodes payload (RGBA, and with threads also with
// Options.UseThreads) at every kernel configuration and compares with the
// first one, the pure-Go level.
func levelsCheck(t *testing.T, name string, payload []byte, threads bool) {
	t.Helper()
	w, h, _, err := DecodeHeader(payload)
	if err != nil {
		return
	}
	decode := func(l simdLevel, useThreads bool) ([]byte, error) {
		defer setLevel(l)()
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		err := DecodeWithOptions(payload, pipeline.NewDirect(img), &Options{UseThreads: useThreads})
		return img.Pix, err
	}
	ls := simdLevels()
	want, wantErr := decode(ls[0], false)
	for _, l := range ls {
		for _, th := range []bool{false, true} {
			if th && !threads {
				continue
			}
			got, err := decode(l, th)
			if fmt.Sprint(err) != fmt.Sprint(wantErr) {
				t.Fatalf("%s at %s (threads %v): error %v, want %v", name, l.name, th, err, wantErr)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s at %s (threads %v): output differs from the Go level", name, l.name, th)
			}
		}
	}
}

// alphaLevelsCheck is levelsCheck for an ALPH payload of a w x h plane.
func alphaLevelsCheck(t *testing.T, name string, payload []byte, w, h int) {
	t.Helper()
	ls := simdLevels()
	var want []byte
	var wantErr error
	for i, l := range ls {
		got := make([]byte, w*h)
		var err error
		func() {
			defer setLevel(l)()
			err = DecodeAlpha(payload, w, h, got)
		}()
		if i == 0 {
			want, wantErr = got, err
			continue
		}
		if fmt.Sprint(err) != fmt.Sprint(wantErr) || !bytes.Equal(got, want) {
			t.Fatalf("%s alpha at %s: output or error (%v, want %v) differs from the Go level", name, l.name, err, wantErr)
		}
	}
}

// TestDecodeLevels decodes the lossless test pictures, whole and
// truncated, their alpha planes, the 3000x2000 lossless logo (256 colors)
// and a lossless encode of a photo (every predictor mode, cross-color,
// subtract-green) at every kernel configuration.
func TestDecodeLevels(t *testing.T) {
	type pic struct {
		name    string
		payload []byte
		threads bool
	}
	var pics []pic
	for _, dir := range []string{"lossless", "alpha"} {
		files, _ := filepath.Glob(filepath.Join("testdata", dir, "*.webp"))
		for _, name := range files {
			b, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			if p := findChunk(b, "VP8L"); p != nil {
				pics = append(pics, pic{name, p, false}, pic{name + "[:2/3]", p[:len(p)*2/3], false})
			}
			if _, p, w, h, ok := alphaFromFile(b); ok {
				alphaLevelsCheck(t, name, p, w, h)
				alphaLevelsCheck(t, name+"[:2/3]", p[:len(p)*2/3], w, h)
			}
		}
	}
	if b, err := os.ReadFile("../../test_data/images/webp-logo-lossless.webp"); err == nil {
		pics = append(pics, pic{"logo", findChunk(b, "VP8L"), true})
	}
	if !testing.Short() {
		if p, err := photoPayload(); err == nil {
			pics = append(pics, pic{"photo", p, true})
		}
	}
	if len(pics) == 0 {
		t.Skip("no test pictures")
	}
	for _, p := range pics {
		levelsCheck(t, p.name, p.payload, p.threads)
	}
}

// FuzzDecodeLevels decodes arbitrary streams at every kernel configuration
// and compares with the pure-Go level.
func FuzzDecodeLevels(f *testing.F) {
	addSeeds(f, func(p []byte, _, _ int) { f.Add(p) })
	f.Fuzz(func(t *testing.T, data []byte) {
		w, h, _, err := DecodeHeader(data)
		if err != nil || w*h > 1<<18 {
			return
		}
		levelsCheck(t, "fuzz", data, false)
	})
}
