//go:build interop

package vp8lenc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// wrapRIFF wraps a VP8L payload into a RIFF/WEBP container.
func wrapRIFF(payload []byte) []byte {
	var b bytes.Buffer
	chunk := uint32(len(payload))
	pad := int(chunk & 1)
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(4+8+len(payload)+pad))
	b.WriteString("WEBPVP8L")
	binary.Write(&b, binary.LittleEndian, chunk)
	b.Write(payload)
	if pad == 1 {
		b.WriteByte(0)
	}
	return b.Bytes()
}

// dwebpDecode decodes a RIFF file with dwebp into raw RGBA.
func dwebpDecode(t *testing.T, data []byte) (int, int, []byte) {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "in.webp")
	out := filepath.Join(dir, "out.pam")
	if err := os.WriteFile(in, data, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("dwebp", in, "-pam", "-o", out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("dwebp failed: %v\n%s", err, msg)
	}
	pam, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	end := bytes.Index(pam, []byte("ENDHDR\n"))
	if end < 0 {
		t.Fatal("bad pam")
	}
	var w, h int
	for _, line := range strings.Split(string(pam[:end]), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && f[0] == "WIDTH" {
			w, _ = strconv.Atoi(f[1])
		}
		if len(f) == 2 && f[0] == "HEIGHT" {
			h, _ = strconv.Atoi(f[1])
		}
	}
	return w, h, pam[end+7:]
}

func expectedRGBA(img *image.NRGBA, exact bool) []byte {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	out := make([]byte, 0, 4*w*h)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+4*w]
		for x := 0; x < w; x++ {
			p := row[4*x : 4*x+4]
			if p[3] == 0 && !exact {
				out = append(out, 0, 0, 0, 0)
			} else {
				out = append(out, p...)
			}
		}
	}
	return out
}

var interopLevels = []struct{ method, quality int }{
	{0, 0}, {1, 20}, {2, 25}, {3, 30}, {3, 50}, {4, 50}, {4, 75}, {4, 90}, {5, 90}, {6, 100},
}

func TestInteropDwebp(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not installed")
	}
	imgs := syntheticImages()
	if extra := os.Getenv("VP8LENC_EXTRA"); extra != "" {
		for _, p := range strings.Split(extra, ",") {
			f, err := os.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			m, _, err := image.Decode(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			n := image.NewNRGBA(m.Bounds())
			for y := m.Bounds().Min.Y; y < m.Bounds().Max.Y; y++ {
				for x := m.Bounds().Min.X; x < m.Bounds().Max.X; x++ {
					n.Set(x, y, m.At(x, y))
				}
			}
			imgs = append(imgs, testImage{filepath.Base(p), n})
		}
	}
	for i, ti := range imgs {
		for _, lv := range interopLevels {
			for _, exact := range []bool{false, true} {
				if !exact && lv.method%2 == 1 {
					continue
				}
				name := fmt.Sprintf("%d-%s-%dx%d-m%dq%d-exact%v", i, ti.name, ti.img.Rect.Dx(), ti.img.Rect.Dy(), lv.method, lv.quality, exact)
				cfg := &Config{Quality: float32(lv.quality), Method: lv.method, Exact: exact, NearLossless: 100, ThreadLevel: true}
				payload, err := Encode(ti.img, cfg)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				w, h, got := dwebpDecode(t, wrapRIFF(payload))
				if w != ti.img.Rect.Dx() || h != ti.img.Rect.Dy() {
					t.Fatalf("%s: size %dx%d", name, w, h)
				}
				want := expectedRGBA(ti.img, exact)
				if !exact {
					// Like libwebp, exact=0 leaves arbitrary RGB under
					// fully transparent pixels: only compare alpha there.
					for k := 0; k < len(want); k += 4 {
						if want[k+3] == 0 && got[k+3] == 0 {
							copy(got[k:k+3], want[k:k+3])
						}
					}
				}
				if !bytes.Equal(got, want) {
					for k := range want {
						if got[k] != want[k] {
							t.Fatalf("%s: mismatch at pixel %d (%d,%d) got %v want %v", name, k/4, (k/4)%w, (k/4)/w, got[k/4*4:k/4*4+4], want[k/4*4:k/4*4+4])
						}
					}
				}
			}
		}
	}
}

// TestInteropTransparentSize compares the size of exact=0 encodes of images
// with random RGB under fully transparent areas against cwebp -lossless.
func TestInteropTransparentSize(t *testing.T) {
	if _, err := exec.LookPath("cwebp"); err != nil {
		t.Skip("cwebp not installed")
	}
	dir := t.TempDir()
	for _, sz := range [][2]int{{129, 65}, {250, 199}, {1000, 1}} {
		w, h := sz[0], sz[1]
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		seed := uint32(12345)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				seed = seed*1664525 + 1013904223
				o := img.PixOffset(x, y)
				a := uint8(255 * x / max(w-1, 1))
				if (x/8+y/8)%3 == 0 {
					a = 0
				}
				img.Pix[o] = uint8(x * 255 / w)
				img.Pix[o+1] = uint8(y * 255 / h)
				img.Pix[o+2] = 128
				if a == 0 {
					img.Pix[o], img.Pix[o+1], img.Pix[o+2] = uint8(seed>>24), uint8(seed>>16), uint8(seed>>8)
				}
				img.Pix[o+3] = a
			}
		}
		in := filepath.Join(dir, "in.png")
		f, err := os.Create(in)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		f.Close()
		for _, z := range []int{1, 3, 6, 9} {
			out := filepath.Join(dir, "out.webp")
			if msg, err := exec.Command("cwebp", "-quiet", "-lossless", "-z", strconv.Itoa(z), in, "-o", out).CombinedOutput(); err != nil {
				t.Fatalf("cwebp: %v %s", err, msg)
			}
			ref, err := os.ReadFile(out)
			if err != nil {
				t.Fatal(err)
			}
			m, q := zLevel(z)
			payload, err := Encode(img, &Config{Quality: float32(q), Method: m, NearLossless: 100})
			if err != nil {
				t.Fatal(err)
			}
			ratio := float64(len(payload)+20) / float64(len(ref))
			t.Logf("%dx%d z%d: libwebp %d, ours %d (x%.3f)", w, h, z, len(ref), len(payload)+20, ratio)
			if ratio > 1.05 {
				t.Errorf("%dx%d z%d: %d bytes vs libwebp %d", w, h, z, len(payload)+20, len(ref))
			}
		}
	}
}

// zLevel maps cwebp's -z level to (method, quality) like
// WebPConfigLosslessPreset.
func zLevel(z int) (int, int) {
	methods := [10]int{0, 1, 2, 3, 3, 4, 4, 4, 4, 6}
	qualities := [10]int{0, 20, 25, 30, 50, 50, 75, 90, 90, 100}
	return methods[z], qualities[z]
}

func TestInteropAlpha(t *testing.T) {
	if _, err := exec.LookPath("dwebp"); err != nil {
		t.Skip("dwebp not installed")
	}
	for i, ti := range syntheticImages() {
		w, h := ti.img.Rect.Dx(), ti.img.Rect.Dy()
		plane := make([]byte, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				plane[y*w+x] = ti.img.Pix[y*ti.img.Stride+4*x+1] // use green as alpha source
			}
		}
		for _, m := range []int{0, 1, 4, 6} {
			stream, err := EncodeAlpha(plane, w, h, w, m, 100)
			if err != nil {
				t.Fatal(err)
			}
			hdr := []byte{vp8lSignature, 0, 0, 0, 0}
			v := uint32(w-1) | uint32(h-1)<<14
			binary.LittleEndian.PutUint32(hdr[1:], v)
			_, _, got := dwebpDecode(t, wrapRIFF(append(hdr, stream...)))
			for k := 0; k < w*h; k++ {
				if got[4*k+1] != plane[k] || got[4*k] != 0 || got[4*k+2] != 0 {
					t.Fatalf("%d %s m%d: alpha mismatch at %d", i, ti.name, m, k)
				}
			}
		}
	}
}
