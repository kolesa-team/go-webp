package vp8

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// goldenModes maps the mode names of testdata/golden.txt to options. The
// hashes are SHA-256 of libwebp 1.6 "dwebp -pam" RGBA output (with A forced
// to 255), generated with the corresponding dwebp flags.
var goldenModes = map[string]Options{
	"default":   {},
	"nofancy":   {NoFancyUpsampling: true},
	"nofilter":  {BypassFiltering: true},
	"dither100": {DitheringStrength: 100},
}

func fixturePath(name string) string {
	if p := filepath.Join("testdata", name); fileExists(p) {
		return p
	}
	return filepath.Join("..", "..", "test_data", "images", name)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func loadVP8(t testing.TB, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(fixturePath(name))
	if err != nil {
		t.Fatal(err)
	}
	ri, err := extractVP8(data)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return ri.vp8
}

// TestConformanceGolden compares the decoded fixtures with libwebp's output
// (testdata/golden.txt) at every SIMD level of the row converters, the loop
// filters and the reconstruction kernels (switched together, so each level
// runs as a whole decoder; one run on non-x86 machines).
func TestConformanceGolden(t *testing.T) {
	saved := filterLevel
	defer func() { filterLevel = saved }()
	withLevels(t, &yuvLevel, func(t *testing.T) {
		filterLevel = yuvLevel
		withReconLevel(yuvLevel, func() { testConformanceGolden(t) })
	})
}

func testConformanceGolden(t *testing.T) {
	f, err := os.Open("testdata/golden.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	n := 0
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) != 3 {
			continue
		}
		name, mode, want := fields[0], fields[1], fields[2]
		opts, ok := goldenModes[mode]
		if !ok {
			t.Fatalf("unknown mode %q", mode)
		}
		for _, threads := range []bool{false, true} {
			opts.UseThreads = threads
			img, err := decodeImage(loadVP8(t, name), nil, &opts)
			if err != nil {
				t.Errorf("%s [%s]: %v", name, mode, err)
				continue
			}
			sum := sha256.Sum256(img.Pix)
			if got := hex.EncodeToString(sum[:]); got != want {
				t.Errorf("%s [%s threads=%v]: sha256 %s, want %s", name, mode, threads, got, want)
			}
		}
		n++
	}
	if n < 40 {
		t.Fatalf("only %d golden entries", n)
	}
}

func TestDecodeHeader(t *testing.T) {
	vp8 := loadVP8(t, "mix_129x65_p2s.webp")
	w, h, err := DecodeHeader(vp8)
	if err != nil || w != 129 || h != 65 {
		t.Fatalf("DecodeHeader = %d, %d, %v", w, h, err)
	}

	bad := func(mut func([]byte) []byte) []byte {
		return mut(append([]byte(nil), vp8...))
	}
	cases := []struct {
		name string
		data []byte
		code utils.VP8StatusCode
	}{
		{"empty", nil, utils.Vp8StatusNotEnoughData},
		{"short", vp8[:9], utils.Vp8StatusNotEnoughData},
		{"signature", bad(func(b []byte) []byte { b[3] = 0; return b }), utils.Vp8StatusBitstreamError},
		{"interframe", bad(func(b []byte) []byte { b[0] |= 1; return b }), utils.Vp8StatusBitstreamError},
		{"profile", bad(func(b []byte) []byte { b[0] |= 7 << 1; b[0] &^= 1; return b }), utils.Vp8StatusBitstreamError},
		{"invisible", bad(func(b []byte) []byte { b[0] &^= 1 << 4; return b }), utils.Vp8StatusBitstreamError},
		{"zero width", bad(func(b []byte) []byte { b[6], b[7] = 0, 0; return b }), utils.Vp8StatusBitstreamError},
		{"zero height", bad(func(b []byte) []byte { b[8], b[9] = 0, 0xc0; return b }), utils.Vp8StatusBitstreamError},
		{"partition0 too big", bad(func(b []byte) []byte { b[0] |= 0xe0; b[1], b[2] = 0xff, 0xff; return b }), utils.Vp8StatusBitstreamError},
		{"truncated after partition 0", vp8[:10+partLen(vp8)], utils.Vp8StatusNotEnoughData},
	}
	for _, c := range cases {
		_, _, err := DecodeHeader(c.data)
		if strings.HasPrefix(c.name, "truncated") {
			if err != nil {
				t.Errorf("%s: DecodeHeader: %v", c.name, err)
			}
			err = Decode(c.data, nil, nil, pipeline.NewDirect(image.NewNRGBA(image.Rect(0, 0, 129, 65))))
		}
		if got := status.CodeOf(err); got != c.code {
			t.Errorf("%s: code %v (%v), want %v", c.name, got, err, c.code)
		}
		var se *status.Error
		if !errors.As(err, &se) {
			t.Errorf("%s: error %T is not *status.Error", c.name, err)
		}
		if err := Decode(c.data, nil, nil, pipeline.NewDirect(image.NewNRGBA(image.Rect(0, 0, 1, 1)))); status.CodeOf(err) != c.code {
			t.Errorf("%s: Decode code %v (%v), want %v", c.name, status.CodeOf(err), err, c.code)
		}
	}
}

func partLen(vp8 []byte) int {
	return int(uint32(vp8[0])|uint32(vp8[1])<<8|uint32(vp8[2])<<16) >> 5
}

func TestDecodeErrors(t *testing.T) {
	vp8 := loadVP8(t, "mix_129x65_p2s.webp")
	if err := Decode(vp8, nil, nil, nil); status.CodeOf(err) != utils.Vp8StatusInvalidParam {
		t.Errorf("nil sink: %v", err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 129, 65))
	if err := Decode(vp8, make([]byte, 10), nil, pipeline.NewDirect(img)); status.CodeOf(err) != utils.Vp8StatusInvalidParam {
		t.Errorf("short alpha: %v", err)
	}
	// Every truncation must fail cleanly (never panic) with a status error.
	for cut := 0; cut < len(vp8); cut += 1 + cut/8 {
		err := Decode(vp8[:cut], nil, nil, pipeline.NewDirect(img))
		if err == nil {
			t.Fatalf("cut %d: no error", cut)
		}
		var se *status.Error
		if !errors.As(err, &se) {
			t.Fatalf("cut %d: %T %v", cut, err, err)
		}
	}
	// A sink error aborts decoding and is returned as is.
	errSink := &failSink{Direct: pipeline.NewDirect(img), err: errors.New("boom")}
	if err := Decode(vp8, nil, nil, errSink); err != errSink.err {
		t.Errorf("sink error: %v", err)
	}
}

type failSink struct {
	*pipeline.Direct
	err error
}

func (f *failSink) Commit(y, n int) error { return f.err }

// recordingSink checks the row protocol and stops early when asked.
type recordingSink struct {
	t       *testing.T
	img     *image.NRGBA
	next    int
	need    int
	pending [2]int
}

func (s *recordingSink) Buffer(y, n int) []byte {
	if y != s.next || n <= 0 || y+n > s.img.Rect.Dy() {
		s.t.Fatalf("Buffer(%d, %d), want start %d", y, n, s.next)
	}
	s.pending = [2]int{y, n}
	return s.img.Pix[y*s.img.Stride : (y+n)*s.img.Stride]
}
func (s *recordingSink) Stride() int { return s.img.Stride }
func (s *recordingSink) Commit(y, n int) error {
	if [2]int{y, n} != s.pending {
		s.t.Fatalf("Commit(%d, %d) does not match Buffer %v", y, n, s.pending)
	}
	s.next = y + n
	return nil
}
func (s *recordingSink) NeedRows() int { return s.need }

func TestSinkProtocolAndEarlyStop(t *testing.T) {
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	for _, opts := range []Options{{}, {NoFancyUpsampling: true}, {BypassFiltering: true}, {UseThreads: true}} {
		full, err := decodeImage(vp8, nil, &opts)
		if err != nil {
			t.Fatal(err)
		}
		for _, need := range []int{1, 2, 15, 16, 17, 40, 100, 198, 199} {
			s := &recordingSink{t: t, img: image.NewNRGBA(full.Rect), need: need}
			if err := Decode(vp8, nil, &opts, s); err != nil {
				t.Fatal(err)
			}
			if s.next < need {
				t.Fatalf("need %d: only %d rows produced", need, s.next)
			}
			if need < 150 && s.next >= 199 {
				t.Errorf("need %d: decoder did not stop early (%d rows)", need, s.next)
			}
			n := s.next * full.Stride
			if !bytes.Equal(s.img.Pix[:n], full.Pix[:n]) {
				t.Errorf("opts %+v need %d: rows differ from a full decode", opts, need)
			}
		}
	}
}

func TestAlphaPlane(t *testing.T) {
	vp8 := loadVP8(t, "mix_31x17_o8.webp")
	opaque, err := decodeImage(vp8, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	alpha := make([]byte, 31*17)
	for i := range alpha {
		alpha[i] = byte(i * 7)
	}
	for _, opts := range []Options{{}, {NoFancyUpsampling: true}} {
		img, err := decodeImage(vp8, alpha, &opts)
		if err != nil {
			t.Fatal(err)
		}
		ref := opaque
		if opts.NoFancyUpsampling {
			if ref, err = decodeImage(vp8, nil, &opts); err != nil {
				t.Fatal(err)
			}
		}
		for i := 0; i < 31*17; i++ {
			if img.Pix[4*i+3] != alpha[i] {
				t.Fatalf("pixel %d: alpha %d, want %d", i, img.Pix[4*i+3], alpha[i])
			}
			if !bytes.Equal(img.Pix[4*i:4*i+3], ref.Pix[4*i:4*i+3]) {
				t.Fatalf("pixel %d: RGB changed by alpha", i)
			}
			if ref.Pix[4*i+3] != 255 {
				t.Fatalf("pixel %d: opaque alpha %d", i, ref.Pix[4*i+3])
			}
		}
	}
}

// TestPoolReuse decodes images of different sizes and options back to back
// so that pooled decoder state is reused, and checks the results stay
// identical to fresh decodes.
func TestPoolReuse(t *testing.T) {
	names := []string{"mix_250x199_o4.webp", "mix_1x1_o0.webp", "noise_1000x1_o2.webp", "mix_1x300_o6.webp", "grad_129x65_o7.webp"}
	want := map[string][]byte{}
	for _, n := range names {
		img, err := decodeImage(loadVP8(t, n), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want[n] = img.Pix
	}
	for round := 0; round < 3; round++ {
		for i, n := range names {
			opts := &Options{DitheringStrength: 100 * (i & 1)}
			if _, err := decodeImage(loadVP8(t, n), nil, opts); err != nil {
				t.Fatal(err)
			}
			img, err := decodeImage(loadVP8(t, n), nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(img.Pix, want[n]) {
				t.Fatalf("round %d %s: output changed after pool reuse", round, n)
			}
		}
	}
}

func TestDitheringStrengthRange(t *testing.T) {
	vp8 := loadVP8(t, "grad_129x65_o7.webp")
	base, _ := decodeImage(vp8, nil, nil)
	neg, _ := decodeImage(vp8, nil, &Options{DitheringStrength: -5})
	if !bytes.Equal(base.Pix, neg.Pix) {
		t.Error("negative dithering strength changed the output")
	}
	d100, _ := decodeImage(vp8, nil, &Options{DitheringStrength: 100})
	d500, _ := decodeImage(vp8, nil, &Options{DitheringStrength: 500})
	if !bytes.Equal(d100.Pix, d500.Pix) {
		t.Error("dithering strength above 100 should behave like 100")
	}
	if bytes.Equal(base.Pix, d100.Pix) {
		t.Error("dithering had no effect on a high quality image")
	}
}

func TestDecodeAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool drops items at random under the race detector")
	}
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	img := image.NewNRGBA(image.Rect(0, 0, 250, 199))
	sink := pipeline.NewDirect(img)
	if err := Decode(vp8, nil, nil, sink); err != nil { // warm up the pool
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := Decode(vp8, nil, nil, sink); err != nil {
			t.Fatal(err)
		}
	})
	if allocs > 1 {
		t.Errorf("Decode allocates %.1f times per call", allocs)
	}
}

// yuvCollector rebuilds an RGBA image from DecodeYUV rows with the
// point-sampling converter, to check the YUV plumbing against Decode.
type yuvCollector struct {
	img  *image.NRGBA
	next int
	t    *testing.T
}

func (c *yuvCollector) PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error {
	if y != c.next || y&1 != 0 {
		c.t.Fatalf("PutYUV(%d, %d), want y=%d (even)", y, n, c.next)
	}
	w := c.img.Rect.Dx()
	for j := 0; j < n; j++ {
		k := (j >> 1) * uvStride
		sampleRow(ys[j*yStride:], us[k:], vs[k:], nil, c.img.Pix[(y+j)*c.img.Stride:], w)
	}
	c.next = y + n
	return nil
}

func (c *yuvCollector) NeedRows() int { return c.img.Rect.Dy() }

func TestDecodeYUVAndRegion(t *testing.T) {
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	opts := &Options{NoFancyUpsampling: true}
	full, err := decodeImage(vp8, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	c := &yuvCollector{img: image.NewNRGBA(full.Rect), t: t}
	if err := DecodeYUV(vp8, nil, nil, c); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(c.img.Pix, full.Pix) {
		t.Error("DecodeYUV + point sampling differs from Decode -nofancy")
	}
	// With point sampling and an even origin, a region is a plain crop.
	r := image.Rect(10, 20, 101, 77)
	got := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	if err := DecodeRegion(vp8, nil, opts, r, pipeline.NewDirect(got)); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < r.Dy(); y++ {
		a := got.Pix[y*got.Stride : y*got.Stride+4*r.Dx()]
		b := full.Pix[(y+r.Min.Y)*full.Stride+4*r.Min.X:][:4*r.Dx()]
		if !bytes.Equal(a, b) {
			t.Fatalf("region row %d differs", y)
		}
	}
	for _, bad := range []image.Rectangle{image.Rect(-2, 0, 5, 5), image.Rect(0, 0, 251, 10), image.Rect(0, 0, 0, 10), image.Rect(248, 0, 252, 10)} {
		err := DecodeRegion(vp8, nil, nil, bad, pipeline.NewDirect(image.NewNRGBA(image.Rect(0, 0, 4, 10))))
		if status.CodeOf(err) != utils.Vp8StatusInvalidParam {
			t.Errorf("crop %v: %v", bad, err)
		}
	}
}
