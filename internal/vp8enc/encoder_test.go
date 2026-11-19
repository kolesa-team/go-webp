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

package vp8enc

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func loadTestImage(tb testing.TB, name string) *image.NRGBA {
	tb.Helper()
	f, err := os.Open("../../test_data/images/" + name)
	if err != nil {
		tb.Skipf("test image not available: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		tb.Fatal(err)
	}
	n := image.NewNRGBA(img.Bounds())
	draw.Draw(n, n.Bounds(), img, img.Bounds().Min, draw.Src)
	return n
}

func synthImage(w, h, kind int, seed int64) *image.NRGBA {
	rng := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var c color.NRGBA
			switch kind {
			case 0: // gradient
				c = color.NRGBA{uint8(x * 255 / max(w-1, 1)), uint8(y * 255 / max(h-1, 1)), uint8((x + y) & 255), 255}
			case 1: // noise
				c = color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255}
			case 2: // flat areas and sharp edges
				v := uint8(0)
				if (x/7+y/5)&1 == 0 {
					v = 255
				}
				c = color.NRGBA{v, 128, 255 - v, 255}
			default: // smooth, with transparent blocks
				v := 128 + 120*math.Sin(float64(x)/9)*math.Cos(float64(y)/13)
				a := uint8(255)
				if (x/16+y/16)&1 == 0 {
					a = 0
				} else if (x+y)%7 == 0 {
					a = 128
				}
				c = color.NRGBA{uint8(v), uint8(255 - v), uint8(v / 2), a}
			}
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// parsedFrame is the result of a light parse of the VP8 frame header.
type parsedFrame struct {
	keyFrame  bool
	profile   int
	show      bool
	size0     int
	width     int
	height    int
	numParts  int
	partSizes []int
}

// parseFrame checks the structure of an encoded VP8 payload: frame tag,
// start code, dimensions and partition layout.
func parseFrame(t *testing.T, data []byte) parsedFrame {
	t.Helper()
	if len(data) < 10 {
		t.Fatalf("payload too short: %d", len(data))
	}
	bits := uint32(data[0]) | uint32(data[1])<<8 | uint32(data[2])<<16
	f := parsedFrame{
		keyFrame: bits&1 == 0,
		profile:  int(bits>>1) & 7,
		show:     bits>>4&1 == 1,
		size0:    int(bits >> 5),
	}
	if data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a {
		t.Fatalf("bad start code % x", data[3:6])
	}
	f.width = int(uint16(data[6])|uint16(data[7])<<8) & 0x3fff
	f.height = int(uint16(data[8])|uint16(data[9])<<8) & 0x3fff
	if 10+f.size0 > len(data) {
		t.Fatalf("partition 0 overflows the payload")
	}
	// The number of partitions is coded after the segment and filter
	// headers; read it with a minimal boolean decoder.
	br := newTestBoolReader(data[10 : 10+f.size0])
	br.readBit(128)           // color space
	br.readBit(128)           // clamping
	if br.readBit(128) == 1 { // segmentation
		updateMap := br.readBit(128)
		if br.readBit(128) == 1 { // update data
			br.readBit(128)
			for i := 0; i < 4; i++ {
				if br.readBit(128) == 1 {
					br.readBits(7)
					br.readBit(128)
				}
			}
			for i := 0; i < 4; i++ {
				if br.readBit(128) == 1 {
					br.readBits(6)
					br.readBit(128)
				}
			}
		}
		if updateMap == 1 {
			for i := 0; i < 3; i++ {
				if br.readBit(128) == 1 {
					br.readBits(8)
				}
			}
		}
	}
	br.readBit(128) // filter type
	br.readBits(6)  // level
	br.readBits(3)  // sharpness
	if br.readBit(128) == 1 {
		if br.readBit(128) == 1 {
			for i := 0; i < 8; i++ {
				if br.readBit(128) == 1 {
					br.readBits(7)
				}
			}
		}
	}
	f.numParts = 1 << br.readBits(2)
	rest := data[10+f.size0:]
	n := 3 * (f.numParts - 1)
	if len(rest) < n {
		t.Fatalf("missing partition sizes")
	}
	total := 0
	for p := 0; p < f.numParts-1; p++ {
		s := int(rest[3*p]) | int(rest[3*p+1])<<8 | int(rest[3*p+2])<<16
		f.partSizes = append(f.partSizes, s)
		total += s
	}
	if total > len(rest)-n {
		t.Fatalf("partitions overflow the payload")
	}
	f.partSizes = append(f.partSizes, len(rest)-n-total)
	return f
}

// testBoolReader is a minimal VP8 boolean decoder (RFC 6386, 7.3).
type testBoolReader struct {
	data     []byte
	pos      int
	value    uint32
	rng      uint32
	bitCount int
}

func newTestBoolReader(data []byte) *testBoolReader {
	br := &testBoolReader{data: data, rng: 255}
	br.value = uint32(br.next())<<8 | uint32(br.next())
	return br
}

func (br *testBoolReader) next() byte {
	if br.pos < len(br.data) {
		b := br.data[br.pos]
		br.pos++
		return b
	}
	return 0
}

func (br *testBoolReader) readBit(prob int) int {
	split := 1 + (((br.rng - 1) * uint32(prob)) >> 8)
	bigSplit := split << 8
	var bit int
	if br.value >= bigSplit {
		bit = 1
		br.rng -= split
		br.value -= bigSplit
	} else {
		br.rng = split
	}
	for br.rng < 128 {
		br.value <<= 1
		br.rng <<= 1
		br.bitCount++
		if br.bitCount == 8 {
			br.bitCount = 0
			br.value |= uint32(br.next())
		}
	}
	return bit
}

func (br *testBoolReader) readBits(n int) int {
	v := 0
	for i := 0; i < n; i++ {
		v = v<<1 | br.readBit(128)
	}
	return v
}

func encodeNRGBA(t testing.TB, img *image.NRGBA, cfg Config) []byte {
	t.Helper()
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	data, err := Encode(pic, &cfg)
	if err != nil {
		t.Fatalf("Encode(%+v): %v", cfg, err)
	}
	return data
}

func TestEncodeStructure(t *testing.T) {
	sizes := [][2]int{{1, 1}, {3, 7}, {15, 17}, {16, 16}, {17, 33}, {100, 75}, {1000, 1}, {1, 300}}
	for _, sz := range sizes {
		for parts := 0; parts < 4; parts++ {
			cfg := DefaultConfig()
			cfg.Partitions = parts
			cfg.LowMemory = true // multiple partitions at method 4
			data := encodeNRGBA(t, synthImage(sz[0], sz[1], 0, 1), cfg)
			f := parseFrame(t, data)
			if !f.keyFrame || !f.show {
				t.Errorf("%v: not a shown key frame", sz)
			}
			if f.width != sz[0] || f.height != sz[1] {
				t.Errorf("dims: got %dx%d want %dx%d", f.width, f.height, sz[0], sz[1])
			}
			if f.numParts != 1<<parts {
				t.Errorf("%v: partitions: got %d want %d", sz, f.numParts, 1<<parts)
			}
			if f.profile != 0 {
				t.Errorf("profile: got %d want 0", f.profile)
			}
		}
	}
}

// expectedParts is the number of token partitions libwebp 1.6.0 writes for
// cfg.
func expectedParts(cfg *Config) int {
	if cfg.Method >= 3 && !cfg.LowMemory {
		return 1
	}
	return 1 << cfg.Partitions
}

// TestPartitionsRule checks the number of token partitions written for each
// (Method, LowMemory, Partitions) combination against libwebp 1.6.0, whose
// token path (methods >= 3 without low_memory) always writes a single
// partition. Expected values verified with webpinfo -bitstream_info on the
// output of v1 (cgo, libwebp 1.6.0).
func TestPartitionsRule(t *testing.T) {
	img := synthImage(64, 48, 0, 1)
	for m := 0; m <= 6; m++ {
		for _, lowMem := range []bool{false, true} {
			for parts := 0; parts <= 3; parts++ {
				cfg := DefaultConfig()
				cfg.Method, cfg.LowMemory, cfg.Partitions = m, lowMem, parts
				want := 1 << parts
				if m >= 3 && !lowMem {
					want = 1
				}
				if e := expectedParts(&cfg); e != want {
					t.Fatalf("expectedParts = %d, want %d", e, want)
				}
				if got := parseFrame(t, encodeNRGBA(t, img, cfg)).numParts; got != want {
					t.Errorf("method=%d low_memory=%v partitions=%d: %d partitions, want %d", m, lowMem, parts, got, want)
				}
			}
		}
	}
}

func TestProfile(t *testing.T) {
	img := synthImage(32, 32, 0, 1)
	for _, tc := range []struct {
		strength, typ, want int
	}{{0, 1, 2}, {60, 1, 0}, {60, 0, 1}} {
		cfg := DefaultConfig()
		cfg.FilterStrength = tc.strength
		cfg.FilterType = tc.typ
		if p := parseFrame(t, encodeNRGBA(t, img, cfg)).profile; p != tc.want {
			t.Errorf("strength=%d type=%d: profile %d want %d", tc.strength, tc.typ, p, tc.want)
		}
	}
}

func TestDeterminism(t *testing.T) {
	img := synthImage(97, 61, 3, 7)
	for m := 0; m <= 6; m++ {
		cfg := DefaultConfig()
		cfg.Method = m
		cfg.Autofilter = m%2 == 0
		cfg.ThreadLevel = true
		a := encodeNRGBA(t, img, cfg)
		b := encodeNRGBA(t, img, cfg)
		if !bytes.Equal(a, b) {
			t.Fatalf("method %d: non deterministic output", m)
		}
	}
	// concurrent use of the pools
	cfg := DefaultConfig()
	want := encodeNRGBA(t, img, cfg)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := encodeNRGBA(t, img, cfg); !bytes.Equal(got, want) {
				t.Error("concurrent encode differs")
			}
		}()
	}
	wg.Wait()
}

func TestManyConfigs(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sizes := [][2]int{{1, 1}, {2, 3}, {15, 17}, {33, 17}, {64, 48}, {130, 9}}
	n := 300
	if testing.Short() {
		n = 60
	}
	for i := 0; i < n; i++ {
		sz := sizes[rng.Intn(len(sizes))]
		img := synthImage(sz[0], sz[1], rng.Intn(4), int64(i))
		cfg := Config{
			Quality:         float32(rng.Intn(101)),
			Method:          rng.Intn(7),
			Segments:        1 + rng.Intn(4),
			SNSStrength:     rng.Intn(101),
			FilterStrength:  rng.Intn(101),
			FilterSharpness: rng.Intn(8),
			FilterType:      rng.Intn(2),
			Autofilter:      rng.Intn(2) == 0,
			Preprocessing:   rng.Intn(4),
			Partitions:      rng.Intn(4),
			PartitionLimit:  rng.Intn(101),
			EmulateJPEGSize: rng.Intn(4) == 0,
			ThreadLevel:     rng.Intn(2) == 0,
			LowMemory:       rng.Intn(2) == 0,
			Pass:            1 + rng.Intn(3),
		}
		if rng.Intn(5) == 0 {
			if rng.Intn(2) == 0 {
				cfg.TargetSize = 50 + rng.Intn(4000)
			} else {
				cfg.TargetPSNR = float32(20 + rng.Intn(25))
			}
		}
		pic := FromNRGBA(img, rng.Intn(3) == 0, rng.Intn(2) == 0)
		data, err := Encode(pic, &cfg)
		pic.Release()
		if err != nil {
			t.Fatalf("config %+v size %v: %v", cfg, sz, err)
		}
		f := parseFrame(t, data)
		if f.width != sz[0] || f.height != sz[1] || f.numParts != expectedParts(&cfg) {
			t.Fatalf("config %+v: bad header %+v", cfg, f)
		}
	}
}

func TestQualityMonotonic(t *testing.T) {
	skipUnderRace(t)
	img := loadTestImage(t, "logo.jpg")
	prev := 0
	for _, q := range []float32{0, 25, 50, 75, 100} {
		cfg := DefaultConfig()
		cfg.Quality = q
		n := len(encodeNRGBA(t, img, cfg))
		if n <= prev {
			t.Errorf("size not increasing with quality: q=%v size=%d prev=%d", q, n, prev)
		}
		prev = n
	}
}

func TestTargetSize(t *testing.T) {
	skipUnderRace(t)
	img := loadTestImage(t, "logo.jpg")
	cfg := DefaultConfig()
	cfg.TargetSize = 20000
	cfg.Pass = 6
	n := len(encodeNRGBA(t, img, cfg)) + 20 // + RIFF overhead
	if math.Abs(float64(n-cfg.TargetSize)) > 0.15*float64(cfg.TargetSize) {
		t.Errorf("target size %d, got %d", cfg.TargetSize, n)
	}
}

// TestTargetSizeSinglePass checks libwebp's behaviour: with one pass a size
// or PSNR target does not trigger a search, so the output equals the plain
// encode at the configured quality.
func TestTargetSizeSinglePass(t *testing.T) {
	img := loadTestImage(t, "logo.jpg")
	cfg := DefaultConfig()
	want := encodeNRGBA(t, img, cfg)
	cfg.TargetSize = 2000
	if got := encodeNRGBA(t, img, cfg); !bytes.Equal(got, want) {
		t.Errorf("TargetSize with Pass=1: %d bytes, want %d", len(got), len(want))
	}
	cfg.TargetSize = 0
	cfg.TargetPSNR = 25
	if got := encodeNRGBA(t, img, cfg); !bytes.Equal(got, want) {
		t.Errorf("TargetPSNR with Pass=1: %d bytes, want %d", len(got), len(want))
	}
}

func TestErrors(t *testing.T) {
	img := synthImage(8, 8, 0, 1)
	pic := FromNRGBA(img, false, true)
	defer pic.Release()
	bad := []func(c *Config){
		func(c *Config) { c.Quality = -1 },
		func(c *Config) { c.Quality = 101 },
		func(c *Config) { c.Method = 7 },
		func(c *Config) { c.Segments = 0 },
		func(c *Config) { c.Segments = 5 },
		func(c *Config) { c.Partitions = 4 },
		func(c *Config) { c.FilterSharpness = 8 },
		func(c *Config) { c.FilterType = 2 },
		func(c *Config) { c.SNSStrength = 101 },
		func(c *Config) { c.Pass = 11 },
		func(c *Config) { c.TargetSize = -1 },
		func(c *Config) { c.PartitionLimit = 101 },
	}
	for i, f := range bad {
		cfg := DefaultConfig()
		f(&cfg)
		_, err := Encode(pic, &cfg)
		if status.CodeOf(err) != utils.Vp8StatusInvalidParam {
			t.Errorf("case %d: got %v, want invalid param", i, err)
		}
	}
	var se *status.Error
	if _, err := Encode(nil, nil); !errors.As(err, &se) {
		t.Errorf("nil picture: got %v", err)
	}
	for _, p := range []*Picture{
		{Width: 0, Height: 1},
		{Width: 16384, Height: 1, Y: make([]byte, 16384), U: make([]byte, 8192), V: make([]byte, 8192), YStride: 16384, UVStride: 8192},
		{Width: 4, Height: 4, Y: make([]byte, 15), U: make([]byte, 4), V: make([]byte, 4), YStride: 4, UVStride: 2},
		{Width: 4, Height: 4, Y: make([]byte, 16), U: make([]byte, 4), V: make([]byte, 4), YStride: 3, UVStride: 2},
	} {
		if _, err := Encode(p, nil); status.CodeOf(err) != utils.Vp8StatusInvalidParam {
			t.Errorf("bad picture %dx%d: got %v", p.Width, p.Height, err)
		}
	}
	// nil config means defaults
	if _, err := Encode(pic, nil); err != nil {
		t.Errorf("nil config: %v", err)
	}
}

func picPSNR(a, b *Picture) float64 {
	var sse float64
	n := 0
	plane := func(pa, pb []byte, sa, sb, w, h int) {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				d := float64(pa[y*sa+x]) - float64(pb[y*sb+x])
				sse += d * d
				n++
			}
		}
	}
	plane(a.Y, b.Y, a.YStride, b.YStride, a.Width, a.Height)
	uvw, uvh := (a.Width+1)/2, (a.Height+1)/2
	plane(a.U, b.U, a.UVStride, b.UVStride, uvw, uvh)
	plane(a.V, b.V, a.UVStride, b.UVStride, uvw, uvh)
	if sse == 0 {
		return 99
	}
	return 10 * math.Log10(255*255*float64(n)/sse)
}

func TestFromYCbCr(t *testing.T) {
	src := loadTestImage(t, "source.jpg")
	f, err := os.Open("../../test_data/images/source.jpg")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	ycc, ok := img.(*image.YCbCr)
	if !ok {
		t.Skip("not a YCbCr image")
	}
	a := FromNRGBA(src, false, true)
	b := FromYCbCr(ycc)
	defer a.Release()
	defer b.Release()
	if p := picPSNR(a, b); p < 45 {
		t.Errorf("YCbCr route PSNR %.2f dB < 45 dB", p)
	}
	// other subsampling ratios and odd origins
	for _, ratio := range []image.YCbCrSubsampleRatio{
		image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422, image.YCbCrSubsampleRatio420,
		image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410,
	} {
		r := image.Rect(0, 0, 37, 23)
		y := image.NewYCbCr(r, ratio)
		// smooth, in-gamut content
		for i := range y.Y {
			y.Y[i] = uint8(80 + (i*3)%97)
		}
		for i := range y.Cb {
			y.Cb[i] = uint8(118 + i%21)
			y.Cr[i] = uint8(138 - i%17)
		}
		for _, sub := range []image.Rectangle{r, image.Rect(1, 3, 36, 22)} {
			yi := y.SubImage(sub).(*image.YCbCr)
			nrgba := image.NewNRGBA(sub)
			draw.Draw(nrgba, sub, yi, sub.Min, draw.Src)
			p1 := FromYCbCr(yi)
			p2 := FromNRGBA(nrgba, false, true)
			if p1.Width != sub.Dx() || p1.Height != sub.Dy() {
				t.Fatalf("bad dims")
			}
			if p := picPSNR(p1, p2); p < 30 {
				t.Errorf("ratio %v rect %v: PSNR %.2f", ratio, sub, p)
			}
			if _, err := Encode(p1, nil); err != nil {
				t.Error(err)
			}
			p1.Release()
			p2.Release()
		}
	}
}

func TestFromNRGBAReference(t *testing.T) {
	// Gray pixels must map to libwebp's studio range luma, neutral chroma.
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	for i := 0; i < 4; i++ {
		copy(img.Pix[4*i:], []byte{0, 0, 0, 255})
	}
	p := FromNRGBA(img, false, true)
	if p.Y[0] != 16 || p.U[0] != 128 || p.V[0] != 128 {
		t.Errorf("black: got Y=%d U=%d V=%d", p.Y[0], p.U[0], p.V[0])
	}
	for i := 0; i < 4; i++ {
		copy(img.Pix[4*i:], []byte{255, 255, 255, 255})
	}
	p = FromNRGBA(img, false, true)
	if p.Y[0] != 235 || p.U[0] != 128 || p.V[0] != 128 {
		t.Errorf("white: got Y=%d U=%d V=%d", p.Y[0], p.U[0], p.V[0])
	}
	// sharp YUV on a large enough flat image gives the same values
	big := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for i := range big.Pix {
		big.Pix[i] = 255
	}
	p = FromNRGBA(big, true, true)
	if p.Y[0] != 235 || p.U[0] != 128 || p.V[0] != 128 {
		t.Errorf("sharp white: got Y=%d U=%d V=%d", p.Y[0], p.U[0], p.V[0])
	}
}

func TestCleanupTransparent(t *testing.T) {
	img := synthImage(64, 64, 3, 1)
	exact := FromNRGBA(img, false, true)
	clean := FromNRGBA(img, false, false)
	defer exact.Release()
	defer clean.Release()
	// the first 16x16 block is fully transparent: it must be flat
	v := clean.Y[0]
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if clean.Y[y*clean.YStride+x] != v {
				t.Fatalf("transparent block not flattened at %d,%d", x, y)
			}
		}
	}
	if bytes.Equal(exact.Y, clean.Y) {
		t.Errorf("cleanup did not change anything")
	}
	cfg := DefaultConfig()
	a, _ := Encode(exact, &cfg)
	b, _ := Encode(clean, &cfg)
	if len(b) >= len(a) {
		t.Errorf("cleanup did not help compression: %d >= %d", len(b), len(a))
	}
}

func TestLargeDimensions(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	for _, sz := range [][2]int{{16383, 1}, {1, 16383}} {
		img := synthImage(sz[0], sz[1], 0, 1)
		data := encodeNRGBA(t, img, DefaultConfig())
		f := parseFrame(t, data)
		if f.width != sz[0] || f.height != sz[1] {
			t.Errorf("bad dims %+v", f)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	img := loadTestImage(b, "source.jpg")
	for _, m := range []int{0, 4, 6} {
		b.Run("m"+string(rune('0'+m)), func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.Method = m
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				pic := FromNRGBA(img, false, false)
				if _, err := Encode(pic, &cfg); err != nil {
					b.Fatal(err)
				}
				pic.Release()
			}
		})
	}
}

func BenchmarkFromNRGBA(b *testing.B) {
	img := loadTestImage(b, "source.jpg")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		FromNRGBA(img, false, false).Release()
	}
}
