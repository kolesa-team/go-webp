package vp8

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

// benchVP8 returns the VP8 payload of a benchmark image. Images that are not
// in the repository (large photos) are looked up in $WEBP_BENCH_CORPUS; the
// benchmark is skipped when the file is not found.
func benchVP8(b *testing.B, name string) []byte {
	b.Helper()
	paths := []string{filepath.Join("..", "..", "test_data", "images", name)}
	if dir := os.Getenv("WEBP_BENCH_CORPUS"); dir != "" {
		paths = append(paths, filepath.Join(dir, name))
	}
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		ri, err := extractVP8(data)
		if err != nil {
			b.Fatal(err)
		}
		return ri.vp8
	}
	b.Skipf("%s not found (set WEBP_BENCH_CORPUS)", name)
	return nil
}

func benchDecode(b *testing.B, name string, opts *Options) {
	vp8 := benchVP8(b, name)
	w, h, err := DecodeHeader(vp8)
	if err != nil {
		b.Fatal(err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	sink := pipeline.NewDirect(img)
	b.SetBytes(int64(w * h * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Decode(vp8, nil, opts, sink); err != nil {
			b.Fatal(err)
		}
	}
}

const (
	benchM4      = "m4_q75.webp"
	benchLogo    = "webp-logo-lossy.webp"
	benchPhoto4K = "photo4k_q75.webp"
	benchTile4K  = "photo4k_tile_q75.webp"
)

var threadsOpts = &Options{UseThreads: true}

func BenchmarkDecodeM4(b *testing.B)        { benchDecode(b, benchM4, nil) }
func BenchmarkDecodeM4Threads(b *testing.B) { benchDecode(b, benchM4, threadsOpts) }
func BenchmarkDecodeM4NoFancy(b *testing.B) {
	benchDecode(b, benchM4, &Options{NoFancyUpsampling: true})
}
func BenchmarkDecodeLogo(b *testing.B)           { benchDecode(b, benchLogo, nil) }
func BenchmarkDecodeLogoThreads(b *testing.B)    { benchDecode(b, benchLogo, threadsOpts) }
func BenchmarkDecodePhoto4K(b *testing.B)        { benchDecode(b, benchPhoto4K, nil) }
func BenchmarkDecodePhoto4KThreads(b *testing.B) { benchDecode(b, benchPhoto4K, threadsOpts) }
func BenchmarkDecodeTile4K(b *testing.B)         { benchDecode(b, benchTile4K, nil) }
func BenchmarkDecodeTile4KThreads(b *testing.B)  { benchDecode(b, benchTile4K, threadsOpts) }

// benchParse measures the bitstream parsing alone (modes and coefficient
// tokens of every macroblock, no reconstruction or output).
func benchParse(b *testing.B, name string) {
	vp8 := benchVP8(b, name)
	d := new(decoder)
	blocks := []mbBlock(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := d.parseHeaders(vp8); err != nil {
			b.Fatal(err)
		}
		d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
		d.outW = d.width
		d.enterCritical()
		d.initFrame()
		if cap(blocks) < d.mbW {
			blocks = make([]mbBlock, d.mbW)
		}
		blocks = blocks[:d.mbW]
		for mbY := 0; mbY < d.mbH; mbY++ {
			if err := d.parseRow(mbY, blocks); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkParseM4(b *testing.B)      { benchParse(b, benchM4) }
func BenchmarkParseLogo(b *testing.B)    { benchParse(b, benchLogo) }
func BenchmarkParsePhoto4K(b *testing.B) { benchParse(b, benchPhoto4K) }
func BenchmarkParseTile4K(b *testing.B)  { benchParse(b, benchTile4K) }

func BenchmarkUpsamplePair(b *testing.B) {
	const w = 1024
	ys := make([]byte, w)
	u0, v0, u1, v1 := make([]byte, w/2), make([]byte, w/2), make([]byte, w/2), make([]byte, w/2)
	for i := range ys {
		ys[i] = byte(i * 7)
	}
	for i := range u0 {
		u0[i], v0[i], u1[i], v1[i] = byte(i*3), byte(i*5), byte(i*11), byte(i*13)
	}
	dst := make([]byte, 4*w)
	uvT, uvB := make([]uint32, w), make([]uint32, w)
	b.SetBytes(2 * w)
	for i := 0; i < b.N; i++ {
		upsamplePair(ys, ys, u0, v0, u1, v1, dst, dst, nil, nil, uvT, uvB, w)
	}
}

func BenchmarkFilterComplex(b *testing.B) {
	const stride = 64
	buf := make([]byte, stride*24)
	for i := range buf {
		buf[i] = byte(128 + (i*7)%9 - 4)
	}
	work := make([]byte, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		o := 4*stride + 8
		loop26H(work, o, stride, 16, 2*(30+4)+1, 10, 1)
		loop24H(work, o+4, stride, 16, 2*30+1, 10, 1)
		loop24H(work, o+8, stride, 16, 2*30+1, 10, 1)
		loop24H(work, o+12, stride, 16, 2*30+1, 10, 1)
		loop26V(work, o, stride, 16, 2*(30+4)+1, 10, 1)
		loop24V(work, o+4*stride, stride, 16, 2*30+1, 10, 1)
		loop24V(work, o+8*stride, stride, 16, 2*30+1, 10, 1)
		loop24V(work, o+12*stride, stride, 16, 2*30+1, 10, 1)
	}
}
