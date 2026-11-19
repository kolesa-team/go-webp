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

// Package suite holds the benchmarks and the peak-memory tool shared by the
// v1 (cgo, libwebp) and v2 (pure Go) adapters in ../v1 and ../v2, so that
// both versions run exactly the same workloads on the same files.
//
// See run.sh in the parent directory and the Benchmarks section of the
// repository README.
package suite

import (
	"errors"
	"image"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Codec adapts one version of the library to the suite.
type Codec interface {
	// Decoder returns a function that decodes data in the given mode
	// (see DecodeModes), or false if this version has no such mode.
	Decoder(data []byte, mode string) (func() error, bool)
	// Decode decodes data with the default options.
	Decode(data []byte) (image.Image, error)
	// Encode encodes img into w with webp.Encode as described by c.
	Encode(w io.Writer, img image.Image, c EncodeCase, threads bool) error
	// DecodeInto decodes data into dst (reused when large enough), or
	// returns ErrUnsupported.
	DecodeInto(data []byte, dst *image.NRGBA) (*image.NRGBA, error)
}

// ErrUnsupported is returned by adapters for calls a version does not have.
var ErrUnsupported = errors.New("not supported by this version")

// DecodeModes are the decode variants:
//   - default: decoder.NewDecoder + Decode with zero Options;
//   - threads: Options.UseThreads;
//   - scale: Options.Scale to 320x240;
//   - into: DecodeInto, reusing one output buffer (v2 only);
//   - ycbcr: DecodeYCbCr, or DecodeNYCbCrA for pictures with alpha (v2,
//     lossy pictures only).
var DecodeModes = []string{"default", "threads", "scale", "into", "ycbcr"}

// DecodeFile is a decode workload.
type DecodeFile struct{ Name, File string }

// DecodeFiles are the pictures of the decode benchmarks, in test_data/images.
var DecodeFiles = []DecodeFile{
	{"lossy675x900", "m4_q75.webp"},
	{"lossyalpha3000x2000", "webp-logo-lossy.webp"},
	{"lossless3000x2000", "webp-logo-lossless.webp"},
}

// EncodeCase is an encode workload.
type EncodeCase struct {
	Name     string
	Source   string // "jpeg": source.jpg (960x1280 *image.YCbCr); "logo": the decoded lossless logo (3000x2000 *image.NRGBA with alpha)
	Lossless bool
	Level    int // lossy quality or lossless level
	Method   int // lossy only
}

// EncodeCases are the encode benchmarks; every case also runs with
// Options.ThreadLevel.
var EncodeCases = []EncodeCase{
	{"lossy-q75-m0", "jpeg", false, 75, 0},
	{"lossy-q75-m4", "jpeg", false, 75, 4},
	{"lossy-q75-m6", "jpeg", false, 75, 6},
	{"lossyalpha-q75-m4", "logo", false, 75, 4},
	{"lossless-L6", "jpeg", true, 6, 0},
	{"lossless-L0", "jpeg", true, 0, 0},
}

// DataDir returns the directory of the test pictures: $WEBP_BENCH_DATA,
// or test_data/images of this repository.
func DataDir() string {
	if d := os.Getenv("WEBP_BENCH_DATA"); d != "" {
		return d
	}
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "test_data", "images")
}

// ReadFile reads a test picture.
func ReadFile(name string) ([]byte, error) {
	return os.ReadFile(filepath.Join(DataDir(), name))
}

// LoadJPEG decodes source.jpg (an *image.YCbCr).
func LoadJPEG() (image.Image, error) {
	f, err := os.Open(filepath.Join(DataDir(), "source.jpg"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return jpeg.Decode(f)
}

func mustRead(b *testing.B, name string) []byte {
	b.Helper()
	d, err := ReadFile(name)
	if err != nil {
		b.Fatal(err)
	}
	return d
}

// sources returns the encode sources by name.
func sources(b *testing.B, c Codec) map[string]image.Image {
	b.Helper()
	jpg, err := LoadJPEG()
	if err != nil {
		b.Fatal(err)
	}
	logo, err := c.Decode(mustRead(b, "webp-logo-lossless.webp"))
	if err != nil {
		b.Fatal(err)
	}
	return map[string]image.Image{"jpeg": jpg, "logo": logo}
}

type counter struct{ n int }

func (c *counter) Write(p []byte) (int, error) { c.n += len(p); return len(p), nil }

// encodedSize encodes img and returns the size of the result.
func encodedSize(c Codec, img image.Image, ec EncodeCase, threads bool) (int, error) {
	var w counter
	err := c.Encode(&w, img, ec, threads)
	return w.n, err
}

// timed runs loop as the measured part of b and reports, besides the wall
// time, the CPU time (user + system, all threads of the process, garbage
// collector included) per operation as cpu-ns/op where the platform has
// getrusage.
func timed(b *testing.B, loop func()) {
	b.ReportAllocs()
	b.ResetTimer()
	c0, ok := cpuTime()
	loop()
	c1, _ := cpuTime()
	b.StopTimer()
	if ok {
		b.ReportMetric(float64((c1-c0).Nanoseconds())/float64(b.N), "cpu-ns/op")
	}
}

// Decode benchmarks one decode per operation: Decode/<file>/<mode>.
func Decode(b *testing.B, c Codec) {
	for _, f := range DecodeFiles {
		data := mustRead(b, f.File)
		for _, mode := range DecodeModes {
			b.Run(f.Name+"/"+mode, func(b *testing.B) {
				run, ok := c.Decoder(data, mode)
				if !ok {
					b.Skip("not available in this version")
				}
				timed(b, func() {
					for i := 0; i < b.N; i++ {
						if err := run(); err != nil {
							b.Fatal(err)
						}
					}
				})
			})
		}
	}
}

func modeName(threads bool) string {
	if threads {
		return "threads"
	}
	return "default"
}

// Encode benchmarks one webp.Encode per operation:
// Encode/<case>/{default,threads}; it reports the encoded size as bytes.
func Encode(b *testing.B, c Codec) {
	src := sources(b, c)
	for _, ec := range EncodeCases {
		for _, threads := range []bool{false, true} {
			b.Run(ec.Name+"/"+modeName(threads), func(b *testing.B) {
				img := src[ec.Source]
				var n int
				timed(b, func() {
					for i := 0; i < b.N; i++ {
						var err error
						if n, err = encodedSize(c, img, ec, threads); err != nil {
							b.Fatal(err)
						}
					}
				})
				b.ReportMetric(float64(n), "bytes")
			})
		}
	}
}

// SatDecode runs GOMAXPROCS decodes at a time (saturation, as on a busy
// server): ns/op is the inverse of the throughput.
func SatDecode(b *testing.B, c Codec) {
	for _, f := range DecodeFiles {
		data := mustRead(b, f.File)
		for _, mode := range []string{"default", "threads"} {
			b.Run(f.Name+"/"+mode, func(b *testing.B) {
				timed(b, func() {
					b.RunParallel(func(pb *testing.PB) {
						run, _ := c.Decoder(data, mode)
						for pb.Next() {
							if err := run(); err != nil {
								b.Error(err)
								return
							}
						}
					})
				})
			})
		}
	}
}

// SatEncode runs GOMAXPROCS encodes at a time (saturation).
func SatEncode(b *testing.B, c Codec) {
	src := sources(b, c)
	for _, ec := range EncodeCases {
		for _, threads := range []bool{false, true} {
			b.Run(ec.Name+"/"+modeName(threads), func(b *testing.B) {
				img := src[ec.Source]
				timed(b, func() {
					b.RunParallel(func(pb *testing.PB) {
						for pb.Next() {
							if _, err := encodedSize(c, img, ec, threads); err != nil {
								b.Error(err)
								return
							}
						}
					})
				})
			})
		}
	}
}
