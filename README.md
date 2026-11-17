![logo](test_data/images/logo.jpg)
# go-webp
[![CI](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml/badge.svg)](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/kolesa-team/go-webp/v2.svg)](https://pkg.go.dev/github.com/kolesa-team/go-webp/v2)
[![Go Report](https://goreportcard.com/badge/github.com/kolesa-team/go-webp)](https://goreportcard.com/report/github.com/kolesa-team/go-webp)
[![codecov](https://codecov.io/gh/kolesa-team/go-webp/branch/master/graph/badge.svg)](https://codecov.io/gh/kolesa-team/go-webp)

WebP encoder and decoder for Go, written in **pure Go** — no cgo, no libwebp,
no assembly, no dependencies outside the standard library.

- Lossy (VP8) and lossless (VP8L) encoding and decoding.
- Alpha channel for lossy images (`ALPH` chunk, raw or compressed, with
  filtering and level quantization) and the extended `VP8X` container.
- Decoder output is **bit-exact with libwebp 1.6** (`dwebp`), including fancy
  upsampling, `NoFancyUpsampling`, `BypassFiltering`, dithering, alpha
  dithering, cropping and flipping. Scaling uses libwebp's rescaler.
- Encoder follows libwebp's algorithms (segmentation, RD-optimised mode
  decision, trellis quantization, LZ77/color cache/meta-Huffman for lossless),
  but the bytes are not identical to `cwebp`'s. Measured against cwebp 1.6.0
  at the same settings (`go test -tags interop -run VsCwebp -v
  ./internal/vp8enc`, two test photos, quality 10–100): lossy methods 4 and 6
  are within −0.8%…+0.3% of cwebp's size at the same PSNR (±0.01 dB);
  method 0 files are 0.1–9% smaller (up to 41% at quality 100) at a PSNR
  within 0.05 dB. Lossless output is 0.2% (level 6) to 1.2% (level 0) larger
  than cwebp's on the benchmark photo (see below).
- Streaming decoder working one macroblock row at a time, pooled scratch
  buffers, zero copies for the common "decode the whole picture" case.
- Works everywhere Go does, including `CGO_ENABLED=0`, cross compilation and
  scratch/distroless containers.

Animated WebP is not supported (as in v1, decoding reports
`VP8_STATUS_UNSUPPORTED_FEATURE`).

## Install
```bash
go get github.com/kolesa-team/go-webp/v2
```

Requires Go 1.27 or newer. Nothing else.

The cgo/libwebp based v1 remains available at tag `v1.0.5` (module path
`github.com/kolesa-team/go-webp`).

## Examples

#### Decode:
```go
package main

import (
	"image/jpeg"
	"log"
	"os"

	"github.com/kolesa-team/go-webp/v2/decoder"
	"github.com/kolesa-team/go-webp/v2/webp"
)

func main() {
	file, err := os.Open("test_data/images/m4_q75.webp")
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	img, err := webp.Decode(file, &decoder.Options{})
	if err != nil {
		log.Fatalln(err)
	}

	if err := os.MkdirAll("example", 0o755); err != nil {
		log.Fatalln(err)
	}
	output, err := os.Create("example/output_decode.jpg")
	if err != nil {
		log.Fatalln(err)
	}

	if err := jpeg.Encode(output, img, &jpeg.Options{Quality: 75}); err != nil {
		output.Close()
		log.Fatalln(err)
	}
	if err := output.Close(); err != nil {
		log.Fatalln(err)
	}
}
```

From the repository root:
```bash
go run ./_examples/decode
```

Importing `github.com/kolesa-team/go-webp/v2/webp` also registers the format
with the standard library, so `image.Decode` and `image.DecodeConfig` work on
WebP files.

#### Encode
```go
package main

import (
	"image/jpeg"
	"log"
	"os"

	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/webp"
)

func main() {
	file, err := os.Open("test_data/images/source.jpg")
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	img, err := jpeg.Decode(file)
	if err != nil {
		log.Fatalln(err)
	}

	options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		log.Fatalln(err)
	}

	if err := os.MkdirAll("example", 0o755); err != nil {
		log.Fatalln(err)
	}
	output, err := os.Create("example/output_encode.webp")
	if err != nil {
		log.Fatalln(err)
	}

	if err := webp.Encode(output, img, options); err != nil {
		output.Close()
		log.Fatalln(err)
	}
	if err := output.Close(); err != nil {
		log.Fatalln(err)
	}
}
```

From the repository root:
```bash
go run ./_examples/encode
```

Lossless: `encoder.NewLosslessEncoderOptions(encoder.PresetDefault, level)`
with `level` from 0 (fastest) to 9 (smallest), like `cwebp -z`.

## Migrating from v1

The v1 (cgo) line is tagged `v1.0.5`. Note that the last v1 commit on
`master`, `64776d9` ("fix: add missing cgo linker flags for Windows"), is
**not** part of `v1.0.5`: tag it `v1.0.6` before merging v2, so that v1 users
on Windows can get the fix with `go get github.com/kolesa-team/go-webp@v1.0.6`.

v2 keeps the v1 API: the same packages, types, fields, constants, functions
and option semantics (presets, lossless levels, `Crop`/`Scale` conventions,
error messages such as `cannot decode picture: VP8_STATUS_BITSTREAM_ERROR` or
`cannot encode webppicture`).

1. Change the import paths from `github.com/kolesa-team/go-webp/...` to
   `github.com/kolesa-team/go-webp/v2/...`.
2. Remove libwebp from your build images and CI; cgo is no longer needed.
3. `encoder.Options.GetConfig()` and `decoder.Options.GetConfig()` used to
   return the cgo types `*C.WebPConfig` / `*C.WebPDecoderConfig`, which could
   not be used outside the packages. They now return the pure Go
   `*encoder.Config` / `*decoder.Config`, with the same fields as libwebp's
   structures and the same validation (`WebPValidateConfig`).
4. The encoder produces valid WebP files of about the same size and quality
   as libwebp (see the measurements above), but not the same bytes. Tests
   comparing encoder output with files produced by libwebp need to compare
   decoded pixels instead. Decoded pixels are bit-identical to v1's.

Other differences worth knowing:

- Errors returned by the decoder can be unwrapped (`errors.As`) to an error
  carrying the libwebp status code and a detailed message; encoder errors
  unwrap (`errors.Unwrap`) to the detailed cause.
- `webp.DecodeConfig` reads only the file headers instead of the whole input.
  As in v1 it reports `color.RGBAModel`, although decoded images are
  `*image.NRGBA`.
- New helper `webp.GetFeatures(io.Reader)` returns `utils.BitstreamFeatures`
  without decoding.
- `*image.YCbCr` sources (e.g. from `image/jpeg`) are encoded lossy directly
  from their Y'CbCr planes (rescaled from full to studio range) instead of
  going through RGB like v1, so the encoded result is slightly different
  from v1's for the same JPEG; `*image.RGBA`, `*image.Gray`,
  `*image.Paletted` and `*image.NYCbCrA` are converted without the generic
  per-pixel path.
- As in v1, `encoder.NewEncoder` takes a snapshot of the source: changing the
  pixels of the source after `NewEncoder` does not change what `Encode`
  writes, except for an `*image.NRGBA`, which (as in v1) is used in place
  until `Encode` returns.
- `Options.ShowCompressed` and `Options.UseDeltaPalette` are accepted and have
  no effect. `Options.LowMemory` does not reduce the lossy encoder's memory
  use; as in libwebp it re-enables `Options.Partitions` (more than one token
  partition) at `Method` >= 3, where libwebp and v2 otherwise write a single
  partition.
- As in libwebp, a NaN `Quality` or `TargetPsnr` passes validation; a NaN
  lossy quality encodes at the finest quantizer (like quality 100).
- `UseThreads` (decoder) and `ThreadLevel` (encoder) use goroutines instead
  of libwebp's worker threads; outputs do not depend on them.
- Allocations are limited like libwebp's `WEBP_MAX_ALLOCABLE_MEMORY`
  (16 GiB on 64-bit targets, just under 2 GiB on 32-bit ones): a `Scale` or
  crop whose output would exceed it fails with `VP8_STATUS_OUT_OF_MEMORY`.
- Raw VP8/VP8L bitstreams without a RIFF header are accepted by
  `decoder.NewDecoder`, `webp.Decode`, `webp.DecodeConfig` and
  `webp.GetFeatures`, as in libwebp; `image.Decode` / `image.DecodeConfig`
  only recognize RIFF files (same registered magic as v1).

## Benchmarks

Apple M5 Pro, Go 1.27, `go test -bench -benchmem`, v1 = v1.0.5 with Homebrew
libwebp 1.6.0 (NEON). Memory for v1 counts only Go allocations (libwebp's own
C heap is not visible to the Go benchmark).

| Benchmark | v1 (libwebp) | v2 (pure Go) |
|---|---|---|
| Decode lossy 675x900 (`m4_q75.webp`) | 3.1 ms, 2.57 MB, 22 allocs | 7.4 ms, 2.50 MB, 11 allocs |
| ↳ with `UseThreads` | | 4.7 ms, 2.50 MB, 16 allocs |
| Decode lossy+alpha 3000x2000 (`webp-logo-lossy.webp`) | 11.1 ms, 24.2 MB, 23 allocs | 26.3 ms, 24.5 MB, 14 allocs |
| ↳ with `UseThreads` | | 20.2 ms, 25.2 MB, 28 allocs |
| Decode lossless 3000x2000 (`webp-logo-lossless.webp`) | 5.9 ms, 24.2 MB, 23 allocs | 6.8 ms, 25.2 MB, 13 allocs |
| Encode lossy q75, 960x1280 JPEG source | 66 ms, 117 858 bytes | 138 ms, 117 918 bytes, 0.9 MB, 11 allocs |
| Encode lossless level 6, same source | 395 ms, 1 098 384 bytes | 386 ms, 1 100 216 bytes |
| Encode lossless level 0, same source | 40 ms, 1 201 738 bytes | 46 ms, 1 216 362 bytes |

The output image (24 MB for 3000x2000) dominates the decoder's memory; the
decoder itself keeps O(width) state. Compared with `golang.org/x/image/webp`
(decoder only), v2 decodes lossy pictures 1.6–2.6x and lossless pictures
1.7–3.4x faster, with far fewer allocations.

```bash
make bench
```

## Tests

```bash
go test ./...                              # unit, round-trip and conformance tests
go test -tags interop ./...                # also compare with cwebp/dwebp (libwebp tools in PATH)
go test -run '^$' -fuzz FuzzDecode ./decoder
```

## License
MIT licensed. See the LICENSE file for details.

Parts of the codec are adapted from [libwebp](https://chromium.googlesource.com/webm/libwebp)
(Copyright 2010 Google Inc., BSD-3-Clause); see the NOTICE file.
