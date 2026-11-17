![logo](test_data/images/logo.jpg)
# go-webp
[![CI](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml/badge.svg)](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/kolesa-team/go-webp/v2.svg)](https://pkg.go.dev/github.com/kolesa-team/go-webp/v2)
[![Go Report](https://goreportcard.com/badge/github.com/kolesa-team/go-webp)](https://goreportcard.com/report/github.com/kolesa-team/go-webp)
[![codecov](https://codecov.io/gh/kolesa-team/go-webp/branch/master/graph/badge.svg)](https://codecov.io/gh/kolesa-team/go-webp)

WebP encoder and decoder written in Go — **no cgo**, no libwebp, no
dependencies outside the standard library. On arm64 the hot loops run as
NEON assembly, with results identical to the pure Go code that other
architectures and `-tags purego` builds use (see
[Assembly and the `purego` tag](#assembly-and-the-purego-tag)).

- Lossy (VP8) and lossless (VP8L) encoding and decoding.
- Alpha channel for lossy images (`ALPH` chunk, raw or compressed, with
  filtering and level quantization) and the extended `VP8X` container.
- Decoder output is **bit-exact with libwebp 1.6** (`dwebp`), including fancy
  upsampling, `NoFancyUpsampling`, `BypassFiltering`, dithering, alpha
  dithering, cropping and flipping. Scaling uses libwebp's rescaler
  (bit-exact with its portable C version).
- Encoder follows libwebp's algorithms (segmentation, RD-optimised mode
  decision, trellis quantization, LZ77/color cache/meta-Huffman for lossless),
  but the bytes are not identical to `cwebp`'s. Measured against cwebp 1.6.0
  at the same settings (`go test -tags interop -run VsCwebp -v
  ./internal/vp8enc`, two test photos, quality 10–100): lossy methods 4 and 6
  are within −0.8%…+0.3% of cwebp's size at the same PSNR (±0.01 dB);
  method 0 files are 0.1–9% smaller (up to 41% at quality 100) at a PSNR
  within 0.05 dB. Lossless output is the same size as cwebp's at level 6
  and 1.2% larger at level 0 on the benchmark photo (see below).
- Streaming decoder working one macroblock row at a time, pooled scratch
  buffers, zero copies for the common "decode the whole picture" case;
  `DecodeInto` reuses the caller's output buffer.
- Optional helper goroutines (`UseThreads`, `ThreadLevel`) from one
  process-wide budget, with output identical to the single-goroutine one.
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

#### Decode into a reused buffer
```go
var img *image.NRGBA // nil: the first decode allocates it
for _, data := range files {
	dec, err := decoder.NewDecoder(bytes.NewReader(data), nil)
	if err != nil {
		return err
	}
	if img, err = dec.DecodeInto(img); err != nil {
		return err
	}
	process(img) // before the next decode overwrites it
}
```

`(*decoder.Decoder).DecodeInto(dst)` and `webp.DecodeInto(r, dst, options)`
decode like `Decode`, but into `dst`:

- `dst` is raw memory: its previous `Rect` and `Stride` are ignored. When
  `cap(dst.Pix) >= 4*w*h` for the output size `w`x`h` (after crop and
  scale), the pixels are written in place, with no allocation and no
  clearing; otherwise a new buffer is allocated.
- On success `dst.Rect` is `(0,0)-(w,h)`, `dst.Stride` is `4*w`, the
  capacity of a reused buffer is kept, and the returned image is `dst`
  itself. A nil `dst` behaves like `Decode`.
- On error the fields of `dst` are unchanged, but its bytes may have been
  partly overwritten.
- Do not pass a `SubImage`, and do not use `dst` from other goroutines
  during the call.

#### Decode to Y'CbCr
`(*decoder.Decoder).DecodeYCbCr()` returns the 4:2:0 planes of a lossy
picture as an `*image.YCbCr` (1.5 bytes per pixel, no conversion to RGB);
`DecodeNYCbCrA()` adds the alpha plane (`*image.NYCbCrA`, 2.5 bytes per
pixel; all 0xff for an opaque picture).

- The samples are those of the VP8 stream (BT.601 limited range), the same
  as libwebp's `WebPDecodeYUV` / `dwebp -yuv` and as
  `golang.org/x/image/webp`. Colors computed through `image.YCbCr`'s color
  model therefore differ slightly from the RGB of `Decode`, which also
  upsamples the chroma smoothly.
- `Crop` (top-left corner rounded down to even coordinates), `Flip` (each
  plane mirrored), `BypassFiltering`, `DitheringStrength`,
  `AlphaDitheringStrength` (for `DecodeNYCbCrA`) and `UseThreads` apply;
  `Scale` is refused with `VP8_STATUS_INVALID_PARAM`.
- Lossless pictures and animations, and pictures with alpha for
  `DecodeYCbCr`, are refused with `VP8_STATUS_UNSUPPORTED_FEATURE`: check
  `GetFeatures()` first.

#### Cancellation
```go
ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
defer cancel()
img, err := webp.DecodeContext(ctx, r, nil)
// errors.Is(err, context.DeadlineExceeded) once the deadline has passed
err = webp.EncodeContext(ctx, w, img, options)
```

- Decoding: `(*decoder.Decoder).DecodeContext(ctx)`,
  `DecodeIntoContext(ctx, dst)` and `webp.DecodeContext(ctx, r, options)`
  check `ctx` between batches of output rows (at most 16 rows; a lossless
  picture reads its headers first). The error satisfies
  `errors.Is(err, context.Canceled)` (or `context.DeadlineExceeded`) and
  carries the status `VP8_STATUS_USER_ABORT`;
  `webp.DecodeContext` returns `ctx.Err()` itself when `ctx` is already
  done before `r` is read.
- Encoding: `(*encoder.Encoder).EncodeContext(ctx, w)` and
  `webp.EncodeContext(ctx, w, src, options)` check `ctx` about once per
  macroblock row (lossy) and between passes, search rows and work chunks
  (lossless), on helper goroutines too. They return `ctx.Err()` itself (not
  wrapped) and write nothing to `w`. The source conversion done by
  `NewEncoder` and the RGB to Y'CbCr conversion are not interruptible.
- Helper goroutines have exited when these functions return. With a
  `context.Background()` there is no overhead.

#### Releasing an encoder
`encoder.NewEncoder` copies every source that is not an `*image.NRGBA`
(as v1 did). `(*encoder.Encoder).Release()` hands that copy back to internal
pools for later encoders; afterwards `Encode` and `EncodeContext` return
`encoder.ErrReleased`. `Release` waits for encodes running on other
goroutines, may be called more than once and is safe on a nil `*Encoder`.
Calling it is optional: otherwise the copy goes back to the pools once the
`Encoder` is unreachable. `webp.Encode` and `webp.EncodeContext` release
automatically. (`Encoder` now holds a `sync.RWMutex`: use it through a
pointer, as `NewEncoder` returns it; `go vet` reports copies.)

## Threads

By default every `Decode`/`Encode` call runs on the calling goroutine only.
Helper goroutines are opt-in, per call:

- `decoder.Options.UseThreads`:
  - lossy pictures of at least 1024 macroblocks (about 512x512): one helper
    reconstructs, filters and converts batches of macroblock rows while the
    caller parses the next ones;
  - lossy pictures with alpha whose rows down to the bottom of the output
    hold 256K pixels or more: another helper decodes the alpha plane ahead
    of the rows that need it;
  - lossless pictures of at least about 256K output pixels: the caller
    entropy-decodes while a helper applies the inverse transforms and
    writes the output rows (libwebp ignores `use_threads` for lossless
    pictures; this is a go-webp extension).
- `encoder.Options.ThreadLevel`:
  - lossy: the analysis pass runs in row bands (384 macroblocks or more,
    1024 at methods 0-1), and the compressed alpha plane of a transparent
    picture (64K pixels or more) is coded while the VP8 frame is;
  - lossless and compressed alpha (256K pixels or more): the hash-chain
    match search, the predictor search (methods 0-4) and the histogram
    remap are split among helpers; level 9 codes at most two candidate
    configurations at a time (one at a time above 8 MP).

All helpers of the process come from one budget of `GOMAXPROCS-1`
goroutines (re-read on every call), shared by all concurrent `Decode` and
`Encode` calls. A call that finds no free slot does the work on its own
goroutine, so a server that already runs one request per core is never
oversubscribed. **The output is identical with or without the flags and
for any budget.** A panic in a helper is re-raised on the calling
goroutine, and helpers have exited when the call returns.

The flags trade CPU for latency (for example lossless level 6 of a 1.2 MP
photo: 308 ms sequential, 158 ms with `ThreadLevel`, for about 10% more CPU
time). A server that is busy with concurrent requests gains little from
them; an application that decodes or encodes one large picture at a time on
an otherwise idle machine gains most.

## Running on a server

The library sets no process-wide limits itself. For a service that decodes
or encodes untrusted pictures:

- Bound the number of concurrent codec calls to about `GOMAXPROCS` (a
  semaphore), and queue or shed the rest. Every call needs its working
  memory at once, and more concurrent calls than cores only add memory.
- Decode with `DecodeInto` into buffers from your own pool (one per worker,
  sized for your largest accepted picture): the output image is nearly all
  of a decode's memory (4 bytes per pixel), and reusing it removes most of
  the garbage collector's work. Check `webp.DecodeConfig` or `GetFeatures`
  against your size limits before decoding.
- Set `GOMEMLIMIT` to about 1.3-1.5x the expected working set (roughly
  4-5 bytes per pixel per concurrent decode, plus encodes), below the
  container's memory limit. It also bounds the peaks of large lossless
  encodes (see [Benchmarks](#benchmarks)). In our measurements of 15
  concurrent decodes `GOMEMLIMIT` alone cost 7-33% more CPU; together with
  `DecodeInto` it cost nothing measurable.
- Between calls the encoders keep their scratch buffers in internal pools:
  up to 256 MiB for the lossy encoder (and `Encoder` snapshots) and
  256 MiB for the lossless encoder (64 MiB each on 32-bit platforms).
  Buffers unused for 15 s are dropped at the next garbage collection, which
  the runtime forces at least every two minutes. Leave room for them in
  the memory limit; they do not follow `GOMEMLIMIT`.
- Profile-guided optimization applies to dependencies: building the
  service with a CPU profile of its own workload (`default.pgo` in the main
  package) also optimizes this library. A library cannot ship a profile.

## Assembly and the `purego` tag

On arm64 (Linux, macOS, Windows) the codec runs its hot loops as NEON
assembly, written for the Go assembler (no cgo):

- lossy decoding: macroblock reconstruction (intra prediction, inverse
  DCT and WHT), the in-loop deblocking filters, fancy upsampling and the
  Y'CbCr to RGBA conversion;
- lossless decoding: the inverse transforms (predictors, cross-color,
  add-green), the RGBA output, palette expansion, and for the alpha plane
  of lossy pictures the palette lookup and the horizontal and vertical
  unfilters;
- scaling (`Scale`): libwebp's rescaler and the alpha premultiplication;
- lossy encoding: forward and inverse DCT, quantization, distortion
  metrics, intra prediction and the analysis histogram.

All other architectures run the same algorithms in pure Go; amd64
assembly is planned. Building with `-tags purego` selects the pure Go code
on arm64 too (for example to rule the assembly out while investigating a
problem).

The results do not depend on it: every assembly function is bit-exact with
the Go function it replaces, so decoded pixels and encoded bytes are the
same with and without `-tags purego`, on every architecture. The tests
compare each assembly function with its Go counterpart (exhaustively where
the input space allows it, otherwise on millions of random and boundary
inputs, with fuzz targets and guard pages that catch out-of-bounds
accesses), and CI runs the whole test suite, conformance tests included,
on both code paths. Nothing else is needed: no build flag, no
`GOEXPERIMENT`, no extra dependency. One caveat: the race detector does
not see memory accesses made by assembly, so also run `go test -race
-tags purego` when looking for races around buffers shared with the codec.

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
- `UseThreads` (decoder) and `ThreadLevel` (encoder) use goroutines from a
  process-wide budget instead of libwebp's worker threads (see
  [Threads](#threads)); outputs do not depend on them. `UseThreads` also
  speeds up lossless pictures, which libwebp decodes on one thread.
- Allocations are limited like libwebp's `WEBP_MAX_ALLOCABLE_MEMORY`
  (16 GiB on 64-bit targets, just under 2 GiB on 32-bit ones): a `Scale` or
  crop whose output would exceed it fails with `VP8_STATUS_OUT_OF_MEMORY`.
- Raw VP8/VP8L bitstreams without a RIFF header are accepted by
  `decoder.NewDecoder`, `webp.Decode`, `webp.DecodeConfig` and
  `webp.GetFeatures`, as in libwebp; `image.Decode` / `image.DecodeConfig`
  only recognize RIFF files (same registered magic as v1).

## Benchmarks

Apple M5 Pro (15 cores), Go 1.27.1, darwin/arm64, otherwise idle. All
columns were measured in one session with prebuilt benchmark binaries run
alternately, 10 rounds each (`-benchmem`; 1 s per decode benchmark, 2 s
per encode benchmark); the table shows medians, and the percentages are
the time difference from v1 (benchstat, p < 0.05; "~" means no
significant difference). The encode rows of the two v2 columns were
re-measured in the same session after the last encoder change (the
unchanged binaries reproduced their first round within 1%).

- **v1**: v1.0.5 with Homebrew libwebp 1.6.0 (cgo; libwebp uses its own
  NEON code). Its memory counts only Go allocations: libwebp's C heap is
  not visible to the Go benchmark.
- **v2, pure Go**: this version built with `-tags purego`, the code that
  runs on amd64 and every other architecture for now (measured here on
  the same arm64 machine).
- **v2 (NEON)**: this version as built by default on arm64.

MB = 10^6 bytes allocated per operation.

| Benchmark | v1 (libwebp) | v2, pure Go (`-tags purego`) | v2 (NEON) |
|---|---|---|---|
| Decode lossy 675x900 (`m4_q75.webp`) | 3.00 ms, 2.57 MB, 21 allocs | 6.60 ms (+120%), 2.50 MB, 10 allocs | 2.99 ms (~), 2.50 MB, 10 allocs |
| ↳ with `UseThreads` | 2.39 ms, 2.57 MB, 21 allocs | 4.45 ms (+87%), 2.50 MB, 18 allocs | 2.47 ms (+4%), 2.50 MB, 18 allocs |
| ↳ scaled to 320x240 | 3.33 ms, 0.45 MB, 21 allocs | 4.30 ms (+29%), 0.38 MB, 12 allocs | 2.88 ms (−13%), 0.38 MB, 12 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 6.42 ms, 0.08 MB, 8 allocs | 2.94 ms, 0.07 MB, 8 allocs |
| ↳ `DecodeYCbCr` |  | 5.25 ms, 0.98 MB, 9 allocs | 2.84 ms, 0.98 MB, 9 allocs |
| Decode lossy+alpha 3000x2000 (`webp-logo-lossy.webp`) | 10.6 ms, 24.2 MB, 22 allocs | 22.3 ms (+110%), 24.2 MB, 15 allocs | 8.61 ms (−19%), 24.1 MB, 15 allocs |
| ↳ with `UseThreads` | 8.54 ms, 24.2 MB, 22 allocs | 17.2 ms (+101%), 24.2 MB, 29 allocs | 5.17 ms (−40%), 24.1 MB, 29 allocs |
| ↳ scaled to 320x240 | 13.1 ms, 0.50 MB, 22 allocs | 14.0 ms (+7%), 0.50 MB, 16 allocs | 6.74 ms (−49%), 0.43 MB, 15 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 21.9 ms, 0.70 MB, 11 allocs | 8.42 ms, 0.29 MB, 11 allocs |
| ↳ `DecodeNYCbCrA` |  | 10.9 ms, 15.2 MB, 14 allocs | 7.45 ms, 15.1 MB, 14 allocs |
| Decode lossless 3000x2000 (`webp-logo-lossless.webp`) | 5.57 ms, 24.2 MB, 22 allocs | 2.61 ms (−53%), 24.1 MB, 10 allocs | 2.66 ms (−52%), 24.1 MB, 10 allocs |
| ↳ with `UseThreads` | 5.56 ms, 24.2 MB, 22 allocs | 1.94 ms (−65%), 24.1 MB, 18 allocs | 1.97 ms (−65%), 24.1 MB, 18 allocs |
| ↳ scaled to 320x240 | 16.8 ms, 0.52 MB, 22 allocs | 5.69 ms (−66%), 0.41 MB, 10 allocs | 3.07 ms (−82%), 0.41 MB, 9 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 2.19 ms, 0.14 MB, 8 allocs | 2.27 ms, 0.14 MB, 8 allocs |
| Encode lossy q75, 960x1280 JPEG source | 64.2 ms, 117 858 bytes, 4.92 MB, 7 allocs | 120 ms (+87%), 117 918 bytes, 0.13 MB, 10 allocs | 46.7 ms (−27%), 117 918 bytes, 0.13 MB, 10 allocs |
| ↳ with `ThreadLevel` | 61.4 ms, 117 858 bytes, 4.92 MB, 7 allocs | 114 ms (+86%), 117 918 bytes, 0.13 MB, 41 allocs | 45.5 ms (−26%), 117 918 bytes, 0.13 MB, 40 allocs |
| ↳ method 0 | 22.8 ms, 150 802 bytes, 4.92 MB, 7 allocs | 26.9 ms (+18%), 149 144 bytes, 0.16 MB, 10 allocs | 17.2 ms (−24%), 149 144 bytes, 0.16 MB, 10 allocs |
| ↳ method 0 with `ThreadLevel` | 21.5 ms, 150 802 bytes, 4.92 MB, 7 allocs | 24.6 ms (+14%), 149 144 bytes, 0.16 MB, 42 allocs | 16.5 ms (−23%), 149 144 bytes, 0.16 MB, 40 allocs |
| ↳ method 6 | 121 ms, 108 840 bytes, 4.92 MB, 7 allocs | 173 ms (+43%), 108 710 bytes, 0.12 MB, 11 allocs | 120 ms (−1%), 108 710 bytes, 0.12 MB, 10 allocs |
| ↳ method 6 with `ThreadLevel` | 118 ms, 108 840 bytes, 4.92 MB, 7 allocs | 166 ms (+41%), 108 710 bytes, 0.12 MB, 42 allocs | 119 ms (~), 108 710 bytes, 0.12 MB, 41 allocs |
| Encode lossy+alpha q75, 3000x2000 logo | 261 ms, 57 584 bytes, 360 B, 5 allocs | 457 ms (+75%), 57 538 bytes, 0.54 MB, 50 allocs | 243 ms (−7%), 57 538 bytes, 0.57 MB, 50 allocs |
| ↳ with `ThreadLevel` | 159 ms, 57 584 bytes, 360 B, 5 allocs | 274 ms (+72%), 57 538 bytes, 0.74 MB, 88 allocs | 154 ms (−4%), 57 538 bytes, 0.93 MB, 88 allocs |
| Encode lossless level 6, same JPEG source | 381 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 309 ms (−19%), 1 098 360 bytes, 7.99 MB, 50 allocs | 309 ms (−19%), 1 098 360 bytes, 7.99 MB, 50 allocs |
| ↳ with `ThreadLevel` | 381 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 159 ms (−58%), 1 098 360 bytes, 2.60 MB, 1434 allocs | 158 ms (−59%), 1 098 360 bytes, 2.60 MB, 1434 allocs |
| Encode lossless level 0, same JPEG source | 38.7 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 46.2 ms (+19%), 1 216 356 bytes, 1.57 MB, 20 allocs | 46.4 ms (+20%), 1 216 356 bytes, 1.55 MB, 20 allocs |
| ↳ with `ThreadLevel` | 38.6 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 37.1 ms (−4%), 1 216 356 bytes, 1.62 MB, 55 allocs | 37.1 ms (−4%), 1 216 356 bytes, 1.63 MB, 55 allocs |

Decoded pixels are identical in all columns (bit-exact with libwebp 1.6
`dwebp`; when scaling, v2 matches libwebp's portable C rescaler, and
libwebp's NEON rescaler used by v1 here rounds a few pixels differently
by 1), and the two v2 columns encode identical bytes. With `UseThreads`
/ `ThreadLevel` the process has 14 helper slots here. The lossless logo is
a 256-colour palette picture, which the NEON kernels do not speed up;
lossless photos and graphics decode 20-55% faster with them. Lossy
pictures scaled to a thumbnail now decode faster than at full size.

Peak resident memory of a whole process (`getrusage`, median of 3 runs, 2
for the lossless encodes, same session, default `GOGC`, no `GOMEMLIMIT`
unless stated):

| Workload | v1 (libwebp) | v2 (NEON) |
|---|---|---|
| One 12 MP lossy encode (4000x3000 `*image.YCbCr`, q75; the source takes 46 MB) | 241 MB, 0.76 s | 78 MB, 0.43 s |
| One 12 MP lossless level 6 encode (upscaled photo) | 336 MB, 4.2 s | 344 MB, 3.2 s |
| Three 12 MP lossless level 6 encodes in a row | 439 MB | 657-705 MB (458-485 MB with `GOMEMLIMIT=450MiB`) |
| One 12 MP lossless decode (output 48 MB) | 113 MB | 69 MB |
| 15 goroutines x 4 decodes of the 6 MP lossless logo | 736 MB | 397 MB |
| 15 goroutines x 4 decodes of the 6 MP lossy+alpha logo | 558 MB | 489 MB |

The output image (24 MB for 3000x2000) dominates the decoder's memory;
the decoder itself keeps O(width) state. v1 converts every source other
than `*image.NRGBA` to `*image.NRGBA` in Go before calling libwebp (the
4.9 MB per encode of the JPEG source above). Large lossless encodes can
peak higher than v1 when several run in a row: the garbage collector runs
while much of their working set (about 30 bytes per pixel at level 6) is
live, and its next target is twice that; `GOMEMLIMIT` bounds it. Compared
with `golang.org/x/image/webp` (decoder only), the first pure Go v2
already decoded lossy pictures 1.6-2.6x and lossless pictures 1.7-3.4x
faster, with far fewer allocations.

```bash
make bench
```

## Tests

```bash
go test ./...                              # unit, round-trip and conformance tests
go test -tags purego ./...                 # the same on the pure Go code path
go test -tags interop ./...                # also compare with cwebp/dwebp (libwebp tools in PATH)
go test -run '^$' -fuzz FuzzDecode ./decoder
```

## License
MIT licensed. See the LICENSE file for details.

Parts of the codec are adapted from [libwebp](https://chromium.googlesource.com/webm/libwebp)
(Copyright 2010 Google Inc., BSD-3-Clause); see the NOTICE file.
