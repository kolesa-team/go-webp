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
  - lossy pictures of at least 1024 macroblocks (about 512x512): the
    caller parses the coefficient tokens of batches of macroblock rows
    while a helper reconstructs and filters them; when that helper is the
    slower stage (the pure Go build, large pictures), the conversion to
    RGBA moves to a second helper, and idle helpers parse the prediction
    modes ahead of the caller;
  - lossy pictures with alpha whose rows down to the bottom of the output
    hold 256K pixels or more: another helper decodes the alpha plane ahead
    of the rows that need it;
  - lossless pictures of at least about 256K output pixels: the caller
    entropy-decodes while a helper applies the inverse transforms and
    writes the output rows (libwebp ignores `use_threads` for lossless
    pictures; this is a go-webp extension).
- `encoder.Options.ThreadLevel`:
  - lossy: the analysis pass runs in row bands (384 macroblocks or more,
    1024 at methods 0-1); the macroblock rows are coded as a wavefront on
    up to 8 goroutines when the picture is large enough to pay for the
    hand-offs (on arm64 about 0.3 MP at methods 3-6, 1700 pixels wide at
    method 2, several MP at methods 0-1; about 0.1 MP at methods 3-6 in the
    pure Go build), taking at most `GOMAXPROCS` / (lossy encodes in
    progress) goroutines; and the compressed alpha plane of a transparent
    picture (64K pixels or more) is coded while the VP8 frame is;
  - lossless and compressed alpha (256K pixels or more): the hash-chain
    match search, the predictor search (methods 0-4) and the histogram
    remap are split among helpers, which are released at the end of each
    of these phases; level 9 codes at most two candidate configurations at
    a time (one at a time above 8 MP).

All helpers of the process come from one budget of `GOMAXPROCS-1`
goroutines (re-read on every call), shared by all concurrent `Decode` and
`Encode` calls. A call that finds no free slot does the work on its own
goroutine, so a server that already runs one request per core is never
oversubscribed. **The output is identical with or without the flags and
for any budget.** A panic in a helper is re-raised on the calling
goroutine, and helpers have exited when the call returns (also on errors
and cancellation). Goroutines wait for each other by blocking on channels
or condition variables, never by spinning. While processors are idle, a
decoder stage that wakes a parked one yields its processor once
(`runtime.Gosched`), so that the woken stage runs at once instead of after
the scheduler's wake-up delay (hundreds of microseconds on Linux).

The flags trade CPU for latency. On the benchmark machine (see
[Benchmarks](#benchmarks); CPU time is the user and system time of the
whole process per call, garbage collection included; the percentages
compare the call with the flag to the same call on one goroutine):

| Call (wall / CPU time) | NEON, one goroutine | NEON, with the flag | pure Go, one goroutine | pure Go, with the flag |
|---|---|---|---|---|
| Decode lossy 675x900 | 3.08 ms / 3.20 ms | 2.00 ms / 3.47 ms (−35% / +8%) | 6.61 ms / 6.73 ms | 4.40 ms / 7.05 ms (−33% / +5%) |
| Decode lossy+alpha 3000x2000 | 8.75 ms / 8.91 ms | 4.35 ms / 10.6 ms (−50% / +19%) | 22.5 ms / 22.7 ms | 13.1 ms / 24.2 ms (−42% / +7%) |
| Decode lossless 3000x2000 | 2.68 ms / 2.86 ms | 2.00 ms / 3.15 ms (−25% / +10%) | 2.68 ms / 2.85 ms | 1.96 ms / 3.08 ms (−27% / +8%) |
| Encode lossy q75 method 0, 1.2 MP | 17.2 ms / 17.2 ms | 16.5 ms / 17.6 ms (−4% / +2%) | 27.1 ms / 27.1 ms | 24.8 ms / 27.8 ms (−9% / +3%) |
| Encode lossy q75 method 4, 1.2 MP | 47.1 ms / 47.1 ms | 21.2 ms / 51.6 ms (−55% / +10%) | 121 ms / 121 ms | 39.4 ms / 132 ms (−67% / +9%) |
| Encode lossy q75 method 6, 1.2 MP | 121 ms / 121 ms | 40.2 ms / 129 ms (−67% / +7%) | 174 ms / 174 ms | 52.4 ms / 187 ms (−70% / +7%) |
| Encode lossy+alpha q75, 3000x2000 | 245 ms / 245 ms | 156 ms / 261 ms (−37% / +6%) | 460 ms / 459 ms | 159 ms / 524 ms (−65% / +14%) |
| Encode lossless level 6, 1.2 MP | 306 ms / 306 ms | 157 ms / 342 ms (−49% / +11%) | 308 ms / 308 ms | 158 ms / 345 ms (−49% / +12%) |
| Encode lossless level 0, 1.2 MP | 45.4 ms / 45.4 ms | 36.2 ms / 47.0 ms (−20% / +3%) | 45.5 ms / 45.5 ms | 36.1 ms / 46.8 ms (−21% / +3%) |

With the flags, a call takes free helper slots only. Under saturation (15
calls at a time on 15 cores) a call rarely finds one: `ThreadLevel` then
changed encode throughput by less than the run-to-run noise (about ±5%),
while `UseThreads` lowered the throughput of lossy decodes by about 5-15%
(the pipeline's hand-offs cost CPU). The flags therefore help most an
application that decodes or encodes one large picture at a time, or a
server with spare cores; a server that already runs about one call per
core may leave them off, which remains the default.

On Linux VMs with coarse timers (for example Docker Desktop's linux/arm64
VM) every hand-off between goroutines costs about a millisecond, so there
the flags pay off on large or slow pictures only. Measured there with 4
CPUs: `UseThreads` decodes the 3000x2000 lossy+alpha logo in 7.1 ms
instead of 9.5 ms and the 675x900 lossy picture in 2.95 ms instead of
3.26 ms, but the lossless logo in 3.2 ms instead of 2.8 ms; `ThreadLevel`
encodes the 1.2 MP photo at method 6 in 88 ms (about 123 ms on one
goroutine) and at method 4 in 47.5 ms (about 49 ms).

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
  container's memory limit; the garbage collector then keeps large
  lossless encodes under it (see [Benchmarks](#benchmarks)). In our
  measurements of 15 concurrent decodes `GOMEMLIMIT` alone cost 7-33%
  more CPU; together with `DecodeInto` it cost nothing measurable.
- Between calls the encoders keep reusable buffers (the lossy and lossless
  working sets, `Encoder` snapshots, alpha planes) under one process-wide
  budget: 256 MiB (64 MiB on 32-bit platforms), or an eighth of the Go
  memory limit (`GOMEMLIMIT`, `debug.SetMemoryLimit`) when that is lower,
  re-read at most once a second. Buffers unused for 15 s are dropped at
  the next garbage collection, which the runtime forces at least every two
  minutes. A single large encode may keep its whole working set for that
  long (about 200 MB after a 12 MP lossless encode): that is what makes
  back-to-back large encodes cheap. Under a tight `GOMEMLIMIT` the eighth
  is too small to keep such a working set, and back-to-back large
  lossless encodes peak higher, though below the limit (see the memory
  table).
- The decoder keeps O(width) state besides the output. The alpha plane of
  a lossy picture is decoded through a ring of rows (about 256 KB, or
  about 2.5 MB at 3000 pixels wide when the `UseThreads` alpha helper
  runs) instead of as a whole plane; only `AlphaDitheringStrength` still
  needs the whole plane.
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
significant difference). The CPU time that `UseThreads` and `ThreadLevel`
cost is in [Threads](#threads).

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
| Decode lossy 675x900 (`m4_q75.webp`) | 3.06 ms, 2.57 MB, 21 allocs | 6.61 ms (+116%), 2.50 MB, 10 allocs | 3.08 ms (~), 2.50 MB, 10 allocs |
| ↳ with `UseThreads` | 2.39 ms, 2.57 MB, 21 allocs | 4.40 ms (+84%), 2.51 MB, 24 allocs | 2.00 ms (−16%), 2.50 MB, 17 allocs |
| ↳ scaled to 320x240 | 3.43 ms, 0.45 MB, 21 allocs | 4.33 ms (+27%), 0.38 MB, 12 allocs | 2.94 ms (−14%), 0.38 MB, 12 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 6.52 ms, 0.08 MB, 8 allocs | 2.97 ms, 0.07 MB, 8 allocs |
| ↳ `DecodeYCbCr` |  | 5.32 ms, 0.98 MB, 9 allocs | 2.87 ms, 0.98 MB, 9 allocs |
| Decode lossy+alpha 3000x2000 (`webp-logo-lossy.webp`) | 10.7 ms, 24.2 MB, 22 allocs | 22.5 ms (+111%), 24.1 MB, 15 allocs | 8.75 ms (−18%), 24.1 MB, 15 allocs |
| ↳ with `UseThreads` | 8.61 ms, 24.2 MB, 22 allocs | 13.1 ms (+52%), 24.1 MB, 36 allocs | 4.35 ms (−50%), 24.1 MB, 36 allocs |
| ↳ scaled to 320x240 | 13.3 ms, 0.50 MB, 22 allocs | 14.0 ms (+6%), 0.46 MB, 16 allocs | 6.81 ms (−49%), 0.41 MB, 16 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 22.2 ms, 0.57 MB, 11 allocs | 8.53 ms, 0.26 MB, 11 allocs |
| ↳ `DecodeNYCbCrA` |  | 11.0 ms, 15.1 MB, 14 allocs | 7.56 ms, 15.1 MB, 14 allocs |
| Decode lossless 3000x2000 (`webp-logo-lossless.webp`) | 5.66 ms, 24.2 MB, 22 allocs | 2.68 ms (−53%), 24.1 MB, 10 allocs | 2.68 ms (−53%), 24.1 MB, 10 allocs |
| ↳ with `UseThreads` | 5.64 ms, 24.2 MB, 22 allocs | 1.96 ms (−65%), 24.1 MB, 18 allocs | 2.00 ms (−65%), 24.1 MB, 18 allocs |
| ↳ scaled to 320x240 | 16.9 ms, 0.52 MB, 22 allocs | 5.73 ms (−66%), 0.42 MB, 10 allocs | 3.12 ms (−82%), 0.41 MB, 9 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 2.30 ms, 0.14 MB, 8 allocs | 2.28 ms, 0.14 MB, 8 allocs |
| Encode lossy q75, 960x1280 JPEG source | 64.8 ms, 117 858 bytes, 4.92 MB, 7 allocs | 121 ms (+86%), 117 918 bytes, 0.13 MB, 10 allocs | 47.1 ms (−27%), 117 918 bytes, 0.13 MB, 10 allocs |
| ↳ with `ThreadLevel` | 61.9 ms, 117 858 bytes, 4.92 MB, 7 allocs | 39.4 ms (−36%), 117 918 bytes, 0.13 MB, 69 allocs | 21.2 ms (−66%), 117 918 bytes, 0.13 MB, 68 allocs |
| ↳ method 0 | 23.1 ms, 150 802 bytes, 4.92 MB, 7 allocs | 27.1 ms (+17%), 149 144 bytes, 0.16 MB, 10 allocs | 17.2 ms (−26%), 149 144 bytes, 0.16 MB, 10 allocs |
| ↳ method 0 with `ThreadLevel` | 21.8 ms, 150 802 bytes, 4.92 MB, 7 allocs | 24.8 ms (+14%), 149 144 bytes, 0.16 MB, 41 allocs | 16.5 ms (−24%), 149 144 bytes, 0.16 MB, 40 allocs |
| ↳ method 6 | 122 ms, 108 840 bytes, 4.92 MB, 7 allocs | 174 ms (+43%), 108 710 bytes, 0.12 MB, 11 allocs | 121 ms (−1%), 108 710 bytes, 0.12 MB, 10 allocs |
| ↳ method 6 with `ThreadLevel` | 119 ms, 108 840 bytes, 4.92 MB, 7 allocs | 52.4 ms (−56%), 108 710 bytes, 0.12 MB, 69 allocs | 40.2 ms (−66%), 108 710 bytes, 0.12 MB, 67 allocs |
| Encode lossy+alpha q75, 3000x2000 logo | 262 ms, 57 584 bytes, 360 B, 5 allocs | 460 ms (+75%), 57 538 bytes, 0.55 MB, 46 allocs | 245 ms (−6%), 57 538 bytes, 0.62 MB, 44 allocs |
| ↳ with `ThreadLevel` | 160 ms, 57 584 bytes, 360 B, 5 allocs | 159 ms (~), 57 538 bytes, 0.98 MB, 121 allocs | 156 ms (−3%), 57 538 bytes, 0.94 MB, 122 allocs |
| Encode lossless level 6, same JPEG source | 389 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 308 ms (−21%), 1 098 360 bytes, 1.63 MB, 24 allocs | 306 ms (−21%), 1 098 360 bytes, 1.63 MB, 23 allocs |
| ↳ with `ThreadLevel` | 389 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 158 ms (−59%), 1 098 360 bytes, 1.93 MB, 117 allocs | 157 ms (−60%), 1 098 360 bytes, 2.01 MB, 116 allocs |
| Encode lossless level 0, same JPEG source | 39.2 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 45.5 ms (+16%), 1 216 356 bytes, 1.41 MB, 19 allocs | 45.4 ms (+16%), 1 216 356 bytes, 1.39 MB, 19 allocs |
| ↳ with `ThreadLevel` | 39.3 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 36.1 ms (−8%), 1 216 356 bytes, 1.49 MB, 48 allocs | 36.2 ms (−8%), 1 216 356 bytes, 1.48 MB, 49 allocs |

Decoded pixels are identical in all columns (bit-exact with libwebp 1.6
`dwebp`; when scaling, v2 matches libwebp's portable C rescaler, and
libwebp's NEON rescaler used by v1 here rounds a few pixels differently
by 1), and the two v2 columns encode identical bytes. With `UseThreads`
/ `ThreadLevel` the process has 14 helper slots here; the lossy+alpha
encode with `ThreadLevel` is then bound by its lossless alpha plane, coded
alongside the VP8 frame, so the NEON build gains nothing over the pure Go
one there. The lossless logo is a 256-colour palette picture, which the
NEON kernels do not speed up; lossless photos and graphics decode 20-55%
faster with them. Lossy pictures scaled to a thumbnail now decode faster
than at full size.

Peak resident memory of a whole process (`getrusage`, median of 3 runs,
same session, default `GOGC`, no `GOMEMLIMIT` unless stated; the pure Go
build peaks within a few MB of the NEON one):

| Workload | v1 (libwebp) | v2 (NEON) |
|---|---|---|
| One 12 MP lossy encode (4000x3000 `*image.YCbCr`, q75; the source takes 46 MB) | 241 MB, 0.79 s | 78 MB, 0.43 s |
| ↳ with `ThreadLevel` | 241 MB, 0.75 s | 80 MB, 0.17 s |
| One 12 MP lossless level 6 encode (upscaled photo) | 336 MB, 4.3 s | 262 MB, 3.2 s |
| Three 12 MP lossless level 6 encodes in a row | 434 MB | 272 MB |
| ↳ with `GOMEMLIMIT=450MiB` | 428 MB | 421 MB |
| One 12 MP lossless decode (output 48 MB) | 113 MB | 69 MB |
| 15 goroutines x 4 decodes of the 6 MP lossless logo | 1115 MB | 760-1030 MB |
| ↳ `DecodeInto`, one buffer per goroutine | | 401 MB |
| 15 goroutines x 4 decodes of the 6 MP lossy+alpha logo | 921 MB | 770-910 MB |
| ↳ `DecodeInto`, one buffer per goroutine | | 407 MB |

The output image (24 MB for 3000x2000) dominates the decoder's memory
(with `Decode`, how many outputs are still waiting for the garbage
collector varies from run to run); the decoder itself keeps O(width)
state. v1 converts every source other than `*image.NRGBA` to
`*image.NRGBA` in Go before calling libwebp (the 4.9 MB per encode of the
JPEG source above). A large lossless encode keeps its working set in the
encoder's retained budget (see [Running on a server](#running-on-a-server)),
so back-to-back encodes reuse it instead of allocating it again before the
garbage collector has reclaimed it; under `GOMEMLIMIT=450MiB` the budget
(an eighth of the limit) is too small to keep it, and the collector paces
the peak instead. Compared with `golang.org/x/image/webp` (decoder only),
the first pure Go v2 already decoded lossy pictures 1.6-2.6x and lossless
pictures 1.7-3.4x faster, with far fewer allocations.

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
