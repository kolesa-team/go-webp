![logo](test_data/images/logo.jpg)
# go-webp

[![Release](https://img.shields.io/github/v/release/kolesa-team/go-webp)](https://github.com/kolesa-team/go-webp/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/kolesa-team/go-webp/v2.svg)](https://pkg.go.dev/github.com/kolesa-team/go-webp/v2)
[![CI](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/kolesa-team/go-webp/actions/workflows/ci.yml?query=branch%3Amaster)
[![Coverage](https://raw.githubusercontent.com/kolesa-team/go-webp/badges/.badges/master/coverage.svg)](https://kolesa-team.github.io/go-webp/)
[![Go version](https://img.shields.io/github/go-mod/go-version/kolesa-team/go-webp)](go.mod)
[![License: MIT](https://img.shields.io/github/license/kolesa-team/go-webp)](LICENSE)
[![Mentioned in Awesome Go](https://awesome.re/mentioned-badge.svg)](https://github.com/avelino/awesome-go)

WebP encoder and decoder written in Go — **no cgo**, no libwebp; the only
dependency outside the standard library is `golang.org/x/sys` (CPU feature
detection). On arm64 and amd64 the hot loops run as assembly (NEON on
arm64; SSE4.1, AVX2 or AVX-512 on amd64, picked at run time), with results
identical to the pure Go code that other architectures and `-tags purego`
builds use (see [Assembly and the `purego` tag](#assembly-and-the-purego-tag)).

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

Requires Go 1.27 or newer. The only other module it pulls in is
`golang.org/x/sys`, which amd64 builds use to read the CPU's features; no C
toolchain or system library is needed.

The cgo/libwebp based v1 remains available at tag `v1.0.6` (module path
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
    pictures; this is a go-webp extension). The caller first writes 32K
    pixels itself and times them; the helper starts only if the rest of
    the picture projects to at least 200 µs of that work (pictures made of
    long copies of one colour do not gain from it). On Linux it also
    starts only when all the CPUs the process may run on share one
    last-level cache: with the rows in another core complex's L3 cache the
    helper made the 3000x2000 logo 10% slower on an 8-core EPYC (two L3
    caches) instead of 12-24% faster, so pin such a process to one L3
    domain (`taskset`, a cpuset) to get the gain.
- `encoder.Options.ThreadLevel`:
  - lossy: the analysis pass runs in row bands (384 macroblocks or more,
    1024 at methods 0-1); the macroblock rows are coded as a wavefront on
    up to 8 goroutines when the picture is large enough to pay for the
    hand-offs (with the assembly kernels of arm64 and amd64 about 0.3 MP
    at methods 3-6, 1700 pixels wide at method 2, several MP at methods
    0-1; about 0.1 MP at methods 3-6 in the pure Go build), taking at
    most `GOMAXPROCS` / (lossy encodes in progress) goroutines; and the
    compressed alpha plane of a transparent picture (64K pixels or more)
    is coded while the VP8 frame is;
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

The flags trade CPU for latency. Measured on an Apple M5 Pro (15 cores;
see [Benchmarks](#benchmarks) for the method; CPU time is the user and
system time of the whole process per call, garbage collection included;
the percentages compare the call with the flag to the same call on one
goroutine):

| Call (wall / CPU time) | NEON, one goroutine | NEON, with the flag | pure Go, one goroutine | pure Go, with the flag |
|---|---|---|---|---|
| Decode lossy 675x900 | 3.02 ms / 3.14 ms | 2.01 ms / 3.47 ms (−34% / +11%) | 6.63 ms / 6.76 ms | 4.40 ms / 7.04 ms (−34% / +4%) |
| Decode lossy+alpha 3000x2000 | 8.72 ms / 8.89 ms | 4.30 ms / 10.5 ms (−51% / +18%) | 22.4 ms / 22.6 ms | 13.1 ms / 24.2 ms (−42% / +7%) |
| Decode lossless 3000x2000 | 2.67 ms / 2.86 ms | 1.97 ms / 3.09 ms (−26% / +8%) | 2.67 ms / 2.85 ms | 1.95 ms / 3.06 ms (−27% / +7%) |
| Encode lossy q75 method 0, 1.2 MP | 17.3 ms / 17.3 ms | 16.5 ms / 17.5 ms (−4% / +1%) | 27.1 ms / 27.1 ms | 24.8 ms / 27.8 ms (−8% / +3%) |
| Encode lossy q75 method 4, 1.2 MP | 47.2 ms / 47.2 ms | 21.3 ms / 51.8 ms (−55% / +10%) | 120 ms / 120 ms | 39.1 ms / 131 ms (−67% / +9%) |
| Encode lossy q75 method 6, 1.2 MP | 120 ms / 120 ms | 40.2 ms / 129 ms (−67% / +7%) | 174 ms / 174 ms | 52.1 ms / 186 ms (−70% / +7%) |
| Encode lossy+alpha q75, 3000x2000 | 245 ms / 245 ms | 155 ms / 260 ms (−37% / +6%) | 458 ms / 458 ms | 158 ms / 522 ms (−65% / +14%) |
| Encode lossless level 6, 1.2 MP | 307 ms / 307 ms | 157 ms / 342 ms (−49% / +11%) | 306 ms / 306 ms | 157 ms / 341 ms (−49% / +11%) |
| Encode lossless level 0, 1.2 MP | 45.1 ms / 45.2 ms | 36.2 ms / 46.8 ms (−20% / +4%) | 45.1 ms / 45.1 ms | 36.1 ms / 46.8 ms (−20% / +4%) |

With the flags, a call takes free helper slots only. Under saturation (15
calls at a time on 15 cores, see [Throughput under
load](#throughput-under-load)) a call rarely finds one: `ThreadLevel` then
changed encode throughput by less than the run-to-run noise (about ±10%
there), while `UseThreads` lowered the throughput of lossy+alpha and
lossless decodes by about 10% (the pipeline's hand-offs cost CPU). The
flags therefore help most an application that decodes or encodes one large
picture at a time, or a server with spare cores; a server that already
runs about one call per core may leave them off, which remains the
default.

On the amd64 server of the [Benchmarks](#amd-epyc-8024p-amd64) section
(8 cores, Linux), measured the same way:

| Call (wall / CPU time) | assembly, one goroutine | assembly, with the flag | pure Go, one goroutine | pure Go, with the flag |
|---|---|---|---|---|
| Decode lossy 675x900 | 5.88 ms / 6.22 ms | 3.97 ms / 6.82 ms (−32% / +10%) | 15.6 ms / 16.1 ms | 11.4 ms / 16.7 ms (−27% / +4%) |
| Decode lossy+alpha 3000x2000 | 15.6 ms / 16.1 ms | 11.9 ms / 26.7 ms (−24% / +66%) | 60.4 ms / 61.2 ms | 37.6 ms / 65.2 ms (−38% / +7%) |
| Decode lossless 3000x2000 | 5.04 ms / 5.41 ms | 5.05 ms / 5.41 ms (+0% / +0%) | 7.74 ms / 8.21 ms | 7.94 ms / 8.42 ms (+3% / +3%) |
| Encode lossy q75 method 0, 1.2 MP | 30.0 ms / 30.0 ms | 32.3 ms / 34.6 ms (+8% / +15%) | 56.9 ms / 57.1 ms | 54.2 ms / 62.1 ms (−5% / +9%) |
| Encode lossy q75 method 4, 1.2 MP | 82.9 ms / 83.1 ms | 38.1 ms / 89.0 ms (−54% / +7%) | 303 ms / 304 ms | 82.8 ms / 309 ms (−73% / +2%) |
| Encode lossy q75 method 6, 1.2 MP | 239 ms / 239 ms | 73.7 ms / 245 ms (−69% / +3%) | 388 ms / 389 ms | 105 ms / 394 ms (−73% / +1%) |
| Encode lossy+alpha q75, 3000x2000 | 517 ms / 518 ms | 327 ms / 533 ms (−37% / +3%) | 1.20 s / 1.21 s | 366 ms / 1.22 s (−70% / +1%) |
| Encode lossless level 6, 1.2 MP | 463 ms / 464 ms | 227 ms / 493 ms (−51% / +6%) | 699 ms / 701 ms | 335 ms / 734 ms (−52% / +5%) |
| Encode lossless level 0, 1.2 MP | 56.0 ms / 56.2 ms | 42.3 ms / 65.0 ms (−24% / +16%) | 103 ms / 103 ms | 79.6 ms / 108 ms (−23% / +4%) |

There `UseThreads` started no helper for the lossless logo, because the
process may run on both of the CPU's L3 caches (see above), and
`ThreadLevel` made the method 0 encode 8% slower, with 15% more CPU time.

On Linux VMs with coarse timers (for example Docker Desktop's linux/arm64
VM) every hand-off between goroutines costs about a millisecond, so there
the flags pay off on large or slow pictures only. Measured there (Docker
Desktop on the same M5 Pro, 4 CPUs; not part of `benchmarks/run.sh`): `UseThreads` decodes the 3000x2000 lossy+alpha logo in 7.1 ms
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
- On amd64 the codec picks its SIMD kernels at run time, so it needs no
  `GOAMD64` setting: building with `GOAMD64=v3` moved its times by −6% to
  +3% (see [Building with GOAMD64=v3](#building-with-goamd64v3)), so
  choose it for the rest of the service, not for the codec.
- On Linux servers with several L3 caches (multi-socket, or EPYC with
  several core complexes), `UseThreads` speeds up lossless decodes only in
  a process confined to one L3 domain (cpuset, `taskset`); see
  [Threads](#threads).

## Assembly and the `purego` tag

On arm64 (Linux, macOS, Windows) and amd64 (every operating system) the
codec runs its hot loops as assembly, written for the Go assembler (no
cgo):

- lossy decoding: macroblock reconstruction (intra prediction, inverse
  DCT and WHT), the in-loop deblocking filters, fancy upsampling and the
  Y'CbCr to RGBA conversion;
- lossless decoding: the inverse transforms (predictors, cross-color,
  add-green), the RGBA output, palette expansion, and for the alpha plane
  of lossy pictures the palette lookup and the horizontal and vertical
  unfilters;
- scaling (`Scale`): libwebp's rescaler and the alpha premultiplication;
- lossy encoding: forward and inverse DCT, quantization, distortion
  metrics, intra prediction and the analysis histogram, and on amd64 also
  the residual cost estimate of the mode decision;
- lossless encoding, on amd64 only (NEON versions gained nothing on
  arm64): predictor residuals, subtract-green, the cross-color and entropy
  estimates, the hash-chain match search, the RLE and colour cache passes,
  the histograms and the entropy-coded writer.

Every arm64 CPU has NEON. On amd64 the kernels come in levels, and the
highest one the CPU supports is picked once, at program start, from CPUID
(through `golang.org/x/sys/cpu`, which also checks that the operating
system saves the vector registers):

| Level | CPUs | What runs at this level |
|---|---|---|
| SSE4.1 | x86-64 CPUs since about 2011 (Intel since 2008, AMD since Bulldozer) | every amd64 kernel |
| AVX2 (x86-64-v3) | Intel Haswell (2013) and later, AMD Excavator and Zen and later: practically every x86 server in use | 256-bit versions where they measured faster: most decoder kernels, the rescaler, the lossy encoder's transforms, quantizer and distortion metrics, three of the lossless encoder's kernels |
| AVX-512 (x86-64-v4) | Intel Xeon since Skylake-SP, AMD Zen 4 and later | the lossless decoder's palette lookups, with AVX-512 VBMI (Intel Ice Lake and later, AMD Zen 4 and later); every other kernel runs its AVX2 version, as wider ones were not at least 10% faster |

CPUs without SSE4.1 (or, for the lossy encoder, POPCNT) run the pure Go
code.
The level does not depend on `GOAMD64`, which only changes the code the Go
compiler generates for the Go parts (see [Running on a
server](#running-on-a-server)). `GODEBUG=cpu.avx512f=off`, `cpu.avx2=off`
or `cpu.sse41=off` lowers the level of a process, for example to compare
levels. Building with `-tags purego` selects the pure Go code on arm64 and
amd64 (for example to rule the assembly out while investigating a
problem); all other architectures always run it.

The results do not depend on it: every assembly function is bit-exact with
the Go function it replaces, at every level, so decoded pixels and encoded
bytes are the same with and without `-tags purego`, on every architecture
and CPU. The tests compare each assembly function with its Go counterpart
at every level the machine supports (exhaustively where the input space
allows it, otherwise on millions of random and boundary inputs, with fuzz
targets and guard pages that catch out-of-bounds accesses), and CI runs
the whole test suite, conformance tests included, on both code paths, on
arm64 and amd64. Nothing else is needed: no build flag, no `GOEXPERIMENT`,
no C toolchain. One caveat: the race detector does not see memory accesses
made by assembly, so also run `go test -race -tags purego` when looking
for races around buffers shared with the codec.

## Migrating from v1

The last v1 (cgo) release is `v1.0.6` (`go get github.com/kolesa-team/go-webp@v1.0.6`).

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

The measurements in this README were taken on two machines with
[`benchmarks/run.sh`](benchmarks/run.sh) (see [Reproducing the
measurements](#reproducing-the-measurements)): an **Apple M5 Pro**
(arm64) and an **AMD EPYC 8024P** server (amd64). Absolute times differ
on other hardware; compare the columns of one table with each other.

- **v1**: v1.0.5 with libwebp 1.6.0 (cgo; libwebp uses its own NEON code
  on arm64 and SSE2/SSE4.1/AVX2 code on amd64). Its memory counts only Go
  allocations: libwebp's C heap is not visible to the Go benchmark.
- **v2, pure Go**: this version built with `-tags purego`, the code that
  every other architecture runs (measured here on the same machines).
- **v2 (NEON)**, **v2 (assembly)**: this version as built by default on
  arm64 and amd64.

Method: every benchmark binary is built once and run in 10 rounds; within a
round the three builds run one after another in a rotating order (1 s per
decode benchmark, 2 s per encode benchmark, `-benchmem`), so that changes of
the machine's state affect all of them alike. The tables show medians; the
percentages are the time difference from v1 (benchstat, p < 0.05; "~"
means no significant difference). The 95% confidence interval of nearly
every time is within ±2% of the median. MB = 10^6 bytes allocated per
operation. The CPU time that `UseThreads` and `ThreadLevel` cost is in
[Threads](#threads).

### Apple M5 Pro (arm64)

15 CPU cores, 24 GB RAM, macOS 27.2, Go 1.27.1, darwin/arm64, Homebrew
libwebp 1.6.0; measured on 2026-11-18 (before the amd64 assembly, which
changed nothing on arm64 but the `UseThreads` start of lossless decodes,
see [Threads](#threads)).

| Benchmark | v1 (libwebp) | v2, pure Go (`-tags purego`) | v2 (NEON) |
|---|---|---|---|
| Decode lossy 675x900 (`m4_q75.webp`) | 3.03 ms, 2.57 MB, 21 allocs | 6.63 ms (+118%), 2.50 MB, 10 allocs | 3.02 ms (~), 2.50 MB, 10 allocs |
| ↳ with `UseThreads` | 2.39 ms, 2.57 MB, 21 allocs | 4.40 ms (+84%), 2.51 MB, 24 allocs | 2.01 ms (−16%), 2.50 MB, 17 allocs |
| ↳ scaled to 320x240 (`Scale`) | 3.39 ms, 0.45 MB, 21 allocs | 4.34 ms (+28%), 0.38 MB, 12 allocs | 2.89 ms (−15%), 0.38 MB, 12 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 6.47 ms, 0.08 MB, 8 allocs | 2.97 ms, 0.07 MB, 8 allocs |
| ↳ `DecodeYCbCr` |  | 5.31 ms, 0.98 MB, 9 allocs | 2.85 ms, 0.98 MB, 9 allocs |
| Decode lossy+alpha 3000x2000 (`webp-logo-lossy.webp`) | 10.7 ms, 24.2 MB, 22 allocs | 22.4 ms (+109%), 24.1 MB, 15 allocs | 8.72 ms (−19%), 24.1 MB, 15 allocs |
| ↳ with `UseThreads` | 8.57 ms, 24.2 MB, 22 allocs | 13.1 ms (+52%), 24.1 MB, 37 allocs | 4.30 ms (−50%), 24.1 MB, 36 allocs |
| ↳ scaled to 320x240 (`Scale`) | 13.3 ms, 0.50 MB, 22 allocs | 14.1 ms (+6%), 0.44 MB, 16 allocs | 6.81 ms (−49%), 0.43 MB, 16 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 22.2 ms, 0.59 MB, 11 allocs | 8.46 ms, 0.26 MB, 11 allocs |
| ↳ `DecodeNYCbCrA` |  | 10.9 ms, 15.1 MB, 14 allocs | 7.46 ms, 15.1 MB, 14 allocs |
| Decode lossless 3000x2000 (`webp-logo-lossless.webp`) | 5.62 ms, 24.2 MB, 22 allocs | 2.67 ms (−53%), 24.1 MB, 10 allocs | 2.67 ms (−52%), 24.1 MB, 10 allocs |
| ↳ with `UseThreads` | 5.66 ms, 24.2 MB, 22 allocs | 1.95 ms (−66%), 24.1 MB, 18 allocs | 1.97 ms (−65%), 24.1 MB, 18 allocs |
| ↳ scaled to 320x240 (`Scale`) | 16.9 ms, 0.52 MB, 22 allocs | 5.71 ms (−66%), 0.42 MB, 10 allocs | 3.09 ms (−82%), 0.41 MB, 9 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 2.24 ms, 0.14 MB, 8 allocs | 2.34 ms, 0.14 MB, 8 allocs |
| Encode lossy q75 method 4, 960x1280 JPEG source (`source.jpg`) | 64.3 ms, 117 858 bytes, 4.92 MB, 7 allocs | 120 ms (+87%), 117 918 bytes, 0.13 MB, 10 allocs | 47.2 ms (−27%), 117 918 bytes, 0.13 MB, 10 allocs |
| ↳ with `ThreadLevel` | 61.8 ms, 117 858 bytes, 4.92 MB, 7 allocs | 39.1 ms (−37%), 117 918 bytes, 0.13 MB, 68 allocs | 21.3 ms (−66%), 117 918 bytes, 0.13 MB, 67 allocs |
| ↳ method 0 | 22.9 ms, 150 802 bytes, 4.92 MB, 7 allocs | 27.1 ms (+18%), 149 144 bytes, 0.16 MB, 10 allocs | 17.3 ms (−25%), 149 144 bytes, 0.16 MB, 10 allocs |
| ↳ method 0 with `ThreadLevel` | 21.6 ms, 150 802 bytes, 4.92 MB, 7 allocs | 24.8 ms (+14%), 149 144 bytes, 0.16 MB, 41 allocs | 16.5 ms (−24%), 149 144 bytes, 0.16 MB, 40 allocs |
| ↳ method 6 | 122 ms, 108 840 bytes, 4.92 MB, 7 allocs | 174 ms (+43%), 108 710 bytes, 0.12 MB, 11 allocs | 120 ms (−1%), 108 710 bytes, 0.12 MB, 10 allocs |
| ↳ method 6 with `ThreadLevel` | 119 ms, 108 840 bytes, 4.92 MB, 7 allocs | 52.1 ms (−56%), 108 710 bytes, 0.12 MB, 68 allocs | 40.2 ms (−66%), 108 710 bytes, 0.12 MB, 67 allocs |
| Encode lossy+alpha q75 method 4, 3000x2000 logo | 261 ms, 57 584 bytes, 360 B, 5 allocs | 458 ms (+75%), 57 538 bytes, 0.74 MB, 46 allocs | 245 ms (−6%), 57 538 bytes, 0.56 MB, 44 allocs |
| ↳ with `ThreadLevel` | 160 ms, 57 584 bytes, 360 B, 5 allocs | 158 ms (~), 57 538 bytes, 0.95 MB, 122 allocs | 155 ms (−3%), 57 538 bytes, 0.95 MB, 120 allocs |
| Encode lossless level 6, same JPEG source | 382 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 306 ms (−20%), 1 098 360 bytes, 1.48 MB, 23 allocs | 307 ms (−20%), 1 098 360 bytes, 1.63 MB, 24 allocs |
| ↳ with `ThreadLevel` | 382 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 157 ms (−59%), 1 098 360 bytes, 1.93 MB, 117 allocs | 157 ms (−59%), 1 098 360 bytes, 2.01 MB, 116 allocs |
| Encode lossless level 0, same JPEG source | 38.9 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 45.1 ms (+16%), 1 216 356 bytes, 1.41 MB, 19 allocs | 45.1 ms (+16%), 1 216 356 bytes, 1.40 MB, 19 allocs |
| ↳ with `ThreadLevel` | 39.0 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 36.1 ms (−8%), 1 216 356 bytes, 1.48 MB, 48 allocs | 36.2 ms (−7%), 1 216 356 bytes, 1.49 MB, 49 allocs |

Decoded pixels are identical in all columns (bit-exact with libwebp 1.6
`dwebp`; when scaling, v2 matches libwebp's portable C rescaler, and
libwebp's NEON rescaler used by v1 here rounds a few pixels differently
by 1), and the two v2 columns encode identical bytes. With `UseThreads`
/ `ThreadLevel` the process has 14 helper slots here; the lossy+alpha
encode with `ThreadLevel` is then bound by its lossless alpha plane, coded
alongside the VP8 frame, so the NEON build gains nothing over the pure Go
one there. The lossless logo is a 256-colour palette picture, which the
NEON kernels do not speed up; lossless photos and graphics decode 20-55%
faster with them. Lossy pictures scaled to a thumbnail decode faster than
at full size.

#### Throughput under load

The same calls, 15 at a time on the 15 cores (`b.RunParallel`, as on a
busy server), in calls per second; the percentages compare the throughput
with v1. Under full load the run-to-run spread is much larger (±10-30%);
"~" marks differences that benchstat does not find significant.

| Call (15 at a time) | v1 (libwebp) | v2, pure Go | v2 (NEON) |
|---|---|---|---|
| Decode lossy 675x900 | 4 197/s | 1 602/s (−62%) | 3 546/s (~) |
| ↳ `UseThreads` | 3 179/s | 1 483/s (−53%) | 3 597/s (+13%) |
| Decode lossy+alpha 3000x2000 | 645/s | 287/s (−55%) | 730/s (+13%) |
| ↳ `UseThreads` | 512/s | 253/s (−51%) | 642/s (+26%) |
| Decode lossless 3000x2000 | 852/s | 1 905/s (+124%) | 1 840/s (+116%) |
| ↳ `UseThreads` | 844/s | 1 558/s (+85%) | 1 681/s (+99%) |
| Encode lossy q75 method 0, 1.2 MP | 316/s | 229/s (−28%) | 427/s (+35%) |
| ↳ `ThreadLevel` | 315/s | 273/s (−14%) | 471/s (+49%) |
| Encode lossy q75 method 4, 1.2 MP | 116/s | 55/s (−53%) | 159/s (+37%) |
| ↳ `ThreadLevel` | 113/s | 56/s (−51%) | 162/s (+43%) |
| Encode lossy q75 method 6, 1.2 MP | 60/s | 41/s (−32%) | 65/s (+8%) |
| ↳ `ThreadLevel` | 61/s | 40/s (−34%) | 66/s (~) |
| Encode lossy+alpha q75, 3000x2000 | 20/s | 12/s (−40%) | 24/s (+18%) |
| ↳ `ThreadLevel` | 23/s | 12/s (−46%) | 26/s (+13%) |
| Encode lossless level 6, 1.2 MP | 18/s | 21/s (+19%) | 22/s (+23%) |
| ↳ `ThreadLevel` | 18/s | 22/s (+22%) | 22/s (+26%) |
| Encode lossless level 0, 1.2 MP | 177/s | 151/s (−15%) | 155/s (−12%) |
| ↳ `ThreadLevel` | 173/s | 143/s (−18%) | 149/s (−14%) |

#### Peak memory

Peak resident memory of a whole process (`getrusage` maxrss, one workload
per process, median of 3 runs with the range when the runs differ by more
than 10%; default `GOGC`, no `GOMEMLIMIT` unless stated; for the encodes
also the wall time of the run). The 12 MP pictures are generated from
`source.jpg`: tiled for the lossy encode, upscaled for the lossless one
(tiling would make a lossless encode unrealistically cheap).

| Workload | v1 (libwebp) | v2, pure Go | v2 (NEON) |
|---|---|---|---|
| One 12 MP lossy encode (4000x3000 `*image.YCbCr`, q75; the source takes 18 MB) | 164 MB, 0.62 s | 66 MB, 1.15 s | 66 MB, 0.42 s |
| ↳ with `ThreadLevel` | 164 MB, 0.59 s | 69 MB, 0.29 s | 69 MB, 0.16 s |
| One 12 MP lossless level 6 encode (upscaled photo) | 340 MB, 4.16 s | 261 MB, 3.13 s | 261 MB, 3.10 s |
| Three 12 MP lossless level 6 encodes in a row | 434 MB, 12.59 s | 270 MB, 9.29 s | 270 MB, 9.21 s |
| ↳ with `GOMEMLIMIT=450MiB` | 434 MB, 12.53 s | 418 MB, 9.38 s | 418 MB, 9.38 s |
| One 12 MP lossless decode (output 48 MB) | 118 MB | 70 MB | 70 MB |
| 15 goroutines x 4 decodes of the 6 MP lossless logo | 1147 MB | 906 MB (832-930) | 858 MB |
| ↳ `DecodeInto`, one buffer per goroutine |  | 401 MB | 401 MB |
| 15 goroutines x 4 decodes of the 6 MP lossy+alpha logo | 1017 MB (897-1114) | 767 MB (767-913) | 864 MB (863-1127) |
| ↳ `DecodeInto`, one buffer per goroutine |  | 408 MB | 408 MB |

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

### AMD EPYC 8024P (amd64)

A bare-metal server: AMD EPYC 8024P (Zen 4c, 8 cores in two core
complexes with a 16 MB L3 cache each, SMT off, `performance` frequency
governor), 62 GiB RAM, Ubuntu 24.04 (Linux 6.8), Go 1.27.1, linux/amd64,
libwebp 1.6.0 built from source; measured on 2026-11-19 with nothing else
running, the same method and the same `benchmarks/run.sh` as above. The
CPU has AVX-512 with VBMI, so v2 runs its AVX-512 level, which for every
kernel but the lossless decoder's palette lookups is the AVX2 code.

| Benchmark | v1 (libwebp) | v2, pure Go (`-tags purego`) | v2 (assembly) |
|---|---|---|---|
| Decode lossy 675x900 (`m4_q75.webp`) | 5.73 ms, 2.57 MB, 21 allocs | 15.6 ms (+173%), 2.50 MB, 10 allocs | 5.88 ms (+3%), 2.50 MB, 10 allocs |
| ↳ with `UseThreads` | 5.78 ms, 2.57 MB, 21 allocs | 11.4 ms (+97%), 2.51 MB, 24 allocs | 3.97 ms (−31%), 2.50 MB, 17 allocs |
| ↳ scaled to 320x240 (`Scale`) | 6.31 ms, 0.45 MB, 21 allocs | 8.94 ms (+42%), 0.38 MB, 12 allocs | 5.12 ms (−19%), 0.38 MB, 12 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 15.1 ms, 0.10 MB, 8 allocs | 5.39 ms, 0.08 MB, 8 allocs |
| ↳ `DecodeYCbCr` |  | 11.9 ms, 0.98 MB, 9 allocs | 5.36 ms, 0.98 MB, 9 allocs |
| Decode lossy+alpha 3000x2000 (`webp-logo-lossy.webp`) | 32.7 ms, 24.2 MB, 22 allocs | 60.4 ms (+85%), 24.2 MB, 17 allocs | 15.6 ms (−52%), 24.1 MB, 15 allocs |
| ↳ with `UseThreads` | 34.3 ms, 24.2 MB, 22 allocs | 37.6 ms (+10%), 24.3 MB, 38 allocs | 11.9 ms (−65%), 24.1 MB, 37 allocs |
| ↳ scaled to 320x240 (`Scale`) | 23.2 ms, 0.50 MB, 22 allocs | 27.7 ms (+20%), 0.44 MB, 16 allocs | 11.5 ms (−50%), 0.41 MB, 16 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 58.6 ms, 1.39 MB, 13 allocs | 14.4 ms, 0.39 MB, 11 allocs |
| ↳ `DecodeNYCbCrA` |  | 27.3 ms, 15.1 MB, 14 allocs | 14.1 ms, 15.1 MB, 14 allocs |
| Decode lossless 3000x2000 (`webp-logo-lossless.webp`) | 12.7 ms, 24.2 MB, 22 allocs | 7.74 ms (−39%), 24.1 MB, 10 allocs | 5.04 ms (−60%), 24.1 MB, 10 allocs |
| ↳ with `UseThreads` | 12.5 ms, 24.2 MB, 22 allocs | 7.94 ms (−37%), 24.1 MB, 10 allocs | 5.05 ms (−60%), 24.1 MB, 10 allocs |
| ↳ scaled to 320x240 (`Scale`) | 17.9 ms, 0.52 MB, 22 allocs | 16.8 ms (−6%), 0.43 MB, 10 allocs | 5.16 ms (−71%), 0.41 MB, 9 allocs |
| ↳ `DecodeInto` (reused buffer) |  | 6.36 ms, 0.23 MB, 8 allocs | 3.81 ms, 0.17 MB, 8 allocs |
| Encode lossy q75 method 4, 960x1280 JPEG source (`source.jpg`) | 127 ms, 117 858 bytes, 4.92 MB, 7 allocs | 303 ms (+139%), 117 918 bytes, 0.13 MB, 12 allocs | 82.9 ms (−35%), 117 918 bytes, 0.13 MB, 10 allocs |
| ↳ with `ThreadLevel` | 122 ms, 117 858 bytes, 4.92 MB, 7 allocs | 82.8 ms (−32%), 117 918 bytes, 0.13 MB, 74 allocs | 38.1 ms (−69%), 117 918 bytes, 0.13 MB, 68 allocs |
| ↳ method 0 | 43.7 ms, 150 802 bytes, 4.92 MB, 7 allocs | 56.9 ms (+30%), 149 144 bytes, 0.16 MB, 10 allocs | 30.0 ms (−31%), 149 144 bytes, 0.16 MB, 10 allocs |
| ↳ method 0 with `ThreadLevel` | 42.1 ms, 150 802 bytes, 4.92 MB, 7 allocs | 54.2 ms (+29%), 149 144 bytes, 0.16 MB, 43 allocs | 32.3 ms (−23%), 149 144 bytes, 0.16 MB, 41 allocs |
| ↳ method 6 | 260 ms, 108 840 bytes, 4.92 MB, 7 allocs | 388 ms (+49%), 108 710 bytes, 0.13 MB, 12 allocs | 239 ms (−8%), 108 710 bytes, 0.12 MB, 11 allocs |
| ↳ method 6 with `ThreadLevel` | 256 ms, 108 840 bytes, 4.92 MB, 7 allocs | 105 ms (−59%), 108 710 bytes, 0.12 MB, 72 allocs | 73.7 ms (−71%), 108 710 bytes, 0.12 MB, 70 allocs |
| Encode lossy+alpha q75 method 4, 3000x2000 logo | 611 ms, 57 584 bytes, 360 B, 5 allocs | 1.20 s (+97%), 57 538 bytes, 0.88 MB, 52 allocs | 517 ms (−15%), 57 538 bytes, 0.60 MB, 47 allocs |
| ↳ with `ThreadLevel` | 399 ms, 57 584 bytes, 360 B, 5 allocs | 366 ms (−8%), 57 538 bytes, 1.42 MB, 124 allocs | 327 ms (−18%), 57 538 bytes, 1.10 MB, 119 allocs |
| Encode lossless level 6, same JPEG source | 693 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 699 ms (~), 1 098 360 bytes, 1.68 MB, 25 allocs | 463 ms (−33%), 1 098 360 bytes, 1.54 MB, 24 allocs |
| ↳ with `ThreadLevel` | 689 ms, 1 098 384 bytes, 4.92 MB, 7 allocs | 335 ms (−51%), 1 098 360 bytes, 2.38 MB, 119 allocs | 227 ms (−67%), 1 098 360 bytes, 2.15 MB, 118 allocs |
| Encode lossless level 0, same JPEG source | 65.4 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 103 ms (+57%), 1 216 356 bytes, 1.41 MB, 19 allocs | 56.0 ms (−14%), 1 216 356 bytes, 1.39 MB, 19 allocs |
| ↳ with `ThreadLevel` | 65.4 ms, 1 201 738 bytes, 4.92 MB, 7 allocs | 79.6 ms (+22%), 1 216 356 bytes, 1.52 MB, 50 allocs | 42.3 ms (−35%), 1 216 356 bytes, 1.50 MB, 50 allocs |

The two v2 columns decode identical pixels and encode identical bytes.
With `UseThreads` / `ThreadLevel` the process has 7 helper slots here. The
lossless decode with `UseThreads` starts no helper on this machine: the
process may run on both core complexes, whose L3 caches are separate (see
[Threads](#threads)); pinned to one complex (4 cores, `taskset`), the
flag made it about 12% faster. The lossless logo is a 256-colour palette
picture, whose lookups use AVX-512 VBMI here (Intel Ice Lake and later,
AMD Zen 4 and later): forced to the AVX2 level, the lossless decoder's
own benchmark of that picture gains 8% over pure Go instead of 39%. A
lossless photo decodes about 30% faster than in pure Go at every level,
SSE4.1 included.

#### Throughput under load, amd64

The same calls, 8 at a time on the 8 cores, in calls per second; the
percentages compare the throughput with v1. With the assembly, v2
completes more calls per second than v1 in every row.

| Call (8 at a time) | v1 (libwebp) | v2, pure Go | v2 (assembly) |
|---|---|---|---|
| Decode lossy 675x900 | 1 177/s | 503/s (−57%) | 1 337/s (+14%) |
| ↳ `UseThreads` | 1 054/s | 511/s (−52%) | 1 355/s (+29%) |
| Decode lossy+alpha 3000x2000 | 225/s | 129/s (−43%) | 457/s (+103%) |
| ↳ `UseThreads` | 216/s | 130/s (−40%) | 441/s (+105%) |
| Decode lossless 3000x2000 | 476/s | 769/s (+62%) | 833/s (+75%) |
| ↳ `UseThreads` | 477/s | 765/s (+60%) | 833/s (+75%) |
| Encode lossy q75 method 0, 1.2 MP | 170/s | 136/s (−20%) | 254/s (+49%) |
| ↳ `ThreadLevel` | 168/s | 137/s (−19%) | 253/s (+50%) |
| Encode lossy q75 method 4, 1.2 MP | 60/s | 25/s (−58%) | 95/s (+57%) |
| ↳ `ThreadLevel` | 60/s | 25/s (−58%) | 94/s (+57%) |
| Encode lossy q75 method 6, 1.2 MP | 30/s | 19/s (−37%) | 31/s (+3%) |
| ↳ `ThreadLevel` | 30/s | 19/s (−37%) | 33/s (+11%) |
| Encode lossy+alpha q75, 3000x2000 | 11/s | 5/s (−53%) | 12/s (+8%) |
| ↳ `ThreadLevel` | 11/s | 6/s (−47%) | 14/s (+23%) |
| Encode lossless level 6, 1.2 MP | 10/s | 9/s (−7%) | 16/s (+69%) |
| ↳ `ThreadLevel` | 10/s | 10/s (+2%) | 16/s (+64%) |
| Encode lossless level 0, 1.2 MP | 113/s | 72/s (−36%) | 136/s (+21%) |
| ↳ `ThreadLevel` | 113/s | 73/s (−35%) | 136/s (+21%) |

#### Peak memory, amd64

Measured as on the M5 Pro (`getrusage` maxrss, median of 3 runs with the
range when the runs differ by more than 10%; wall time of the encodes).
v2 peaks lower than v1 in every row.

| Workload | v1 (libwebp) | v2, pure Go | v2 (assembly) |
|---|---|---|---|
| One 12 MP lossy encode (4000x3000 `*image.YCbCr`, q75; the source takes 18 MB) | 162 MB, 1.30 s | 64 MB, 2.96 s | 64 MB, 0.79 s |
| ↳ with `ThreadLevel` | 162 MB, 1.26 s | 66 MB, 0.60 s | 66 MB, 0.27 s |
| One 12 MP lossless level 6 encode (upscaled photo) | 325 MB, 7.08 s | 257 MB, 6.61 s | 256 MB, 4.19 s |
| Three 12 MP lossless level 6 encodes in a row | 529 MB (448-533), 21.45 s | 267 MB, 19.76 s | 267 MB, 12.37 s |
| ↳ with `GOMEMLIMIT=450MiB` | 449 MB, 21.25 s | 406 MB, 19.79 s | 403 MB, 12.49 s |
| One 12 MP lossless decode (output 48 MB) | 116 MB | 67 MB | 67 MB |
| 15 goroutines x 4 decodes of the 6 MP lossless logo | 1267 MB | 651 MB (571-654) | 672 MB (598-723) |
| ↳ `DecodeInto`, one buffer per goroutine |  | 395 MB | 395 MB |
| 15 goroutines x 4 decodes of the 6 MP lossy+alpha logo | 864 MB | 774 MB | 647 MB (600-804) |
| ↳ `DecodeInto`, one buffer per goroutine |  | 403 MB | 401 MB |

#### Building with GOAMD64=v3

`GOAMD64=v3` lets the Go compiler use x86-64-v3 instructions (AVX2,
BMI2, FMA...) in Go code, and the binary then runs only on such CPUs. It
does not change the assembly kernels, which are picked at run time either
way. One call at a time on the same server (8 rounds, same method; the
percentages compare with v1):

| Benchmark | v1 (libwebp) | v2, pure Go | v2, pure Go, `GOAMD64=v3` | v2 (assembly) | v2 (assembly), `GOAMD64=v3` |
|---|---|---|---|---|---|
| Decode lossy 675x900 | 5.70 ms | 15.7 ms (+175%) | 15.4 ms (+170%) | 5.90 ms (+4%) | 5.56 ms (−2%) |
| Decode lossy+alpha 3000x2000 | 33.0 ms | 60.6 ms (+83%) | 59.9 ms (+81%) | 15.8 ms (−52%) | 15.6 ms (−53%) |
| Decode lossless 3000x2000 | 12.6 ms | 7.76 ms (−38%) | 7.60 ms (−40%) | 5.12 ms (−59%) | 5.15 ms (−59%) |
| Encode lossy q75 method 0, 1.2 MP | 43.7 ms | 57.0 ms (+30%) | 56.8 ms (+30%) | 30.0 ms (−31%) | 30.8 ms (−30%) |
| Encode lossy q75 method 4, 1.2 MP | 127 ms | 303 ms (+138%) | 302 ms (+138%) | 83.0 ms (−35%) | 83.1 ms (−35%) |
| Encode lossy q75 method 6, 1.2 MP | 261 ms | 387 ms (+49%) | 387 ms (+48%) | 239 ms (−8%) | 238 ms (−9%) |
| Encode lossless level 6, 1.2 MP | 692 ms | 700 ms (+1%) | 693 ms (~) | 463 ms (−33%) | 462 ms (−33%) |
| Encode lossless level 0, 1.2 MP | 65.3 ms | 103 ms (+58%) | 103 ms (+58%) | 56.0 ms (−14%) | 55.6 ms (−15%) |

The pure Go build gains 0-2% from it (geomean −0.9%). With the assembly,
the lossy decode, whose bitstream parsing is Go code, gains 6%, the
method 0 encode loses 3%, and the rest moves by 1% or less (geomean
−0.6%).

### Reproducing the measurements

The benchmarks live in [`benchmarks/`](benchmarks), a separate Go module
(so that the library itself depends neither on v1 nor on cgo): the
workloads are defined once in [`benchmarks/suite`](benchmarks/suite/suite.go)
and run through two thin adapters, `benchmarks/v1` (v1.0.5 + libwebp) and
`benchmarks/v2` (this repository).

```bash
# v1 needs cgo and libwebp with its pkg-config file:
brew install webp pkg-config                 # macOS
go install golang.org/x/perf/cmd/benchstat@latest

make bench-compare        # runs benchmarks/run.sh: every table above, about an hour
```

On Linux, build libwebp 1.6.0 from source: distributions ship older
releases (Ubuntu 24.04 has 1.3.2), and the numbers above are for 1.6.0.
`/usr/local` comes first in `pkg-config`'s and the dynamic linker's search
paths, so v1 then uses it even with the distribution's `libwebp-dev`
installed (`env.txt` reports the version found):

```bash
sudo apt install build-essential cmake pkg-config
curl -fsSLO https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz
curl -fsSLO https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz.asc
gpg --verify libwebp-1.6.0.tar.gz.asc       # WebP release key 6B0E6B70976DE303EDF2F601F9C3D6BDB8232B5D
tar -xzf libwebp-1.6.0.tar.gz && cd libwebp-1.6.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j"$(nproc)" && sudo cmake --install build && sudo ldconfig
```

The amd64 tables were measured that way, with the whole machine to
themselves (`benchmarks/run.sh` uses every core; nothing else ran).

The results go to `benchmarks/out/`:

- `benchstat.txt`: the main table and the [Threads](#threads) table.
  Columns `v1.txt`, `v2-purego.txt` and `v2.txt` are v1, v2 pure Go and
  v2 (NEON or assembly); `sec/op` is the time, `cpu-sec/op` the CPU
  time, `B/op` and `allocs/op` the allocations, `bytes` the size of the
  encoded file; "vs base" is the difference from v1 with its p-value;
- `benchstat-saturation.txt`: [Throughput under load](#throughput-under-load)
  (sec/op there is the inverse of the throughput);
- `rss.txt`: [Peak memory](#peak-memory);
- `env.txt`: the machine, Go and libwebp versions and the load average;
  `raw/`: the raw `go test -bench` outputs.

Shorter runs and single benchmarks:

```bash
ROUNDS=3 RSS_ROUNDS=1 SAT=0 benchmarks/run.sh    # a quick look, about 10 minutes
BUILDS="v2-purego v2" benchmarks/run.sh          # without libwebp (v2 only)

cd benchmarks
go test -run '^$' -bench 'Decode/lossy675x900' -benchmem -count 6 ./v1 ./v2
go test -run '^$' -bench 'Encode/lossy-q75-m4' -benchmem -count 6 -tags purego ./v2
```

Expect other absolute numbers on other machines and a few percent of
difference between runs on the same one: close other heavy programs
during a run, and compare the columns of one run rather than numbers from
different machines. The correctness statements above are checked by the
tests: `go test -tags interop ./...` (libwebp tools in `PATH`) compares
decoded pixels with libwebp's `dwebp`, decodes the files v2 encodes with
`dwebp`, and compares their sizes and quality with `cwebp`'s; `go test
./...` checks that the assembly (at every level the machine supports) and
pure Go code paths produce the same output.

For a quick check of v2 alone, `make bench` runs the `webp` package
benchmarks.

## Tests

```bash
go test ./...                              # unit, round-trip and conformance tests
go test -tags purego ./...                 # the same on the pure Go code path
go test -tags interop ./...                # also compare with cwebp/dwebp (libwebp tools in PATH)
go test -run '^$' -fuzz FuzzDecode ./decoder
make cover                                 # coverage of the whole module, as CI reports it
go tool cover -html=coverage.txt           # browse that coverage report
```

CI measures the coverage of its linux/amd64 run (the Coverage badge at the
top) and fails when the total drops below 95%. The report of master, the
same as `go tool cover -html` shows, is at
[kolesa-team.github.io/go-webp](https://kolesa-team.github.io/go-webp/).

## License
MIT licensed. See the LICENSE file for details.

Parts of the codec are adapted from [libwebp](https://chromium.googlesource.com/webm/libwebp)
(Copyright 2010 Google Inc., BSD-3-Clause); see the NOTICE file.
