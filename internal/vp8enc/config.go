// Package vp8enc implements a pure Go VP8 (lossy WebP) encoder.
//
// The encoder follows the architecture of libwebp's VP8 encoder: analysis
// pass with k-means segmentation on macroblock susceptibility, per-segment
// quantizers derived from the quality factor with libwebp's constants,
// rate-distortion optimised intra mode selection (i16/i4/UV), trellis
// quantization at methods 5-6, DC error diffusion for chroma, adaptive token
// probabilities, SSIM-based autofilter and multi-pass rate control towards a
// target size or PSNR. At equal settings the output size and quality are
// within a fraction of a percent of cwebp's for methods 1-6.
//
// Encode returns the payload of a "VP8 " RIFF chunk; wrapping it into a
// WebP container (and coding the alpha plane) is left to the caller.
//
// Differences from libwebp 1.6.0, all of them producing valid bitstreams:
//   - All methods code the frame from a token buffer (libwebp does so for
//     methods >= 3 only, with a single partition). Methods 0-2 therefore get
//     exact token probabilities, which makes their output smaller than
//     cwebp's; multiple token partitions are supported with every method.
//   - The skip flag is used whenever enough macroblocks have no
//     coefficients (libwebp only uses it for methods 0-2).
//   - With a single segment, the autofilter result is applied (libwebp
//     leaves the initial level in the frame header).
//   - Methods 0-2 propagate the chroma DC diffusion error between
//     macroblocks.
//   - Config.Preprocessing bits 1 (dithering) and 2 (sharp YUV) act on the
//     RGB->YUV conversion, which happens before Encode: use
//     FromNRGBADithered with DitheringFor(cfg), and sharpYUV for bit 2.
//   - Config.LowMemory has no effect.
//
// Algorithm adapted from libwebp (BSD-3-Clause).
package vp8enc

import (
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// MaxDimension is the largest width or height the VP8 bitstream can carry.
const MaxDimension = 16383

// Config holds the lossy encoding parameters. The semantics follow libwebp's
// WebPConfig for the corresponding fields.
type Config struct {
	Quality float32 // 0..100, between smallest size and best quality
	Method  int     // 0..6 speed/quality trade-off (0=fast, 6=slowest)
	// ImageHint is accepted for API compatibility; it does not influence
	// lossy coding (as in libwebp).
	ImageHint int
	// TargetSize, when non-zero, is the desired output size in bytes. With
	// Pass > 1 it overrides Quality through a multi-pass search; with a
	// single pass it has no effect (as in libwebp's WebPEncode).
	TargetSize int
	// TargetPSNR, when non-zero, is the desired distortion in dB. Like
	// TargetSize it needs Pass > 1 (TargetSize has precedence).
	TargetPSNR float32
	// Pass is the number of entropy-analysis passes (1..10). Values <= 0
	// are treated as 1.
	Pass            int
	Segments        int  // number of segments, 1..4
	SNSStrength     int  // spatial noise shaping, 0..100
	FilterStrength  int  // loop filter strength 0..100 (0 = off)
	FilterSharpness int  // 0..7, 0 = most sharp
	FilterType      int  // 0 = simple, 1 = strong (complex)
	Autofilter      bool // auto-adjust the filter strengths per segment
	// Preprocessing: bit 0 enables segment-map smoothing. Bits 1
	// (dithered RGB->YUV conversion) and 2 (sharp YUV) concern the
	// conversion done by FromNRGBADithered before Encode and are ignored
	// here.
	Preprocessing   int
	Partitions      int // log2 of the number of token partitions, 0..3
	PartitionLimit  int // 0..100: quality degradation allowed to fit the 512k limit on partition #0
	EmulateJPEGSize bool
	// ThreadLevel allows the analysis pass to run on two goroutines.
	ThreadLevel bool
	// LowMemory is accepted for compatibility. The encoder always uses a
	// compact token buffer, so it has no effect.
	LowMemory bool
}

// DefaultConfig returns libwebp's default lossy configuration (quality 75,
// method 4, the "default" preset).
func DefaultConfig() Config {
	return Config{
		Quality:         75,
		Method:          4,
		Pass:            1,
		Segments:        4,
		SNSStrength:     50,
		FilterStrength:  60,
		FilterSharpness: 0,
		FilterType:      1,
	}
}

func (c *Config) validate() error {
	switch {
	case c.Quality < 0 || c.Quality > 100 || c.Quality != c.Quality:
		return status.InvalidParam("vp8enc: quality out of range [0,100]")
	case c.Method < 0 || c.Method > 6:
		return status.InvalidParam("vp8enc: method out of range [0,6]")
	case c.TargetSize < 0:
		return status.InvalidParam("vp8enc: negative target size")
	case c.TargetPSNR < 0 || c.TargetPSNR != c.TargetPSNR:
		return status.InvalidParam("vp8enc: negative target PSNR")
	case c.Pass > 10:
		return status.InvalidParam("vp8enc: pass out of range [1,10]")
	case c.Segments < 1 || c.Segments > 4:
		return status.InvalidParam("vp8enc: segments out of range [1,4]")
	case c.SNSStrength < 0 || c.SNSStrength > 100:
		return status.InvalidParam("vp8enc: sns strength out of range [0,100]")
	case c.FilterStrength < 0 || c.FilterStrength > 100:
		return status.InvalidParam("vp8enc: filter strength out of range [0,100]")
	case c.FilterSharpness < 0 || c.FilterSharpness > 7:
		return status.InvalidParam("vp8enc: filter sharpness out of range [0,7]")
	case c.FilterType < 0 || c.FilterType > 1:
		return status.InvalidParam("vp8enc: filter type must be 0 or 1")
	case c.Preprocessing < 0 || c.Preprocessing > 7:
		return status.InvalidParam("vp8enc: preprocessing out of range [0,7]")
	case c.Partitions < 0 || c.Partitions > 3:
		return status.InvalidParam("vp8enc: partitions out of range [0,3]")
	case c.PartitionLimit < 0 || c.PartitionLimit > 100:
		return status.InvalidParam("vp8enc: partition limit out of range [0,100]")
	case c.ImageHint < 0 || c.ImageHint > 3:
		return status.InvalidParam("vp8enc: image hint out of range [0,3]")
	}
	return nil
}
