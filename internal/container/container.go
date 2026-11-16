// Package container parses and writes the RIFF/WebP container.
//
// Parsing follows libwebp's ParseHeadersInternal (src/dec/webp_dec.c) step by
// step, so that the returned status codes and features match what
// WebPGetFeatures / WebPDecode report for the same input, including the
// "internal" inputs libwebp accepts: raw VP8 / VP8L bitstreams without a RIFF
// header and "ALPH + VP8" sequences.
//
// Algorithm adapted from libwebp (BSD-3-Clause).
package container

import (
	"encoding/binary"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

const (
	tagSize             = 4
	chunkHeaderSize     = 8
	riffHeaderSize      = 12
	vp8xChunkSize       = 10
	vp8FrameHeaderSize  = 10
	vp8lFrameHeaderSize = 5
	vp8lMagicByte       = 0x2f

	// maxChunkPayload mirrors libwebp's MAX_CHUNK_PAYLOAD.
	maxChunkPayload = ^uint32(0) - chunkHeaderSize - 1
	// maxImageArea mirrors libwebp's MAX_IMAGE_AREA.
	maxImageArea = uint64(1) << 32

	// VP8X feature flags.
	animationFlag = 0x02
	xmpFlag       = 0x04
	exifFlag      = 0x08
	alphaFlag     = 0x10
	iccpFlag      = 0x20

	// MaxDimension is the largest width or height of a VP8 frame (and of a
	// VP8X canvas written by WriteLossy). VP8L headers can encode 16384,
	// which libwebp accepts.
	MaxDimension = 16383
)

// Features describes a WebP file, with the same semantics as libwebp's
// WebPBitstreamFeatures as filled by WebPGetFeatures.
type Features struct {
	Width, Height int
	// HasAlpha is set from the VP8X alpha flag, overridden by the VP8L
	// header alpha bit for lossless frames, and forced on by the presence of
	// an ALPH chunk.
	HasAlpha bool
	// HasAnimation is set from the VP8X animation flag.
	HasAnimation bool
	// Format is FormatLossy for VP8, FormatLossless for VP8L and
	// FormatUndefined for animated files.
	Format utils.FormatType
}

// Image is a parsed WebP file. The payload slices alias the input passed to
// Parse; nothing is copied.
type Image struct {
	Features
	// VP8 is the lossy bitstream, nil for lossless images. Like libwebp
	// (ParseHeadersInternal hands the frame decoder io->data_size = the
	// remaining input), it starts at the "VP8 " chunk payload and extends
	// to the end of the input, past the declared chunk size; ChunkSize
	// holds the declared size.
	VP8 []byte
	// VP8L is the lossless bitstream, nil for lossy images. It extends to
	// the end of the input like VP8.
	VP8L []byte
	// ChunkSize is the declared size of the image chunk (or the size of
	// the raw bitstream).
	ChunkSize int
	// ALPH is the payload of the last ALPH chunk found before the image
	// chunk, including its 1-byte header; nil if there is none. libwebp only
	// uses it for lossy images.
	ALPH []byte
}

// Chunk returns the image chunk payload limited to its declared size (VP8
// or VP8L without the bytes that follow the chunk).
func (img *Image) Chunk() []byte {
	p := img.VP8
	if p == nil {
		p = img.VP8L
	}
	if img.ChunkSize < len(p) {
		p = p[:img.ChunkSize:img.ChunkSize]
	}
	return p
}

// Parse parses a WebP file (or a raw VP8/VP8L bitstream) and locates the
// image payload. The whole file must be available (like libwebp's
// WebPDecode): truncated chunks yield a NotEnoughData error.
//
// For animated files (VP8X with the animation flag) only Features is filled
// and the payload slices are nil; the caller should report
// status.Unsupported like libwebp's WebPDecode does.
//
// Errors are *status.Error with the code libwebp would return.
func Parse(data []byte) (*Image, error) {
	img := new(Image)
	if err := parse(data, true, img, nil); err != nil {
		return nil, err
	}
	return img, nil
}

// GetFeatures returns the features of a WebP file, like libwebp's
// WebPGetFeatures. It only needs the headers: a file truncated inside the
// image payload is still accepted.
func GetFeatures(data []byte) (Features, error) {
	var img Image
	if err := parse(data, false, &img, nil); err != nil {
		return Features{}, err
	}
	return img.Features, nil
}

// GetFeaturesPrefix is GetFeatures for a prefix of a file that is still
// being read. done reports whether the prefix was long enough for the result
// to be final, i.e. to equal GetFeatures of the whole file: parsing reached
// the VP8/VP8L frame header (and checked it against the VP8X canvas), found
// an animation, or failed with an error other than NotEnoughData. When done
// is false the caller should read more data, or use the result as is at the
// end of the input.
func GetFeaturesPrefix(prefix []byte) (f Features, done bool, err error) {
	var img Image
	err = parse(prefix, false, &img, &done)
	if err != nil {
		return Features{}, status.CodeOf(err) != utils.Vp8StatusNotEnoughData, err
	}
	return img.Features, done || img.HasAnimation, nil
}

func le24(b []byte) uint32 {
	_ = b[2]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
}

func tagIs(b []byte, tag string) bool {
	return len(b) >= tagSize && b[0] == tag[0] && b[1] == tag[1] && b[2] == tag[2] && b[3] == tag[3]
}

// parse mirrors ParseHeadersInternal (and WebPParseHeaders when haveAll is
// set). haveAll selects the WebPDecode behaviour (headers != NULL,
// have_all_data = 1) versus the WebPGetFeatures one (headers == NULL).
func parse(data []byte, haveAll bool, img *Image, reached *bool) error {
	var anim Features
	err := parseHeaders(data, haveAll, img, &anim, reached)
	if haveAll && anim.HasAnimation && (err == nil || status.CodeOf(err) == utils.Vp8StatusNotEnoughData) {
		// WebPParseHeaders reports UNSUPPORTED_FEATURE here; we hand back the
		// VP8X features and let the caller decide.
		*img = Image{Features: anim}
		return nil
	}
	return err
}

// reached, if not nil, is set when the frame header has been parsed.
func parseHeaders(data []byte, haveAll bool, img *Image, anim *Features, reached *bool) error {
	if len(data) < riffHeaderSize {
		return status.NotEnoughData("not enough data for a WebP header")
	}

	// RIFF header.
	riffSize := uint32(0)
	if tagIs(data, "RIFF") {
		if !tagIs(data[8:], "WEBP") {
			return status.Bitstream("wrong RIFF signature")
		}
		size := binary.LittleEndian.Uint32(data[4:])
		if size < tagSize+chunkHeaderSize {
			return status.Bitstream("RIFF size too small")
		}
		if size > maxChunkPayload {
			return status.Bitstream("RIFF size too large")
		}
		if haveAll && uint64(size) > uint64(len(data)-chunkHeaderSize) {
			return status.NotEnoughData("truncated RIFF")
		}
		riffSize = size
		data = data[riffHeaderSize:]
	}
	foundRIFF := riffSize > 0

	// VP8X.
	if len(data) < chunkHeaderSize {
		return status.NotEnoughData("not enough data for a chunk header")
	}
	foundVP8X := false
	animation := false
	canvasW, canvasH := 0, 0
	if tagIs(data, "VP8X") {
		if binary.LittleEndian.Uint32(data[4:]) != vp8xChunkSize {
			return status.Bitstream("wrong VP8X chunk size")
		}
		if len(data) < chunkHeaderSize+vp8xChunkSize {
			return status.NotEnoughData("truncated VP8X chunk")
		}
		flags := binary.LittleEndian.Uint32(data[8:])
		canvasW = 1 + int(le24(data[12:]))
		canvasH = 1 + int(le24(data[15:]))
		if uint64(canvasW)*uint64(canvasH) >= maxImageArea {
			return status.Bitstream("canvas too large")
		}
		data = data[chunkHeaderSize+vp8xChunkSize:]
		foundVP8X = true
		animation = flags&animationFlag != 0
		img.HasAlpha = flags&alphaFlag != 0
	}
	if !foundRIFF && foundVP8X {
		return status.Bitstream("VP8X chunk without RIFF header")
	}
	img.HasAnimation = animation
	img.Format = utils.FormatUndefined
	img.Width, img.Height = canvasW, canvasH
	if foundVP8X && animation {
		*anim = img.Features
		if !haveAll {
			// WebPGetFeatures returns the VP8X features right away.
			return nil
		}
	}

	// finish implements libwebp's "ReturnWidthHeight" exit.
	finish := func(err error, alpha []byte) error {
		if err == nil || (!haveAll && foundVP8X && status.CodeOf(err) == utils.Vp8StatusNotEnoughData) {
			if alpha != nil {
				img.HasAlpha = true
			}
			return nil
		}
		return err
	}

	if len(data) < tagSize {
		return finish(status.NotEnoughData("not enough data for a chunk tag"), nil)
	}

	// Optional chunks (only after RIFF+VP8X, or for a raw "ALPH" sequence).
	var alpha []byte
	if (foundRIFF && foundVP8X) || (!foundRIFF && !foundVP8X && tagIs(data, "ALPH")) {
		// libwebp accumulates total_size in a uint32_t, which wraps for
		// chunk sizes close to MAX_CHUNK_PAYLOAD; do the same so that the
		// same files are accepted.
		totalSize := uint32(tagSize + chunkHeaderSize + vp8xChunkSize)
		for {
			if len(data) < chunkHeaderSize {
				return finish(status.NotEnoughData("truncated chunk header"), alpha)
			}
			chunkSize := binary.LittleEndian.Uint32(data[4:])
			if chunkSize > maxChunkPayload {
				return finish(status.Bitstream("invalid chunk size"), alpha)
			}
			diskSize := (uint64(chunkHeaderSize) + uint64(chunkSize) + 1) &^ 1
			totalSize += uint32(diskSize)
			if riffSize > 0 && totalSize > riffSize {
				return finish(status.Bitstream("chunk exceeds RIFF size"), alpha)
			}
			if tagIs(data, "VP8 ") || tagIs(data, "VP8L") {
				break
			}
			if uint64(len(data)) < diskSize {
				return finish(status.NotEnoughData("truncated chunk"), alpha)
			}
			if tagIs(data, "ALPH") {
				alpha = data[chunkHeaderSize : chunkHeaderSize+int(chunkSize) : chunkHeaderSize+int(chunkSize)]
			}
			data = data[diskSize:]
		}
	}

	// VP8 / VP8L chunk header.
	if len(data) < chunkHeaderSize {
		return finish(status.NotEnoughData("truncated image chunk header"), alpha)
	}
	var compressedSize uint64
	lossless := false
	if isVP8, isVP8L := tagIs(data, "VP8 "), tagIs(data, "VP8L"); isVP8 || isVP8L {
		size := binary.LittleEndian.Uint32(data[4:])
		const minimalSize = tagSize + chunkHeaderSize
		if riffSize >= minimalSize && size > riffSize-minimalSize {
			return finish(status.Bitstream("image chunk exceeds RIFF size"), alpha)
		}
		if haveAll && uint64(size) > uint64(len(data)-chunkHeaderSize) {
			return finish(status.NotEnoughData("truncated image chunk"), alpha)
		}
		compressedSize = uint64(size)
		data = data[chunkHeaderSize:]
		lossless = isVP8L
	} else {
		lossless = isVP8LSignature(data)
		compressedSize = uint64(len(data))
	}
	if compressedSize > uint64(maxChunkPayload) {
		return status.Bitstream("image chunk too large")
	}
	if !animation {
		if lossless {
			img.Format = utils.FormatLossless
		} else {
			img.Format = utils.FormatLossy
		}
	}

	var w, h int
	if !lossless {
		if len(data) < vp8FrameHeaderSize {
			return finish(status.NotEnoughData("truncated VP8 frame header"), alpha)
		}
		var ok bool
		if w, h, ok = vp8Info(data, compressedSize); !ok {
			return status.Bitstream("invalid VP8 frame header")
		}
	} else {
		if len(data) < vp8lFrameHeaderSize {
			return finish(status.NotEnoughData("truncated VP8L header"), alpha)
		}
		var a, ok bool
		if w, h, a, ok = vp8lInfo(data); !ok {
			return status.Bitstream("invalid VP8L header")
		}
		img.HasAlpha = a
	}
	if foundVP8X && (canvasW != w || canvasH != h) {
		return status.Bitstream("VP8X canvas size does not match the frame size")
	}
	img.Width, img.Height = w, h
	if reached != nil {
		*reached = true
	}

	img.ChunkSize = int(compressedSize)
	payload := data
	if lossless {
		img.VP8L = payload
	} else {
		img.VP8 = payload
	}
	img.ALPH = alpha
	return finish(nil, alpha)
}

// isVP8LSignature mirrors VP8LCheckSignature.
func isVP8LSignature(b []byte) bool {
	return len(b) >= vp8lFrameHeaderSize && b[0] == vp8lMagicByte && b[4]>>5 == 0
}

// vp8Info mirrors VP8GetInfo: it validates the 10-byte VP8 frame header and
// returns the frame dimensions (scale bits dropped).
func vp8Info(b []byte, chunkSize uint64) (w, h int, ok bool) {
	if len(b) < vp8FrameHeaderSize {
		return 0, 0, false
	}
	_ = b[9]
	if b[3] != 0x9d || b[4] != 0x01 || b[5] != 0x2a {
		return 0, 0, false
	}
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
	w = int(uint32(b[6])|uint32(b[7])<<8) & 0x3fff
	h = int(uint32(b[8])|uint32(b[9])<<8) & 0x3fff
	switch {
	case bits&1 != 0: // not a key frame
		return 0, 0, false
	case (bits>>1)&7 > 3: // unknown profile
		return 0, 0, false
	case (bits>>4)&1 == 0: // invisible frame
		return 0, 0, false
	case uint64(bits>>5) >= chunkSize: // partition length
		return 0, 0, false
	case w == 0 || h == 0:
		return 0, 0, false
	}
	return w, h, true
}

// vp8lInfo mirrors VP8LGetInfo: 0x2f signature, 14-bit width-1, 14-bit
// height-1, alpha bit, 3-bit version (must be 0).
func vp8lInfo(b []byte) (w, h int, alpha, ok bool) {
	if !isVP8LSignature(b) {
		return 0, 0, false, false
	}
	v := binary.LittleEndian.Uint32(b[1:5])
	w = int(v&0x3fff) + 1
	h = int((v>>14)&0x3fff) + 1
	alpha = (v>>28)&1 != 0
	if v>>29 != 0 {
		return 0, 0, false, false
	}
	return w, h, alpha, true
}
