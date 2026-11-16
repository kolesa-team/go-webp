package container

import (
	"encoding/binary"
	"io"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// maxRIFFPayload bounds the total RIFF payload so that its size fits in the
// 32-bit RIFF size field (same limit as libwebp's MAX_CHUNK_PAYLOAD).
const maxRIFFPayload = uint64(maxChunkPayload)

// WriteLossy writes a lossy WebP file. With alph == nil it is a simple
// "RIFF/WEBP/VP8 " file; otherwise an extended file made of a VP8X chunk
// (alpha flag, canvas width x height), the ALPH chunk (alph must include its
// 1-byte header) and the "VP8 " chunk, as libwebp's encoder lays it out.
// Odd-sized chunks are padded with a zero byte.
func WriteLossy(w io.Writer, width, height int, vp8, alph []byte) error {
	if len(vp8) == 0 {
		return status.InvalidParam("empty VP8 payload")
	}
	if alph == nil {
		riff := uint64(tagSize) + chunkHeaderSize + padded(len(vp8))
		if riff > maxRIFFPayload {
			return status.InvalidParam("VP8 payload too large")
		}
		var hdr [riffHeaderSize + chunkHeaderSize]byte
		putRIFF(hdr[:], uint32(riff))
		putChunkHeader(hdr[riffHeaderSize:], "VP8 ", len(vp8))
		return writeAll(w, hdr[:], vp8, pad(len(vp8)))
	}
	if width < 1 || height < 1 || width > MaxDimension || height > MaxDimension {
		return status.InvalidParam("invalid canvas dimensions")
	}
	if len(alph) == 0 {
		return status.InvalidParam("empty ALPH payload")
	}
	riff := uint64(tagSize) + chunkHeaderSize + vp8xChunkSize +
		chunkHeaderSize + padded(len(alph)) + chunkHeaderSize + padded(len(vp8))
	if riff > maxRIFFPayload {
		return status.InvalidParam("payload too large")
	}
	var hdr [riffHeaderSize + chunkHeaderSize + vp8xChunkSize + chunkHeaderSize]byte
	putRIFF(hdr[:], uint32(riff))
	b := hdr[riffHeaderSize:]
	putChunkHeader(b, "VP8X", vp8xChunkSize)
	b[8] = alphaFlag
	putLE24(b[12:], uint32(width-1))
	putLE24(b[15:], uint32(height-1))
	putChunkHeader(b[chunkHeaderSize+vp8xChunkSize:], "ALPH", len(alph))

	// The padding byte of ALPH (if any) and the "VP8 " chunk header go in a
	// single small write between the two payloads.
	var mid [1 + chunkHeaderSize]byte
	m := mid[:0]
	if len(alph)&1 != 0 {
		m = append(m, 0)
	}
	m = m[:len(m)+chunkHeaderSize]
	putChunkHeader(m[len(m)-chunkHeaderSize:], "VP8 ", len(vp8))
	return writeAll(w, hdr[:], alph, m, vp8, pad(len(vp8)))
}

// WriteLossless writes a simple lossless "RIFF/WEBP/VP8L" file.
func WriteLossless(w io.Writer, vp8l []byte) error {
	if len(vp8l) == 0 {
		return status.InvalidParam("empty VP8L payload")
	}
	riff := uint64(tagSize) + chunkHeaderSize + padded(len(vp8l))
	if riff > maxRIFFPayload {
		return status.InvalidParam("VP8L payload too large")
	}
	var hdr [riffHeaderSize + chunkHeaderSize]byte
	putRIFF(hdr[:], uint32(riff))
	putChunkHeader(hdr[riffHeaderSize:], "VP8L", len(vp8l))
	return writeAll(w, hdr[:], vp8l, pad(len(vp8l)))
}

// LosslessHeaderSize is the size of the RIFF header plus the VP8L chunk
// header of a simple lossless file.
const LosslessHeaderSize = riffHeaderSize + chunkHeaderSize

// FrameLossless turns buf, made of LosslessHeaderSize bytes of headroom
// followed by a VP8L chunk payload, into a complete lossless WebP file in
// place: it fills in the headers and appends the padding byte if needed
// (reusing buf's capacity when possible). The result is what WriteLossless
// writes for the same payload, without copying it.
func FrameLossless(buf []byte) ([]byte, error) {
	if len(buf) <= LosslessHeaderSize {
		return nil, status.InvalidParam("empty VP8L payload")
	}
	n := len(buf) - LosslessHeaderSize
	riff := uint64(tagSize) + chunkHeaderSize + padded(n)
	if riff > maxRIFFPayload {
		return nil, status.InvalidParam("VP8L payload too large")
	}
	putRIFF(buf, uint32(riff))
	putChunkHeader(buf[riffHeaderSize:], "VP8L", n)
	if n&1 != 0 {
		buf = append(buf, 0)
	}
	return buf, nil
}

var zeroPad = []byte{0}

func pad(n int) []byte {
	if n&1 != 0 {
		return zeroPad
	}
	return nil
}

func padded(n int) uint64 { return (uint64(n) + 1) &^ 1 }

func putRIFF(b []byte, size uint32) {
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], size)
	copy(b[8:], "WEBP")
}

func putChunkHeader(b []byte, tag string, size int) {
	copy(b, tag)
	binary.LittleEndian.PutUint32(b[4:], uint32(size))
}

func putLE24(b []byte, v uint32) {
	_ = b[2]
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
}

// writeAll writes the non-empty parts in order. The headers are already
// assembled into as few parts as possible, so a file costs 3 to 5 Write calls.
func writeAll(w io.Writer, parts ...[]byte) error {
	for _, p := range parts {
		if len(p) == 0 {
			continue
		}
		if err := write(w, p); err != nil {
			return err
		}
	}
	return nil
}

func write(w io.Writer, b []byte) error {
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}
