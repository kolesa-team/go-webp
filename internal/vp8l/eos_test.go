package vp8l

import (
	"encoding/binary"
	"image"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// countSink accepts any number of rows into one small reusable buffer and
// counts the rows committed.
type countSink struct {
	stride int
	buf    []byte
	rows   int
}

func (s *countSink) Buffer(y, n int) []byte {
	if need := n * s.stride; cap(s.buf) < need {
		s.buf = make([]byte, need)
	}
	return s.buf[:n*s.stride]
}
func (s *countSink) Stride() int           { return s.stride }
func (s *countSink) Commit(y, n int) error { s.rows += n; return nil }
func (s *countSink) NeedRows() int         { return 1 << 30 }

// TestDecodeStopsAtEndOfData checks that a stream whose header declares far
// more rows than its data holds fails as soon as the data runs out, like
// libwebp, instead of decoding the whole frame from zero bits (back
// references past the end used to fill all 16383 rows of a 264-byte file).
func TestDecodeStopsAtEndOfData(t *testing.T) {
	const w, h = 16383, 16
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = byte(i * 7)
	}
	file, err := vp8lenc.Encode(img, &vp8lenc.Config{Quality: 0, Method: 0})
	must(t, err)
	payload := file
	if c := findChunk(file, "VP8L"); c != nil {
		payload = c
	}
	payload = append([]byte(nil), payload...)
	// Patch the 14-bit height field (bits 14..27 after the signature byte).
	bits := binary.LittleEndian.Uint32(payload[1:])
	bits = bits&^(0x3fff<<14) | (16383-1)<<14
	binary.LittleEndian.PutUint32(payload[1:], bits)

	s := &countSink{stride: 4 * w}
	err = Decode(payload, s)
	if err == nil {
		t.Fatal("decoding a truncated stream succeeded")
	}
	if c := status.CodeOf(err); c != utils.Vp8StatusBitstreamError && c != utils.Vp8StatusNotEnoughData {
		t.Errorf("status %v (%v)", c, err)
	}
	t.Logf("payload %d bytes, rows %d, err %v", len(payload), s.rows, err)
	if s.rows > 2*h+32 {
		t.Errorf("committed %d rows from a %d-row stream", s.rows, h)
	}
}
