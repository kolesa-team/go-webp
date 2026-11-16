package decoder

import (
	"bytes"
	"encoding/binary"
	"image"
	"os"
	"runtime"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// riffFile wraps chunks into a RIFF/WEBP container.
func riffFile(chunks ...[]byte) []byte {
	n := 4
	for _, c := range chunks {
		n += len(c)
	}
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, uint32(n))
	b = append(b, "WEBP"...)
	for _, c := range chunks {
		b = append(b, c...)
	}
	return b
}

func riffChunk(tag string, p []byte) []byte {
	b := []byte(tag)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(p)))
	b = append(b, p...)
	if len(p)&1 != 0 {
		b = append(b, 0)
	}
	return b
}

// vp8Frame returns a key frame header for a w x h frame with the given first
// partition length, followed by payload.
func vp8Frame(w, h, partLen int, payload []byte) []byte {
	bits := uint32(1<<4 | partLen<<5)
	p := []byte{byte(bits), byte(bits >> 8), byte(bits >> 16), 0x9d, 0x01, 0x2a, byte(w), byte(w >> 8), byte(h), byte(h >> 8)}
	return append(p, payload...)
}

func vp8lFrame(w, h int, payload []byte) []byte {
	p := []byte{0x2f}
	p = binary.LittleEndian.AppendUint32(p, uint32(w-1)|uint32(h-1)<<14)
	return append(p, payload...)
}

// TestTinyInputHugeFrame checks that tiny broken files declaring a huge
// frame fail without allocating the output image or the alpha plane:
// like libwebp, the bitstream must be validated first.
func TestTinyInputHugeFrame(t *testing.T) {
	const W = 16383
	cases := map[string][]byte{
		"vp8 zeros":       riffFile(riffChunk("VP8 ", vp8Frame(W, W, 20, make([]byte, 30)))),
		"vp8 ff":          riffFile(riffChunk("VP8 ", vp8Frame(W, W, 20, bytes.Repeat([]byte{0xff}, 30)))),
		"vp8l one byte":   riffFile(riffChunk("VP8L", vp8lFrame(16384, 16384, []byte{0}))),
		"vp8l zero bytes": riffFile(riffChunk("VP8L", vp8lFrame(16384, 16384, make([]byte, 8)))),
	}
	for _, name := range []string{"huge_lossy_patched.webp", "huge_lossless_patched.webp", "huge_alpha.webp"} {
		data, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		cases[name] = data
	}
	for name, data := range cases {
		for _, threads := range []bool{false, true} {
			runtime.GC()
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			d, err := NewDecoder(bytes.NewReader(data), &Options{UseThreads: threads})
			if err == nil {
				_, err = d.Decode()
			}
			runtime.ReadMemStats(&m1)
			if err == nil {
				t.Errorf("%s: decoded without error", name)
			}
			if a := m1.TotalAlloc - m0.TotalAlloc; a > 16<<20 {
				t.Errorf("%s (threads=%v): allocated %d MB, want < 16 MB", name, threads, a>>20)
			}
		}
	}
}

// TestAlphaDecodedWithVP8Rows checks that the alpha plane is decoded only
// as far as the VP8 frame goes, like libwebp's VP8DecompressAlphaRows.
// alpha_bomb_vp8_truncated.webp (552 bytes) declares a 16383x16383 frame
// whose VP8 data ends after the first macroblock row, with an ALPH stream
// whose pixels cost no bits at all: decoding the whole alpha plane first
// would take hundreds of milliseconds and allocate hundreds of megabytes.
// alpha_truncated_both.webp has both streams truncated: the VP8 error
// comes first, as in libwebp (dwebp reports NOT_ENOUGH_DATA).
func TestAlphaDecodedWithVP8Rows(t *testing.T) {
	for _, name := range []string{"alpha_bomb_vp8_truncated.webp", "alpha_truncated_both.webp"} {
		data, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		opts := []*Options{
			{Crop: image.Rect(0, 0, 16, 16)},
			{Crop: image.Rect(0, 0, 16, 16), UseThreads: true},
			{Scale: image.Rect(0, 0, 64, 64)},
			{Crop: image.Rect(0, 0, 64, 64), AlphaDitheringStrength: 50},
		}
		for _, o := range opts {
			runtime.GC()
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			d, err := NewDecoder(bytes.NewReader(data), o)
			if err == nil {
				_, err = d.Decode()
			}
			runtime.ReadMemStats(&m1)
			if code := status.CodeOf(err); code != utils.Vp8StatusNotEnoughData {
				t.Errorf("%s %+v: got %v (%v), want VP8_STATUS_NOT_ENOUGH_DATA", name, *o, code, err)
			}
			if a := m1.TotalAlloc - m0.TotalAlloc; a > 32<<20 {
				t.Errorf("%s %+v: allocated %d MB, want < 32 MB", name, *o, a>>20)
			}
		}
		// Without crop or scaling the (lazily allocated) output frame is
		// legitimately big; only check the status, for both modes.
		for _, threads := range []bool{false, true} {
			d, err := NewDecoder(bytes.NewReader(data), &Options{UseThreads: threads})
			if err == nil {
				_, err = d.Decode()
			}
			if code := status.CodeOf(err); code != utils.Vp8StatusNotEnoughData {
				t.Errorf("%s threads=%v: got %v, want VP8_STATUS_NOT_ENOUGH_DATA", name, threads, code)
			}
		}
	}
}
