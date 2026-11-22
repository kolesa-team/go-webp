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

package container

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func readFixture(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading fixture %s: %v", name, err)
	}
	return b
}

// Expected values were produced with libwebp 1.6.0 (WebPGetFeatures and
// WebPParseHeaders) on the same files.
func TestFixtures(t *testing.T) {
	tests := []struct {
		file             string
		feat             Features
		off, size        int // payload offset and size
		alphaOff, alphaN int
	}{
		{"grad_17x33_q70.webp", Features{17, 33, false, false, utils.FormatLossy}, 20, 144, -1, 0},
		{"grada_17x33_q70.webp", Features{17, 33, true, false, utils.FormatLossy}, 174, 174, 38, 127},
		{"grada_3x7_q70.webp", Features{3, 7, true, false, utils.FormatLossy}, 68, 94, 38, 22},
		{"grada_17x33_ll.webp", Features{17, 33, true, false, utils.FormatLossless}, 20, 800, -1, 0},
		{"grada_1x1_ll.webp", Features{1, 1, false, false, utils.FormatLossless}, 20, 15, -1, 0},
		{"meta_exif.webp", Features{257, 129, true, false, utils.FormatLossy}, 2404, 1484, 38, 2358},
		{"meta_xmp.webp", Features{17, 33, false, false, utils.FormatLossy}, 38, 144, -1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			data := readFixture(t, tt.file)
			f, err := GetFeatures(data)
			if err != nil {
				t.Fatalf("GetFeatures: %v", err)
			}
			if f != tt.feat {
				t.Errorf("GetFeatures = %+v, want %+v", f, tt.feat)
			}

			img, err := Parse(data)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			if img.Features != tt.feat {
				t.Errorf("Parse: Features = %+v, want %+v", img.Features, tt.feat)
			}
			payload := img.Chunk()
			if tt.feat.Format == utils.FormatLossless {
				if img.VP8 != nil {
					t.Errorf("Parse: VP8 has %d bytes, want nil for a lossless image", len(img.VP8))
				}
				if !bytes.Equal(img.VP8L, data[tt.off:]) {
					t.Errorf("Parse: VP8L (len %d) is not data[%d:] (len %d)", len(img.VP8L), tt.off, len(data)-tt.off)
				}
			} else {
				if img.VP8L != nil {
					t.Errorf("Parse: VP8L has %d bytes, want nil for a lossy image", len(img.VP8L))
				}
				if !bytes.Equal(img.VP8, data[tt.off:]) {
					t.Errorf("Parse: VP8 (len %d) is not data[%d:] (len %d)", len(img.VP8), tt.off, len(data)-tt.off)
				}
			}
			if !bytes.Equal(payload, data[tt.off:tt.off+tt.size]) {
				t.Errorf("Chunk (len %d) is not data[%d:%d]", len(payload), tt.off, tt.off+tt.size)
			}
			if tt.alphaOff < 0 {
				if img.ALPH != nil {
					t.Errorf("Parse: ALPH has %d bytes, want nil", len(img.ALPH))
				}
			} else if !bytes.Equal(img.ALPH, data[tt.alphaOff:tt.alphaOff+tt.alphaN]) {
				t.Errorf("Parse: ALPH (len %d) is not data[%d:%d]", len(img.ALPH), tt.alphaOff, tt.alphaOff+tt.alphaN)
			}
			// Payloads alias the input.
			if &payload[0] != &data[tt.off] {
				t.Errorf("&Chunk()[0] = %p (%#x), want &data[%d] = %p (%#x)", &payload[0], payload[0], tt.off, &data[tt.off], data[tt.off])
			}
		})
	}
}

func TestAnimated(t *testing.T) {
	data := readFixture(t, "anim.webp")
	want := Features{Width: 3, Height: 7, HasAlpha: true, HasAnimation: true, Format: utils.FormatUndefined}
	f, err := GetFeatures(data)
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if f != want {
		t.Errorf("GetFeatures = %+v, want %+v", f, want)
	}
	img, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if wantImg := (&Image{Features: want}); !reflect.DeepEqual(img, wantImg) {
		t.Errorf("Parse = %+v, want %+v", img, wantImg)
	}
	// Header-only is enough for GetFeatures.
	f, err = GetFeatures(data[:30])
	if err != nil {
		t.Fatalf("GetFeatures(data[:30]): %v", err)
	}
	if f != want {
		t.Errorf("GetFeatures(data[:30]) = %+v, want %+v", f, want)
	}
}

func TestRawBitstreams(t *testing.T) {
	lossy := readFixture(t, "grad_17x33_q70.webp")
	img, err := Parse(lossy[20:])
	if err != nil {
		t.Fatalf("Parse(raw VP8): %v", err)
	}
	if want := (Features{17, 33, false, false, utils.FormatLossy}); img.Features != want {
		t.Errorf("Parse(raw VP8): Features = %+v, want %+v", img.Features, want)
	}
	if !bytes.Equal(img.VP8, lossy[20:]) {
		t.Errorf("Parse(raw VP8): VP8 (len %d) is not the input (len %d)", len(img.VP8), len(lossy[20:]))
	}

	ll := readFixture(t, "grada_17x33_ll.webp")
	img, err = Parse(ll[20:])
	if err != nil {
		t.Fatalf("Parse(raw VP8L): %v", err)
	}
	if want := (Features{17, 33, true, false, utils.FormatLossless}); img.Features != want {
		t.Errorf("Parse(raw VP8L): Features = %+v, want %+v", img.Features, want)
	}
	if !bytes.Equal(img.VP8L, ll[20:]) {
		t.Errorf("Parse(raw VP8L): VP8L (len %d) is not the input (len %d)", len(img.VP8L), len(ll[20:]))
	}

	// "ALPH" + "VP8 " without RIFF is accepted (libwebp internal format).
	withAlpha := readFixture(t, "grada_17x33_q70.webp")
	img, err = Parse(withAlpha[30:])
	if err != nil {
		t.Fatalf("Parse(ALPH + VP8 without RIFF): %v", err)
	}
	if want := (Features{17, 33, true, false, utils.FormatLossy}); img.Features != want {
		t.Errorf("Parse(ALPH + VP8 without RIFF): Features = %+v, want %+v", img.Features, want)
	}
	if len(img.ALPH) != 127 {
		t.Errorf("Parse(ALPH + VP8 without RIFF): len(ALPH) = %d, want 127", len(img.ALPH))
	}

	// VP8X without RIFF is rejected.
	_, err = Parse(withAlpha[12:])
	if got := status.CodeOf(err); got != utils.Vp8StatusBitstreamError {
		t.Errorf("Parse(VP8X without RIFF): status = %v, want %v (err: %v)", got, utils.Vp8StatusBitstreamError, err)
	}
}

func code(err error) utils.VP8StatusCode { return status.CodeOf(err) }

func mutate(b []byte, f func(c []byte) []byte) []byte {
	return f(append([]byte(nil), b...))
}

// Expected codes were cross-checked against libwebp 1.6.0.
func TestErrors(t *testing.T) {
	lossy := readFixture(t, "grad_17x33_q70.webp")
	ext := readFixture(t, "grada_17x33_q70.webp")
	ll := readFixture(t, "grada_17x33_ll.webp")
	const (
		ok  = utils.Vp8StatusOk
		bs  = utils.Vp8StatusBitstreamError
		ned = utils.Vp8StatusNotEnoughData
	)
	tests := []struct {
		name          string
		data          []byte
		feat, parse   utils.VP8StatusCode
		checkFeatures bool
	}{
		{"empty", nil, ned, ned, false},
		{"11 bytes", lossy[:11], ned, ned, false},
		{"bad WEBP tag", mutate(lossy, func(c []byte) []byte { c[8] = 'X'; return c }), bs, bs, false},
		{"RIFF size too small", mutate(lossy, func(c []byte) []byte { binary.LittleEndian.PutUint32(c[4:], 11); return c }), bs, bs, false},
		{"RIFF size huge", mutate(lossy, func(c []byte) []byte { binary.LittleEndian.PutUint32(c[4:], 0xfffffff7); return c }), bs, bs, false},
		{"RIFF size beyond data", mutate(lossy, func(c []byte) []byte { binary.LittleEndian.PutUint32(c[4:], uint32(len(c))); return c }), ok, ned, true},
		{"truncated payload", lossy[:40], ok, ned, true},
		{"truncated VP8 header", lossy[:25], ned, ned, false},
		{"bad VP8 start code", mutate(lossy, func(c []byte) []byte { c[23] = 0; return c }), bs, bs, false},
		{"VP8 not keyframe", mutate(lossy, func(c []byte) []byte { c[20] |= 1; return c }), bs, bs, false},
		{"VP8 invisible", mutate(lossy, func(c []byte) []byte { c[20] &^= 0x10; return c }), bs, bs, false},
		{"VP8 bad profile", mutate(lossy, func(c []byte) []byte { c[20] |= 0x0e; return c }), bs, bs, false},
		{"VP8 zero width", mutate(lossy, func(c []byte) []byte { c[26], c[27] = 0, 0xc0; return c }), bs, bs, false},
		{"VP8 chunk exceeds RIFF", mutate(lossy, func(c []byte) []byte { binary.LittleEndian.PutUint32(c[16:], 1000); return c }), bs, bs, false},
		{"VP8X wrong size", mutate(ext, func(c []byte) []byte { c[16] = 11; return c }), bs, bs, false},
		{"VP8X truncated", ext[:25], ned, ned, false},
		{"VP8X canvas mismatch", mutate(ext, func(c []byte) []byte { c[24]++; return c }), bs, bs, false},
		{"VP8X truncated ALPH", ext[:60], ok, ned, true},
		{"chunk exceeds RIFF", mutate(ext, func(c []byte) []byte { binary.LittleEndian.PutUint32(c[34:], 5000); return c }), bs, bs, false},
		{"VP8L bad version", mutate(ll, func(c []byte) []byte { c[24] |= 0x20; return c }), bs, bs, false},
		{"VP8L bad signature", mutate(ll, func(c []byte) []byte { c[20] = 0x2e; return c }), bs, bs, false},
		{"garbage", bytes.Repeat([]byte{0xAB}, 64), bs, bs, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := GetFeatures(tt.data)
			if got := code(err); got != tt.feat {
				t.Errorf("GetFeatures: status = %v, want %v (err: %v)", got, tt.feat, err)
			}
			if tt.checkFeatures {
				if f.Width != 17 {
					t.Errorf("GetFeatures: Width = %d, want 17", f.Width)
				}
				if f.Height != 33 {
					t.Errorf("GetFeatures: Height = %d, want 33", f.Height)
				}
			}
			_, err = Parse(tt.data)
			if got := code(err); got != tt.parse {
				t.Errorf("Parse: status = %v, want %v (err: %v)", got, tt.parse, err)
			}
			if err != nil {
				var se *status.Error
				if !errors.As(err, &se) {
					t.Errorf("Parse: error %v (%T) is not a *status.Error", err, err)
				}
			}
		})
	}
}

func TestWriteRoundTrip(t *testing.T) {
	// cwebp writes exactly the layout WriteLossy / WriteLossless produce, so
	// re-muxing the parsed payloads must give back the original files.
	for _, name := range []string{"grad_17x33_q70.webp", "grada_17x33_q70.webp", "grada_3x7_q70.webp", "grada_17x33_ll.webp", "grada_1x1_ll.webp"} {
		t.Run(name, func(t *testing.T) {
			data := readFixture(t, name)
			img, err := Parse(data)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			var buf bytes.Buffer
			if img.VP8L != nil {
				if err := WriteLossless(&buf, img.Chunk()); err != nil {
					t.Fatalf("WriteLossless: %v", err)
				}
			} else if err := WriteLossy(&buf, img.Width, img.Height, img.Chunk(), img.ALPH); err != nil {
				t.Fatalf("WriteLossy: %v", err)
			}
			if !bytes.Equal(buf.Bytes(), data) {
				t.Errorf("re-muxed file (len %d) differs from the original (len %d)", buf.Len(), len(data))
			}
		})
	}
}

func TestWriteOddSizes(t *testing.T) {
	lossy := readFixture(t, "grad_17x33_q70.webp")
	vp8 := lossy[20:]
	alph := []byte{0, 1, 2} // odd ALPH payload, uncompressed
	for _, v := range [][]byte{vp8, append(append([]byte(nil), vp8...), 0)} {
		var buf bytes.Buffer
		if err := WriteLossy(&buf, 17, 33, v, alph); err != nil {
			t.Fatalf("WriteLossy(%d-byte VP8): %v", len(v), err)
		}
		out := buf.Bytes()
		if len(out)&1 != 0 {
			t.Errorf("WriteLossy(%d-byte VP8): file size %d, want even", len(v), len(out))
		}
		if got, want := binary.LittleEndian.Uint32(out[4:]), uint32(len(out)-8); got != want {
			t.Errorf("WriteLossy(%d-byte VP8): RIFF size = %d, want %d", len(v), got, want)
		}
		img, err := Parse(out)
		if err != nil {
			t.Fatalf("Parse(WriteLossy(%d-byte VP8)): %v", len(v), err)
		}
		if got := img.Chunk(); !bytes.Equal(got, v) {
			t.Errorf("Parse(WriteLossy(%d-byte VP8)): Chunk has %d bytes and differs from the VP8 payload", len(v), len(got))
		}
		if !bytes.Equal(img.ALPH, alph) {
			t.Errorf("Parse(WriteLossy(%d-byte VP8)): ALPH = %v, want %v", len(v), img.ALPH, alph)
		}
		if !img.HasAlpha {
			t.Errorf("Parse(WriteLossy(%d-byte VP8)): HasAlpha = false, want true", len(v))
		}
	}
	var buf bytes.Buffer
	if err := WriteLossless(&buf, []byte{0x2f, 0, 0, 0, 0, 1, 2}); err != nil {
		t.Fatalf("WriteLossless: %v", err)
	}
	if buf.Len() != 28 {
		t.Errorf("WriteLossless wrote %d bytes, want 28", buf.Len())
	}
	f, err := GetFeatures(buf.Bytes())
	if err != nil {
		t.Fatalf("GetFeatures(WriteLossless): %v", err)
	}
	if want := (Features{1, 1, false, false, utils.FormatLossless}); f != want {
		t.Errorf("GetFeatures(WriteLossless) = %+v, want %+v", f, want)
	}
}

func TestWriteInvalid(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteLossy(&buf, 1, 1, nil, nil); err == nil {
		t.Error("WriteLossy(1x1, nil VP8, nil ALPH) succeeded, want error")
	}
	if err := WriteLossy(&buf, 0, 1, []byte{1}, []byte{0}); err == nil {
		t.Error("WriteLossy(0x1) succeeded, want error")
	}
	if err := WriteLossy(&buf, 16384, 1, []byte{1}, []byte{0}); err == nil {
		t.Error("WriteLossy(16384x1) succeeded, want error")
	}
	if err := WriteLossless(&buf, nil); err == nil {
		t.Error("WriteLossless(nil) succeeded, want error")
	}
	if buf.Len() != 0 {
		t.Errorf("invalid writes wrote %d bytes, want 0", buf.Len())
	}
}

type failWriter struct{ n int }

func (w *failWriter) Write(p []byte) (int, error) {
	if w.n == 0 {
		return 0, errors.New("boom")
	}
	w.n--
	return len(p), nil
}

func TestWriteErrors(t *testing.T) {
	vp8 := readFixture(t, "grad_17x33_q70.webp")[20:]
	for n := 0; n < 4; n++ {
		err := WriteLossy(&failWriter{n: n}, 17, 33, vp8, []byte{0, 1, 2})
		if err == nil {
			t.Errorf("WriteLossy to a writer that fails after %d writes succeeded, want error", n)
		}
	}
}

func FuzzParse(f *testing.F) {
	for _, name := range []string{"grad_17x33_q70.webp", "grada_17x33_q70.webp", "grada_17x33_ll.webp", "anim.webp", "meta_exif.webp", "meta_xmp.webp"} {
		f.Add(readFixture(f, name))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		feat, ferr := GetFeatures(data)
		img, err := Parse(data)
		if err != nil {
			if code(err) == utils.Vp8StatusOk {
				t.Fatalf("error without status code: %v", err)
			}
			return
		}
		// Whenever Parse succeeds, GetFeatures succeeds with the same result.
		if ferr != nil || feat != img.Features {
			t.Fatalf("GetFeatures %+v %v vs Parse %+v", feat, ferr, img.Features)
		}
		if img.HasAnimation {
			return
		}
		if (img.VP8 == nil) == (img.VP8L == nil) {
			t.Fatal("exactly one payload expected")
		}
		// VP8 frames are limited to 14 bits; VP8L stores width-1 / height-1
		// on 14 bits, so libwebp accepts up to 16384.
		limit := MaxDimension
		if img.VP8L != nil {
			limit = MaxDimension + 1
		}
		if img.Width < 1 || img.Height < 1 || img.Width > limit || img.Height > limit {
			t.Fatalf("bad dimensions %dx%d", img.Width, img.Height)
		}
	})
}

func BenchmarkParse(b *testing.B) {
	data := readFixture(b, "meta_exif.webp")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Parse(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGetFeatures(b *testing.B) {
	data := readFixture(b, "meta_exif.webp")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := GetFeatures(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWriteLossy(b *testing.B) {
	data := readFixture(b, "meta_exif.webp")
	img, _ := Parse(data)
	var buf bytes.Buffer
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		if err := WriteLossy(&buf, img.Width, img.Height, img.Chunk(), img.ALPH); err != nil {
			b.Fatal(err)
		}
	}
}

// TestChunkSizeWrap checks libwebp's uint32 total_size accounting: an
// optional chunk declaring a size close to 4 GiB wraps the running total
// instead of exceeding the RIFF size, and parsing stops with "not enough
// data" (which WebPGetFeatures turns into the VP8X features).
func TestChunkSizeWrap(t *testing.T) {
	data := []byte("RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x10\x00\x00\x00\x03\x00\x00\x06\x00\x00")
	data = append(data, "ALPH\x02\x00\x00\x00\x00\x00"...)
	data = append(data, "XXXX\xf0\xff\xff\xff"...)
	data = append(data, make([]byte, 200)...)
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	f, err := GetFeatures(data)
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if want := (Features{Width: 4, Height: 7, HasAlpha: true, Format: utils.FormatUndefined}); f != want {
		t.Errorf("GetFeatures = %+v, want %+v", f, want)
	}
}

// TestFrameLossless checks that framing a payload in place gives exactly
// what WriteLossless writes.
func TestFrameLossless(t *testing.T) {
	for _, n := range []int{5, 6, 7, 100, 101} {
		payload := make([]byte, n)
		for i := range payload {
			payload[i] = byte(i*7 + 1)
		}
		var want bytes.Buffer
		if err := WriteLossless(&want, payload); err != nil {
			t.Fatalf("WriteLossless(%d bytes): %v", n, err)
		}
		buf := append(make([]byte, LosslessHeaderSize), payload...)
		got, err := FrameLossless(buf)
		if err != nil {
			t.Fatalf("FrameLossless(%d bytes): %v", n, err)
		}
		if !bytes.Equal(got, want.Bytes()) {
			t.Errorf("FrameLossless(%d bytes) (len %d) differs from WriteLossless (len %d)", n, len(got), want.Len())
		}
	}
	_, err := FrameLossless(make([]byte, LosslessHeaderSize))
	if err == nil {
		t.Fatal("FrameLossless(header only) succeeded, want error")
	}
}

// TestFrameLossy checks that framing a VP8 payload in place gives exactly
// what WriteLossy writes without alpha, for odd and even sizes, with and
// without spare capacity for the padding byte.
func TestFrameLossy(t *testing.T) {
	for _, n := range []int{1, 5, 6, 7, 100, 101} {
		payload := make([]byte, n)
		for i := range payload {
			payload[i] = byte(i*5 + 3)
		}
		var want bytes.Buffer
		if err := WriteLossy(&want, 3, 3, payload, nil); err != nil {
			t.Fatalf("WriteLossy(n=%d): %v", n, err)
		}
		for _, spare := range []int{0, 1, 16} {
			buf := make([]byte, LossyHeaderSize+n, LossyHeaderSize+n+spare)
			copy(buf[LossyHeaderSize:], payload)
			got, err := FrameLossy(buf)
			if err != nil {
				t.Fatalf("FrameLossy(n=%d spare=%d): %v", n, spare, err)
			}
			if !bytes.Equal(got, want.Bytes()) {
				t.Errorf("n=%d spare=%d: FrameLossy (len %d) differs from WriteLossy (len %d)", n, spare, len(got), want.Len())
			}
		}
	}
	_, err := FrameLossy(make([]byte, LossyHeaderSize))
	if want := "VP8_STATUS_INVALID_PARAM: empty VP8 payload"; err == nil || err.Error() != want {
		t.Fatalf("FrameLossy(header only) error = %v, want %q", err, want)
	}
	_, err = FrameLossless(nil)
	if want := "VP8_STATUS_INVALID_PARAM: empty VP8L payload"; err == nil || err.Error() != want {
		t.Fatalf("FrameLossless(nil) error = %v, want %q", err, want)
	}
}

// TestGetFeaturesPrefixRawVP8 checks that for a raw VP8 bitstream (whose
// size, used by the first partition length check, is that of the whole
// input) GetFeaturesPrefix only reports a final result once the prefix
// extends past the first partition, and that every final result equals
// GetFeatures of the whole input.
func TestGetFeaturesPrefixRawVP8(t *testing.T) {
	data, err := os.ReadFile("../../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}
	img, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	raw := img.Chunk()
	part0 := int(uint32(raw[0])|uint32(raw[1])<<8|uint32(raw[2])<<16) >> 5
	if part0 <= 64 {
		t.Fatalf("first partition is %d bytes, want > 64: the first partition must be longer than a small prefix", part0)
	}

	want, err := GetFeatures(raw)
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	for n := 1; n <= len(raw); n++ {
		f, done, err := GetFeaturesPrefix(raw[:n])
		if n <= part0 {
			if done {
				t.Fatalf("GetFeaturesPrefix(prefix %d of %d): done = true, want false (first partition %d)", n, len(raw), part0)
			}
			continue
		}
		if !done {
			t.Fatalf("GetFeaturesPrefix(prefix %d): done = false, want true", n)
		}
		if err != nil {
			t.Fatalf("GetFeaturesPrefix(prefix %d): %v", n, err)
		}
		if f != want {
			t.Fatalf("GetFeaturesPrefix(prefix %d) = %+v, want %+v", n, f, want)
		}
	}
	// An invalid header is final right away.
	bad := append([]byte(nil), raw[:64]...)
	bad[3] = 0
	_, done, err := GetFeaturesPrefix(bad)
	if !done {
		t.Error("GetFeaturesPrefix(invalid header): done = false, want true")
	}
	if got := status.CodeOf(err); got != utils.Vp8StatusBitstreamError {
		t.Errorf("GetFeaturesPrefix(invalid header): status = %v, want %v (err: %v)", got, utils.Vp8StatusBitstreamError, err)
	}
}
