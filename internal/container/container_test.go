package container

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func readFixture(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	require.NoError(t, err)
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
			require.NoError(t, err)
			assert.Equal(t, tt.feat, f)

			img, err := Parse(data)
			require.NoError(t, err)
			assert.Equal(t, tt.feat, img.Features)
			payload := img.Chunk()
			if tt.feat.Format == utils.FormatLossless {
				assert.Nil(t, img.VP8)
				assert.Equal(t, data[tt.off:], img.VP8L)
			} else {
				assert.Nil(t, img.VP8L)
				assert.Equal(t, data[tt.off:], img.VP8)
			}
			assert.Equal(t, data[tt.off:tt.off+tt.size], payload)
			if tt.alphaOff < 0 {
				assert.Nil(t, img.ALPH)
			} else {
				assert.Equal(t, data[tt.alphaOff:tt.alphaOff+tt.alphaN], img.ALPH)
			}
			// Payloads alias the input.
			assert.Equal(t, &data[tt.off], &payload[0])
		})
	}
}

func TestAnimated(t *testing.T) {
	data := readFixture(t, "anim.webp")
	want := Features{Width: 3, Height: 7, HasAlpha: true, HasAnimation: true, Format: utils.FormatUndefined}
	f, err := GetFeatures(data)
	require.NoError(t, err)
	assert.Equal(t, want, f)
	img, err := Parse(data)
	require.NoError(t, err)
	assert.Equal(t, &Image{Features: want}, img)
	// Header-only is enough for GetFeatures.
	f, err = GetFeatures(data[:30])
	require.NoError(t, err)
	assert.Equal(t, want, f)
}

func TestRawBitstreams(t *testing.T) {
	lossy := readFixture(t, "grad_17x33_q70.webp")
	img, err := Parse(lossy[20:])
	require.NoError(t, err)
	assert.Equal(t, Features{17, 33, false, false, utils.FormatLossy}, img.Features)
	assert.Equal(t, lossy[20:], img.VP8)

	ll := readFixture(t, "grada_17x33_ll.webp")
	img, err = Parse(ll[20:])
	require.NoError(t, err)
	assert.Equal(t, Features{17, 33, true, false, utils.FormatLossless}, img.Features)
	assert.Equal(t, ll[20:], img.VP8L)

	// "ALPH" + "VP8 " without RIFF is accepted (libwebp internal format).
	withAlpha := readFixture(t, "grada_17x33_q70.webp")
	img, err = Parse(withAlpha[30:])
	require.NoError(t, err)
	assert.Equal(t, Features{17, 33, true, false, utils.FormatLossy}, img.Features)
	assert.Equal(t, 127, len(img.ALPH))

	// VP8X without RIFF is rejected.
	_, err = Parse(withAlpha[12:])
	assert.Equal(t, utils.Vp8StatusBitstreamError, status.CodeOf(err))
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
			assert.Equal(t, tt.feat, code(err), "GetFeatures: %v", err)
			if tt.checkFeatures {
				assert.Equal(t, 17, f.Width)
				assert.Equal(t, 33, f.Height)
			}
			_, err = Parse(tt.data)
			assert.Equal(t, tt.parse, code(err), "Parse: %v", err)
			if err != nil {
				var se *status.Error
				assert.True(t, errors.As(err, &se))
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
			require.NoError(t, err)
			var buf bytes.Buffer
			if img.VP8L != nil {
				require.NoError(t, WriteLossless(&buf, img.Chunk()))
			} else {
				require.NoError(t, WriteLossy(&buf, img.Width, img.Height, img.Chunk(), img.ALPH))
			}
			assert.Equal(t, data, buf.Bytes())
		})
	}
}

func TestWriteOddSizes(t *testing.T) {
	lossy := readFixture(t, "grad_17x33_q70.webp")
	vp8 := lossy[20:]
	alph := []byte{0, 1, 2} // odd ALPH payload, uncompressed
	for _, v := range [][]byte{vp8, append(append([]byte(nil), vp8...), 0)} {
		var buf bytes.Buffer
		require.NoError(t, WriteLossy(&buf, 17, 33, v, alph))
		out := buf.Bytes()
		assert.Equal(t, 0, len(out)&1)
		assert.Equal(t, uint32(len(out)-8), binary.LittleEndian.Uint32(out[4:]))
		img, err := Parse(out)
		require.NoError(t, err)
		assert.Equal(t, v, img.Chunk())
		assert.Equal(t, alph, img.ALPH)
		assert.True(t, img.HasAlpha)
	}
	var buf bytes.Buffer
	require.NoError(t, WriteLossless(&buf, []byte{0x2f, 0, 0, 0, 0, 1, 2}))
	assert.Equal(t, 28, buf.Len())
	f, err := GetFeatures(buf.Bytes())
	require.NoError(t, err)
	assert.Equal(t, Features{1, 1, false, false, utils.FormatLossless}, f)
}

func TestWriteInvalid(t *testing.T) {
	var buf bytes.Buffer
	assert.Error(t, WriteLossy(&buf, 1, 1, nil, nil))
	assert.Error(t, WriteLossy(&buf, 0, 1, []byte{1}, []byte{0}))
	assert.Error(t, WriteLossy(&buf, 16384, 1, []byte{1}, []byte{0}))
	assert.Error(t, WriteLossless(&buf, nil))
	assert.Equal(t, 0, buf.Len())
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
		assert.Error(t, err, "fail after %d writes", n)
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
	require.NoError(t, err)
	assert.Equal(t, Features{Width: 4, Height: 7, HasAlpha: true, Format: utils.FormatUndefined}, f)
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
		require.NoError(t, WriteLossless(&want, payload))
		buf := append(make([]byte, LosslessHeaderSize), payload...)
		got, err := FrameLossless(buf)
		require.NoError(t, err)
		assert.Equal(t, want.Bytes(), got)
	}
	_, err := FrameLossless(make([]byte, LosslessHeaderSize))
	require.Error(t, err)
}
