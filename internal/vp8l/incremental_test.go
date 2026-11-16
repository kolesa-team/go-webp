package vp8l

import (
	"bytes"
	"encoding/hex"
	"math/rand"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

// decodeAlphaInBatches decodes an alpha stream with AlphaDecoder in row
// batches of the given sizes (cycled), growing the destination like the
// WebP decoder does for huge planes.
func decodeAlphaInBatches(t *testing.T, payload []byte, w, h int, batches []int) []byte {
	t.Helper()
	a, err := NewAlphaDecoder(payload, w, h)
	must(t, err)
	defer a.Close()
	var dst []byte
	for i, y := 0, 0; y < h; i++ {
		y = min(y+batches[i%len(batches)], h)
		if len(dst) < y*w {
			// Grow by copying, so that the decoder must not rely on
			// earlier destination slices.
			nd := make([]byte, y*w)
			copy(nd, dst)
			dst = nd
		}
		must(t, a.DecodeRows(dst, y))
		check(t, a.Rows() == y, "rows")
	}
	return dst
}

func TestAlphaDecoderIncremental(t *testing.T) {
	type stream struct {
		name    string
		payload []byte
		w, h    int
	}
	var streams []stream
	for name := range alphaFixtures {
		hdr, payload, w, h, ok := alphaFromFile(readFixture(t, "alpha", name))
		if !ok || hdr&3 != 1 {
			t.Fatal("bad ALPH chunk")
		}
		streams = append(streams, stream{name, payload, w, h})
	}
	for _, name := range []string{"noise_64x64_z3.webp", "alpha_257x129_z6.webp", "pal4_17x33_exact_z9.webp", "pal16_500x375_z2.webp"} {
		payload := findChunk(readFixture(t, "lossless", name), "VP8L")
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		streams = append(streams, stream{name, payload[headerSize:], w, h})
	}
	// Planes bigger than the LZ77 window, so that both the 8-bit
	// (palette) and the 32-bit paths slide their buffers.
	const bw, bh = 1500, 1200
	rng := rand.New(rand.NewSource(1))
	few := make([]byte, bw*bh)
	many := make([]byte, bw*bh)
	for y := 0; y < bh; y++ {
		for x := 0; x < bw; x++ {
			few[y*bw+x] = byte((x*x/37 + y*y/23) % 20 * 12)
			many[y*bw+x] = byte(x*y/97) + byte(rng.Intn(4))
		}
	}
	for _, p := range []struct {
		name   string
		plane  []byte
		method int
		use8b  bool
	}{{"big-palette", few, 0, true}, {"big-noise", many, 1, false}} {
		enc, err := vp8lenc.EncodeAlpha(p.plane, bw, bh, bw, p.method, 100)
		must(t, err)
		a, err := NewAlphaDecoder(enc, bw, bh)
		must(t, err)
		check(t, a.d.sliding && a.d.use8b == p.use8b, p.name+": expected a sliding window on the intended path")
		a.Close()
		streams = append(streams, stream{p.name, enc, bw, bh})
	}
	for _, s := range streams {
		want := make([]byte, s.w*s.h)
		must(t, DecodeAlpha(s.payload, s.w, s.h, want), s.name)
		for _, batches := range [][]int{{1}, {3, 16, 1}, {16}, {17, 40}, {s.h}} {
			if batches[0] == 1 && s.h > 300 {
				continue // too slow for no gain
			}
			got := decodeAlphaInBatches(t, s.payload, s.w, s.h, batches)
			if !bytes.Equal(got, want) {
				t.Fatalf("%s %v: incremental decode differs", s.name, batches)
			}
		}
	}
}

// TestAlphaDecoderBomb checks that a stream whose pixels cost no bits (five
// single-symbol codes) is only decoded as far as requested: a 16383x16383
// plane of which only the first rows are asked for must be cheap (the
// whole plane would take hundreds of milliseconds).
func TestAlphaDecoderBomb(t *testing.T) {
	payload, _ := hex.DecodeString("28604404")
	const w, h = 16383, 16383
	a, err := NewAlphaDecoder(payload, w, h)
	must(t, err)
	defer a.Close()
	dst := make([]byte, 32*w)
	start := time.Now()
	must(t, a.DecodeRows(dst, 16))
	must(t, a.DecodeRows(dst, 32))
	if d := time.Since(start); d > time.Second {
		t.Fatalf("decoding 32 rows took %v", d)
	}
	for _, v := range dst {
		check(t, v == dst[0], "constant plane expected")
	}
}
