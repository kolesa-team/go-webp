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

package decoder

import (
	"bytes"
	"encoding/binary"
	"image"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

type chunk struct {
	tag  string
	data []byte
}

// splitChunks returns the chunks of a RIFF/WEBP file.
func splitChunks(t testing.TB, file []byte) []chunk {
	t.Helper()
	if tag := string(file[:4]); tag != "RIFF" {
		t.Fatalf("file starts with %q, want RIFF", tag)
	}
	var out []chunk
	for p := 12; p+8 <= len(file); {
		n := int(binary.LittleEndian.Uint32(file[p+4:]))
		if p+8+n > len(file) {
			t.Fatalf("chunk %q at offset %d: %d payload bytes overrun the %d-byte file", file[p:p+4], p, n, len(file))
		}
		out = append(out, chunk{string(file[p : p+4]), bytes.Clone(file[p+8 : p+8+n])})
		p += 8 + n + n&1
	}
	return out
}

func joinChunks(chunks []chunk) []byte {
	var parts [][]byte
	for _, c := range chunks {
		parts = append(parts, riffChunk(c.tag, c.data))
	}
	return riffFile(parts...)
}

// modifyChunk returns file with f applied to the payload of its tag chunk.
func modifyChunk(t testing.TB, file []byte, tag string, f func([]byte) []byte) []byte {
	t.Helper()
	chunks := splitChunks(t, file)
	found := false
	for i := range chunks {
		if chunks[i].tag == tag {
			chunks[i].data = f(chunks[i].data)
			found = true
		}
	}
	if !found {
		t.Fatalf("no %s chunk", tag)
	}
	return joinChunks(chunks)
}

// alphaModes are the decoding modes whose alpha errors must agree.
var alphaModes = []*Options{
	nil,
	{UseThreads: true},
	{Scale: image.Rect(0, 0, 300, 200)},
	{Scale: image.Rect(0, 0, 300, 200), UseThreads: true},
	{Crop: image.Rect(100, 100, 500, 300), UseThreads: true},
	{AlphaDitheringStrength: 50, UseThreads: true},
}

// firstPartitionOnly truncates a VP8 key frame payload after its first
// partition.
func firstPartitionOnly(p []byte) []byte {
	partLen := int(uint32(p[0])|uint32(p[1])<<8|uint32(p[2])<<16) >> 5
	return p[:10+partLen]
}

func decodeErr(t testing.TB, data []byte, o *Options) error {
	t.Helper()
	d, err := NewDecoder(bytes.NewReader(data), o)
	if err != nil {
		return err
	}
	_, err = d.Decode()
	return err
}

// TestAlphaInitErrorOrder checks the error order of a broken ALPH header
// when the alpha decoder is set up before the VP8 decode starts (UseThreads):
// a VP8 error that comes before the first emitted row still wins.
func TestAlphaInitErrorOrder(t *testing.T) {
	logo := readFile(t, "webp-logo-lossy.webp")
	badHeader := func(p []byte) []byte { p[0] = 0xc1; return p }
	cases := []struct {
		name string
		data []byte
		code utils.VP8StatusCode
	}{
		{"bad header", modifyChunk(t, logo, "ALPH", badHeader), utils.Vp8StatusOutOfMemory},
		{"short chunk", modifyChunk(t, logo, "ALPH", func(p []byte) []byte { return p[:1] }), utils.Vp8StatusOutOfMemory},
		{"broken codes", modifyChunk(t, logo, "ALPH", func(p []byte) []byte { return p[:3] }), utils.Vp8StatusOutOfMemory},
		// The VP8 frame keeps its first partition (modes) but loses the
		// token partition: it fails before its first row.
		{"bad header, broken vp8", modifyChunk(t, modifyChunk(t, logo, "ALPH", badHeader), "VP8 ", firstPartitionOnly), utils.Vp8StatusNotEnoughData},
	}
	for _, c := range cases {
		for _, o := range alphaModes {
			err := decodeErr(t, c.data, o)
			if code := status.CodeOf(err); code != c.code {
				t.Errorf("%s %+v: %v: code %v, want %v", c.name, o, err, code, c.code)
			}
		}
	}
}

// TestAlphaSetUpOnCaller checks that with UseThreads the pooled alpha
// decoder state is taken on the calling goroutine (which also returns it),
// not on the VP8 worker: a sync.Pool Get on another P misses the state the
// previous decode put back (+1.5 MB per decode for the vp8l alpha window).
// Without UseThreads the set-up stays lazy.
func TestAlphaSetUpOnCaller(t *testing.T) {
	chunks := splitChunks(t, readFile(t, "webp-logo-lossy.webp"))
	var alph []byte
	for _, c := range chunks {
		if c.tag == "ALPH" {
			alph = c.data
		}
	}
	if alph == nil {
		t.Fatal("webp-logo-lossy.webp has no ALPH chunk")
	}
	frame := image.Rect(0, 0, 3000, 2000)

	a := newLazyAlpha(alph, 3000, 2000, frame, &Config{UseThreads: true})
	if !a.started {
		t.Error("UseThreads: alpha decoder not set up by newLazyAlpha")
	}
	if a.dec == nil {
		t.Error("UseThreads: alpha decoder state not taken by newLazyAlpha")
	}
	if a.err != nil {
		t.Fatalf("UseThreads: newLazyAlpha: %v", a.err)
	}
	a.release()

	a = newLazyAlpha(alph, 3000, 2000, frame, &Config{})
	if a.started {
		t.Error("without UseThreads: alpha decoder set up by newLazyAlpha, want lazy")
	}
	if a.dec != nil {
		t.Error("without UseThreads: alpha decoder state taken by newLazyAlpha, want lazy")
	}
	a.release()

	// A broken header is reported by the first get, not before.
	bad := append([]byte{0xc1}, alph[1:]...)
	a = newLazyAlpha(bad, 3000, 2000, frame, &Config{UseThreads: true})
	if !a.started {
		t.Error("broken header: alpha decoder not set up by newLazyAlpha")
	}
	_, _, err := a.get(0, 16)
	if code := status.CodeOf(err); code != utils.Vp8StatusOutOfMemory {
		t.Errorf("broken header: get(0, 16) = %v: code %v, want %v", err, code, utils.Vp8StatusOutOfMemory)
	}
	a.release()
}
