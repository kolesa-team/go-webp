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
	"os"
	"testing"
)

// TestImageChunkSizeTooSmall checks that, like libwebp, the frame decoders
// get the data up to the end of the input rather than the declared chunk
// size: a simple-format file whose VP8/VP8L chunk size is understated still
// decodes to the same pixels.
func TestImageChunkSizeTooSmall(t *testing.T) {
	// The deltas keep the declared size above the VP8 first partition
	// length, which libwebp checks against the chunk size.
	for name, deltas := range map[string][]uint32{
		"../internal/vp8/testdata/grad_250x199_o9.webp": {1, 8},
		"../test_data/images/100x150_lossless.webp":     {1, 8},
		"../test_data/images/m4_q75.webp":               {1, 8},
	} {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		want := decodeBytes(t, data)
		for _, delta := range deltas {
			m := append([]byte(nil), data...)
			sz := binary.LittleEndian.Uint32(m[16:])
			binary.LittleEndian.PutUint32(m[16:], sz-delta)
			got := decodeBytes(t, m)
			if !bytes.Equal(want.Pix, got.Pix) {
				t.Errorf("%s: chunk size -%d decodes differently", name, delta)
			}
		}
	}
}

func decodeBytes(t *testing.T, data []byte) *image.NRGBA {
	t.Helper()
	d, err := NewDecoder(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatal(err)
	}
	img, err := d.Decode()
	if err != nil {
		t.Fatal(err)
	}
	return img.(*image.NRGBA)
}
