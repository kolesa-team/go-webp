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
	"crypto/sha256"
	"encoding/hex"
	"image"
	"os"
	"strings"
	"testing"
)

// Corrupt VP8 partitions can push the bool decoder's value more than 2^31
// above split. libwebp's VP8GetBit compares unsigned, so a signed 32-bit
// difference would decode different bits (or succeed where libwebp fails).
// Expected outputs come from a plain-C (non-NEON) libwebp 1.6 dwebp build.
func TestBoolDecoderUnsignedCompare(t *testing.T) {
	t.Run("pixels match libwebp", func(t *testing.T) {
		f, err := os.Open("testdata/bigvalue_flip.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d, err := NewDecoder(f, nil)
		if err != nil {
			t.Fatal(err)
		}
		img, err := d.Decode()
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(img.(*image.NRGBA).Pix)
		const want = "3f080bc905aa17b0421d020389801b4f5a837b71756f0a682edc56cafb27b8f2"
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("pixel hash %s, want %s", got, want)
		}
	})
	t.Run("truncation detected like libwebp", func(t *testing.T) {
		f, err := os.Open("testdata/bigvalue_truncated.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d, err := NewDecoder(f, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = d.Decode(); err == nil || !strings.Contains(err.Error(), "VP8_STATUS_NOT_ENOUGH_DATA") {
			t.Fatalf("got %v, want VP8_STATUS_NOT_ENOUGH_DATA", err)
		}
	})
}
