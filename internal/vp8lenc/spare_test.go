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

package vp8lenc

import (
	"bytes"
	"testing"
)

// TestAlphaEncoderRecycle checks that a recycled stream buffer is reused by
// the next Encode and that reuse does not change the output.
func TestAlphaEncoderRecycle(t *testing.T) {
	const w, h = 97, 61
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = byte(i*i>>5) & 0xf0
	}
	want, err := EncodeAlpha(plane, w, h, w, 4, 100)
	if err != nil {
		t.Fatal(err)
	}
	var a AlphaEncoder
	defer a.Release()
	for _, method := range []int{4, 6, 0, 4} {
		b1, err := a.Encode(plane, w, h, w, method, 100)
		if err != nil {
			t.Fatal(err)
		}
		saved := append([]byte(nil), b1...)
		a.Recycle(b1)
		if cap(a.spare) < cap(b1) {
			t.Fatalf("method %d: no spare buffer at least as big as the recycled one", method)
		}
		spare := a.spare
		// Dirty the spare buffer: the encoder must not depend on it.
		for i := range a.spare[:cap(a.spare)] {
			a.spare[:cap(a.spare)][i] = 0xa5
		}
		b2, err := a.Encode(plane, w, h, w, method, 100)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b2, saved) {
			t.Fatalf("method %d: output changed with a recycled buffer", method)
		}
		if method == 4 && !bytes.Equal(b2, want) {
			t.Fatal("output differs from EncodeAlpha")
		}
		// A single-configuration encode writes into the spare buffer.
		same := func(x, y []byte) bool { return cap(x) > 0 && cap(y) > 0 && &x[:1][0] == &y[:1][0] }
		if method == 0 && !same(spare, b2) {
			t.Fatalf("method %d: recycled buffer was not reused", method)
		}
		a.Recycle(b2)
	}
}

// TestEncodeARGBSpare checks that the multi-configuration search (method 5
// and 6 at high quality) hands back a losing buffer distinct from the
// result, and accepts it as the output buffer of the next call with an
// identical result.
func TestEncodeARGBSpare(t *testing.T) {
	const w, h = 64, 48
	argb := make([]uint32, w*h)
	for i := range argb {
		argb[i] = 0xff000000 | uint32(i)*2654435761&0x00f0f0f0
	}
	cfg := &Config{Quality: 100, Method: 6}
	enc := func(spare []byte) ([]byte, []byte) {
		src := append([]uint32(nil), argb...)
		return encodeARGB(src, w, h, cfg, true, 0, nil, spare, nil)
	}
	best, spare := enc(nil)
	if spare != nil && &best[:cap(best)][0] == &spare[:cap(spare)][0] {
		t.Fatal("spare aliases the result")
	}
	saved := append([]byte(nil), best...)
	best2, _ := enc(spare)
	if !bytes.Equal(best2, saved) {
		t.Fatal("output changed with a spare buffer")
	}
}
