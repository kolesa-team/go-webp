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

import "image"

// DecodeInto decodes the picture like Decode, but into dst, so that a caller
// decoding many pictures can reuse one output buffer instead of allocating
// (and having the garbage collector reclaim) a new image per picture. It
// returns dst.
//
// dst is treated as raw memory: its previous Rect and Stride are ignored.
// For an output of w x h pixels (after cropping and scaling), when
// cap(dst.Pix) >= 4*w*h the pixels are written to dst.Pix[:4*w*h] without
// allocating or clearing anything; otherwise a new buffer is allocated. On
// success dst.Pix is set to that w x h image (keeping the capacity of a
// reused buffer), dst.Stride to 4*w and dst.Rect to (0, 0)-(w, h); crop,
// scale and flip options apply as for Decode. Do not pass a SubImage: its
// Pix starts at the sub-image origin, so the decoded pixels would overwrite
// the rest of the parent image.
//
// On error the dst fields are left unchanged, but the bytes of the memory
// dst.Pix refers to (up to its capacity) may have been partially
// overwritten.
//
// The returned image is dst itself: decoding into dst again overwrites it.
// dst must not be used by other goroutines during the call. A nil dst
// makes DecodeInto behave like Decode (a new image is allocated).
func (d *Decoder) DecodeInto(dst *image.NRGBA) (*image.NRGBA, error) {
	return d.decodeInto(call{dst: dst})
}

func (d *Decoder) decodeInto(c call) (*image.NRGBA, error) {
	img, err := d.decode(c)
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	if c.dst == nil {
		return img, nil
	}
	c.dst.Pix, c.dst.Stride, c.dst.Rect = img.Pix, img.Stride, img.Rect
	return c.dst, nil
}
