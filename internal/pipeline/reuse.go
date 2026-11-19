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

package pipeline

import "image"

// newOutput returns the w x h output image of a sink. When reuse is not nil
// and its Pix has the capacity for 4*w*h bytes, the image is backed by that
// memory (reuse.Pix[:4*w*h], keeping its capacity so that a later, larger
// decode can reuse it too) and is not cleared: every sink writes each output
// pixel exactly once before the decode succeeds, so stale bytes never show.
// Otherwise a new image is allocated. reuse itself (its Pix slice header,
// Stride and Rect) is never modified; the decoder copies the result into it
// only once decoding has succeeded.
func newOutput(w, h int, reuse *image.NRGBA) *image.NRGBA {
	n := 4 * w * h
	if reuse != nil && cap(reuse.Pix) >= n {
		return &image.NRGBA{Pix: reuse.Pix[:n], Stride: 4 * w, Rect: image.Rect(0, 0, w, h)}
	}
	return image.NewNRGBA(image.Rect(0, 0, w, h))
}
