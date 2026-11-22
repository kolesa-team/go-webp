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

package webp

import (
	"context"
	"image"
	"io"

	"github.com/kolesa-team/go-webp/v2/decoder"
)

// DecodeInto decodes a picture from r into dst, reusing the memory of
// dst.Pix when it is large enough, and returns dst (a new image when dst is
// nil). See decoder.Decoder.DecodeInto for the exact semantics.
func DecodeInto(r io.Reader, dst *image.NRGBA, options *decoder.Options) (*image.NRGBA, error) {
	dec, err := decoder.NewDecoder(r, options)
	if err != nil {
		return nil, err
	}
	return dec.DecodeInto(dst)
}

// DecodeContext is Decode with cancellation: ctx is checked before r is
// read and between batches of output rows, see
// decoder.Decoder.DecodeContext.
func DecodeContext(ctx context.Context, r io.Reader, options *decoder.Options) (image.Image, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dec, err := decoder.NewDecoder(r, options)
	if err != nil {
		return nil, err
	}
	return dec.DecodeContext(ctx)
}
