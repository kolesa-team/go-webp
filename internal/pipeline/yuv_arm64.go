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

//go:build arm64 && !purego

package pipeline

// yuvToRGBARowNEON converts pixels [0, n) of full resolution Y, U, V rows
// (and alpha, or opaque when a is nil) to RGBA, n >= 16. It reads y, u, v
// and a [:n] and writes dst[:4*n].
//
//go:noescape
func yuvToRGBARowNEON(y, u, v, a, dst *byte, n int)

// yuvToRGBARow converts full resolution Y'CbCr rows to RGBA with NEON; it
// matches yuvToRGBARowGo bit for bit.
func yuvToRGBARow(yr, ur, vr, ar, out []byte) {
	w := len(yr)
	if w < 16 {
		yuvToRGBARowGo(yr, ur, vr, ar, out)
		return
	}
	ur, vr = ur[:w], vr[:w]
	out = out[:4*w]
	var pa *byte
	if ar != nil {
		ar = ar[:w]
		pa = &ar[0]
	}
	yuvToRGBARowNEON(&yr[0], &ur[0], &vr[0], pa, &out[0], w)
}
