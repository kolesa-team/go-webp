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

// PremultipliedSink is implemented by sinks that alpha-premultiply the rows
// they receive before resampling them (Params.Premultiply). A decoder that
// can premultiply more cheaply than per pixel, such as a palette image that
// premultiplies its palette once, calls AcceptPremultiplied before its first
// Buffer call: when it returns true, the sink no longer premultiplies, and
// every row the decoder delivers must already be premultiplied exactly as
// resample.PremultiplyRGBA would do it.
type PremultipliedSink interface {
	AcceptPremultiplied() bool
}

// AcceptPremultiplied implements PremultipliedSink.
func (s *scaleSink) AcceptPremultiplied() bool {
	if !s.premult || s.scratch != nil || s.done {
		// Not premultiplying, or rows were already delivered.
		return false
	}
	s.preDone = true
	return true
}
