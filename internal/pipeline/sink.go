// Package pipeline connects frame decoders (lossy VP8, lossless VP8L) to the
// final output image. Decoders produce full-width, non-premultiplied RGBA rows
// (4 bytes per pixel, R,G,B,A order) strictly top to bottom; a Sink decides
// where those rows go (straight into the output image, or through
// crop / scale stages).
//
// Zero-copy protocol: the decoder asks the sink for a destination buffer for
// the next n source rows via Buffer, writes the pixels there, then calls
// Commit with the same (y, n). For the common "no crop, no scale" case the
// returned buffer is a slice of the final image's Pix, so no copy happens.
package pipeline

import "image"

// Sink receives decoded rows.
type Sink interface {
	// Buffer returns a writable byte slice able to hold n full-width source
	// rows starting at source row y. Row r (0 <= r < n) starts at offset
	// r*Stride(). The slice is valid until the matching Commit call.
	// y must equal the y+n of the previous Commit (rows arrive in order).
	Buffer(y, n int) []byte
	// Stride returns the row stride, in bytes, of slices returned by Buffer.
	Stride() int
	// Commit marks rows [y, y+n) previously obtained from Buffer as final.
	Commit(y, n int) error
	// NeedRows reports how many source rows (from the top) the sink still
	// needs in total; decoders MAY stop after producing that many rows
	// (e.g. when cropping away the bottom of the image).
	NeedRows() int
}

// Direct writes rows straight into an *image.NRGBA whose bounds equal the
// source frame size (Min must be 0,0). It performs no copy.
type Direct struct {
	Img *image.NRGBA
}

// NewDirect returns a Direct sink for img.
func NewDirect(img *image.NRGBA) *Direct { return &Direct{Img: img} }

// Buffer implements Sink.
func (d *Direct) Buffer(y, n int) []byte {
	s := d.Img.Stride
	end := (y + n) * s
	if end > len(d.Img.Pix) {
		end = len(d.Img.Pix)
	}
	return d.Img.Pix[y*s : end]
}

// Stride implements Sink.
func (d *Direct) Stride() int { return d.Img.Stride }

// Commit implements Sink.
func (d *Direct) Commit(y, n int) error { return nil }

// NeedRows implements Sink.
func (d *Direct) NeedRows() int { return d.Img.Rect.Dy() }
