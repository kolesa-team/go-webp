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
