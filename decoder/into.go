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
	img, err := d.decode(call{dst: dst})
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	if dst == nil {
		return img, nil
	}
	dst.Pix, dst.Stride, dst.Rect = img.Pix, img.Stride, img.Rect
	return dst, nil
}
