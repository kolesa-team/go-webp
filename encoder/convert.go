package encoder

import (
	"image"
	"image/color"
	"sync"
)

// toNRGBA returns src as an *image.NRGBA. An *image.NRGBA is returned as is
// (shared, as in v1); every other image is copied (see convertNRGBA), so
// that the result is a snapshot of src.
func toNRGBA(src image.Image) *image.NRGBA {
	if v, ok := src.(*image.NRGBA); ok {
		return v
	}
	b := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	convertNRGBA(dst, src)
	return dst
}

// maxPooledPixels bounds the size of the snapshots whose storage is pooled
// (as in the internal encoders): the buffers of larger pictures are left to
// the garbage collector instead of being retained between calls.
const maxPooledPixels = 2 << 20

var pixPool sync.Pool // of *[]byte, NRGBA snapshot storage

// pooledNRGBA returns a w x h *image.NRGBA whose pixels (unspecified, to be
// overwritten) may come from a pool, and the storage to hand back to
// putPix when the image is no longer used.
func pooledNRGBA(w, h int) (*image.NRGBA, *[]byte) {
	n := 4 * w * h
	var pp *[]byte
	// A picture too large to be pooled leaves the pooled buffers alone
	// (taking one would discard it).
	if n <= 4*maxPooledPixels {
		if p, ok := pixPool.Get().(*[]byte); ok && cap(*p) >= n {
			pp = p
		}
	}
	if pp == nil {
		b := make([]byte, n)
		pp = &b
	}
	*pp = (*pp)[:n]
	return &image.NRGBA{Pix: *pp, Stride: 4 * w, Rect: image.Rect(0, 0, w, h)}, pp
}

// putPix returns storage obtained from pooledNRGBA to the pool (unless it
// is too large to be retained).
func putPix(pp *[]byte) {
	if pp == nil || cap(*pp) > 4*maxPooledPixels {
		return
	}
	pixPool.Put(pp)
}

// convertNRGBA overwrites every pixel of dst (whose size is that of src's
// bounds, origin at 0,0) with src converted to non-premultiplied RGBA. The
// common standard library image types are converted with dedicated loops
// (the results equal draw.Draw(dst, r, src, r.Min, draw.Src), i.e.
// color.NRGBAModel conversion, but without its per-pixel interface calls);
// an opaque *image.RGBA, whose bytes are the same premultiplied or not
// (png.Decode returns those for RGB files), is copied row by row.
func convertNRGBA(dst *image.NRGBA, src image.Image) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()

	switch v := src.(type) {
	case *image.RGBA:
		opaque := v.Opaque()
		for y := 0; y < h; y++ {
			s := v.Pix[v.PixOffset(b.Min.X, b.Min.Y+y):][:4*w]
			d := dst.Pix[y*dst.Stride:][:4*w]
			if opaque {
				copy(d, s)
			} else {
				unpremultiplyRow(d, s)
			}
		}
	case *image.Gray:
		for y := 0; y < h; y++ {
			s := v.Pix[v.PixOffset(b.Min.X, b.Min.Y+y):][:w]
			d := dst.Pix[y*dst.Stride:][:4*w]
			for x, g := range s {
				p := d[4*x : 4*x+4 : 4*x+4]
				p[0], p[1], p[2], p[3] = g, g, g, 0xff
			}
		}
	case *image.YCbCr:
		for y := 0; y < h; y++ {
			d := dst.Pix[y*dst.Stride:][:4*w]
			yi := v.YOffset(b.Min.X, b.Min.Y+y)
			ys := v.Y[yi : yi+w]
			for x := range ys {
				ci := v.COffset(b.Min.X+x, b.Min.Y+y)
				r, g, bb := color.YCbCrToRGB(ys[x], v.Cb[ci], v.Cr[ci])
				p := d[4*x : 4*x+4 : 4*x+4]
				p[0], p[1], p[2], p[3] = r, g, bb, 0xff
			}
		}
	case *image.NYCbCrA:
		for y := 0; y < h; y++ {
			d := dst.Pix[y*dst.Stride:][:4*w]
			yi := v.YOffset(b.Min.X, b.Min.Y+y)
			ai := v.AOffset(b.Min.X, b.Min.Y+y)
			ys := v.Y[yi : yi+w]
			as := v.A[ai : ai+w]
			for x := range ys {
				ci := v.COffset(b.Min.X+x, b.Min.Y+y)
				r, g, bb := color.YCbCrToRGB(ys[x], v.Cb[ci], v.Cr[ci])
				p := d[4*x : 4*x+4 : 4*x+4]
				p[0], p[1], p[2], p[3] = r, g, bb, as[x]
			}
		}
	case *image.Paletted:
		var lut [256][4]byte
		for i, c := range v.Palette {
			if i >= len(lut) {
				break
			}
			n := color.NRGBAModel.Convert(c).(color.NRGBA)
			lut[i] = [4]byte{n.R, n.G, n.B, n.A}
		}
		for y := 0; y < h; y++ {
			s := v.Pix[v.PixOffset(b.Min.X, b.Min.Y+y):][:w]
			d := dst.Pix[y*dst.Stride:][:4*w]
			for x, ix := range s {
				copy(d[4*x:4*x+4], lut[ix][:])
			}
		}
	default:
		for y := 0; y < h; y++ {
			d := dst.Pix[y*dst.Stride:][:4*w]
			for x := 0; x < w; x++ {
				n := color.NRGBAModel.Convert(src.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				p := d[4*x : 4*x+4 : 4*x+4]
				p[0], p[1], p[2], p[3] = n.R, n.G, n.B, n.A
			}
		}
	}
}

// unpremultiplyRow converts alpha-premultiplied RGBA samples to
// non-premultiplied ones exactly like color.NRGBAModel.
func unpremultiplyRow(d, s []byte) {
	_ = d[len(s)-1]
	for i := 0; i+3 < len(s); i += 4 {
		a := s[i+3]
		switch a {
		case 0xff:
			d[i], d[i+1], d[i+2], d[i+3] = s[i], s[i+1], s[i+2], 0xff
		case 0:
			d[i], d[i+1], d[i+2], d[i+3] = 0, 0, 0, 0
		default:
			a32 := uint32(a)
			d[i] = uint8((uint32(s[i]) * 0xffff / a32) >> 8)
			d[i+1] = uint8((uint32(s[i+1]) * 0xffff / a32) >> 8)
			d[i+2] = uint8((uint32(s[i+2]) * 0xffff / a32) >> 8)
			d[i+3] = a
		}
	}
}

// hasTransparency reports whether any pixel of img has alpha below 255
// (libwebp's WebPPictureHasTransparency).
func hasTransparency(img *image.NRGBA) bool {
	b := img.Rect
	w := b.Dx()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		row := img.Pix[img.PixOffset(b.Min.X, y):][:4*w]
		for x := 3; x < len(row); x += 4 {
			if row[x] != 0xff {
				return true
			}
		}
	}
	return false
}
