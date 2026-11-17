package decoder

import (
	"image"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/vp8"
)

// DecodeYCbCr decodes a lossy picture without alpha to its Y'CbCr 4:2:0
// planes, without converting them to RGB: the output takes 1.5 bytes per
// pixel instead of 4, and decoding is faster. The planes hold the samples
// of the VP8 stream as they are (BT.601 "limited range"; libwebp's
// WebPDecodeYUV output), like the *image.YCbCr of golang.org/x/image/webp,
// so the colors that image.YCbCr's color model computes (full-range JFIF)
// differ slightly from the RGB that Decode returns, which also upsamples
// the chroma smoothly ("fancy upsampling") instead of repeating it.
//
// Options: Crop (with its top-left corner rounded down to even
// coordinates, as for Decode), Flip (each plane is mirrored separately,
// like libwebp does for Y'CbCr output), BypassFiltering, DitheringStrength
// and UseThreads apply; NoFancyUpsampling has no effect. Scale is not
// supported (VP8_STATUS_INVALID_PARAM).
//
// Pictures with alpha (use DecodeNYCbCrA), lossless pictures (they have no
// Y'CbCr form; use Decode) and animations are refused with
// VP8_STATUS_UNSUPPORTED_FEATURE: check GetFeatures first.
func (d *Decoder) DecodeYCbCr() (*image.YCbCr, error) {
	s, err := d.decodeYUV(false)
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	return s.yuv, nil
}

// DecodeNYCbCrA is DecodeYCbCr for lossy pictures with or without alpha: the
// alpha plane (the decoded ALPH chunk, with AlphaDitheringStrength applied;
// all 0xff for an opaque picture) comes with the Y'CbCr 4:2:0 planes, 2.5
// bytes per pixel in all. Lossless pictures are refused like by
// DecodeYCbCr.
func (d *Decoder) DecodeNYCbCrA() (*image.NYCbCrA, error) {
	s, err := d.decodeYUV(true)
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	return s.nyuv, nil
}

func (d *Decoder) decodeYUV(withAlpha bool) (*ycbcrSink, error) {
	parsed, err := container.Parse(d.data)
	if err != nil {
		return nil, err
	}
	switch {
	case parsed.HasAnimation:
		return nil, status.Unsupported("animated WebP is not supported")
	case parsed.VP8L != nil:
		return nil, status.Unsupported("lossless pictures have no Y'CbCr form")
	case parsed.ALPH != nil && !withAlpha:
		return nil, status.Unsupported("the picture has alpha: decode it with DecodeNYCbCrA")
	}
	c := d.config
	if c.UseScaling {
		return nil, status.InvalidParam("scaling is not supported for Y'CbCr output")
	}
	w, h := parsed.Width, parsed.Height
	crop := image.Rect(0, 0, w, h)
	if c.UseCropping {
		crop = pipeline.SnapCropForYUV(image.Rect(c.CropLeft, c.CropTop, c.CropLeft+c.CropWidth, c.CropTop+c.CropHeight))
		if crop.Empty() || crop.Min.X < 0 || crop.Min.Y < 0 || crop.Max.X > w || crop.Max.Y > h {
			return nil, status.InvalidParam("crop rectangle outside of the frame")
		}
	}
	s := &ycbcrSink{
		w: crop.Dx(), h: crop.Dy(),
		flip:      c.Flip,
		withAlpha: withAlpha,
		top:       crop.Min.Y, left: crop.Min.X, alphaStride: w,
	}
	if parsed.ALPH != nil {
		s.alpha = newLazyAlpha(parsed.ALPH, w, h, crop, c)
		defer s.alpha.release()
	}
	opts := vp8.Options{
		BypassFiltering:   c.BypassFiltering,
		DitheringStrength: c.DitheringStrength,
		UseThreads:        c.UseThreads,
	}
	if err := vp8.DecodeYUV(parsed.VP8, &opts, &crop, s); err != nil {
		return nil, err
	}
	if s.yuv == nil {
		return nil, status.NotEnoughData("no rows decoded")
	}
	return s, nil
}

// ycbcrSink copies the Y'CbCr rows of the VP8 decoder (and the alpha rows)
// into an *image.YCbCr or *image.NYCbCrA, allocated with the first rows.
type ycbcrSink struct {
	w, h        int // output size
	flip        bool
	withAlpha   bool
	alpha       *lazyAlpha // nil for opaque pictures
	top, left   int        // crop origin in the alpha plane
	alphaStride int

	yuv  *image.YCbCr
	nyuv *image.NYCbCrA // nil without alpha output; yuv is &nyuv.YCbCr
}

// PutYUV implements vp8.YUVSink.
func (s *ycbcrSink) PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error {
	var aplane []byte
	if s.alpha != nil {
		p, err := s.alpha.get(s.top + y + n)
		if err != nil {
			return err
		}
		aplane = p
	}
	if s.yuv == nil {
		r := image.Rect(0, 0, s.w, s.h)
		if s.withAlpha {
			s.nyuv = image.NewNYCbCrA(r, image.YCbCrSubsampleRatio420)
			s.yuv = &s.nyuv.YCbCr
		} else {
			s.yuv = image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
		}
	}
	img := s.yuv
	w := s.w
	for j := range n {
		copy(img.Y[s.row(y+j, s.h)*img.YStride:][:w], ys[j*yStride:][:w])
	}
	cw, ch := (w+1)/2, (s.h+1)/2
	for k := range (n + 1) / 2 {
		o := s.row(y/2+k, ch) * img.CStride
		copy(img.Cb[o:][:cw], us[k*uvStride:][:cw])
		copy(img.Cr[o:][:cw], vs[k*uvStride:][:cw])
	}
	if s.nyuv == nil {
		return nil
	}
	a := s.nyuv
	for j := range n {
		dst := a.A[s.row(y+j, s.h)*a.AStride:][:w]
		if aplane == nil {
			for i := range dst {
				dst[i] = 0xff
			}
		} else {
			copy(dst, aplane[(s.top+y+j)*s.alphaStride+s.left:][:w])
		}
	}
	return nil
}

// row returns where row r of a plane of h rows goes.
func (s *ycbcrSink) row(r, h int) int {
	if s.flip {
		return h - 1 - r
	}
	return r
}

// NeedRows implements vp8.YUVSink.
func (s *ycbcrSink) NeedRows() int { return s.h }
