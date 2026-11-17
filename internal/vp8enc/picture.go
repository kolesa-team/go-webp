package vp8enc

import (
	"image"
	"math"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Picture is a YUV 4:2:0 picture (BT.601 "studio" range as libwebp: Y in
// [16,235]). Y has YStride bytes per row; U and V have UVStride bytes per row
// and (Height+1)/2 rows of (Width+1)/2 samples.
type Picture struct {
	Width, Height     int
	Y, U, V           []byte
	YStride, UVStride int

	buf []byte // pooled backing storage, if any
}

var pictureBufPool sync.Pool // of *[]byte

// maxPooledPixels bounds the picture size whose buffers (pictures, encoder
// state) are pooled: the memory of larger encodes is left to the garbage
// collector instead of being retained between calls.
const maxPooledPixels = 2 << 20

func newPicture(w, h int) *Picture {
	uvw, uvh := (w+1)>>1, (h+1)>>1
	ySize := w * h
	uvSize := uvw * uvh
	n := ySize + 2*uvSize
	var buf []byte
	if v := pictureBufPool.Get(); v != nil {
		b := *(v.(*[]byte))
		if cap(b) >= n {
			buf = b[:n]
		}
	}
	if buf == nil {
		buf = make([]byte, n)
	}
	return &Picture{
		Width: w, Height: h,
		Y:       buf[:ySize:ySize],
		U:       buf[ySize : ySize+uvSize : ySize+uvSize],
		V:       buf[ySize+uvSize : n : n],
		YStride: w, UVStride: uvw,
		buf: buf,
	}
}

// unpooledPicture is newPicture with freshly allocated storage that Release
// does not return to the pool.
func unpooledPicture(w, h int) *Picture {
	uvw, uvh := (w+1)>>1, (h+1)>>1
	ySize := w * h
	uvSize := uvw * uvh
	n := ySize + 2*uvSize
	buf := make([]byte, n)
	return &Picture{
		Width: w, Height: h,
		Y:       buf[:ySize:ySize],
		U:       buf[ySize : ySize+uvSize : ySize+uvSize],
		V:       buf[ySize+uvSize : n : n],
		YStride: w, UVStride: uvw,
	}
}

// Release returns the picture's storage to an internal pool. The picture
// must not be used afterwards. It is safe to call on pictures not created
// by this package (it then does nothing) and on nil.
func (p *Picture) Release() {
	if p == nil || p.buf == nil {
		return
	}
	b := p.buf[:0]
	p.buf = nil
	p.Y, p.U, p.V = nil, nil, nil
	if p.Width*p.Height > maxPooledPixels {
		return
	}
	pictureBufPool.Put(&b)
}

func (p *Picture) validate() error {
	w, h := p.Width, p.Height
	if w < 1 || h < 1 || w > MaxDimension || h > MaxDimension {
		return status.InvalidParam("vp8enc: bad picture dimensions")
	}
	uvw, uvh := (w+1)>>1, (h+1)>>1
	if p.YStride < w || p.UVStride < uvw {
		return status.InvalidParam("vp8enc: bad picture stride")
	}
	if len(p.Y) < (h-1)*p.YStride+w || len(p.U) < (uvh-1)*p.UVStride+uvw || len(p.V) < (uvh-1)*p.UVStride+uvw {
		return status.InvalidParam("vp8enc: picture buffers too small")
	}
	return nil
}

//------------------------------------------------------------------------------
// RGB -> YUV conversion (libwebp's "fast" path with gamma-compressed chroma
// averaging). Algorithm adapted from libwebp (BSD-3-Clause).

const (
	yuvFix        = 16
	yuvHalf       = 1 << (yuvFix - 1)
	gammaFix      = 12
	gammaTabFix   = 7
	gammaTabSize  = 1 << (gammaFix - gammaTabFix)
	kGammaF       = 0.80
	kGammaScale   = (1 << gammaFix) - 1
	kGammaTabScl  = 1 << gammaTabFix
	kGammaRounder = kGammaTabScl >> 1
	kAlphaFix     = 19
)

var (
	csTablesOnce      sync.Once
	kGammaToLinearTab [256]uint32
	kLinearToGammaTab [gammaTabSize + 1]int32
	kInvAlpha         [4*0xff + 1]uint32
)

func initCSTables() {
	csTablesOnce.Do(func() {
		scale := float64(1<<gammaTabFix) / kGammaScale
		norm := 1. / 255.
		for v := 0; v <= 255; v++ {
			kGammaToLinearTab[v] = uint32(math.Pow(norm*float64(v), kGammaF)*kGammaScale + .5)
		}
		for v := 0; v <= gammaTabSize; v++ {
			kLinearToGammaTab[v] = int32(255.*math.Pow(scale*float64(v), 1./kGammaF) + .5)
		}
		for a := 1; a < len(kInvAlpha); a++ {
			kInvAlpha[a] = (1 << kAlphaFix) / uint32(a)
		}
	})
}

func linearToGamma(base uint32, shift uint) int32 {
	v := int32(base << shift)
	tabPos := (v >> (gammaTabFix + 2)) & 31
	x := v & ((kGammaTabScl << 2) - 1)
	v0 := kLinearToGammaTab[tabPos]
	v1 := kLinearToGammaTab[tabPos+1]
	y := v1*x + v0*((kGammaTabScl<<2)-x)
	return (y + kGammaRounder) >> gammaTabFix
}

func rgbToY(r, g, b int32) uint8 { return rgbToYR(r, g, b, yuvHalf) }

// rgbToYR is VP8RGBToY with an explicit rounding term.
func rgbToYR(r, g, b, rounding int32) uint8 {
	return uint8((16839*r + 33059*g + 6420*b + rounding + (16 << yuvFix)) >> yuvFix)
}

func clipUV(uv int32) uint8 { return clipUVR(uv, yuvHalf<<2) }

// clipUVR is VP8ClipUV with an explicit rounding term.
func clipUVR(uv, rounding int32) uint8 {
	uv = (uv + rounding + (128 << (yuvFix + 2))) >> (yuvFix + 2)
	if uint32(uv) <= 255 {
		return uint8(uv)
	}
	if uv < 0 {
		return 0
	}
	return 255
}

func rgbToU(r, g, b int32) uint8 { return clipUV(-9719*r - 19081*g + 28800*b) }
func rgbToV(r, g, b int32) uint8 { return clipUV(28800*r - 24116*g - 4684*b) }

// FromNRGBA converts img to a YUV 4:2:0 picture exactly like libwebp's
// RGBA import followed by WebPPictureARGBToYUVA (gamma-compressed, alpha
// weighted chroma averaging). If sharpYUV is true the iterative "sharp"
// conversion of libsharpyuv is used instead (for pictures of at least 4x4).
// If exact is false and the image has transparent pixels, the luma/chroma
// of fully transparent areas is flattened (WebPCleanupTransparentArea) to
// help compression. The returned picture may be pooled: call Release when
// done.
func FromNRGBA(img *image.NRGBA, sharpYUV bool, exact bool) *Picture {
	return FromNRGBADithered(img, sharpYUV, exact, 0)
}

// FromNRGBADithered is FromNRGBA with libwebp's dithered conversion
// (WebPPictureARGBToYUVADithered): when dithering > 0 and sharpYUV is not
// used, the rounding term of every RGB->Y/U/V conversion is randomized
// with amplitude dithering (0..1, see DitheringFor), which only affects the
// sub-LSB rounding of the samples.
func FromNRGBADithered(img *image.NRGBA, sharpYUV bool, exact bool, dithering float32) *Picture {
	initCSTables()
	b := img.Rect
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return &Picture{}
	}
	pic := newPicture(w, h)
	pix := img.Pix[img.PixOffset(b.Min.X, b.Min.Y):]
	stride := img.Stride
	hasAlpha := false
	for y := 0; y < h && !hasAlpha; y++ {
		row := pix[y*stride : y*stride+4*w]
		for x := 3; x < len(row); x += 4 {
			if row[x] != 0xff {
				hasAlpha = true
				break
			}
		}
	}
	if sharpYUV && w >= 4 && h >= 4 {
		sharpRGBToYUV(pix, stride, w, h, pic)
	} else {
		var rg *vp8Random
		if dithering > 0 {
			rg = newRandom(dithering)
		}
		importRGBA(pix, stride, w, h, hasAlpha, pic, rg)
	}
	if hasAlpha && !exact {
		cleanupTransparentArea(pic, pix[3:], stride)
	}
	return pic
}

func rowsOpaque(pix []byte, stride, w, n int) bool {
	for y := 0; y < n; y++ {
		row := pix[y*stride : y*stride+4*w]
		for x := 3; x < len(row); x += 4 {
			if row[x] != 0xff {
				return false
			}
		}
	}
	return true
}

// importRGBA converts the RGB samples to Y'CbCr 4:2:0; rg, if not nil,
// randomizes the rounding (libwebp's ConvertRowToY/ConvertRowsToUV with a
// VP8Random, in the same draw order).
func importRGBA(pix []byte, stride, w, h int, hasAlpha bool, pic *Picture, rg *vp8Random) {
	uvw := (w + 1) >> 1
	tmp := make([]int32, 4*uvw) // r,g,b accumulators
	for y := 0; y < h; y += 2 {
		rows := 2
		rs := stride // stride to the second row
		if y+1 >= h {
			rows = 1
			rs = 0
		}
		r0 := pix[y*stride:]
		for k := 0; k < rows; k++ {
			src := pix[(y+k)*stride : (y+k)*stride+4*w]
			dst := pic.Y[(y+k)*pic.YStride : (y+k)*pic.YStride+w]
			if rg != nil {
				for x := range dst {
					p := src[4*x : 4*x+3 : 4*x+3]
					dst[x] = rgbToYR(int32(p[0]), int32(p[1]), int32(p[2]), rg.bits(yuvFix))
				}
				continue
			}
			for x := range dst {
				p := src[4*x : 4*x+3 : 4*x+3]
				dst[x] = rgbToY(int32(p[0]), int32(p[1]), int32(p[2]))
			}
		}
		rowsHaveAlpha := hasAlpha && !rowsOpaque(r0, stride, w, rows)
		accumulateRGBA(r0, rs, w, rowsHaveAlpha, tmp)
		cy := y >> 1
		du := pic.U[cy*pic.UVStride : cy*pic.UVStride+uvw]
		dv := pic.V[cy*pic.UVStride : cy*pic.UVStride+uvw]
		if rg != nil {
			for i := range du {
				t := tmp[4*i : 4*i+3 : 4*i+3]
				r, g, b := t[0], t[1], t[2]
				du[i] = clipUVR(-9719*r-19081*g+28800*b, rg.bits(yuvFix+2))
				dv[i] = clipUVR(28800*r-24116*g-4684*b, rg.bits(yuvFix+2))
			}
			continue
		}
		for i := range du {
			t := tmp[4*i : 4*i+3 : 4*i+3]
			du[i] = rgbToU(t[0], t[1], t[2])
			dv[i] = rgbToV(t[0], t[1], t[2])
		}
	}
}

func g2l(v byte) uint32 { return kGammaToLinearTab[v] }

func linearToGammaWeighted(src []byte, step, rs int, a0, a1, a2, a3 uint32, totalA uint32) int32 {
	sum := a0*g2l(src[0]) + a1*g2l(src[step]) + a2*g2l(src[rs]) + a3*g2l(src[rs+step])
	return linearToGamma((sum*kInvAlpha[totalA])>>(kAlphaFix-2), 0)
}

// accumulateRGBA averages 2x2 blocks (gamma-compressed, alpha-weighted if
// withAlpha) into tmp as 4x-scaled r,g,b values.
func accumulateRGBA(p []byte, rs, w int, withAlpha bool, tmp []int32) {
	j := 0
	i := 0
	for ; i < w>>1; i, j = i+1, j+8 {
		d := tmp[4*i : 4*i+3 : 4*i+3]
		if withAlpha {
			a0, a1, a2, a3 := uint32(p[j+3]), uint32(p[j+7]), uint32(p[j+rs+3]), uint32(p[j+rs+7])
			a := a0 + a1 + a2 + a3
			if a != 4*0xff && a != 0 {
				for c := 0; c < 3; c++ {
					d[c] = linearToGammaWeighted(p[j+c:], 4, rs, a0, a1, a2, a3, a)
				}
				continue
			}
		}
		for c := 0; c < 3; c++ {
			s := p[j+c:]
			d[c] = linearToGamma(g2l(s[0])+g2l(s[4])+g2l(s[rs])+g2l(s[rs+4]), 0)
		}
	}
	if w&1 != 0 {
		d := tmp[4*i : 4*i+3 : 4*i+3]
		if withAlpha {
			a0, a2 := uint32(p[j+3]), uint32(p[j+rs+3])
			a := 2 * (a0 + a2)
			if a != 4*0xff && a != 0 {
				for c := 0; c < 3; c++ {
					d[c] = linearToGammaWeighted(p[j+c:], 0, rs, a0, a0, a2, a2, a)
				}
				return
			}
		}
		for c := 0; c < 3; c++ {
			s := p[j+c:]
			d[c] = linearToGamma(g2l(s[0])+g2l(s[rs]), 1)
		}
	}
}

//------------------------------------------------------------------------------
// Transparent area cleanup (WebPCleanupTransparentArea on YUV).

// smoothenBlock replaces the luma of transparent pixels by the average of the
// non-transparent ones. Returns true if the whole block is transparent.
func smoothenBlock(a []byte, aStride int, yp []byte, yStride, w, h int) bool {
	sum, count := 0, 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if a[y*aStride+4*x] != 0 {
				count++
				sum += int(yp[y*yStride+x])
			}
		}
	}
	if count > 0 && count < w*h {
		avg := byte(sum / count)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if a[y*aStride+4*x] == 0 {
					yp[y*yStride+x] = avg
				}
			}
		}
	}
	return count == 0
}

func flatten(p []byte, v byte, stride, size int) {
	for y := 0; y < size; y++ {
		row := p[y*stride : y*stride+size]
		for i := range row {
			row[i] = v
		}
	}
}

// cleanupTransparentArea flattens fully transparent 8x8 blocks. 'a' points
// to the first alpha sample (4 bytes per pixel, aStride bytes per row).
func cleanupTransparentArea(pic *Picture, a []byte, aStride int) {
	const size, size2 = 8, 4
	width, height := pic.Width, pic.Height
	var values [3]byte
	y := 0
	for ; y+size <= height; y += size {
		needReset := true
		ap := a[y*aStride:]
		yp := pic.Y[y*pic.YStride:]
		up := pic.U[(y>>1)*pic.UVStride:]
		vp := pic.V[(y>>1)*pic.UVStride:]
		x := 0
		for ; x+size <= width; x += size {
			if smoothenBlock(ap[4*x:], aStride, yp[x:], pic.YStride, size, size) {
				if needReset {
					values[0] = yp[x]
					values[1] = up[x>>1]
					values[2] = vp[x>>1]
					needReset = false
				}
				flatten(yp[x:], values[0], pic.YStride, size)
				flatten(up[x>>1:], values[1], pic.UVStride, size2)
				flatten(vp[x>>1:], values[2], pic.UVStride, size2)
			} else {
				needReset = true
			}
		}
		if x < width {
			smoothenBlock(ap[4*x:], aStride, yp[x:], pic.YStride, width-x, size)
		}
	}
	if y < height {
		sub := height - y
		ap := a[y*aStride:]
		yp := pic.Y[y*pic.YStride:]
		x := 0
		for ; x+size <= width; x += size {
			smoothenBlock(ap[4*x:], aStride, yp[x:], pic.YStride, size, sub)
		}
		if x < width {
			smoothenBlock(ap[4*x:], aStride, yp[x:], pic.YStride, width-x, sub)
		}
	}
}

//------------------------------------------------------------------------------
// YCbCr import

// FromYCbCr builds a Picture from an *image.YCbCr. Go's YCbCr is full-range
// (JFIF) while VP8 expects BT.601 studio range, so samples are rescaled with
// the same fixed-point coefficients as libwebp's RGB->YUV matrix
// (Y' = 16 + Y*219/255, C' = 128 + (C-128)*224/255). Chroma planes with a
// subsampling other than 4:2:0 are averaged down to 4:2:0.
//
// The returned picture may be pooled: call Release when done with it.
func FromYCbCr(img *image.YCbCr) *Picture {
	return fromYCbCr(img, true)
}

// SnapshotYCbCr is FromYCbCr with storage that is never pooled: the picture
// can be kept indefinitely (and encoded several times) without a Release,
// which is then a no-op.
func SnapshotYCbCr(img *image.YCbCr) *Picture {
	return fromYCbCr(img, false)
}

func fromYCbCr(img *image.YCbCr, pooled bool) *Picture {
	b := img.Rect
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return &Picture{}
	}
	var pic *Picture
	if pooled {
		pic = newPicture(w, h)
	} else {
		pic = unpooledPicture(w, h)
	}
	var lumaLUT, chromaLUT [256]uint8
	for v := 0; v < 256; v++ {
		lumaLUT[v] = uint8((56318*v + (16 << 16) + (1 << 15)) >> 16)
		c := 128 + ((v-128)*57600+(1<<15))>>16
		chromaLUT[v] = uint8(clipInt(c, 0, 255))
	}
	for y := 0; y < h; y++ {
		src := img.Y[img.YOffset(b.Min.X, b.Min.Y+y):]
		src = src[:w]
		dst := pic.Y[y*pic.YStride : y*pic.YStride+w]
		for x, v := range src {
			dst[x] = lumaLUT[v]
		}
	}
	uvw, uvh := (w+1)>>1, (h+1)>>1
	if img.SubsampleRatio == image.YCbCrSubsampleRatio420 && b.Min.X&1 == 0 && b.Min.Y&1 == 0 {
		for y := 0; y < uvh; y++ {
			co := img.COffset(b.Min.X, b.Min.Y+2*y)
			cb := img.Cb[co : co+uvw]
			cr := img.Cr[co : co+uvw]
			du := pic.U[y*pic.UVStride : y*pic.UVStride+uvw]
			dv := pic.V[y*pic.UVStride : y*pic.UVStride+uvw]
			for x := range du {
				du[x] = chromaLUT[cb[x]]
				dv[x] = chromaLUT[cr[x]]
			}
		}
		return pic
	}
	// generic path: average the chroma of each 2x2 luma block
	for y := 0; y < uvh; y++ {
		y0 := b.Min.Y + 2*y
		y1 := min(y0+1, b.Max.Y-1)
		for x := 0; x < uvw; x++ {
			x0 := b.Min.X + 2*x
			x1 := min(x0+1, b.Max.X-1)
			o00, o01 := img.COffset(x0, y0), img.COffset(x1, y0)
			o10, o11 := img.COffset(x0, y1), img.COffset(x1, y1)
			sb := int(img.Cb[o00]) + int(img.Cb[o01]) + int(img.Cb[o10]) + int(img.Cb[o11])
			sr := int(img.Cr[o00]) + int(img.Cr[o01]) + int(img.Cr[o10]) + int(img.Cr[o11])
			pic.U[y*pic.UVStride+x] = chromaLUT[(sb+2)>>2]
			pic.V[y*pic.UVStride+x] = chromaLUT[(sr+2)>>2]
		}
	}
	return pic
}
