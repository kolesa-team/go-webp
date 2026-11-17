package pipeline

import (
	"image"
	"strconv"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// maxAllocableMemory is libwebp's WEBP_MAX_ALLOCABLE_MEMORY (default
// build): 1<<34 bytes where size_t is 64-bit, (1<<31) - (1<<16) on 32-bit
// targets (below INT_MAX). On 32-bit Go targets this also keeps the output
// image below math.MaxInt bytes, which image.NewNRGBA would panic on.
const maxAllocableMemory = uint64(1<<34)*(strconv.IntSize/64) + uint64(1<<31-1<<16)*(1-strconv.IntSize/64)

// maxOutputBytes bounds the output image allocation: libwebp's
// WebPAllocateDecBuffer reports VP8_STATUS_OUT_OF_MEMORY above
// WEBP_MAX_ALLOCABLE_MEMORY. It is a variable so that tests can lower it.
var maxOutputBytes = maxAllocableMemory

// Params describes the transformations applied to a decoded frame, in
// libwebp's order: crop, then scale, then vertical flip.
type Params struct {
	// SrcW, SrcH are the dimensions of the decoded frame.
	SrcW, SrcH int
	// Crop selects the rectangle to keep (proper rectangle semantics: Min is
	// the top-left corner, Dx/Dy the size). An empty rectangle disables
	// cropping. A rectangle not fully inside the frame is an InvalidParam
	// error, like libwebp's WebPCheckCropDimensions. For lossy (VP8)
	// frames libwebp snaps the corner to even coordinates first; apply
	// SnapCropForYUV before calling NewSink to match it.
	Crop image.Rectangle
	// ScaleW, ScaleH > 0 enable libwebp's rescaler on the (cropped) frame.
	// When exactly one of them is 0 it is derived from the other keeping
	// the aspect ratio (WebPRescalerGetScaledDimensions). Negative values
	// are an InvalidParam error.
	ScaleW, ScaleH int
	// Flip mirrors the output vertically.
	Flip bool
	// Premultiply rescales alpha-premultiplied samples (then
	// un-premultiplies the result), which is what libwebp does for lossless
	// (VP8L) frames. libwebp rescales lossy frames in the YUV domain with
	// non-premultiplied alpha; leave Premultiply false for them.
	Premultiply bool
}

// SnapCropForYUV returns r with its top-left corner rounded down to even
// coordinates and the same size, as libwebp does for crops of lossy (YUV
// 4:2:0) frames ("crop_left & ~1", "crop_top & ~1").
func SnapCropForYUV(r image.Rectangle) image.Rectangle {
	if r.Empty() {
		return r
	}
	x, y := r.Min.X&^1, r.Min.Y&^1
	return image.Rect(x, y, x+r.Dx(), y+r.Dy())
}

// CropFromV1 converts the v1 decoder.Options.Crop convention (Min = left/top,
// Max = width/height, passed to libwebp as crop_left/crop_top/crop_width/
// crop_height) to a proper rectangle for Params.Crop.
func CropFromV1(r image.Rectangle) image.Rectangle {
	return image.Rectangle{Min: r.Min, Max: r.Min.Add(r.Max)}
}

// NewSink returns a sink receiving the full-width rows of a SrcW x SrcH frame
// and producing the final image, which the returned function gives back once
// all needed rows are committed.
//
// Without crop, scale or flip the sink is a Direct sink over a freshly
// allocated image (zero copy). Flip alone also decodes in place and mirrors
// the rows once the last one is committed. Crop copies only the kept part of
// the needed rows (or writes in place when the full width is kept) and
// NeedRows reports the crop bottom. Scaling keeps only O(width) state besides
// the output image.
func NewSink(p Params) (Sink, func() *image.NRGBA, error) {
	if p.SrcW <= 0 || p.SrcH <= 0 {
		return nil, nil, status.InvalidParam("invalid source dimensions")
	}
	crop := image.Rect(0, 0, p.SrcW, p.SrcH)
	if !p.Crop.Empty() {
		x, y, w, h := p.Crop.Min.X, p.Crop.Min.Y, p.Crop.Dx(), p.Crop.Dy()
		if x < 0 || y < 0 || x >= p.SrcW || w > p.SrcW-x || y >= p.SrcH || h > p.SrcH-y {
			return nil, nil, status.InvalidParam("crop rectangle outside of the frame")
		}
		crop = p.Crop
	}
	cw, ch := crop.Dx(), crop.Dy()

	if p.ScaleW < 0 || p.ScaleH < 0 {
		return nil, nil, status.InvalidParam("negative scaled dimensions")
	}
	if p.ScaleW > 0 || p.ScaleH > 0 {
		w, h, ok := resample.ScaledDimensions(cw, ch, p.ScaleW, p.ScaleH)
		if !ok {
			return nil, nil, status.InvalidParam("invalid scaled dimensions")
		}
		if err := checkImageSize(w, h); err != nil {
			return nil, nil, err
		}
		if err := checkRescalerSize(w); err != nil {
			return nil, nil, err
		}
		s := &scaleSink{
			base:    newBase(p.SrcW, crop, w, h),
			flip:    p.Flip,
			premult: p.Premultiply,
		}
		if err := s.rs.Init(cw, ch, w, h, 4, nil); err != nil {
			return nil, nil, status.InvalidParam("invalid scaled dimensions")
		}
		return s, s.image, nil
	}

	if err := checkImageSize(cw, ch); err != nil {
		return nil, nil, err
	}
	if !p.Flip && cw == p.SrcW && ch == p.SrcH {
		d := &lazyDirect{w: cw, h: ch}
		return d, d.image, nil
	}
	s := &cropSink{
		base:      newBase(p.SrcW, crop, cw, ch),
		flip:      p.Flip,
		fullWidth: cw == p.SrcW,
	}
	return s, s.image, nil
}

// checkImageSize reports an OutOfMemory error for output images libwebp
// would refuse to allocate.
func checkImageSize(w, h int) error {
	if uint64(w)*uint64(h)*4 > maxOutputBytes {
		return status.New(utils.Vp8StatusOutOfMemory, "output image too large")
	}
	return nil
}

// checkRescalerSize reports an OutOfMemory error when the rescaler state
// for an output w pixels wide exceeds maxOutputBytes: libwebp allocates it
// with WebPSafeMalloc (2 * 4 channels * w uint32 work cells, plus a row of
// w RGBA pixels), which fails above WEBP_MAX_ALLOCABLE_MEMORY. Both the RGBA
// and the Y'CbCr(A) rescalers below need that much.
func checkRescalerSize(w int) error {
	if uint64(w)*(4*uint64(resample.WorkSize(1, 4))+4) > maxOutputBytes {
		return status.New(utils.Vp8StatusOutOfMemory, "rescaler too large")
	}
	return nil
}

// lazyDirect is a Direct sink whose image is allocated on the first Buffer
// call, so that a bitstream rejected before its first row is decoded never
// allocates the (possibly huge) output frame. libwebp likewise validates the
// frame headers before WebPAllocateDecBuffer.
type lazyDirect struct {
	img  *image.NRGBA
	w, h int
}

func (d *lazyDirect) image() *image.NRGBA {
	if d.img == nil {
		d.img = image.NewNRGBA(image.Rect(0, 0, d.w, d.h))
	}
	return d.img
}

// Buffer implements Sink.
func (d *lazyDirect) Buffer(y, n int) []byte {
	img := d.image()
	s := img.Stride
	end := min((y+n)*s, len(img.Pix))
	return img.Pix[y*s : end]
}

// Stride implements Sink.
func (d *lazyDirect) Stride() int { return d.w * 4 }

// Commit implements Sink.
func (d *lazyDirect) Commit(y, n int) error { return nil }

// NeedRows implements Sink.
func (d *lazyDirect) NeedRows() int { return d.h }

// scratchPool recycles row-batch buffers between decodes.
var scratchPool sync.Pool // of *[]byte

// base holds the geometry and scratch buffer shared by the sinks.
type base struct {
	srcStride   int
	left4, w4   int // crop left and width, in bytes
	top, bottom int
	scratch     *[]byte
	done        bool
	// dst is the output image, allocated on first use (see lazyDirect).
	dst        *image.NRGBA
	dstW, dstH int
}

func newBase(srcW int, crop image.Rectangle, dstW, dstH int) base {
	return base{
		srcStride: srcW * 4,
		left4:     crop.Min.X * 4,
		w4:        crop.Dx() * 4,
		top:       crop.Min.Y,
		bottom:    crop.Max.Y,
		dstW:      dstW,
		dstH:      dstH,
	}
}

// out returns the output image, allocating it on the first call.
func (b *base) out() *image.NRGBA {
	if b.dst == nil {
		b.dst = image.NewNRGBA(image.Rect(0, 0, b.dstW, b.dstH))
	}
	return b.dst
}

// Stride implements Sink.
func (b *base) Stride() int { return b.srcStride }

// NeedRows implements Sink.
func (b *base) NeedRows() int { return b.bottom }

// scratchRows returns a scratch buffer for n full-width rows.
func (b *base) scratchRows(n int) []byte {
	need := n * b.srcStride
	if b.scratch == nil {
		if v, ok := scratchPool.Get().(*[]byte); ok {
			b.scratch = v
		} else {
			b.scratch = new([]byte)
		}
	}
	if cap(*b.scratch) < need {
		*b.scratch = make([]byte, need)
	}
	return (*b.scratch)[:need]
}

// release returns the scratch buffer to the pool.
func (b *base) release() {
	if b.scratch != nil {
		scratchPool.Put(b.scratch)
		b.scratch = nil
	}
}

// keptRows returns the source rows of the batch [y, y+n) that lie inside the
// crop.
func (b *base) keptRows(y, n int) (int, int) {
	return max(y, b.top), min(y+n, b.bottom)
}

// cropSink implements crop and/or flip without scaling.
type cropSink struct {
	base
	flip      bool
	fullWidth bool
	direct    bool // the last Buffer call returned dst memory
}

func (s *cropSink) image() *image.NRGBA { return s.out() }

// Buffer implements Sink.
func (s *cropSink) Buffer(y, n int) []byte {
	if s.fullWidth && y >= s.top && y+n <= s.bottom {
		// Rows land at their final place (flip is applied at the end).
		s.direct = true
		off := (y - s.top) * s.srcStride
		return s.out().Pix[off : off+n*s.srcStride]
	}
	s.direct = false
	return s.scratchRows(n)
}

// Commit implements Sink.
func (s *cropSink) Commit(y, n int) error {
	if !s.direct && s.scratch != nil {
		src := *s.scratch
		dst := s.out()
		ch := dst.Rect.Dy()
		from, to := s.keptRows(y, n)
		for r := from; r < to; r++ {
			dy := r - s.top
			if s.flip && !s.fullWidth {
				dy = ch - 1 - dy
			}
			o := (r - y) * s.srcStride
			copy(dst.Pix[dy*dst.Stride:dy*dst.Stride+s.w4], src[o+s.left4:o+s.left4+s.w4])
		}
	}
	if !s.done && y+n >= s.bottom {
		s.done = true
		s.release()
		if s.flip && s.fullWidth {
			flipRows(s.out())
		}
	}
	return nil
}

// flipRows mirrors img vertically in place.
func flipRows(img *image.NRGBA) {
	h, w4 := img.Rect.Dy(), img.Rect.Dx()*4
	tmp := make([]byte, w4)
	for a, b := 0, h-1; a < b; a, b = a+1, b-1 {
		ra := img.Pix[a*img.Stride : a*img.Stride+w4]
		rb := img.Pix[b*img.Stride : b*img.Stride+w4]
		copy(tmp, ra)
		copy(ra, rb)
		copy(rb, tmp)
	}
}

// scaleSink implements crop + rescale (+ flip).
type scaleSink struct {
	base
	flip    bool
	premult bool
	rs      resample.Rescaler
	yuv     yuvScaler // YUV-domain path, see YUVSink
}

func (s *scaleSink) image() *image.NRGBA { return s.out() }

// Buffer implements Sink.
func (s *scaleSink) Buffer(y, n int) []byte { return s.scratchRows(n) }

// Commit implements Sink.
func (s *scaleSink) Commit(y, n int) error {
	if s.scratch == nil || s.done {
		return nil
	}
	src := *s.scratch
	from, to := s.keptRows(y, n)
	for r := from; r < to; r++ {
		o := (r-y)*s.srcStride + s.left4
		row := src[o : o+s.w4]
		if s.premult {
			resample.PremultiplyRGBA(row)
		}
		s.rs.ImportRow(row)
		s.export()
	}
	if y+n >= s.bottom {
		s.done = true
		s.release()
	}
	return nil
}

// export drains the rescaler's pending output rows into dst.
func (s *scaleSink) export() {
	dst := s.out()
	h, w4 := dst.Rect.Dy(), dst.Rect.Dx()*4
	for s.rs.HasPendingOutput() {
		dy := s.rs.DstY()
		if s.flip {
			dy = h - 1 - dy
		}
		out := dst.Pix[dy*dst.Stride : dy*dst.Stride+w4]
		s.rs.ExportRow(out)
		if s.premult {
			resample.UnpremultiplyRGBA(out)
		}
	}
}
