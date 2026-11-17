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

// Package decoder decodes WebP images (lossy, lossless, with alpha) with a
// pure Go implementation that produces the same pixels as libwebp.
package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"image"
	"io"
	"os"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/vp8"
	"github.com/kolesa-team/go-webp/v2/internal/vp8l"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// Decoder stores information to decode picture
type Decoder struct {
	data     []byte
	options  *Options
	config   *Config
	features container.Features
}

// statusError formats like v1 ("<prefix>: VP8_STATUS_...") and unwraps to
// the underlying *status.Error, which carries the libwebp status code and a
// more detailed message.
type statusError struct {
	prefix string
	err    error
}

func (e *statusError) Error() string {
	code := status.CodeOf(e.err)
	if code == utils.Vp8StatusUserAbort {
		// Cancelled through a context: say why.
		return e.prefix + ": " + code.String() + ": " + e.err.Error()
	}
	return e.prefix + ": " + code.String()
}

func (e *statusError) Unwrap() error { return e.err }

// NewDecoder return new decoder instance
func NewDecoder(r io.Reader, options *Options) (d *Decoder, err error) {
	var data []byte

	if data, err = readAll(r); err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	if options == nil {
		options = &Options{}
	}
	d = &Decoder{data: data, options: options}

	if d.config, err = d.options.GetConfig(); err != nil {
		return nil, err
	}

	// получаем WebPBitstreamFeatures
	if d.features, err = container.GetFeatures(d.data); err != nil {
		return nil, &statusError{prefix: "cannot fetch features", err: err}
	}

	return d, nil
}

// readAll reads r to EOF. When the size is known in advance (bytes.Reader,
// bytes.Buffer, strings.Reader, regular files) the buffer is allocated once.
// Otherwise (bufio.Reader as used by image.Decode, network bodies) the size
// declared in the RIFF header is used only as a growth target: the first
// allocation is at most initialReadSize and the buffer then grows by at most
// readGrowth times the bytes actually received, so a short input with a
// lying header costs at most a few times its own size.
func readAll(r io.Reader) ([]byte, error) {
	const maxHint = 1 << 30
	n := 0
	switch v := r.(type) {
	case interface{ Len() int }:
		n = v.Len()
	case *os.File:
		if fi, err := v.Stat(); err == nil && fi.Mode().IsRegular() && fi.Size() < maxHint {
			n = int(fi.Size())
		}
	}
	if n > 0 && n < maxHint {
		return readFull(r, make([]byte, 0, n+1), n)
	}
	// Unknown size: read the RIFF header first.
	var hdr [12]byte
	m, err := io.ReadFull(r, hdr[:])
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return hdr[:m:m], nil
	}
	if err != nil {
		return nil, err
	}
	hint := 0
	if string(hdr[0:4]) == "RIFF" && string(hdr[8:12]) == "WEBP" {
		hint = int(min(int64(binary.LittleEndian.Uint32(hdr[4:]))+8, maxHint))
	}
	b := make([]byte, 0, min(max(hint, 12), initialReadSize)+1)
	b = append(b, hdr[:]...)
	return readFull(r, b, hint)
}

const (
	// initialReadSize bounds the first allocation for readers of unknown size.
	initialReadSize = 64 << 10
	// readGrowth bounds each buffer growth relative to the bytes read so far.
	readGrowth = 4
)

// readFull appends the rest of r to b. hint is the expected total length
// (0 if unknown): growth aims at hint+1 so that a truthful header needs only
// a few allocations, but never exceeds readGrowth times the data received.
// One spare byte of capacity lets the final read observe io.EOF without
// growing the buffer.
func readFull(r io.Reader, b []byte, hint int) ([]byte, error) {
	for {
		if len(b) == cap(b) {
			want := 2 * cap(b)
			if hint >= len(b) {
				want = min(hint+1, readGrowth*len(b))
			}
			nb := make([]byte, len(b), max(want, len(b)+512))
			copy(nb, b)
			b = nb
		}
		m, err := r.Read(b[len(b):cap(b)])
		b = b[:len(b)+m]
		if err == io.EOF {
			return b, nil
		}
		if err != nil {
			return b, err
		}
	}
}

// Decode picture from reader
func (d *Decoder) Decode() (image.Image, error) {
	img, err := d.decode(call{})
	if err != nil {
		return nil, &statusError{prefix: "cannot decode picture", err: err}
	}
	return img, nil
}

// call holds the per-call arguments of the Decode variants.
type call struct {
	// dst lends its pixel memory to the output image (DecodeInto); nil
	// allocates a new image.
	dst *image.NRGBA
	// ctx cancels the decode (DecodeContext); nil if it cannot be
	// cancelled.
	ctx context.Context
}

func (d *Decoder) decode(c call) (*image.NRGBA, error) {
	if err := canceled(c.ctx); err != nil {
		return nil, err
	}
	parsed, err := container.Parse(d.data)
	if err != nil {
		return nil, err
	}
	if parsed.HasAnimation {
		return nil, status.Unsupported("animated WebP is not supported")
	}
	if parsed.VP8L != nil {
		return d.decodeLossless(parsed, c)
	}
	return d.decodeLossy(parsed, c)
}

// GetFeatures return information about picture: width, height ...
func (d *Decoder) GetFeatures() utils.BitstreamFeatures {
	return utils.BitstreamFeatures{
		Width:        d.features.Width,
		Height:       d.features.Height,
		HasAlpha:     d.features.HasAlpha,
		HasAnimation: d.features.HasAnimation,
		Format:       d.features.Format,
	}
}

func (d *Decoder) decodeLossless(parsed *container.Image, cl call) (*image.NRGBA, error) {
	c := d.config
	p := pipeline.Params{
		SrcW:        parsed.Width,
		SrcH:        parsed.Height,
		Flip:        c.Flip,
		Premultiply: true,
		Dst:         cl.dst,
	}
	if c.UseCropping {
		p.Crop = image.Rect(c.CropLeft, c.CropTop, c.CropLeft+c.CropWidth, c.CropTop+c.CropHeight)
	}
	if c.UseScaling {
		p.ScaleW, p.ScaleH = c.ScaledWidth, c.ScaledHeight
	}
	sink, result, err := pipeline.NewSink(p)
	if err != nil {
		return nil, err
	}
	if err := vp8l.DecodeWithOptions(parsed.VP8L, cl.wrap(sink), &vp8l.Options{UseThreads: c.UseThreads}); err != nil {
		return nil, err
	}
	return result(), nil
}

func (d *Decoder) decodeLossy(parsed *container.Image, cl call) (*image.NRGBA, error) {
	c := d.config
	w, h := parsed.Width, parsed.Height

	crop := image.Rect(0, 0, w, h)
	if c.UseCropping {
		crop = pipeline.SnapCropForYUV(image.Rect(c.CropLeft, c.CropTop, c.CropLeft+c.CropWidth, c.CropTop+c.CropHeight))
		if crop.Empty() || crop.Min.X < 0 || crop.Min.Y < 0 || crop.Max.X > w || crop.Max.Y > h {
			return nil, status.InvalidParam("crop rectangle outside of the frame")
		}
	}
	cw, ch := crop.Dx(), crop.Dy()

	opts := vp8.Options{
		BypassFiltering:   c.BypassFiltering,
		NoFancyUpsampling: c.NoFancyUpsampling,
		DitheringStrength: c.DitheringStrength,
		UseThreads:        c.UseThreads,
	}

	var (
		sw, sh int
		sink   pipeline.Sink
		result func() *image.NRGBA
		err    error
	)
	if c.UseScaling {
		var ok bool
		if sw, sh, ok = resample.ScaledDimensions(cw, ch, c.ScaledWidth, c.ScaledHeight); !ok {
			return nil, status.InvalidParam("invalid scaled dimensions")
		}
		// libwebp (WebPIoInitFromOptions) skips the loop filter for large
		// downscaling ratios.
		if sw < w*3/4 && sh < h*3/4 {
			opts.BypassFiltering = true
		}
	}
	if sink, result, err = pipeline.NewSink(pipeline.Params{SrcW: cw, SrcH: ch, ScaleW: sw, ScaleH: sh, Flip: c.Flip, Dst: cl.dst}); err != nil {
		return nil, err
	}

	// The alpha plane is decoded incrementally, as the VP8 decoder emits
	// rows (like libwebp's VP8DecompressAlphaRows), so that a truncated or
	// broken VP8 stream never pays for a full-frame alpha decode.
	var la *lazyAlpha
	if parsed.ALPH != nil {
		la = newLazyAlpha(parsed.ALPH, w, h, crop, c)
		defer la.release()
		opts.Alpha = la.get
	}

	switch {
	case c.UseScaling:
		ys, ok := sink.(pipeline.YUVSink)
		if !ok {
			return nil, status.InvalidParam("scaling sink does not accept YUV")
		}
		ad := &yuvAdapter{sink: ys, alpha: la, top: crop.Min.Y, left: crop.Min.X, alphaStride: w, ctx: cl.ctx}
		opts.Alpha = nil
		err = vp8.DecodeYUV(parsed.VP8, &opts, &crop, ad)
	case cw == w && ch == h:
		err = vp8.Decode(parsed.VP8, nil, &opts, cl.wrap(sink))
	default:
		err = vp8.DecodeRegion(parsed.VP8, nil, &opts, crop, cl.wrap(sink))
	}
	if err != nil {
		return nil, err
	}
	return result(), nil
}

// yuvAdapter forwards the Y'CbCr rows of the VP8 decoder (and the matching
// alpha rows) to a scaling sink, which rescales the planes separately like
// libwebp.
type yuvAdapter struct {
	sink        pipeline.YUVSink
	alpha       *lazyAlpha // nil for opaque images
	top, left   int        // crop origin in the alpha plane
	alphaStride int
	ctx         context.Context // DecodeContext, may be nil
}

func (a *yuvAdapter) PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error {
	if err := canceled(a.ctx); err != nil {
		return err
	}
	var ap []byte
	if a.alpha != nil {
		plane, base, err := a.alpha.get(a.top+y, a.top+y+n)
		if err != nil {
			return err
		}
		ap = plane[(a.top+y-base)*a.alphaStride+a.left:]
	}
	return a.sink.CommitYUV(y, n, ys, yStride, us, vs, uvStride, ap, a.alphaStride)
}

func (a *yuvAdapter) NeedRows() int { return a.sink.NeedRows() }
