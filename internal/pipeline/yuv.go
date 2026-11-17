package pipeline

// Algorithm adapted from libwebp (BSD-3-Clause): src/dec/io_dec.c (EmitRescaledRGB) and
// src/dsp/yuv.h (VP8YuvToRgba, VP8Clip8).

import (
	"github.com/kolesa-team/go-webp/v2/internal/resample"
	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// YUVSink is implemented by the scaling sinks returned by NewSink. A lossy
// (VP8) decoder can hand its YUV 4:2:0 (+ alpha) output to CommitYUV instead
// of converting to RGBA and using Buffer/Commit: the planes are then rescaled
// separately and converted afterwards, which is bit-exact with libwebp's
// scaled RGBA output (EmitRescaledRGB / EmitRescaledAlphaRGB in
// src/dec/io_dec.c). To fully match libwebp the decoder must also, when
// scaling, bypass the loop filter when both scaled dimensions are below 3/4
// of the frame's (WebPIoInitFromOptions); crop rectangles should be snapped
// with SnapCropForYUV.
//
// A sink receives either RGBA rows (Buffer/Commit) or YUV rows (CommitYUV)
// for a given frame, never both. NeedRows applies to both.
type YUVSink interface {
	Sink
	// CommitYUV consumes the full-width luma rows [y, y+n) (row i at
	// yp[i*yStride:]), the chroma rows [y/2, (y+n+1)/2) (row j at
	// u[j*uvStride:] and v[j*uvStride:], (SrcW+1)/2 samples each) and the
	// alpha rows [y, y+n) (a, aStride; a == nil on the first call for frames
	// without alpha, which then yields opaque output). Batches must arrive
	// in order; a chroma row already passed with the previous batch (odd y)
	// is ignored. The buffers are not retained after the call returns.
	CommitYUV(y, n int, yp []byte, yStride int, u, v []byte, uvStride int, a []byte, aStride int) error
}

// yuvScaler holds the state of the YUV-domain rescaling path.
type yuvScaler struct {
	ry, ru, rv, ra resample.Rescaler
	init           bool
	hasAlpha       bool
	uvLeft         int // crop left, in chroma samples
	uvTop, uvH     int // crop top and height, in chroma rows
	cw, cuw        int // cropped luma / chroma widths
	yDone, uvDone  int // rows imported, relative to the crop
	rowY, rowU     []byte
	rowV, rowA     []byte
	// Rows that could not be imported during their batch (the U/V rescalers
	// may lag one row behind the Y one, or the reverse). carryY/carryA hold
	// rows yDone.., carryU/carryV rows uvDone...
	carryY, carryA [][]byte
	carryU, carryV [][]byte
}

func (s *scaleSink) initYUV(hasAlpha bool) error {
	q := &s.yuv
	q.init = true
	q.hasAlpha = hasAlpha
	cw, ch := s.w4/4, s.bottom-s.top
	dw, dh := s.dstW, s.dstH
	q.cw = cw
	q.cuw = (cw + 1) >> 1
	q.uvLeft = (s.left4 / 4) >> 1
	q.uvTop = s.top >> 1
	q.uvH = (ch + 1) >> 1
	work := make([]uint32, 4*resample.WorkSize(dw, 1))
	ws := resample.WorkSize(dw, 1)
	err := q.ry.Init(cw, ch, dw, dh, 1, work[0:ws:ws])
	if err == nil {
		err = q.ru.Init(q.cuw, q.uvH, dw, dh, 1, work[ws:2*ws:2*ws])
	}
	if err == nil {
		err = q.rv.Init(q.cuw, q.uvH, dw, dh, 1, work[2*ws:3*ws:3*ws])
	}
	if err == nil && hasAlpha {
		err = q.ra.Init(cw, ch, dw, dh, 1, work[3*ws:])
	}
	if err != nil {
		return status.InvalidParam("invalid scaled dimensions")
	}
	rows := make([]byte, 4*dw)
	q.rowY, q.rowU, q.rowV, q.rowA = rows[:dw:dw], rows[dw:2*dw:2*dw], rows[2*dw:3*dw:3*dw], rows[3*dw:]
	return nil
}

// CommitYUV implements YUVSink.
func (s *scaleSink) CommitYUV(y, n int, yp []byte, yStride int, u, v []byte, uvStride int, a []byte, aStride int) error {
	q := &s.yuv
	if !q.init {
		if err := s.initYUV(a != nil); err != nil {
			return err
		}
	}
	if s.done || n <= 0 {
		return nil
	}
	if q.hasAlpha && a == nil {
		return status.InvalidParam("missing alpha rows")
	}
	left := s.left4 / 4

	// Rows of this batch inside the crop, relative to the crop.
	yFrom, yTo := s.keptRows(y, n)
	yFrom, yTo = yFrom-s.top, yTo-s.top
	uvFirst := y >> 1 // chroma row at u[0]
	uvFrom := max(uvFirst, q.uvTop) - q.uvTop
	uvTo := min((y+n+1)>>1, q.uvTop+q.uvH) - q.uvTop
	if y+n >= s.bottom {
		uvTo = q.uvH
	}
	batchY := func(k int) []byte {
		o := (k+s.top-y)*yStride + left
		return yp[o : o+q.cw]
	}
	batchA := func(k int) []byte {
		o := (k+s.top-y)*aStride + left
		return a[o : o+q.cw]
	}
	batchU := func(k int) []byte {
		o := (k+q.uvTop-uvFirst)*uvStride + q.uvLeft
		return u[o : o+q.cuw]
	}
	batchV := func(k int) []byte {
		o := (k+q.uvTop-uvFirst)*uvStride + q.uvLeft
		return v[o : o+q.cuw]
	}

	// Carried rows come first: luma/alpha rows [startY, carryYEnd), chroma
	// rows [startUV, carryUVEnd); the batch supplies the following ones.
	startY, startUV := q.yDone, q.uvDone
	carryYEnd, carryUVEnd := startY+len(q.carryY), startUV+len(q.carryU)
	if (yTo > yFrom && yFrom > carryYEnd) || (uvTo > uvFrom && uvFrom > carryUVEnd) {
		return status.InvalidParam("YUV rows out of order")
	}
	yAvail, uvAvail := max(yTo, carryYEnd), max(uvTo, carryUVEnd)

	for progress := true; progress; {
		progress = false
		for !q.ry.HasPendingOutput() && q.yDone < yAvail {
			k := q.yDone
			if k < carryYEnd {
				q.ry.ImportRow(q.carryY[k-startY])
				if q.hasAlpha {
					q.ra.ImportRow(q.carryA[k-startY])
				}
			} else {
				q.ry.ImportRow(batchY(k))
				if q.hasAlpha {
					q.ra.ImportRow(batchA(k))
				}
			}
			q.yDone++
			progress = true
		}
		for !q.ru.HasPendingOutput() && q.uvDone < uvAvail {
			k := q.uvDone
			if k < carryUVEnd {
				q.ru.ImportRow(q.carryU[k-startUV])
				q.rv.ImportRow(q.carryV[k-startUV])
			} else {
				q.ru.ImportRow(batchU(k))
				q.rv.ImportRow(batchV(k))
			}
			q.uvDone++
			progress = true
		}
		for q.ry.HasPendingOutput() && q.ru.HasPendingOutput() {
			s.exportYUV()
			progress = true
		}
	}

	q.carryY = carry(q.carryY, q.yDone-startY, carryYEnd, yAvail, batchY)
	if q.hasAlpha {
		q.carryA = carry(q.carryA, q.yDone-startY, carryYEnd, yAvail, batchA)
	}
	q.carryU = carry(q.carryU, q.uvDone-startUV, carryUVEnd, uvAvail, batchU)
	q.carryV = carry(q.carryV, q.uvDone-startUV, carryUVEnd, uvAvail, batchV)

	if q.ry.OutputDone() {
		s.done = true
		s.release()
	}
	return nil
}

// carry returns the rows still to import: the unconsumed tail of old (its
// first consumed rows were imported, possibly followed by batch rows) and
// copies of the remaining batch rows up to to. batchFrom is the index of
// the first row that was not in old.
func carry(old [][]byte, consumed, batchFrom, to int, batch func(int) []byte) [][]byte {
	if consumed >= len(old) {
		batchFrom += consumed - len(old)
		old = old[:0]
	} else {
		old = old[consumed:]
	}
	if batchFrom >= to {
		return old
	}
	out := make([][]byte, 0, len(old)+to-batchFrom)
	out = append(out, old...)
	for k := batchFrom; k < to; k++ {
		out = append(out, append([]byte(nil), batch(k)...))
	}
	return out
}

// exportYUV exports one row from each rescaler and converts it to RGBA
// (libwebp's VP8YuvToRgba).
func (s *scaleSink) exportYUV() {
	q := &s.yuv
	dy := q.ry.DstY()
	q.ry.ExportRow(q.rowY)
	q.ru.ExportRow(q.rowU)
	q.rv.ExportRow(q.rowV)
	dst := s.out()
	h, w := dst.Rect.Dy(), dst.Rect.Dx()
	if s.flip {
		dy = h - 1 - dy
	}
	out := dst.Pix[dy*dst.Stride : dy*dst.Stride+w*4]
	var ar []byte
	if q.hasAlpha {
		q.ra.ExportRow(q.rowA)
		ar = q.rowA[:w]
	}
	yuvToRGBARow(q.rowY[:w], q.rowU[:w], q.rowV[:w], ar, out)
}

// yuvToRGBARowGo converts full resolution Y, U and V rows to RGBA
// (libwebp's VP8YuvToRgba) into out[:4*len(yr)]. ar holds the alpha
// values, or is nil for opaque output. This is the portable reference;
// yuvToRGBARow (yuv_arm64.go, yuv_generic.go) is the entry point.
func yuvToRGBARowGo(yr, ur, vr, ar, out []byte) {
	w := len(yr)
	ur, vr = ur[:w], vr[:w]
	out = out[:4*w]
	if ar != nil {
		ar = ar[:w]
	}
	for x := range yr {
		o := out[x*4 : x*4+4 : x*4+4]
		yy, uu, vv := int(yr[x]), int(ur[x]), int(vr[x])
		yh := (yy * 19077) >> 8
		o[0] = clip8YUV(yh + (vv*26149)>>8 - 14234)
		o[1] = clip8YUV(yh - (uu*6419)>>8 - (vv*13320)>>8 + 8708)
		o[2] = clip8YUV(yh + (uu*33050)>>8 - 17685)
		o[3] = 255
		if ar != nil {
			o[3] = ar[x]
		}
	}
}

// clip8YUV is libwebp's VP8Clip8 (YUV_FIX2 = 6).
func clip8YUV(v int) byte {
	const mask = (256 << 6) - 1
	if v&^mask == 0 {
		return byte(v >> 6)
	}
	if v < 0 {
		return 0
	}
	return 255
}
