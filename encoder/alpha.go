package encoder

// Alpha plane coding for lossy images (the ALPH chunk).
// Algorithm adapted from libwebp (BSD-3-Clause): src/enc/alpha_enc.c.

import (
	"errors"
	"image"
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
)

const (
	alphaFilterFast = 1 // libwebp's WEBP_FILTER_FAST alpha_filtering value

	// GetFilterMap thresholds (libwebp).
	minColorsForFilterNone = 16
	maxColorsForFilterNone = 192
)

var planePool sync.Pool // of *[]byte

func getPlane(n int) *[]byte {
	if p, ok := planePool.Get().(*[]byte); ok && cap(*p) >= n {
		*p = (*p)[:n]
		return p
	}
	b := make([]byte, n)
	return &b
}

// encodeAlpha returns the ALPH chunk payload (header byte included) of img's
// alpha channel, following libwebp's EncodeAlpha.
func encodeAlpha(img *image.NRGBA, c *Config) ([]byte, error) {
	method := c.AlphaCompression
	if method > alpha.CompressionLossless {
		return nil, errors.New("invalid alpha compression method")
	}
	quality := c.AlphaQuality
	effort := c.Method
	filter := c.AlphaFiltering
	if method == alpha.CompressionNone {
		filter = alpha.FilterNone
	}

	w, h := img.Rect.Dx(), img.Rect.Dy()
	pp := getPlane(w * h)
	defer planePool.Put(pp)
	plane := *pp
	for y := 0; y < h; y++ {
		row := img.Pix[img.PixOffset(img.Rect.Min.X, img.Rect.Min.Y+y):][:4*w]
		dst := plane[y*w : y*w+w]
		for x := range dst {
			dst[x] = row[4*x+3]
		}
	}

	reduceLevels := quality < 100
	if reduceLevels {
		if err := alpha.QuantizeLevels(plane, w, h, w, alpha.AlphaLevels(quality)); err != nil {
			return nil, err
		}
	}

	// Candidate filters (libwebp's GetFilterMap).
	var tryMap uint
	switch filter {
	case alpha.FilterNone:
		tryMap = 1 << alpha.FilterNone
	case alphaFilterFast:
		n := numColors(plane)
		f := alpha.FilterNone
		if n > minColorsForFilterNone {
			f = alpha.EstimateBestFilter(plane, w, h, w)
		}
		tryMap = 1 << uint(f)
		if effort > 3 || n > maxColorsForFilterNone {
			tryMap |= 1 << alpha.FilterNone
		}
	default: // best: try all
		tryMap = 1<<alpha.FilterNone | 1<<alpha.FilterHorizontal | 1<<alpha.FilterVertical | 1<<alpha.FilterGradient
	}

	var filtered []byte
	if tryMap != 1<<alpha.FilterNone {
		fp := getPlane(w * h)
		defer planePool.Put(fp)
		filtered = *fp
	}

	pre := 0
	if reduceLevels {
		pre = alpha.PreprocessingLevels
	}
	// One lossless encoder for all candidate filters: its working buffers
	// (hundreds of MB for large images) are allocated once per image.
	var lenc vp8lenc.AlphaEncoder
	defer lenc.Release()
	var best []byte
	for f := alpha.FilterNone; tryMap != 0; f, tryMap = f+1, tryMap>>1 {
		if tryMap&1 == 0 {
			continue
		}
		src := plane
		if f != alpha.FilterNone {
			alpha.Filter(f, plane, w, h, w, filtered, w)
			src = filtered
		}
		m := method
		data := src
		var enc []byte
		if m != alpha.CompressionNone {
			var err error
			if enc, err = lenc.Encode(src, w, h, w, effort, float32(quality)); err != nil {
				return nil, err
			}
			if len(enc) > len(src) {
				// Compressed is larger than the source: store raw, as libwebp.
				m = alpha.CompressionNone
			} else {
				data = enc
			}
		}
		if best == nil || 1+len(data) < len(best) {
			best = make([]byte, 1+len(data))
			copy(best[1:], data)
			best[0] = alpha.Header{Compression: m, Filter: f, PreProcessing: pre}.Byte()
		}
		// The stream has been copied: its buffer can hold the next one.
		lenc.Recycle(enc)
	}
	return best, nil
}

// numColors counts the distinct values of an alpha plane.
func numColors(plane []byte) int {
	var seen [256]bool
	n := 0
	for _, v := range plane {
		if !seen[v] {
			seen[v] = true
			n++
		}
	}
	return n
}
