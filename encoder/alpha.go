package encoder

// Alpha plane coding for lossy images (the ALPH chunk).
// Algorithm adapted from libwebp (BSD-3-Clause): src/enc/alpha_enc.c.

import (
	"errors"
	"image"
	"sync"
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/vp8enc"
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

const (
	alphaFilterFast = 1 // libwebp's WEBP_FILTER_FAST alpha_filtering value

	// GetFilterMap thresholds (libwebp).
	minColorsForFilterNone = 16
	maxColorsForFilterNone = 192

	// minParallelAlphaPixels is the smallest picture whose (compressed)
	// alpha plane is worth a helper goroutine: about a millisecond of work
	// even at method 0.
	minParallelAlphaPixels = 1 << 16
)

// errAlphaInterrupted is returned by encodeAlpha when its interrupt hook
// asked it to stop between two candidate filters.
var errAlphaInterrupted = errors.New("alpha encoding interrupted")

// The encoders, as variables so that tests can inject failures.
var (
	vp8Encode   = vp8enc.Encode
	alphaEncode = encodeAlpha
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
// alpha channel, following libwebp's EncodeAlpha. interrupt, if not nil, is
// polled between the passes; the encode stops when it returns true.
func encodeAlpha(img *image.NRGBA, c *Config, interrupt func() bool) ([]byte, error) {
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
	lenc := vp8lenc.AlphaEncoder{Interrupt: interrupt}
	defer lenc.Release()
	var best []byte
	for f := alpha.FilterNone; tryMap != 0; f, tryMap = f+1, tryMap>>1 {
		if tryMap&1 == 0 {
			continue
		}
		if interrupt != nil && interrupt() {
			return nil, errAlphaInterrupted
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

// alphaTask codes the alpha plane on a helper goroutine (see startAlpha).
type alphaTask struct {
	task *workers.Task
	// stop asks the helper to give up (the VP8 frame failed, or the call
	// is returning).
	stop atomic.Bool
	alph []byte // the result, written by the helper
}

// startAlpha starts coding img's alpha plane on a helper goroutine, like
// libwebp's alpha worker. It returns nil, and the caller must call
// encodeAlpha itself, when that is not worth it (raw or small alpha planes)
// or when the helper budget is exhausted. Otherwise the caller must call
// cancel (deferred) and, to get the result, wait.
func startAlpha(img *image.NRGBA, c *Config, interrupt func() bool) *alphaTask {
	// Raw alpha is a copy and a filter: not worth a goroutine. Invalid
	// methods are reported by the sequential path, which checks them first.
	if c.AlphaCompression != alpha.CompressionLossless || img.Rect.Dx()*img.Rect.Dy() < minParallelAlphaPixels {
		return nil
	}
	t := &alphaTask{}
	stop := func() bool { return t.stop.Load() || interrupt != nil && interrupt() }
	t.task = workers.Spawn(func() error {
		var err error
		t.alph, err = alphaEncode(img, c, stop)
		return err
	})
	if t.task == nil {
		return nil
	}
	return t
}

// wait joins the helper and returns its result, re-raising its panic on the
// calling goroutine. If abandon is set (the result is not needed), the
// helper is asked to stop first.
func (t *alphaTask) wait(abandon bool) ([]byte, error) {
	if abandon {
		t.stop.Store(true)
	}
	err := t.task.Wait()
	return t.alph, err
}

// cancel asks the helper to stop and waits until it has returned, without
// re-raising its panic: deferred by the caller so that no helper outlives
// the call, even when the caller panics.
func (t *alphaTask) cancel() {
	t.stop.Store(true)
	<-t.task.Done()
}
