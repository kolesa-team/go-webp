package vp8enc

import (
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// Algorithm adapted from libwebp (BSD-3-Clause).

const errorDiffusionQuality = 98 // quality below which error-diffusion is enabled

type encoder struct {
	cfg Config
	pic *Picture

	mbW, mbH int
	predsW   int
	numParts int

	method          int
	rdOptLevel      int
	maxI4HeaderBits int
	mbHeaderLimit   int64
	doSearch        bool
	profile         int

	// filter header
	filterSimple    bool
	filterLevel     int
	filterSharpness int

	// segment header
	numSegments int
	updateMap   bool
	segHdrSize  int

	dqm       [numMBSegments]segmentInfo
	baseQuant int
	alpha     int // global susceptibility
	uvAlpha   int // U/V quantization susceptibility
	dqY1DC    int
	dqY2DC    int
	dqY2AC    int
	dqUVDC    int
	dqUVAC    int

	proba encProba

	mbInfo  []mbInfo
	preds   []uint8  // (4*mbW+1)*(4*mbH+1) intra modes, origin at predsW+1
	nz      []uint32 // mbW+1 non-zero contexts, nz[0] is the constant left border
	yTop    []byte   // mbW*16 top luma samples
	uvTop   []byte   // mbW*16 top chroma samples (8 U + 8 V per MB)
	topDerr [][2][2]int8

	lfStats    *[numMBSegments][64]float64
	lfStatsMem [numMBSegments][64]float64

	tokens     tokenBuffer
	mbTokStart []int32 // start of each macroblock's tokens

	parts [8]bitWriter
	bw0   bitWriter
	flat  [numProbaSlots]uint8

	it iterator
}

var encoderPool = sync.Pool{New: func() any { return new(encoder) }}

func resize[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// Encode encodes pic and returns the VP8 chunk payload (frame header,
// partition #0 and token partitions). The returned slice is owned by the
// caller.
func Encode(pic *Picture, cfg *Config) ([]byte, error) {
	if pic == nil {
		return nil, status.InvalidParam("vp8enc: nil picture")
	}
	c := DefaultConfig()
	if cfg != nil {
		c = *cfg
	}
	if c.Pass <= 1 {
		// Like libwebp's WebPEncode (and unlike the cwebp tool, which
		// raises the pass count), a size/PSNR target with a single pass
		// does not search: the quality is used as is.
		c.Pass = 1
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if err := pic.validate(); err != nil {
		return nil, err
	}
	e := encoderPool.Get().(*encoder)
	defer func() {
		e.pic = nil
		e.tokens.release()
		if pic.Width*pic.Height <= maxPooledPixels {
			encoderPool.Put(e)
		}
	}()
	e.setup(pic, &c)
	e.analyze()
	if err := e.tokenLoop(); err != nil {
		return nil, err
	}
	return e.write()
}

func (e *encoder) setup(pic *Picture, cfg *Config) {
	e.cfg = *cfg
	e.pic = pic
	e.mbW = (pic.Width + 15) >> 4
	e.mbH = (pic.Height + 15) >> 4
	e.predsW = 4*e.mbW + 1
	predsH := 4*e.mbH + 1
	e.numParts = 1 << uint(cfg.Partitions)
	e.mbInfo = resize(e.mbInfo, e.mbW*e.mbH)
	e.preds = resize(e.preds, e.predsW*predsH)
	e.nz = resize(e.nz, e.mbW+1)
	e.yTop = resize(e.yTop, e.mbW*16)
	e.uvTop = resize(e.uvTop, e.mbW*16)
	if cfg.Quality <= errorDiffusionQuality || cfg.Pass > 1 {
		e.topDerr = resize(e.topDerr, e.mbW)
	} else {
		e.topDerr = nil
	}
	e.lfStats = nil
	if cfg.Autofilter {
		e.lfStats = &e.lfStatsMem
	}
	useFilter := cfg.FilterStrength > 0 || cfg.Autofilter
	switch {
	case !useFilter:
		e.profile = 2
	case cfg.FilterType == 1:
		e.profile = 0
	default:
		e.profile = 1
	}

	// MapConfigToTools
	e.method = cfg.Method
	switch {
	case e.method >= 6:
		e.rdOptLevel = rdOptTrellisAll
	case e.method >= 5:
		e.rdOptLevel = rdOptTrellis
	case e.method >= 3:
		e.rdOptLevel = rdOptBasic
	default:
		e.rdOptLevel = rdOptNone
	}
	// libwebp 1.6 codes methods >= 3 from its token buffer unless
	// low_memory is set, and the token path supports a single partition
	// only (MapConfigToTools: "doesn't work with multi-partition"): the
	// Partitions setting is then ignored. Mirror that, so that the number
	// of partitions in the output is the one cwebp writes.
	if e.method >= 3 && !cfg.LowMemory {
		e.numParts = 1
	}
	limit := 100 - cfg.PartitionLimit
	e.maxI4HeaderBits = 256 * 16 * 16 * (limit * limit) / (100 * 100)
	e.mbHeaderLimit = int64(256*510*8*1024) / int64(e.mbW*e.mbH)
	e.doSearch = cfg.TargetSize > 0 || cfg.TargetPSNR > 0

	e.proba.reset()
	e.numSegments = cfg.Segments
	e.updateMap = e.numSegments > 1
	e.segHdrSize = 0
	e.filterSimple = true
	e.filterLevel = 0
	e.filterSharpness = 0
	e.dqm = [numMBSegments]segmentInfo{}
	e.alpha, e.uvAlpha = 0, 0

	e.tokens.reset()
	e.mbTokStart = resize(e.mbTokStart, e.mbW*e.mbH+1)
}
