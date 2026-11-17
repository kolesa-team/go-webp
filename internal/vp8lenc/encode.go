package vp8lenc

// Algorithm adapted from libwebp (BSD-3-Clause): src/enc/vp8l_enc.c (entropy analysis, crunch
// configurations, stream layout).

import (
	"errors"
	"image"
	"math"
	"sync"
)

// Config controls the lossless encoder. Field semantics follow libwebp's
// WebPConfig for lossless encoding.
type Config struct {
	// Quality in 0..100 is the effort within a method: higher is slower and
	// produces smaller output.
	Quality float32
	// Method in 0..6 selects the speed/size trade-off (0 fastest, 6 best).
	Method int
	// Exact preserves the RGB values under fully transparent pixels. When
	// false, such pixels are replaced by transparent black for better
	// compression.
	Exact bool
	// NearLossless in 0..100 enables near-lossless preprocessing (100 = off).
	// It is ignored for images that can be palettized.
	NearLossless int
	// UseDeltaPalette is accepted for compatibility with libwebp and ignored
	// (it is an experimental libwebp option).
	UseDeltaPalette bool
	// LowMemory reduces peak memory (smaller LZ77 window, fewer candidate
	// encodings kept in memory) at some cost in size.
	LowMemory bool
	// ThreadLevel allows evaluating candidate encodings concurrently.
	ThreadLevel bool
	// ImageHint is libwebp's WebPImageHint (0 default, 1 picture, 2 photo,
	// 3 graph). With 3 (graph), palette images additionally try the
	// palette+spatial combination.
	ImageHint int
	// Interrupt, if not nil, is polled between the major passes of the
	// encode; when it returns true, the encode stops and returns
	// ErrInterrupted. It must be safe for concurrent use and, once it has
	// returned true, keep doing so.
	Interrupt func() bool
}

// ErrInterrupted is returned when Config.Interrupt (or
// AlphaEncoder.Interrupt) asked the encoder to stop.
var ErrInterrupted = errors.New("vp8lenc: interrupted")

var (
	errNilImage   = errors.New("vp8lenc: nil image")
	errDimensions = errors.New("vp8lenc: image dimensions must be in 1..16383")
	errPlane      = errors.New("vp8lenc: alpha plane too small for the given dimensions")
	errPixels     = errors.New("vp8lenc: image pixel buffer too small for its bounds")
)

// entropy modes (libwebp EntropyIx).
const (
	modeDirect = iota
	modeSpatial
	modeSubGreen
	modeSpatialSubGreen
	modePalette
	modePaletteAndSpatial
	numEntropyModes
)

const (
	sortDefault = iota // ascending
	sortMinimizeDelta
)

type crunchConfig struct {
	entropy     int
	paletteSort int
}

// params are the derived encoding parameters shared by all crunch configs.
type params struct {
	quality      int
	method       int
	lowEffort    bool
	histoBits    int
	transBits    int
	useCache     bool
	tryNoCache   bool
	tryBox       bool // also try the box LZ77 parse (at most 16 colors)
	rbZero       bool // red and blue always zero (skip cross color)
	windowCap    int
	nearLossless int
	// cleanTransparent enables libwebp's exact=0 predictor behaviour: the
	// RGB residual of a fully transparent pixel is zeroed and the pixel
	// replaced by its prediction (only set when such pixels exist).
	cleanTransparent bool
	parallel         bool
	paletteSorts     [2][]uint32
	interrupt        func() bool // Config.Interrupt
}

// interrupted polls Config.Interrupt. After it returned true the output of
// the current encode is garbage: the caller reports ErrInterrupted instead.
func (p *params) interrupted() bool { return p.interrupt != nil && p.interrupt() }

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func normalize(cfg *Config) (quality, method int) {
	q := float32(75)
	m := 4
	if cfg != nil {
		q = cfg.Quality
		m = cfg.Method
	}
	if q != q { // NaN
		q = 75
	}
	quality = clampInt(int(q), 0, 100)
	method = clampInt(m, 0, 6)
	return
}

// Encode encodes img as a VP8L chunk payload: the 0x2f signature, the 14-bit
// width-1 and height-1, the alpha_is_used bit, a 3-bit version (0) and the
// lossless bitstream. The image bounds may have a non-zero origin.
func Encode(img *image.NRGBA, cfg *Config) ([]byte, error) {
	return EncodeWithHeadroom(img, cfg, 0)
}

// EncodeWithHeadroom is Encode, but the returned buffer starts with
// headroom unspecified bytes followed by the VP8L chunk payload, and has at
// least one byte of spare capacity, so that a container header (and a
// padding byte) can be added without copying the payload.
func EncodeWithHeadroom(img *image.NRGBA, cfg *Config, headroom int) ([]byte, error) {
	if headroom < 0 {
		headroom = 0
	}
	if img == nil {
		return nil, errNilImage
	}
	b := img.Rect
	w, h := b.Dx(), b.Dy()
	if w < 1 || h < 1 || w > maxDim || h > maxDim {
		return nil, errDimensions
	}
	if img.Stride < 4*w || len(img.Pix) < (h-1)*img.Stride+4*w {
		return nil, errPixels
	}
	if cfg == nil {
		cfg = &Config{Quality: 75, Method: 4, NearLossless: 100}
	}
	n := w * h
	argb := getU32(n)
	defer putU32(argb)
	hasAlpha := false
	for y := 0; y < h; y++ {
		src := img.Pix[y*img.Stride : y*img.Stride+4*w]
		dst := argb[y*w : (y+1)*w]
		for x := range dst {
			s := src[4*x : 4*x+4 : 4*x+4]
			a := uint32(s[3])
			if a != 0xff {
				hasAlpha = true
			}
			if a == 0 && !cfg.Exact {
				dst[x] = 0
				continue
			}
			dst[x] = a<<24 | uint32(s[0])<<16 | uint32(s[1])<<8 | uint32(s[2])
		}
	}
	out, _ := encodeARGB(argb, w, h, cfg, true, headroom+5, nil, nil)
	if cfg.Interrupt != nil && cfg.Interrupt() {
		return nil, ErrInterrupted
	}
	hdr := out[headroom : headroom+5]
	hdr[0] = vp8lSignature
	v := uint32(w-1) | uint32(h-1)<<14
	if hasAlpha {
		v |= 1 << 28
	}
	hdr[1] = byte(v)
	hdr[2] = byte(v >> 8)
	hdr[3] = byte(v >> 16)
	hdr[4] = byte(v >> 24)
	return out, nil
}

// EncodeAlpha encodes an alpha plane as an ALPH compression-method-1 stream:
// a headerless VP8L image stream whose green channel carries the alpha values.
// method (0..6) and quality (0..100) map like libwebp's alpha encoder: the
// lossless quality is 8*method, or 100 when method is 6 and quality is 100.
// As in libwebp 1.6 (whose VP8LEncodeStream no longer has a use_cache
// argument), the color cache may be used.
func EncodeAlpha(plane []byte, width, height, stride int, method int, quality float32) ([]byte, error) {
	var a AlphaEncoder
	defer a.Release()
	return a.Encode(plane, width, height, stride, method, quality)
}

// AlphaEncoder encodes several alpha planes (typically the candidate
// filterings of one image) with one set of working buffers, so that only
// the first plane pays for allocating them. The zero value is ready to use;
// Release must be called when done. An AlphaEncoder is not safe for
// concurrent use.
type AlphaEncoder struct {
	sc    *scratch
	argb  []uint32
	spare []byte // output buffer for the next encodeStream
	// Interrupt is Config.Interrupt for Encode.
	Interrupt func() bool
}

// Encode is EncodeAlpha using the encoder's buffers.
func (a *AlphaEncoder) Encode(plane []byte, width, height, stride int, method int, quality float32) ([]byte, error) {
	if width < 1 || height < 1 || width > maxDim || height > maxDim {
		return nil, errDimensions
	}
	if stride < width || len(plane) < (height-1)*stride+width {
		return nil, errPlane
	}
	method = clampInt(method, 0, 6)
	q := float32(8 * method)
	if method == 6 && quality >= 100 {
		q = 100
	}
	n := width * height
	if cap(a.argb) < n {
		putU32(a.argb)
		a.argb = getU32(n)
	}
	argb := a.argb[:n]
	// Like libwebp's WebPDispatchAlphaToGreen, the alpha, red and blue
	// channels are left to 0 (the alpha channel is not transmitted).
	for y := 0; y < height; y++ {
		row := plane[y*stride : y*stride+width]
		dst := argb[y*width : (y+1)*width]
		for x, a := range row {
			dst[x] = uint32(a) << 8
		}
	}
	if a.sc == nil {
		a.sc = getScratch()
	}
	cfg := &Config{Quality: q, Method: method, Exact: true, NearLossless: 100, Interrupt: a.Interrupt}
	out, spare := encodeARGB(argb, width, height, cfg, true, 0, a.sc, a.spare)
	a.spare = spare
	if cfg.Interrupt != nil && cfg.Interrupt() {
		a.Recycle(out)
		return nil, ErrInterrupted
	}
	return out, nil
}

// Recycle hands a stream returned by Encode back to the encoder once the
// caller is done with it (e.g. after copying it, or when another filter
// candidate won), so that the next Encode writes into it instead of
// allocating a new output buffer.
func (a *AlphaEncoder) Recycle(b []byte) {
	if cap(b) > cap(a.spare) {
		a.spare = b[:0]
	}
}

// Release returns the encoder's buffers to the pools (large ones are left
// to the garbage collector). The encoder may be reused afterwards.
func (a *AlphaEncoder) Release() {
	if a.sc != nil {
		putScratch(a.sc)
		a.sc = nil
	}
	if a.argb != nil {
		putU32(a.argb)
		a.argb = nil
	}
	a.spare = nil
}

// clampBits clamps histogram and transform bits (libwebp ClampBits): bits
// is raised until the sub-image has at most imageSizeMax pixels, then
// lowered to the smallest value still giving a 1-pixel sub-image.
func clampBits(width, height, bits, minBits, maxBits, imageSizeMax int) int {
	bits = clampInt(bits, minBits, maxBits)
	size := subSampleSize(width, bits) * subSampleSize(height, bits)
	for bits < maxBits && size > imageSizeMax {
		bits++
		size = subSampleSize(width, bits) * subSampleSize(height, bits)
	}
	for bits > minBits && size == 1 {
		size = subSampleSize(width, bits-1) * subSampleSize(height, bits-1)
		if size != 1 {
			break
		}
		bits--
	}
	return bits
}

func getHistoBits(method int, usePalette bool, width, height int) int {
	hb := 7 - method
	if usePalette {
		hb = 9 - method
	}
	return clampBits(width, height, hb, minHuffmanBits, maxHuffmanBits, maxHuffImageSize)
}

func getTransformBits(method, histoBits int) int {
	m := 5
	if method < 4 {
		m = 6
	} else if method > 4 {
		m = 4
	}
	if histoBits > m {
		return m
	}
	return histoBits
}

// encodeARGB encodes argb (which may be modified: the caller must not use
// it afterwards) and
// returns a freshly allocated slice holding prefix zero bytes followed by the
// image stream. sc, if not nil, is the scratch to use (owned by the caller);
// otherwise one is taken from the pool.
//
// spare, if not nil, is an output buffer that may be reused (see
// encodeStream). Besides the smallest stream, encodeARGB returns a buffer of
// a losing candidate (or spare itself if unused), which the caller may pass
// as spare to a later call.
func encodeARGB(argb []uint32, w, h int, cfg *Config, useCache bool, prefix int, sc *scratch, spare []byte) (best, spareOut []byte) {
	quality, method := normalize(cfg)
	p := params{
		quality:   quality,
		method:    method,
		lowEffort: method == 0,
		useCache:  useCache,
		parallel:  cfg.ThreadLevel && !cfg.LowMemory,
		interrupt: cfg.Interrupt,
	}
	if !cfg.Exact {
		for _, c := range argb[:w*h] {
			if c < 0x01000000 {
				p.cleanTransparent = true
				break
			}
		}
	}
	if cfg.LowMemory {
		p.windowCap = 1 << 16
	}
	n := w * h
	var palBuf [maxPaletteSize]uint32
	palette := collectPalette(argb[:n], palBuf[:0])
	usePalette := palette != nil
	p.nearLossless = clampInt(cfg.NearLossless, 0, 100)
	p.histoBits = getHistoBits(method, usePalette, w, h)
	p.transBits = getTransformBits(method, p.histoBits)

	var configs []crunchConfig
	if p.lowEffort {
		if usePalette {
			configs = append(configs, crunchConfig{modePalette, sortDefault})
		} else {
			configs = append(configs, crunchConfig{modeSpatialSubGreen, sortDefault})
		}
		p.rbZero = true
	} else {
		best, rbZero := analyzeEntropy(argb[:n], w, h, usePalette, len(palette), p.transBits)
		p.rbZero = rbZero
		if method == 6 && quality == 100 {
			p.tryNoCache = true
			for m := 0; m < numEntropyModes; m++ {
				if m == modePalette || m == modePaletteAndSpatial {
					if usePalette {
						configs = append(configs, crunchConfig{m, sortDefault}, crunchConfig{m, sortMinimizeDelta})
					}
					continue
				}
				configs = append(configs, crunchConfig{m, sortDefault})
			}
			// As in libwebp 1.6, rbZero (from the analysis of the guessed
			// best mode) also applies to the brute-force configurations.
		} else {
			configs = append(configs, crunchConfig{best, sortMinimizeDelta})
			if quality >= 75 && method == 5 {
				p.tryNoCache = true
				if best == modePalette {
					configs = append(configs, crunchConfig{modePaletteAndSpatial, sortMinimizeDelta})
				}
			} else if cfg.ImageHint == 3 && best == modePalette && method >= 4 {
				configs = append(configs, crunchConfig{modePaletteAndSpatial, sortMinimizeDelta})
			}
		}
	}
	if cfg.LowMemory {
		p.tryNoCache = false
	}
	// Like libwebp, images with few colors also try the box LZ77 parse
	// (which needs a second match table: skipped with LowMemory).
	p.tryBox = !p.lowEffort && !cfg.LowMemory && usePalette && len(palette) <= 16
	if usePalette {
		p.paletteSorts[sortDefault] = palette
		p.paletteSorts[sortMinimizeDelta] = paletteSortMinimizeDeltas(palette, make([]uint32, 0, len(palette)))
	}

	if sc == nil {
		sc = getScratch()
		defer putScratch(sc)
	}
	if len(configs) == 1 {
		// argb is private to the caller and not needed afterwards: let the
		// transforms work in place (libwebp also transforms the picture's
		// argb in place).
		return encodeStream(argb[:n], true, w, h, &p, configs[0], prefix, sc, spare), nil
	}
	results := make([][]byte, len(configs))
	if p.parallel {
		var wg sync.WaitGroup
		for i := range configs {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				s := sc
				if i > 0 {
					s = getScratch()
					defer putScratch(s)
				}
				var buf []byte
				if i == 0 {
					buf = spare
				}
				results[i] = encodeStream(argb[:n], false, w, h, &p, configs[i], prefix, s, buf)
			}(i)
		}
		wg.Wait()
	} else {
		// Keep only the best stream and one spare buffer: each trial
		// writes into the buffer of the previous loser.
		for i := range configs {
			r := encodeStream(argb[:n], false, w, h, &p, configs[i], prefix, sc, spare)
			if best == nil || len(r) < len(best) {
				best, spare = r, best
			} else {
				spare = r
			}
		}
		return best, spare
	}
	bi := 0
	for i, r := range results {
		if len(r) < len(results[bi]) {
			bi = i
		}
	}
	for i, r := range results {
		if i != bi && cap(r) > cap(spareOut) {
			spareOut = r
		}
	}
	return results[bi], spareOut
}

// hashPix approximates palette entropy with a multiplicative hash.
func hashPix(pix uint32) uint32 {
	return uint32(((uint64(pix) + uint64(pix>>19)) * 0x39c5fba7) & 0xffffffff >> 24)
}

const (
	hAlpha = iota
	hRed
	hGreen
	hBlue
	hAlphaPred
	hRedPred
	hGreenPred
	hBluePred
	hRedSubGreen
	hBlueSubGreen
	hRedPredSubGreen
	hBluePredSubGreen
	hPalette
	hTotal
)

// analyzeEntropy guesses the best entropy mode (libwebp AnalyzeEntropy).
func analyzeEntropy(argb []uint32, w, h int, usePalette bool, paletteSize, transBits int) (int, bool) {
	if usePalette && paletteSize <= 16 {
		return modePalette, true
	}
	histo := getU32(hTotal * 256)
	defer putU32(histo)
	for i := range histo {
		histo[i] = 0
	}
	var hh [hTotal][]uint32
	for i := range hh {
		hh[i] = histo[i*256 : (i+1)*256 : (i+1)*256]
	}
	pixPrev := argb[0]
	for y := 0; y < h; y++ {
		cur := argb[y*w : (y+1)*w]
		var prevRow []uint32
		if y > 0 {
			prevRow = argb[(y-1)*w : y*w]
		}
		for x, pix := range cur {
			diff := subPixels(pix, pixPrev)
			pixPrev = pix
			if diff == 0 || (prevRow != nil && pix == prevRow[x]) {
				continue
			}
			hh[hAlpha][pix>>24]++
			hh[hRed][(pix>>16)&0xff]++
			hh[hGreen][(pix>>8)&0xff]++
			hh[hBlue][pix&0xff]++
			hh[hAlphaPred][diff>>24]++
			hh[hRedPred][(diff>>16)&0xff]++
			hh[hGreenPred][(diff>>8)&0xff]++
			hh[hBluePred][diff&0xff]++
			g := pix >> 8
			hh[hRedSubGreen][((pix>>16)-g)&0xff]++
			hh[hBlueSubGreen][(pix-g)&0xff]++
			dg := diff >> 8
			hh[hRedPredSubGreen][((diff>>16)-dg)&0xff]++
			hh[hBluePredSubGreen][(diff-dg)&0xff]++
			hh[hPalette][hashPix(pix)]++
		}
	}
	hh[hRedPredSubGreen][0]++
	hh[hBluePredSubGreen][0]++
	hh[hRedPred][0]++
	hh[hGreenPred][0]++
	hh[hBluePred][0]++
	hh[hAlphaPred][0]++
	var ec [hTotal]float64
	for i := range ec {
		ec[i] = bitsEntropy(hh[i])
	}
	var e [modePalette + 1]float64
	e[modeDirect] = ec[hAlpha] + ec[hRed] + ec[hGreen] + ec[hBlue]
	e[modeSpatial] = ec[hAlphaPred] + ec[hRedPred] + ec[hGreenPred] + ec[hBluePred]
	e[modeSubGreen] = ec[hAlpha] + ec[hRedSubGreen] + ec[hGreen] + ec[hBlueSubGreen]
	e[modeSpatialSubGreen] = ec[hAlphaPred] + ec[hRedPredSubGreen] + ec[hGreenPred] + ec[hBluePredSubGreen]
	e[modePalette] = ec[hPalette]
	tiles := float64(subSampleSize(w, transBits) * subSampleSize(h, transBits))
	e[modeSpatial] += tiles * math.Log2(14)
	e[modeSpatialSubGreen] += tiles * math.Log2(24)
	e[modePalette] += float64(paletteSize * 8)
	last := modeSpatialSubGreen
	if usePalette {
		last = modePalette
	}
	best := modeDirect
	for k := modeDirect + 1; k <= last; k++ {
		if e[best] > e[k] {
			best = k
		}
	}
	pairs := [5][2]int{{hRed, hBlue}, {hRedPred, hBluePred}, {hRedSubGreen, hBlueSubGreen},
		{hRedPredSubGreen, hBluePredSubGreen}, {hRed, hBlue}}
	rb := true
	r, bl := hh[pairs[best][0]], hh[pairs[best][1]]
	for i := 1; i < 256; i++ {
		if r[i]|bl[i] != 0 {
			rb = false
			break
		}
	}
	return best, rb
}

// encodeStream encodes src with one crunch configuration and returns a
// byte slice with prefix zero bytes followed by the image stream (with one
// spare byte of capacity for a RIFF padding byte). The stream is written
// into buf if its capacity is at least the initial estimate, else into a
// new buffer. src is modified only if mutable is set.
func encodeStream(src []uint32, mutable bool, w, h int, p *params, cc crunchConfig, prefix int, sc *scratch, buf []byte) []byte {
	sc.noteSize(w * h)
	sc.interrupt = p.interrupt
	// The output is written straight into the returned buffer (no final
	// copy), presized from a rough estimate of 1.5 bytes per pixel.
	if need := prefix + w*h*3/2 + 4096; cap(buf) >= need {
		buf = buf[:prefix]
		clear(buf)
	} else {
		buf = make([]byte, prefix, need)
	}
	if p.interrupted() {
		return buf
	}
	bw := &bitWriter{buf: buf}
	usePalette := cc.entropy == modePalette || cc.entropy == modePaletteAndSpatial
	useSubGreen := cc.entropy == modeSubGreen || cc.entropy == modeSpatialSubGreen
	usePredict := cc.entropy == modeSpatial || cc.entropy == modeSpatialSubGreen || cc.entropy == modePaletteAndSpatial
	useCrossColor := !p.lowEffort && !usePalette && usePredict && !p.rbZero
	quality := p.quality
	width := w
	var cur []uint32
	cacheBitsMax := 0
	if p.useCache {
		cacheBitsMax = maxColorCacheBits
	}
	if usePalette {
		pal := p.paletteSorts[cc.paletteSort]
		xbits := paletteXBits(len(pal))
		width = subSampleSize(w, xbits)
		cur = sc.curBuf(width * h)
		applyPalette(src, w, h, pal, cur, sc.idx(w))
		// Write the palette transform.
		bw.putBits(1, 1)
		bw.putBits(colorIndexingTransform, 2)
		bw.putBits(uint32(len(pal)-1), 8)
		delta := make([]uint32, len(pal))
		delta[0] = pal[0]
		for i := 1; i < len(pal); i++ {
			delta[i] = subPixels(pal[i], pal[i-1])
		}
		encodeImageNoHuffman(bw, delta, len(pal), 1, 20, p.lowEffort, sc.child())
		if p.useCache && len(pal) < 1<<maxColorCacheBits {
			cacheBitsMax = log2Floor(uint32(len(pal))) + 1
		}
	} else if mutable {
		cur = src
	} else {
		cur = sc.curBuf(len(src))
		copy(cur, src)
	}

	if !usePalette && !usePredict && p.nearLossless < 100 {
		applyNearLossless(width, h, p.nearLossless, cur)
	}
	if useSubGreen {
		for i, px := range cur {
			g := (px >> 8) & 0xff
			cur[i] = px&0xff00ff00 | (((px >> 16) - g) & 0xff << 16) | ((px - g) & 0xff)
		}
		bw.putBits(1, 1)
		bw.putBits(subtractGreenTransform, 2)
	}
	if usePredict {
		// As in libwebp 1.6 (ApplyPredictFilter), methods 5 and 6 also
		// search smaller predictor tiles.
		maxBits := clampBits(width, h, p.transBits, minTransformBits, maxTransformBits, maxPredictorImageSize)
		minBits := clampBits(width, h, maxBits-2*max(p.method-4, 0), minTransformBits, maxTransformBits, maxPredictorImageSize)
		modes := make([]uint32, subSampleSize(width, minBits)*subSampleSize(h, minBits))
		maxQuant := 1
		if !usePalette && p.nearLossless < 100 {
			maxQuant = 1 << uint(nearLosslessBits(p.nearLossless))
		}
		bits := predictorImage(width, h, minBits, maxBits, p.lowEffort, maxQuant, useSubGreen, p.cleanTransparent, cur, modes, sc)
		tw, th := subSampleSize(width, bits), subSampleSize(h, bits)
		bw.putBits(1, 1)
		bw.putBits(predictorTransform, 2)
		bw.putBits(uint32(bits-minTransformBits), 3)
		encodeImageNoHuffman(bw, modes[:tw*th], tw, th, quality, p.lowEffort, sc.child())
		if p.interrupted() {
			return bw.buf
		}
	}
	if useCrossColor {
		bits := p.transBits
		img := make([]uint32, subSampleSize(width, bits)*subSampleSize(h, bits))
		bits = colorSpaceTransform(width, h, bits, quality, cur, img, sc)
		tw, th := subSampleSize(width, bits), subSampleSize(h, bits)
		bw.putBits(1, 1)
		bw.putBits(crossColorTransform, 2)
		bw.putBits(uint32(bits-minTransformBits), 3)
		encodeImageNoHuffman(bw, img[:tw*th], tw, th, quality, p.lowEffort, sc.child())
		if p.interrupted() {
			return bw.buf
		}
	}
	bw.putBits(0, 1) // no more transforms
	encodeImageInternal(bw, cur, width, h, p, cacheBitsMax, sc)
	out := bw.finish()
	if len(out) == cap(out) {
		out = append(out, 0)[:len(out)] // room for a RIFF padding byte
	}
	return out
}

// refsCandidate is one backward reference parse with its cache size.
type refsCandidate struct {
	refs      []pixOrCopy
	cacheBits int
}

// histoCost estimates the size of a parse. With raw set, distances are
// counted as linear distances like libwebp does when it compares candidate
// parses (see addRefsRawDist); this decides, as in libwebp, whether the
// cost-based parse is tried. Otherwise they are counted as the plane codes
// actually written, a better estimate used when no cost-based parse
// follows.
func histoCost(refs []pixOrCopy, xsize, cacheBits int, raw bool, sc *scratch) float64 {
	h := sc.tmpHisto(cacheBits)
	if raw {
		sc.tmpSet.addRefsRawDist(h, refs)
	} else {
		sc.tmpSet.addRefs(h, refs, xsize)
	}
	return sc.tmpSet.estimate(h)
}

// Parse types (libwebp VP8LLZ77Type).
const (
	lz77Standard = 1
	lz77RLE      = 2
	lz77Box      = 4
)

// getBackwardRefs computes the best references for argb, like libwebp 1.6's
// VP8LGetBackwardReferences (except for lowEffort, see below), from the
// matches of sc.hc (which the caller fills). With box, only the "box" LZ77
// parse (matches restricted to short 2D distances) is tried, otherwise the
// standard LZ77 and RLE ones. It returns the best parse with its color
// cache size and, when tryNoCache is set and that cache size is not 0, the
// best parse without color cache (searched separately).
func getBackwardRefs(argb []uint32, w, h int, quality int, lowEffort, box bool, cacheBitsMax int,
	tryNoCache bool, sc *scratch, cands []refsCandidate) []refsCandidate {
	n := w * h
	refsCap := initRefsCap(argb, w, h)
	types := []int{lz77Standard, lz77RLE}
	if box {
		types = []int{lz77Box}
		sc.hcBox.fillBox(argb, w, n, &sc.hc, sc)
	}
	// Unlike libwebp, the low effort mode also tries the RLE parse and a
	// color cache (but not the cost-based parse): the low-effort hash chain
	// does not find runs, and this keeps method 0 output much smaller than
	// libwebp's for a small cost.
	cacheBuf := sc.cacheBuf(1 << maxColorCacheBits)
	trace := quality >= 25 && !lowEffort
	// sc.refs slots: best[0] with cache, best[1] without (tryNoCache), tmp.
	best := [2]int{0, 1}
	tmp := 2
	bestCost := [2]float64{math.Inf(1), math.Inf(1)}
	bestType := [2]int{0, 0}
	bestCB := 0
	for _, typ := range types {
		var r []pixOrCopy
		switch typ {
		case lz77Standard:
			r = backwardRefsLz77(n, argb, &sc.hc, sc.refsBuf(tmp, refsCap))
		case lz77RLE:
			r = backwardRefsRle(w, n, argb, sc.refsBuf(tmp, refsCap))
		default:
			r = backwardRefsLz77(n, argb, &sc.hcBox, sc.refsBuf(tmp, refsCap))
		}
		sc.refs[tmp] = r
		var noCacheCost float64
		if tryNoCache {
			noCacheCost = histoCost(r, w, 0, trace, sc)
			if noCacheCost < bestCost[1] {
				sc.refs[best[1]] = append(sc.refsBuf(best[1], len(r)), r...)
				bestCost[1] = noCacheCost
				bestType[1] = typ
			}
		}
		cb := bestCacheBits(argb, quality, r, cacheBitsMax, sc)
		var cost float64
		if tryNoCache && cb == 0 {
			cost = noCacheCost
		} else {
			applyColorCache(argb, cb, r, cacheBuf)
			cost = histoCost(r, w, cb, trace, sc)
		}
		if cost < bestCost[0] {
			best[0], tmp = tmp, best[0]
			bestCost[0] = cost
			bestType[0] = typ
			bestCB = cb
		}
	}
	// Improve on the simple LZ77 with the cost-based parse (only for high
	// quality, as it is costly).
	for i := 1; i >= 0; i-- {
		if i == 1 && !tryNoCache {
			continue
		}
		cb := 0
		if i == 0 {
			cb = bestCB
		}
		if (bestType[i] == lz77Standard || bestType[i] == lz77Box) && trace {
			hc := &sc.hc
			if bestType[i] == lz77Box {
				hc = &sc.hcBox
			}
			r := traceBackwards(w, n, argb, cb, hc, sc.refs[best[i]], sc.refsBuf(tmp, 0), sc)
			applyColorCache(argb, cb, r, cacheBuf)
			sc.refs[tmp] = r
			if c := histoCost(r, w, cb, true, sc); c < bestCost[i] {
				best[i], tmp = tmp, best[i]
			}
		}
		if i == 1 && bestType[0] == bestType[1] && bestCB == 0 {
			// Same parse type without cache: use the no-cache result.
			sc.refs[best[0]] = append(sc.refsBuf(best[0], len(sc.refs[best[1]])), sc.refs[best[1]]...)
			break
		}
	}
	cands = append(cands[:0], refsCandidate{sc.refs[best[0]], bestCB})
	if tryNoCache && bestCB > 0 {
		cands = append(cands, refsCandidate{sc.refs[best[1]], 0})
	}
	return cands
}

// encodeImageNoHuffman encodes a sub-image (transform data, entropy image or
// palette) without color cache and meta Huffman codes. As in libwebp,
// lowEffort only affects the match search (hash chain).
func encodeImageNoHuffman(bw *bitWriter, argb []uint32, w, h int, quality int, lowEffort bool, sc *scratch) {
	n := w * h
	sc.hc.fill(argb, w, h, quality, lowEffort, 0)
	refsCap := initRefsCap(argb, w, h)
	r0 := backwardRefsLz77(n, argb, &sc.hc, sc.refsBuf(0, refsCap))
	sc.refs[0] = r0
	r1 := backwardRefsRle(w, n, argb, sc.refsBuf(1, refsCap))
	sc.refs[1] = r1
	best := r0
	trace := quality >= 25
	bestCost := histoCost(r0, w, 0, trace, sc)
	if c := histoCost(r1, w, 0, trace, sc); c < bestCost {
		best, bestCost = r1, c
	} else if trace {
		r2 := traceBackwards(w, n, argb, 0, &sc.hc, r0, sc.refsBuf(2, 0), sc)
		sc.refs[2] = r2
		if c := histoCost(r2, w, 0, true, sc); c < bestCost {
			best = r2
		}
	}
	sc.out.init(1, 0)
	sc.out.addRefs(&sc.out.h[0], best, w)
	bw.putBits(0, 1) // no color cache
	codes := buildCodes(&sc.out, sc)
	for i := range codes {
		sc.huff.storeHuffmanCode(bw, &codes[i])
		clearIfOneSymbol(&codes[i])
	}
	writeImageData(bw, w, best, 0, nil, codes)
}

// buildCodes creates the 5 Huffman codes of every histogram in s.
func buildCodes(s *histoSet, sc *scratch) []huffCode {
	ng := len(s.h)
	per := s.litLen + 3*256 + numDistanceCodes
	total := ng * per
	if cap(sc.lens) < total {
		sc.lens = make([]uint8, total)
		sc.codes = make([]uint16, total)
	}
	lens := sc.lens[:total]
	codes := sc.codes[:total]
	if cap(sc.hcodes) < 5*ng {
		sc.hcodes = make([]huffCode, 5*ng)
	}
	hc := sc.hcodes[:5*ng]
	sizes := [5]int{s.litLen, 256, 256, 256, numDistanceCodes}
	for g := 0; g < ng; g++ {
		d := s.h[g].d
		off := g * per
		hoff := 0
		for k := 0; k < 5; k++ {
			sz := sizes[k]
			c := &hc[5*g+k]
			c.lengths = lens[off : off+sz]
			c.codes = codes[off : off+sz]
			sc.huff.createHuffmanTree(d[hoff:hoff+sz], 15, c)
			off += sz
			hoff += sz
		}
	}
	return hc
}

// encodeImageInternal encodes the main (transformed) image with color cache
// and meta Huffman codes, choosing the smallest of the candidate parses.
// Like libwebp, images with at most 16 colors also try the "box" LZ77
// parse.
func encodeImageInternal(bw *bitWriter, argb []uint32, w, h int, p *params, cacheBitsMax int, sc *scratch) {
	sc.hc.fill(argb, w, h, p.quality, p.lowEffort, p.windowCap)
	if p.interrupted() {
		return
	}
	var candBuf [2]refsCandidate
	cands := getBackwardRefs(argb, w, h, p.quality, p.lowEffort, false, cacheBitsMax, p.tryNoCache, sc, candBuf[:0])
	if p.interrupted() {
		return
	}
	if len(cands) == 1 && !p.tryBox {
		writeMainImage(bw, w, h, p, cands[0], sc)
		return
	}
	// Write each candidate into the spare writer, keeping the smallest.
	bestIdx := -1
	try := func(c refsCandidate) {
		i := 0
		if bestIdx == 0 {
			i = 1
		}
		t := &sc.bwTmp[i]
		t.reset()
		writeMainImage(t, w, h, p, c, sc)
		if bestIdx < 0 || t.numBits() < sc.bwTmp[bestIdx].numBits() {
			bestIdx = i
		}
	}
	for _, c := range cands {
		try(c)
	}
	if p.tryBox && !p.interrupted() {
		cands = getBackwardRefs(argb, w, h, p.quality, p.lowEffort, true, cacheBitsMax, p.tryNoCache, sc, candBuf[:0])
		for _, c := range cands {
			try(c)
		}
	}
	bw.appendWriter(&sc.bwTmp[bestIdx])
}

func writeMainImage(bw *bitWriter, w, h int, p *params, c refsCandidate, sc *scratch) {
	if c.cacheBits > 0 {
		bw.putBits(1, 1)
		bw.putBits(uint32(c.cacheBits), 4)
	} else {
		bw.putBits(0, 1)
	}
	hb := p.histoBits
	sc.clus.interrupt = p.interrupt
	ng := sc.clus.buildHistoImage(w, h, c.refs, p.quality, p.lowEffort, hb, c.cacheBits, &sc.out)
	sc.clus.interrupt = nil
	if p.interrupted() {
		return
	}
	var symbols []uint16
	if ng > 1 {
		// Reduce the sampling of the entropy image if it is made of uniform
		// squares (libwebp 1.6 VP8LOptimizeSampling).
		syms := sc.clus.symbols
		hb = optimizeSampling(syms, w, h, hb, maxHuffmanBits)
		syms = syms[:subSampleSize(w, hb)*subSampleSize(h, hb)]
		bw.putBits(1, 1)
		bw.putBits(uint32(hb-2), 3)
		img := make([]uint32, len(syms))
		for i, s := range syms {
			img[i] = uint32(s) << 8
		}
		encodeImageNoHuffman(bw, img, subSampleSize(w, hb), subSampleSize(h, hb), p.quality, p.lowEffort, sc.child())
		symbols = syms
	} else {
		bw.putBits(0, 1)
	}
	codes := buildCodes(&sc.out, sc)
	for i := range codes {
		sc.huff.storeHuffmanCode(bw, &codes[i])
		clearIfOneSymbol(&codes[i])
	}
	if ng > 1 {
		writeImageData(bw, w, c.refs, hb, symbols, codes)
	} else {
		writeImageData(bw, w, c.refs, 0, nil, codes)
	}
}

// writeImageData writes the entropy-coded pixels.
func writeImageData(bw *bitWriter, w int, refs []pixOrCopy, histoBits int, symbols []uint16, codes []huffCode) {
	x, y := 0, 0
	tileMask := 0
	hx := 0
	if symbols != nil {
		tileMask = ^((1 << uint(histoBits)) - 1)
		hx = subSampleSize(w, histoBits)
	}
	tileX, tileY := -1, -1
	g := codes[0:5]
	if symbols == nil {
		tileX, tileY = 0, 0
	}
	for _, r := range refs {
		if symbols != nil && (x&tileMask != tileX || y&tileMask != tileY) {
			tileX = x & tileMask
			tileY = y & tileMask
			ix := int(symbols[(y>>uint(histoBits))*hx+(x>>uint(histoBits))])
			g = codes[5*ix : 5*ix+5]
		}
		l := 1
		switch r.mode() {
		case modeLiteral:
			a := r.val
			gr := (a >> 8) & 0xff
			rd := (a >> 16) & 0xff
			bl := a & 0xff
			al := a >> 24
			c0, c1, c2, c3 := &g[0], &g[1], &g[2], &g[3]
			n0, n1 := uint(c0.lengths[gr]), uint(c1.lengths[rd])
			bw.putBits(uint32(c0.codes[gr])|uint32(c1.codes[rd])<<n0, n0+n1)
			n2, n3 := uint(c2.lengths[bl]), uint(c3.lengths[al])
			bw.putBits(uint32(c2.codes[bl])|uint32(c3.codes[al])<<n2, n2+n3)
		case modeCache:
			s := numLiteralCodes + numLengthCodes + r.val
			bw.putBits(uint32(g[0].codes[s]), uint(g[0].lengths[s]))
		default:
			l = r.length()
			code, nb, extra := prefixEncode(l)
			s := numLiteralCodes + code
			bw.putBits(uint32(g[0].codes[s]), uint(g[0].lengths[s]))
			bw.putBits(uint32(extra), uint(nb))
			dcode, dnb, dextra := prefixEncode(distanceToPlaneCode(w, int(r.val)))
			bw.putBits(uint32(g[4].codes[dcode]), uint(g[4].lengths[dcode]))
			bw.putBits(uint32(dextra), uint(dnb))
		}
		x += l
		for x >= w {
			x -= w
			y++
		}
	}
}
