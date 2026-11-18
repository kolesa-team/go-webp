package decoder

import (
	"bytes"
	"fmt"
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// setRing sets the ring of sequential alpha decodes to rows rows (0: the
// whole plane, as libwebp keeps it) for the rest of the test.
func setRing(t testing.TB, rows int) {
	oldBytes, oldRows := alphaRingBytes, minAlphaRingRows
	if rows == 0 {
		alphaRingBytes = 1 << 30
	} else {
		alphaRingBytes, minAlphaRingRows = 0, rows
	}
	t.Cleanup(func() { alphaRingBytes, minAlphaRingRows = oldBytes, oldRows })
}

// ringOptions are the decode options of the ring tests for a w x h picture:
// fancy and point upsampling, crops at the top, in the middle and at the
// bottom (a few rows, so that the first request reaches the last row),
// scaling, flipping and alpha dithering.
func ringOptions(w, h int) []*Options {
	return []*Options{
		{},
		{NoFancyUpsampling: true},
		{Crop: cropRect(3, 0, w-7, 41)},
		{Crop: cropRect(1, h/3+1, w/2, h/3)},
		{Crop: cropRect(0, h-9, w, 9)},
		{Crop: cropRect(0, h-9, w, 9), NoFancyUpsampling: true},
		{Crop: cropRect(5, h-70, w-10, 70), Flip: true},
		{Scale: image.Rect(0, 0, w/2+1, h/3+1)},
		{Crop: cropRect(2, 31, w-4, h/2), Scale: image.Rect(0, 0, w/3, h/5)},
		{Crop: cropRect(2, h-12, w-4, 12), Scale: image.Rect(0, 0, w/3, 5)},
		{AlphaDitheringStrength: 80},
		{BypassFiltering: true, DitheringStrength: 50},
	}
}

// cropRect returns Options.Crop for the cw x ch rectangle at (x, y)
// (libwebp's convention: Max holds the size).
func cropRect(x, y, cw, ch int) image.Rectangle {
	return image.Rectangle{Min: image.Pt(x, y), Max: image.Pt(cw, ch)}
}

// TestAlphaRing checks that decoding the alpha plane through rings of
// various sizes (requests that wrap around the ring, requests for the
// last row that the ring cannot hold) gives the pixels of the whole-plane
// decode, sequentially and with the alpha helper, for Decode and
// DecodeNYCbCrA.
func TestAlphaRing(t *testing.T) {
	const w, h = 61, 300
	files := alphaFiles(t, w, h)
	rings := []int{26, 27, 33, 64}
	if testing.Short() {
		rings = []int{27, 64}
	}
	for name, data := range files {
		for oi, o := range ringOptions(w, h) {
			setRing(t, 0)
			want := decodeOutcome(t, data, o)
			require.Empty(t, want.err, name)
			wantNA, naErr := newDec(t, data, o).DecodeNYCbCrA()
			for _, rows := range rings {
				for _, threads := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/opts%d/ring%d/threads%v", name, oi, rows, threads), func(t *testing.T) {
						setRing(t, rows)
						smallBands(t, 3, w)
						ot := *o
						ot.UseThreads = threads
						defer func(old int) { workers.SetLimitForTest(old) }(workers.SetLimitForTest(8))
						got := decodeOutcome(t, data, &ot)
						require.Equal(t, want, got)
						na, err := newDec(t, data, &ot).DecodeNYCbCrA()
						if naErr != nil {
							require.Error(t, err)
							return
						}
						require.NoError(t, err)
						require.Equal(t, wantNA, na)
					})
				}
			}
		}
	}
	assert.Equal(t, 0, workers.InUse())
}

// TestAlphaRingErrors corrupts and truncates the ALPH data in many places
// (including its last bytes, where libwebp's end-of-stream rule for the
// last row applies) and checks that decoding through rings gives the
// errors and pixels of the whole-plane decode.
func TestAlphaRingErrors(t *testing.T) {
	const w, h = 40, 200
	files := alphaFiles(t, w, h)
	type variant struct {
		name string
		data []byte
	}
	var variants []variant
	for name, data := range files {
		alph := chunkData(t, data, "ALPH")
		step := max(len(alph)/30, 1)
		for i := 1; i < len(alph); i += step {
			variants = append(variants,
				variant{fmt.Sprintf("%s xor@%d", name, i), modifyChunk(t, data, "ALPH", func(p []byte) []byte { p[i] ^= 0x5a; return p })},
				variant{fmt.Sprintf("%s cut@%d", name, i), modifyChunk(t, data, "ALPH", func(p []byte) []byte { return p[:i] })},
			)
		}
		for i := max(len(alph)-6, 1); i < len(alph); i++ {
			variants = append(variants, variant{fmt.Sprintf("%s cut@%d", name, i), modifyChunk(t, data, "ALPH", func(p []byte) []byte { return p[:i] })})
		}
	}
	if testing.Short() {
		variants = variants[:len(variants)/4]
	}
	opts := []*Options{{}, {Crop: cropRect(0, 0, w, h/2)}, {Crop: cropRect(0, h-5, w, 5)}, {Crop: cropRect(0, h-40, w, 40)}, {Scale: image.Rect(0, 0, 20, 20)}}
	failures := 0
	for _, v := range variants {
		for _, o := range opts {
			setRing(t, 0)
			want := decodeOutcome(t, v.data, o)
			if want.err != "" {
				failures++
			}
			for _, rows := range []int{26, 31} {
				for _, threads := range []bool{false, true} {
					setRing(t, rows)
					smallBands(t, 2, w)
					ot := *o
					ot.UseThreads = threads
					restore := budgets[2].set()
					got := decodeOutcome(t, v.data, &ot)
					restore()
					require.Equal(t, want, got, "%s %+v ring %d threads %v", v.name, o, rows, threads)
				}
			}
		}
	}
	t.Logf("%d variants, %d failing decodes", len(variants), failures)
	assert.Positive(t, failures)
	assert.Equal(t, 0, workers.InUse())
}

// TestAlphaRingSize checks the buffers of a large lossy+alpha decode: a
// ring of about alphaRingBytes sequentially, and of the helper's
// look-ahead with it, not the whole plane; alpha dithering keeps the plane.
func TestAlphaRingSize(t *testing.T) {
	const w, h = 3000, 2000
	frame := image.Rect(0, 0, w, h)
	alph := []byte{0x05} // lossless, horizontal filter; the payload is not read
	a := &lazyAlpha{alph: alph, w: w, h: h, crop: frame}
	a.hdr.Filter = 1
	r := a.ringRows(false)
	assert.Equal(t, (alphaRingBytes+w-1)/w, r)
	rh := a.ringRows(true)
	assert.Less(t, rh*w, w*h/2, "helper ring of %d rows", rh)
	assert.Greater(t, rh, r)
	a.alloc(false)
	assert.Equal(t, (r+alphaTailRows)*w, len(a.buf))
	assert.Zero(t, (h+a.off)%a.ring, "the last row must end the ring")
	a.release()

	a = &lazyAlpha{alph: alph, w: w, h: h, crop: frame, dithering: 50}
	a.hdr.PreProcessing = 1
	a.alloc(false)
	assert.Equal(t, w*h, len(a.buf))
	a.release()
}

// FuzzAlphaRing compares decoding an arbitrary ALPH payload (on a valid VP8
// frame, possibly truncated) through a small ring, sequentially and with
// the alpha helper, with the whole-plane decode: pixels and errors must be
// identical.
func FuzzAlphaRing(f *testing.F) {
	const w, h = 31, 140
	files := alphaFiles(f, w, h)
	base := files[fmt.Sprintf("%dx%d comp1 filter1", w, h)]
	for _, data := range files {
		f.Add(chunkData(f, data, "ALPH"), uint16(0), uint8(0))
		f.Add(chunkData(f, data, "ALPH"), uint16(500), uint8(0x5a))
	}
	vp8 := chunkData(f, base, "VP8 ")
	f.Fuzz(func(t *testing.T, alph []byte, vp8cut uint16, flags uint8) {
		data := modifyChunk(t, base, "ALPH", func([]byte) []byte { return alph })
		if cut := int(vp8cut); cut > 0 && cut < len(vp8) {
			data = modifyChunk(t, data, "VP8 ", func(p []byte) []byte { return p[:cut] })
		}
		o := &Options{NoFancyUpsampling: flags&4 != 0, AlphaDitheringStrength: int(flags>>3&1) * 60}
		switch flags >> 5 & 3 {
		case 1:
			o.Crop = cropRect(0, h/3, w, h/2+1)
		case 2:
			o.Scale = image.Rect(0, 0, w/2+1, h/3+1)
		case 3:
			o.Crop = cropRect(1, h-int(flags&3)*5-3, w-1, int(flags&3)*5+3)
		}
		setRing(t, 0)
		want := decodeOutcome(t, data, o)
		setRing(t, 26+int(flags&3)*3)
		smallBands(t, 1+int(flags&3), w)
		got := decodeOutcome(t, data, o)
		o.UseThreads = true
		restore := budgets[2].set()
		gotHelper := decodeOutcome(t, data, o)
		restore()
		for _, g := range []outcome{got, gotHelper} {
			if want.err != g.err || !bytes.Equal(want.pix, g.pix) {
				t.Fatalf("ring %q, plane %q (pixels equal: %v)", g.err, want.err, bytes.Equal(want.pix, g.pix))
			}
		}
	})
}
