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

package decoder

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/encoder"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// budgets runs f with the helper budget exhausted (Spawn returns nil, the
// sequential path), with one helper slot and with plenty of them.
var budgets = []struct {
	name string
	set  func() (restore func())
}{
	{"exhausted", func() func() {
		old := workers.SetLimitForTest(1)
		taken := workers.TryAcquire()
		return func() {
			if taken {
				workers.Release()
			}
			workers.SetLimitForTest(old)
		}
	}},
	{"one", func() func() {
		old := workers.SetLimitForTest(1)
		return func() { workers.SetLimitForTest(old) }
	}},
	{"many", func() func() {
		old := workers.SetLimitForTest(8)
		return func() { workers.SetLimitForTest(old) }
	}},
}

// smallBands makes every frame use the helper, with bands of a few rows, so
// that band boundaries fall everywhere relative to the VP8 output batches.
func smallBands(t testing.TB, rows, w int) {
	oldBand, oldStep, oldMin := alphaBandPixels, alphaStepPixels, minHelperPixels
	alphaBandPixels, alphaStepPixels, minHelperPixels = rows*w, (rows+1)/2*w, 0
	t.Cleanup(func() { alphaBandPixels, alphaStepPixels, minHelperPixels = oldBand, oldStep, oldMin })
}

// alphaPicture returns a synthetic picture with a varied alpha channel.
func alphaPicture(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8((x*x + 3*y*y) / 7)
			switch (x/5 + y/3) % 4 {
			case 0:
				a = 0
			case 1:
				a = 255
			}
			img.SetNRGBA(x, y, color.NRGBA{uint8(3 * x), uint8(5 * y), uint8(x ^ y), a})
		}
	}
	return img
}

// alphaFiles encodes alphaPicture with every ALPH variant: compression
// none/lossless, each filter, and the "quantized levels" flag (which makes
// AlphaDitheringStrength smooth the whole plane at once).
func alphaFiles(t testing.TB, w, h int) map[string][]byte {
	img := alphaPicture(w, h)
	out := map[string][]byte{}
	for _, comp := range []int{0, 1} {
		for _, filt := range []int{0, 1, 2} {
			o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 70)
			require.NoError(t, err)
			o.AlphaCompression, o.AlphaFiltering = comp, filt
			buf := &bytes.Buffer{}
			require.NoError(t, encodeTo(buf, img, o))
			name := fmt.Sprintf("%dx%d comp%d filter%d", w, h, comp, filt)
			out[name] = buf.Bytes()
			out[name+" levels"] = modifyChunk(t, buf.Bytes(), "ALPH", func(p []byte) []byte { p[0] |= 0x10; return p })
		}
	}
	return out
}

type outcome struct {
	err string
	pix []byte
}

func decodeOutcome(t testing.TB, data []byte, o *Options) outcome {
	t.Helper()
	d, err := NewDecoder(bytes.NewReader(data), o)
	if err != nil {
		return outcome{err: "new: " + err.Error()}
	}
	img, err := d.Decode()
	if err != nil {
		return outcome{err: fmt.Sprintf("%v (%v)", err, status.CodeOf(err))}
	}
	return outcome{pix: img.(*image.NRGBA).Pix}
}

func helperOptions(w, h int) []*Options {
	return []*Options{
		{UseThreads: true},
		{UseThreads: true, NoFancyUpsampling: true},
		{UseThreads: true, Crop: image.Rect(2, 4, w/2, h/2)},
		{UseThreads: true, Crop: image.Rect(0, h/3, w, h/3)},
		{UseThreads: true, Scale: image.Rect(0, 0, w/2+1, h/3+1), Flip: true},
		{UseThreads: true, AlphaDitheringStrength: 80},
	}
}

// TestAlphaHelperIdentical checks that decoding the alpha plane on a helper
// gives the same pixels as the sequential decode, for every helper budget.
func TestAlphaHelperIdentical(t *testing.T) {
	const w, h = 97, 61
	smallBands(t, 3, w)
	files := alphaFiles(t, w, h)
	for name, data := range files {
		for _, o := range helperOptions(w, h) {
			seq := *o
			seq.UseThreads = false
			want := decodeOutcome(t, data, &seq)
			require.Empty(t, want.err, name)
			for _, b := range budgets {
				restore := b.set()
				got := decodeOutcome(t, data, o)
				na, err := newDec(t, data, o).DecodeNYCbCrA()
				restore()
				require.Equal(t, want, got, "%s %+v budget %s", name, o, b.name)
				if o.Scale.Empty() {
					require.NoError(t, err)
					wantNA, err := newDec(t, data, &seq).DecodeNYCbCrA()
					require.NoError(t, err)
					require.Equal(t, wantNA, na, "%s %+v budget %s", name, o, b.name)
				}
			}
		}
	}
	assert.Equal(t, 0, workers.InUse())
}

// TestAlphaHelperLogo runs the real-size path (default band size) on the
// 3000x2000 logo.
func TestAlphaHelperLogo(t *testing.T) {
	data := readFile(t, "webp-logo-lossy.webp")
	want := decodeOutcome(t, data, nil)
	for _, o := range []*Options{{UseThreads: true}, {UseThreads: true, Crop: image.Rect(0, 0, 3000, 300)}} {
		if !o.Crop.Empty() {
			want = decodeOutcome(t, data, &Options{Crop: o.Crop})
		}
		for _, b := range budgets {
			restore := b.set()
			got := decodeOutcome(t, data, o)
			restore()
			require.Equal(t, want, got, "budget %s %+v", b.name, o)
		}
	}
	assert.Equal(t, 0, workers.InUse())
}

// TestAlphaHelperStopsAtCrop checks that the helper does not decode alpha
// rows below the crop (no request goes there), and that a crop whose rows
// hold too little alpha work does not start it at all.
func TestAlphaHelperStopsAtCrop(t *testing.T) {
	const w, h = 97, 61
	data := alphaFiles(t, w, h)[fmt.Sprintf("%dx%d comp1 filter1", w, h)]
	defer func(old int) { workers.SetLimitForTest(old) }(workers.SetLimitForTest(8))
	steps := 0
	testHookHelperStep = func() { steps++ }
	defer func() { testHookHelperStep = nil }()

	smallBands(t, 3, w) // steps of 2 rows
	for _, o := range []*Options{
		{UseThreads: true, Crop: image.Rect(0, 0, w, 20)},
		{UseThreads: true, Crop: image.Rect(10, 4, 30, 16), Flip: true},
		{UseThreads: true, Crop: image.Rect(0, 0, w, 20), Scale: image.Rect(0, 0, 50, 10)},
	} {
		steps = 0
		seq := *o
		seq.UseThreads = false
		require.Equal(t, decodeOutcome(t, data, &seq), decodeOutcome(t, data, o), "%+v", o)
		bottom := o.Crop.Min.Y&^1 + o.Crop.Max.Y
		assert.Positive(t, steps, "%+v: helper not used", o)
		assert.LessOrEqual(t, steps, (bottom+1)/2, "%+v: helper decoded rows below the crop", o)
	}

	// 97 x 20 alpha pixels to decode: below the threshold.
	minHelperPixels = w*20 + 1
	steps = 0
	o := &Options{UseThreads: true, Crop: image.Rect(0, 0, w, 20)}
	require.Equal(t, decodeOutcome(t, data, &Options{Crop: o.Crop}), decodeOutcome(t, data, o))
	assert.Zero(t, steps, "helper started for a small crop")
	assert.Equal(t, 0, workers.InUse())
}

// TestAlphaHelperErrors corrupts and truncates the ALPH (and VP8) data in
// many places: the helper must report exactly the sequential errors (and
// pixels), whatever the band in which the decoder fails.
func TestAlphaHelperErrors(t *testing.T) {
	const w, h = 64, 48
	smallBands(t, 2, w)
	files := alphaFiles(t, w, h)
	type variant struct {
		name string
		data []byte
	}
	var variants []variant
	for name, data := range files {
		alph := chunkData(t, data, "ALPH")
		step := max(len(alph)/40, 1)
		for i := 1; i < len(alph); i += step {
			variants = append(variants,
				variant{fmt.Sprintf("%s xor@%d", name, i), modifyChunk(t, data, "ALPH", func(p []byte) []byte { p[i] ^= 0x5a; return p })},
				variant{fmt.Sprintf("%s cut@%d", name, i), modifyChunk(t, data, "ALPH", func(p []byte) []byte { return p[:i] })},
			)
		}
		vp8 := chunkData(t, data, "VP8 ")
		for _, n := range []int{len(vp8) / 3, len(vp8) * 2 / 3} {
			variants = append(variants, variant{fmt.Sprintf("%s vp8cut@%d", name, n), modifyChunk(t, data, "VP8 ", func(p []byte) []byte { return p[:n] })})
		}
	}
	if testing.Short() {
		variants = variants[:len(variants)/4]
	}
	failures := 0
	for _, v := range variants {
		for _, o := range []*Options{{UseThreads: true}, {UseThreads: true, Scale: image.Rect(0, 0, 20, 20)}, {UseThreads: true, Crop: image.Rect(0, 0, w, h/2)}} {
			restore := budgets[0].set()
			want := decodeOutcome(t, v.data, o)
			restore()
			if want.err != "" {
				failures++
			}
			for _, b := range budgets[1:] {
				restore := b.set()
				got := decodeOutcome(t, v.data, o)
				restore()
				require.Equal(t, want, got, "%s %+v budget %s", v.name, o, b.name)
			}
		}
	}
	t.Logf("%d variants, %d failing decodes", len(variants), failures)
	assert.Positive(t, failures)
	assert.Equal(t, 0, workers.InUse())
}

func chunkData(t testing.TB, file []byte, tag string) []byte {
	for _, c := range splitChunks(t, file) {
		if c.tag == tag {
			return c.data
		}
	}
	t.Fatalf("no %s chunk", tag)
	return nil
}

// TestAlphaHelperPanic checks that a panic on the helper is re-raised on
// the decoding goroutine, after the helper has exited, without leaking the
// helper slot or goroutines.
func TestAlphaHelperPanic(t *testing.T) {
	data := readFile(t, "webp-logo-lossy.webp")
	defer func(old int) { workers.SetLimitForTest(old) }(workers.SetLimitForTest(8))
	g := runtime.NumGoroutine()
	for _, after := range []int{0, 2} {
		n := 0
		testHookHelperStep = func() {
			if n++; n > after {
				panic("boom")
			}
		}
		var pe *workers.PanicError
		func() {
			defer func() {
				r := recover()
				var ok bool
				pe, ok = r.(*workers.PanicError)
				require.True(t, ok, "recovered %v", r)
			}()
			_, _ = newDec(t, data, &Options{UseThreads: true}).Decode()
		}()
		testHookHelperStep = nil
		require.NotNil(t, pe)
		assert.Equal(t, "boom", pe.Value)
	}
	waitGoroutines(t, g)
	// The decoder still works (no broken state was pooled).
	assert.Equal(t, decodeOutcome(t, data, nil), decodeOutcome(t, data, &Options{UseThreads: true}))
}

// FuzzAlphaHelper compares the alpha helper (tiny bands and steps) with the
// sequential alpha decoding on a valid VP8 frame carrying an arbitrary ALPH
// payload (and, with vp8cut, a truncated VP8 frame): pixels and errors must
// be identical.
func FuzzAlphaHelper(f *testing.F) {
	files := alphaFiles(f, 41, 23)
	base := files["41x23 comp1 filter1"]
	for _, data := range files {
		f.Add(chunkData(f, data, "ALPH"), uint16(0), uint8(0))
		f.Add(chunkData(f, data, "ALPH"), uint16(300), uint8(0x5a))
	}
	vp8 := chunkData(f, base, "VP8 ")
	f.Fuzz(func(t *testing.T, alph []byte, vp8cut uint16, flags uint8) {
		data := modifyChunk(t, base, "ALPH", func([]byte) []byte { return alph })
		if cut := int(vp8cut); cut > 0 && cut < len(vp8) {
			data = modifyChunk(t, data, "VP8 ", func(p []byte) []byte { return p[:cut] })
		}
		d, err := NewDecoder(bytes.NewReader(data), nil)
		if err != nil {
			return
		}
		feat := d.GetFeatures()
		smallBands(t, 1+int(flags&3), feat.Width)
		o := &Options{UseThreads: true, NoFancyUpsampling: flags&4 != 0, AlphaDitheringStrength: int(flags>>3&1) * 60}
		switch flags >> 5 & 3 {
		case 1:
			o.Crop = image.Rect(0, feat.Height/3, feat.Width, feat.Height/2+1)
		case 2:
			o.Scale = image.Rect(0, 0, feat.Width/2+1, feat.Height/3+1)
		}
		restore := budgets[0].set()
		want := decodeOutcome(t, data, o)
		restore()
		restore = budgets[2].set()
		got := decodeOutcome(t, data, o)
		restore()
		if want.err != got.err || !bytes.Equal(want.pix, got.pix) {
			t.Fatalf("helper %q, sequential %q (pixels equal: %v)", got.err, want.err, bytes.Equal(want.pix, got.pix))
		}
	})
}

// BenchmarkAlphaHelper compares lossy+alpha decoding with UseThreads with
// and without the alpha helper (budget exhausted: the alpha plane is
// decoded by the VP8 output stage).
func BenchmarkAlphaHelper(b *testing.B) {
	data := readFile(b, "webp-logo-lossy.webp")
	for _, bud := range []int{0, 2} {
		b.Run(budgets[bud].name, func(b *testing.B) {
			restore := budgets[bud].set()
			defer restore()
			b.ReportAllocs()
			var dst image.NRGBA
			for i := 0; i < b.N; i++ {
				d, err := NewDecoder(bytes.NewReader(data), &Options{UseThreads: true})
				if err != nil {
					b.Fatal(err)
				}
				if _, err := d.DecodeInto(&dst); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
