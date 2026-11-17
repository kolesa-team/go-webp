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
func smallBands(t *testing.T, rows, w int) {
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
func alphaFiles(t *testing.T, w, h int) map[string][]byte {
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

func decodeOutcome(t *testing.T, data []byte, o *Options) outcome {
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

func chunkData(t *testing.T, file []byte, tag string) []byte {
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
