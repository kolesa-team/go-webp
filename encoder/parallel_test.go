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

package encoder

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/vp8enc"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// alphaImage returns a w x h picture whose alpha plane mixes transparent,
// opaque, gradient and noisy areas.
func alphaImage(w, h int, seed int64) *image.NRGBA {
	rnd := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p := img.Pix[y*img.Stride+4*x:][:4:4]
			p[0], p[1], p[2] = uint8(x*3+y), uint8(y*5), uint8(rnd.Intn(32)+x)
			switch (x/8 + y/8) % 4 {
			case 0:
				p[3] = 0
			case 1:
				p[3] = 255
			case 2:
				p[3] = uint8(x + y)
			default:
				p[3] = uint8(rnd.Intn(256))
			}
		}
	}
	return img
}

// encodeBytes runs NewEncoder and Encode.
func encodeBytes(t testing.TB, img image.Image, o *Options) []byte {
	t.Helper()
	e, err := NewEncoder(img, o)
	require.NoError(t, err)
	defer e.Release()
	var buf bytes.Buffer
	require.NoError(t, e.Encode(&buf))
	return buf.Bytes()
}

// budgets are the helper budgets the threaded paths are tested with:
// one helper, several helpers, and none available (Spawn returns nil).
var budgets = []string{"1", "8", "exhausted"}

// withBudget runs f with the process-wide helper budget set up as mode.
func withBudget(t testing.TB, mode string, f func()) {
	t.Helper()
	switch mode {
	case "1", "8":
		n := 1
		if mode == "8" {
			n = 8
		}
		prev := workers.SetLimitForTest(n)
		defer workers.SetLimitForTest(prev)
	case "exhausted":
		prev := workers.SetLimitForTest(1)
		defer workers.SetLimitForTest(prev)
		block := make(chan struct{})
		task := workers.Spawn(func() error { <-block; return nil })
		require.NotNil(t, task)
		defer func() {
			close(block)
			require.NoError(t, task.Wait())
		}()
		require.Nil(t, workers.Spawn(func() error { return nil }), "the budget must be exhausted")
	default:
		t.Fatalf("unknown budget %q", mode)
	}
	f()
}

// onHelperGoroutine reports whether the caller runs on a helper goroutine
// started by workers.Spawn.
func onHelperGoroutine() bool {
	buf := make([]byte, 64<<10)
	n := runtime.Stack(buf, false)
	return bytes.Contains(buf[:n], []byte("internal/workers.(*Task).run"))
}

// checkNoLeak fails if the number of goroutines does not come back to its
// value at the start of the test.
func checkNoLeak(t *testing.T) {
	t.Helper()
	before := runtime.NumGoroutine()
	t.Cleanup(func() {
		deadline := time.Now().Add(5 * time.Second)
		for runtime.NumGoroutine() > before {
			if time.Now().After(deadline) {
				buf := make([]byte, 1<<20)
				t.Fatalf("goroutine leak: %d > %d\n%s", runtime.NumGoroutine(), before, buf[:runtime.Stack(buf, true)])
			}
			time.Sleep(time.Millisecond)
		}
		require.Zero(t, workers.InUse(), "helper slots still taken")
	})
}

// TestParallelAlphaIdentical checks that coding the alpha plane on a helper
// goroutine (ThreadLevel) gives the same bytes as the sequential path, for
// any helper budget, and that the helper is used exactly when expected.
func TestParallelAlphaIdentical(t *testing.T) {
	checkNoLeak(t)
	var onHelper atomic.Bool
	alphaEncode = func(img *image.NRGBA, c *Config, interrupt func() bool) ([]byte, error) {
		if onHelperGoroutine() {
			onHelper.Store(true)
		}
		return encodeAlpha(img, c, interrupt)
	}
	defer func() { alphaEncode = encodeAlpha }()

	// 255x255 is just below minParallelAlphaPixels.
	sizes := [][2]int{{257, 257}, {16383, 5}, {5, 16383}, {255, 255}}
	variants := []struct {
		name string
		mut  func(o *Options)
	}{
		{"default", func(o *Options) {}},
		{"m0", func(o *Options) { o.Method = 0 }},
		{"best filter", func(o *Options) { o.AlphaFiltering = 2 }},
		{"no filter q50", func(o *Options) { o.AlphaFiltering = 0; o.alphaQuality = 50 }},
		{"exact sharp", func(o *Options) { o.Exact = 1; o.UseSharpYuv = true }},
		{"raw alpha", func(o *Options) { o.AlphaCompression = 0 }},
		{"m6", func(o *Options) { o.Method = 6 }}, // slow alpha: first size only
	}
	if testing.Short() {
		sizes = sizes[:2]
		variants = variants[:3]
	}
	for i, size := range sizes {
		img := alphaImage(size[0], size[1], int64(size[0]))
		for _, v := range variants {
			if v.name == "m6" && i > 0 {
				continue
			}
			o, err := NewLossyEncoderOptions(PresetDefault, 75)
			require.NoError(t, err)
			v.mut(o)
			onHelper.Store(false)
			want := encodeBytes(t, img, o)
			require.False(t, onHelper.Load())
			o.ThreadLevel = true
			for _, b := range budgets {
				name := fmt.Sprintf("%dx%d/%s/budget %s", size[0], size[1], v.name, b)
				onHelper.Store(false)
				withBudget(t, b, func() {
					require.Equal(t, want, encodeBytes(t, img, o), name)
				})
				expect := b != "exhausted" && size[0]*size[1] >= minParallelAlphaPixels && o.AlphaCompression == 1
				require.Equal(t, expect, onHelper.Load(), "%s: alpha coded on a helper", name)
				require.Zero(t, workers.InUse(), name)
			}
		}
	}
}

func lossyAlphaEncoder(t *testing.T, threads bool) *Encoder {
	t.Helper()
	o, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	o.ThreadLevel = threads
	e, err := NewEncoder(alphaImage(300, 260, 1), o)
	require.NoError(t, err)
	return e
}

// stoppableAlpha is an alpha encoder hook that runs until it is told to
// stop through its interrupt hook (or 10 s pass), then returns
// errAlphaInterrupted. started is closed when it starts, returned when it
// returns.
func stoppableAlpha(started, returned chan struct{}) func(*image.NRGBA, *Config, func() bool) ([]byte, error) {
	return func(_ *image.NRGBA, _ *Config, interrupt func() bool) ([]byte, error) {
		defer close(returned)
		close(started)
		for deadline := time.Now().Add(10 * time.Second); !interrupt(); {
			if time.Now().After(deadline) {
				return nil, errors.New("the alpha helper was not stopped")
			}
			time.Sleep(time.Millisecond)
		}
		return nil, errAlphaInterrupted
	}
}

func restoreHooks() {
	alphaEncode = encodeAlpha
	vp8Encode = vp8enc.EncodeWithHeadroom
}

// TestParallelAlphaVP8Error checks that a failing VP8 frame stops the alpha
// helper, joins it and reports the VP8 error.
func TestParallelAlphaVP8Error(t *testing.T) {
	checkNoLeak(t)
	defer restoreHooks()
	errVP8 := errors.New("vp8 failure")
	started, returned := make(chan struct{}), make(chan struct{})
	alphaEncode = stoppableAlpha(started, returned)
	vp8Encode = func(*vp8enc.Picture, *vp8enc.Config, int) ([]byte, error) {
		<-started
		return nil, errVP8
	}
	e := lossyAlphaEncoder(t, true)
	defer e.Release()
	withBudget(t, "1", func() {
		var cw countWriter
		err := e.Encode(&cw)
		require.ErrorIs(t, err, errVP8)
		require.EqualError(t, err, "cannot encode webppicture")
		require.Zero(t, cw.calls)
	})
	select {
	case <-returned:
	default:
		t.Fatal("Encode returned before the alpha helper")
	}
}

// TestParallelAlphaErrorPrecedence checks that, as in the sequential order
// (alpha plane first), an alpha error takes precedence over a VP8 error,
// whatever the helper budget.
func TestParallelAlphaErrorPrecedence(t *testing.T) {
	checkNoLeak(t)
	defer restoreHooks()
	errAlpha, errVP8 := errors.New("alpha failure"), errors.New("vp8 failure")
	for _, tc := range []struct {
		alphaErr, vp8Err, want error
	}{
		{errAlpha, nil, errAlpha},
		{nil, errVP8, errVP8},
		{errAlpha, errVP8, errAlpha},
	} {
		alphaEncode = func(img *image.NRGBA, c *Config, interrupt func() bool) ([]byte, error) {
			if tc.alphaErr != nil {
				return nil, tc.alphaErr
			}
			return encodeAlpha(img, c, interrupt)
		}
		vp8Encode = func(pic *vp8enc.Picture, cfg *vp8enc.Config, headroom int) ([]byte, error) {
			if tc.vp8Err != nil {
				return nil, tc.vp8Err
			}
			return vp8enc.EncodeWithHeadroom(pic, cfg, headroom)
		}
		for _, threads := range []bool{false, true} {
			for _, b := range budgets {
				withBudget(t, b, func() {
					e := lossyAlphaEncoder(t, threads)
					defer e.Release()
					require.ErrorIs(t, e.Encode(&bytes.Buffer{}), tc.want, "threads %v budget %s", threads, b)
				})
			}
		}
	}
}

// TestParallelAlphaPanics checks that a panic of the alpha helper surfaces
// on the caller, and that a panic of the VP8 frame still joins the helper.
func TestParallelAlphaPanics(t *testing.T) {
	checkNoLeak(t)
	defer restoreHooks()

	t.Run("alpha helper", func(t *testing.T) {
		alphaEncode = func(*image.NRGBA, *Config, func() bool) ([]byte, error) { panic("alpha boom") }
		defer restoreHooks()
		withBudget(t, "1", func() {
			e := lossyAlphaEncoder(t, true)
			defer e.Release()
			v := recoverValue(func() { e.Encode(&bytes.Buffer{}) })
			pe, ok := v.(*workers.PanicError)
			require.True(t, ok, "got %T %v", v, v)
			require.Equal(t, "alpha boom", pe.Value)
		})
		withBudget(t, "exhausted", func() {
			e := lossyAlphaEncoder(t, true)
			defer e.Release()
			require.Equal(t, "alpha boom", recoverValue(func() { e.Encode(&bytes.Buffer{}) }))
		})
		require.Zero(t, workers.InUse())
	})

	t.Run("vp8 frame", func(t *testing.T) {
		started, returned := make(chan struct{}), make(chan struct{})
		alphaEncode = stoppableAlpha(started, returned)
		vp8Encode = func(*vp8enc.Picture, *vp8enc.Config, int) ([]byte, error) {
			<-started
			panic("vp8 boom")
		}
		defer restoreHooks()
		withBudget(t, "1", func() {
			e := lossyAlphaEncoder(t, true)
			defer e.Release()
			require.Equal(t, "vp8 boom", recoverValue(func() { e.Encode(&bytes.Buffer{}) }))
		})
		select {
		case <-returned:
		default:
			t.Fatal("the panic left the alpha helper running")
		}
		require.Zero(t, workers.InUse())
	})
}

func recoverValue(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}
