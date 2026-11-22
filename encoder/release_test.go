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
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
)

// sourceImages returns a picture of the given size as each source type
// NewEncoder handles differently.
func sourceImages(w, h int, seed int64) []image.Image {
	rnd := rand.New(rand.NewSource(seed))
	r := image.Rect(3, 1, 3+w, 1+h) // non-zero origin
	nrgba := image.NewNRGBA(r)
	rnd.Read(nrgba.Pix)
	rgba := image.NewRGBA(r)
	rnd.Read(rgba.Pix)
	for i := 0; i < len(rgba.Pix); i += 4 { // valid premultiplied colors
		for c := 0; c < 3; c++ {
			rgba.Pix[i+c] = min(rgba.Pix[i+c], rgba.Pix[i+3])
		}
	}
	gray := image.NewGray(r)
	rnd.Read(gray.Pix)
	ycc := image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
	rnd.Read(ycc.Y)
	rnd.Read(ycc.Cb)
	rnd.Read(ycc.Cr)
	nycca := image.NewNYCbCrA(r, image.YCbCrSubsampleRatio444)
	rnd.Read(nycca.Y)
	rnd.Read(nycca.Cb)
	rnd.Read(nycca.Cr)
	rnd.Read(nycca.A)
	pal := image.NewPaletted(r, color.Palette{color.Black, color.NRGBA{10, 20, 30, 40}, color.White})
	for i := range pal.Pix {
		pal.Pix[i] = uint8(rnd.Intn(3))
	}
	return []image.Image{nrgba, rgba, gray, ycc, nycca, pal}
}

func lossyLossless(t testing.TB) []*Options {
	t.Helper()
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	lossless, err := NewLosslessEncoderOptions(PresetDefault, 6)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions: %v", err)
	}
	return []*Options{lossy, lossless}
}

func TestRelease(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for _, o := range lossyLossless(t) {
		for _, src := range sourceImages(37, 29, 1) {
			name := fmt.Sprintf("%T lossless=%v", src, o.Lossless)
			e, err := NewEncoder(src, o)
			if err != nil {
				t.Fatalf("%s: NewEncoder: %v", name, err)
			}
			var want bytes.Buffer
			if err := e.Encode(&want); err != nil {
				t.Fatalf("%s: Encode: %v", name, err)
			}
			e.Release()
			var cw countWriter
			if err := e.Encode(&cw); !errors.Is(err, ErrReleased) {
				t.Fatalf("%s: Encode after Release = %v, want %v", name, err, ErrReleased)
			}
			if err := e.EncodeContext(ctx, &cw); !errors.Is(err, ErrReleased) {
				t.Fatalf("%s: EncodeContext after Release = %v, want %v", name, err, ErrReleased)
			}
			if cw.calls != 0 {
				t.Fatalf("%s: %d Write calls after Release, want 0", name, cw.calls)
			}
			e.Release() // no-op
			if e.img != nil {
				t.Fatalf("%s: img = %p after Release, want nil", name, e.img)
			}
			if e.pic != nil {
				t.Fatalf("%s: pic = %p after Release, want nil", name, e.pic)
			}
		}
	}
	// No snapshot to release.
	e, err := NewEncoder(image.NewNRGBA(image.Rect(0, 0, 0, 7)), nil)
	if err != nil {
		t.Fatalf("NewEncoder(0x7): %v", err)
	}
	e.Release()
	if err := e.Encode(&bytes.Buffer{}); !errors.Is(err, ErrReleased) {
		t.Fatalf("Encode(0x7) after Release = %v, want %v", err, ErrReleased)
	}
	var nilEncoder *Encoder
	nilEncoder.Release()
}

// TestReleaseRecycles checks that snapshot storage handed back by Release
// and reused by the next encoders (pooled buffers are not cleared) never
// leaks into their output, for all source types and sizes in any order.
func TestReleaseRecycles(t *testing.T) {
	sizes := [][2]int{{1, 1}, {1, 300}, {300, 1}, {17, 33}, {64, 48}, {16, 16}}
	for _, o := range lossyLossless(t) {
		var srcs []image.Image
		for i, s := range sizes {
			srcs = append(srcs, sourceImages(s[0], s[1], int64(i))...)
			srcs = append(srcs, sourceImages(s[0], s[1], int64(i+100))...)
		}
		want := make([][]byte, len(srcs))
		for i, src := range srcs {
			e, err := NewEncoder(src, o)
			if err != nil {
				t.Fatalf("%T %v lossless=%v: NewEncoder: %v", src, src.Bounds(), o.Lossless, err)
			}
			var buf bytes.Buffer
			if err := e.Encode(&buf); err != nil {
				t.Fatalf("%T %v lossless=%v: Encode: %v", src, src.Bounds(), o.Lossless, err)
			}
			want[i] = buf.Bytes() // e is not released: fresh storage for all
		}
		// Now with recycling, twice in different orders.
		rnd := rand.New(rand.NewSource(5))
		for round := 0; round < 2; round++ {
			for _, i := range rnd.Perm(len(srcs)) {
				e, err := NewEncoder(srcs[i], o)
				if err != nil {
					t.Fatalf("%T %v lossless=%v: NewEncoder: %v", srcs[i], srcs[i].Bounds(), o.Lossless, err)
				}
				for k := 0; k < 2; k++ { // Encode can be called again
					var buf bytes.Buffer
					if err := e.Encode(&buf); err != nil {
						t.Fatalf("%T %v lossless=%v: Encode: %v", srcs[i], srcs[i].Bounds(), o.Lossless, err)
					}
					if !bytes.Equal(buf.Bytes(), want[i]) {
						t.Fatalf("%T %v lossless=%v: output with recycled storage differs (%d bytes, want %d)", srcs[i], srcs[i].Bounds(), o.Lossless, buf.Len(), len(want[i]))
					}
				}
				e.Release()
			}
		}
	}
}

// TestReleaseWaitsForEncode runs Release concurrently with Encode (run it
// with -race): Encode either completes normally or reports ErrReleased.
func TestReleaseWaitsForEncode(t *testing.T) {
	src := sourceImages(200, 150, 3)[1] // *image.RGBA: pooled snapshot
	for _, o := range lossyLossless(t) {
		want, err := encodeOnce(src, o)
		if err != nil {
			t.Fatalf("lossless=%v: encode: %v", o.Lossless, err)
		}
		for i := 0; i < 10; i++ {
			e, err := NewEncoder(src, o)
			if err != nil {
				t.Fatalf("lossless=%v: NewEncoder: %v", o.Lossless, err)
			}
			var wg sync.WaitGroup
			results := make([]error, 3)
			outs := make([][]byte, 3)
			for g := range results {
				wg.Go(func() {
					var buf bytes.Buffer
					results[g] = e.Encode(&buf)
					outs[g] = buf.Bytes()
				})
			}
			runtime.Gosched()
			e.Release()
			wg.Wait()
			for g, err := range results {
				if errors.Is(err, ErrReleased) {
					continue
				}
				if err != nil {
					t.Fatalf("lossless=%v round %d: Encode %d: %v", o.Lossless, i, g, err)
				}
				if !bytes.Equal(outs[g], want) {
					t.Fatalf("lossless=%v round %d: Encode %d: output differs (%d bytes, want %d)", o.Lossless, i, g, len(outs[g]), len(want))
				}
			}
		}
	}
}

func encodeOnce(src image.Image, o *Options) ([]byte, error) {
	e, err := NewEncoder(src, o)
	if err != nil {
		return nil, err
	}
	defer e.Release()
	var buf bytes.Buffer
	err = e.Encode(&buf)
	return buf.Bytes(), err
}

// TestCleanupRecyclesSnapshot checks the safety net: the snapshot of an
// Encoder that was never released goes back to the pool once the Encoder
// is unreachable.
func TestCleanupRecyclesSnapshot(t *testing.T) {
	src := sourceImages(37, 29, 7)[2] // *image.Gray
	n := 4 * 37 * 29
	bufpool.Trim() // drain
	func() {
		e, err := NewEncoder(src, nil)
		if err != nil {
			t.Fatalf("NewEncoder: %v", err)
		}
		if err := e.Encode(&bytes.Buffer{}); err != nil {
			t.Fatalf("Encode: %v", err)
		}
	}()
	want := toNRGBA(src).Pix
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); {
		runtime.GC()
		time.Sleep(10 * time.Millisecond) // let the cleanup goroutine run
		// pixPool holds nothing but snapshots: before the cleanup ran, Get
		// allocates a zeroed buffer.
		if p := pixPool.Get(n); bytes.Equal(p, want) {
			return
		}
	}
	t.Fatal("the cleanup did not recycle the snapshot")
}

// scribble takes buffers of n bytes from the snapshot pool and overwrites
// them with v, until stop is closed.
func scribble(n int, v byte, stop <-chan struct{}, gc bool) {
	for {
		select {
		case <-stop:
			return
		default:
		}
		if gc {
			runtime.GC()
		}
		p := pixPool.Get(n)
		for i := range p {
			p[i] = v
		}
		runtime.Gosched()
	}
}

// TestCleanupNotEarly encodes while another goroutine forces garbage
// collections and scribbles over every buffer it takes from the pool: if
// the cleanup recycled a snapshot still in use, the output would change.
func TestCleanupNotEarly(t *testing.T) {
	srcs := sourceImages(64, 40, 8)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() { scribble(4*64*40, 0xa5, stop, true) })
	defer func() {
		close(stop)
		wg.Wait()
	}()
	for _, o := range lossyLossless(t) {
		for _, src := range srcs {
			e, err := NewEncoder(src, o)
			if err != nil {
				t.Fatalf("%T lossless=%v: NewEncoder: %v", src, o.Lossless, err)
			}
			var ref bytes.Buffer
			if err := e.Encode(&ref); err != nil {
				t.Fatalf("%T lossless=%v: Encode: %v", src, o.Lossless, err)
			}
			for i := 0; i < 10; i++ {
				e, err := NewEncoder(src, o) // never released
				if err != nil {
					t.Fatalf("%T lossless=%v: NewEncoder: %v", src, o.Lossless, err)
				}
				var buf bytes.Buffer
				if err := e.Encode(&buf); err != nil {
					t.Fatalf("%T lossless=%v: Encode: %v", src, o.Lossless, err)
				}
				if !bytes.Equal(buf.Bytes(), ref.Bytes()) {
					t.Fatalf("%T lossless=%v: output differs (%d bytes, want %d)", src, o.Lossless, buf.Len(), ref.Len())
				}
			}
		}
	}
}
