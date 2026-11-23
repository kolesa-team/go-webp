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

package webp

import (
	"context"
	"fmt"
	"image"
	"io"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/encoder"
)

// BenchmarkEncodeContext is BenchmarkEncode through EncodeContext with a
// context that can be canceled (the encoders then poll it).
func BenchmarkEncodeContext(b *testing.B) {
	img := loadJPEG(b, "source.jpg")
	o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		b.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b.ReportAllocs()
	for b.Loop() {
		if err := EncodeContext(ctx, io.Discard, img, o); err != nil {
			b.Fatal(err)
		}
	}
}

type countingWriter struct{ n int }

func (c *countingWriter) Write(p []byte) (int, error) { c.n += len(p); return len(p), nil }

func TestEncodeContext(t *testing.T) {
	img := loadJPEG(t, "source.jpg")
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	var cw countingWriter
	if err := EncodeContext(canceled, &cw, img, nil); !reflect.DeepEqual(context.Canceled, err) {
		t.Fatalf("EncodeContext(canceled) = %v, want %v", err, context.Canceled)
	}
	if cw.n != 0 {
		t.Fatalf("EncodeContext(canceled) wrote %d bytes, want 0", cw.n)
	}

	// Option and source errors are reported as by Encode.
	bad := &encoder.Options{Quality: 200}
	err := EncodeContext(context.Background(), &cw, img, bad)
	if want := Encode(&cw, img, bad).Error(); err == nil || err.Error() != want {
		t.Fatalf("EncodeContext(Quality 200) error = %v, want %q", err, want)
	}
	err = EncodeContext(context.Background(), &cw, nil, nil)
	if want := Encode(&cw, nil, nil).Error(); err == nil || err.Error() != want {
		t.Fatalf("EncodeContext(nil image) error = %v, want %q", err, want)
	}
	big := image.NewNRGBA(image.Rect(0, 0, 20000, 1))
	if err := EncodeContext(context.Background(), &cw, big, nil); err == nil || err.Error() != "cannot encode webppicture" {
		t.Fatalf("EncodeContext(20000x1) error = %v, want %q", err, "cannot encode webppicture")
	}
	if cw.n != 0 {
		t.Fatalf("EncodeContext wrote %d bytes on errors, want 0", cw.n)
	}
}

// TestEncodeContextCancelLatency cancels slow encodes while they run: they
// must give up promptly, return context.Canceled and write nothing.
//
// The test is timing-based. A loaded machine (other processes, the race
// detector) slows the encode and its polling alike, so a cancellation that
// takes 2 s or more to return is compared with the time the whole encode
// takes at that moment: it passes when it took less than a quarter of that.
// An encode that ends before the cancellation (a fast machine) is canceled
// earlier the next time. A case fails after three misses.
func TestEncodeContextCancelLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	logo := loadLogo(t)
	lossy, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	lossy.Method = 6
	threads, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		t.Fatalf("NewLossyEncoderOptions: %v", err)
	}
	threads.ThreadLevel = true
	lossless, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 9)
	if err != nil {
		t.Fatalf("NewLosslessEncoderOptions: %v", err)
	}
	before := runtime.NumGoroutine()
	for _, tc := range []struct {
		name string
		o    *encoder.Options
	}{{"lossy m6", lossy}, {"lossy alpha threads", threads}, {"lossless z9", lossless}} {
		for _, after := range []time.Duration{0, 20 * time.Millisecond, 150 * time.Millisecond} {
			var misses []string
			for attempt := 0; ; attempt++ {
				ctx, cancel := context.WithCancel(context.Background())
				timer := time.AfterFunc(after, cancel)
				var cw countingWriter
				start := time.Now()
				err := EncodeContext(ctx, &cw, logo, tc.o)
				latency := time.Since(start) - after
				timer.Stop()
				cancel()
				if err == nil {
					misses = append(misses, fmt.Sprintf("ended %v before the cancellation after %v", -latency, after))
					after /= 2
				} else {
					if !reflect.DeepEqual(context.Canceled, err) {
						t.Fatalf("%s: EncodeContext = %v, want %v", tc.name, err, context.Canceled)
					}
					if cw.n != 0 {
						t.Fatalf("%s: canceled EncodeContext wrote %d bytes, want 0", tc.name, cw.n)
					}
					t.Logf("%s: canceled after %v, returned %v later", tc.name, after, latency)
					if latency < 2*time.Second {
						break
					}
					start = time.Now()
					if err := Encode(io.Discard, logo, tc.o); err != nil {
						t.Fatalf("%s: Encode: %v", tc.name, err)
					}
					whole := time.Since(start)
					if latency < whole/4 {
						break
					}
					misses = append(misses, fmt.Sprintf("canceled after %v, returned %v later; the whole encode takes %v", after, latency, whole))
				}
				if attempt >= 2 {
					t.Fatalf("%s: %d attempts missed: %s", tc.name, attempt+1, strings.Join(misses, "; "))
				}
			}
		}
	}
	for deadline := time.Now().Add(20 * time.Second); runtime.NumGoroutine() > before; {
		if !time.Now().Before(deadline) {
			t.Fatalf("goroutine leak: %d goroutines after 20s, %d before the encodes", runtime.NumGoroutine(), before)
		}
		time.Sleep(time.Millisecond)
	}
	if err := Encode(io.Discard, logo, threads); err != nil {
		t.Fatalf("Encode(threads): %v", err)
	}
}
