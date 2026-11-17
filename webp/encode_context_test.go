package webp

import (
	"context"
	"image"
	"io"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/encoder"
)

// BenchmarkEncodeContext is BenchmarkEncode through EncodeContext with a
// context that can be canceled (the encoders then poll it).
func BenchmarkEncodeContext(b *testing.B) {
	img := loadJPEG(b, "source.jpg")
	o, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	require.NoError(b, err)
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
	require.Equal(t, context.Canceled, EncodeContext(canceled, &cw, img, nil))
	require.Zero(t, cw.n)

	// Option and source errors are reported as by Encode.
	bad := &encoder.Options{Quality: 200}
	require.EqualError(t, EncodeContext(context.Background(), &cw, img, bad), Encode(&cw, img, bad).Error())
	require.EqualError(t, EncodeContext(context.Background(), &cw, nil, nil), Encode(&cw, nil, nil).Error())
	big := image.NewNRGBA(image.Rect(0, 0, 20000, 1))
	require.EqualError(t, EncodeContext(context.Background(), &cw, big, nil), "cannot encode webppicture")
	require.Zero(t, cw.n)
}

// TestEncodeContextCancelLatency cancels slow encodes while they run: they
// must give up promptly, return context.Canceled and write nothing.
func TestEncodeContextCancelLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	logo := loadLogo(t)
	lossy, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	require.NoError(t, err)
	lossy.Method = 6
	threads, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	require.NoError(t, err)
	threads.ThreadLevel = true
	lossless, err := encoder.NewLosslessEncoderOptions(encoder.PresetDefault, 9)
	require.NoError(t, err)
	before := runtime.NumGoroutine()
	for _, tc := range []struct {
		name string
		o    *encoder.Options
	}{{"lossy m6", lossy}, {"lossy alpha threads", threads}, {"lossless z9", lossless}} {
		for _, after := range []time.Duration{0, 20 * time.Millisecond, 150 * time.Millisecond} {
			ctx, cancel := context.WithCancel(context.Background())
			timer := time.AfterFunc(after, cancel)
			var cw countingWriter
			start := time.Now()
			err := EncodeContext(ctx, &cw, logo, tc.o)
			latency := time.Since(start) - after
			timer.Stop()
			cancel()
			require.Equal(t, context.Canceled, err, tc.name)
			require.Zero(t, cw.n)
			t.Logf("%s: canceled after %v, returned %v later", tc.name, after, latency)
			require.Less(t, latency, 2*time.Second, tc.name)
		}
	}
	for deadline := time.Now().Add(5 * time.Second); runtime.NumGoroutine() > before; {
		require.True(t, time.Now().Before(deadline), "goroutine leak")
		time.Sleep(time.Millisecond)
	}
	require.NoError(t, Encode(io.Discard, logo, threads))
}
