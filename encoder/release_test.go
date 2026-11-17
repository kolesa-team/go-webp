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

	"github.com/stretchr/testify/require"

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
	lossy, err := NewLossyEncoderOptions(PresetDefault, 75)
	require.NoError(t, err)
	lossless, err := NewLosslessEncoderOptions(PresetDefault, 6)
	require.NoError(t, err)
	return []*Options{lossy, lossless}
}

func TestRelease(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for _, o := range lossyLossless(t) {
		for _, src := range sourceImages(37, 29, 1) {
			name := fmt.Sprintf("%T lossless=%v", src, o.Lossless)
			e, err := NewEncoder(src, o)
			require.NoError(t, err)
			var want bytes.Buffer
			require.NoError(t, e.Encode(&want), name)
			e.Release()
			var cw countWriter
			require.ErrorIs(t, e.Encode(&cw), ErrReleased, name)
			require.ErrorIs(t, e.EncodeContext(ctx, &cw), ErrReleased, name)
			require.Zero(t, cw.calls, name)
			e.Release() // no-op
			require.Nil(t, e.img, name)
			require.Nil(t, e.pic, name)
		}
	}
	// No snapshot to release.
	e, err := NewEncoder(image.NewNRGBA(image.Rect(0, 0, 0, 7)), nil)
	require.NoError(t, err)
	e.Release()
	require.ErrorIs(t, e.Encode(&bytes.Buffer{}), ErrReleased)
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
			require.NoError(t, err)
			var buf bytes.Buffer
			require.NoError(t, e.Encode(&buf))
			want[i] = buf.Bytes() // e is not released: fresh storage for all
		}
		// Now with recycling, twice in different orders.
		rnd := rand.New(rand.NewSource(5))
		for round := 0; round < 2; round++ {
			for _, i := range rnd.Perm(len(srcs)) {
				e, err := NewEncoder(srcs[i], o)
				require.NoError(t, err)
				for k := 0; k < 2; k++ { // Encode can be called again
					var buf bytes.Buffer
					require.NoError(t, e.Encode(&buf))
					require.Equal(t, want[i], buf.Bytes(), "%T %v lossless=%v", srcs[i], srcs[i].Bounds(), o.Lossless)
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
		require.NoError(t, err)
		for i := 0; i < 10; i++ {
			e, err := NewEncoder(src, o)
			require.NoError(t, err)
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
				require.NoError(t, err)
				require.Equal(t, want, outs[g])
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
		require.NoError(t, err)
		require.NoError(t, e.Encode(&bytes.Buffer{}))
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
			require.NoError(t, err)
			var ref bytes.Buffer
			require.NoError(t, e.Encode(&ref))
			for i := 0; i < 10; i++ {
				e, err := NewEncoder(src, o) // never released
				require.NoError(t, err)
				var buf bytes.Buffer
				require.NoError(t, e.Encode(&buf))
				require.Equal(t, ref.Bytes(), buf.Bytes(), "%T lossless=%v", src, o.Lossless)
			}
		}
	}
}
