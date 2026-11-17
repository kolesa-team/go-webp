package decoder

import (
	"context"
	"fmt"
	"image"
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// TestUseThreadsAllPathsAllBudgets runs every UseThreads helper together
// (the VP8 batch helper and the alpha helper of a large lossy+alpha
// picture, the two-stage helper of a large lossless one) through Decode,
// DecodeInto with a dirty buffer and a cancellable DecodeContext, under
// every helper budget: the output must equal the sequential decode, and no
// helper slot or goroutine may be left behind.
func TestUseThreadsAllPathsAllBudgets(t *testing.T) {
	g := runtime.NumGoroutine()
	files := []string{"webp-logo-lossy.webp", "webp-logo-lossless.webp", "m4_q75.webp"}
	opts := []*Options{
		{UseThreads: true},
		{UseThreads: true, Crop: image.Rect(101, 33, 2100, 1201), Flip: true},
		{UseThreads: true, Scale: image.Rect(0, 0, 600, 401)},
	}
	if testing.Short() {
		opts = opts[:2]
	}
	for _, file := range files {
		data := readFile(t, file)
		for _, o := range opts {
			if file == "m4_q75.webp" && !o.Crop.Empty() {
				continue // smaller than the crop
			}
			seq := *o
			seq.UseThreads = false
			want := decode(t, data, &seq)
			for _, b := range budgets {
				name := fmt.Sprintf("%s %+v %s", file, *o, b.name)
				restore := b.set()
				base := workers.InUse()

				got := decode(t, data, o)
				assert.Equal(t, want, got, name)

				dst := dirty(4 * want.Rect.Dx() * want.Rect.Dy())
				into, err := newDec(t, data, o).DecodeInto(dst)
				require.NoError(t, err, name)
				assert.Same(t, dst, into, name)
				assert.Equal(t, want.Pix, into.Pix, name)

				ctx, cancel := context.WithCancel(context.Background())
				img, err := newDec(t, data, o).DecodeContext(ctx)
				cancel()
				require.NoError(t, err, name)
				assert.Equal(t, want, img, name)

				assert.Equal(t, base, workers.InUse(), "%s: helper slots left taken", name)
				restore()
			}
		}
	}
	waitGoroutines(t, g)
}

// TestUseThreadsConcurrentSharedBudget decodes lossy, lossy+alpha and
// lossless pictures with UseThreads from several goroutines sharing a
// small helper budget (run with -race): some calls get helpers, others
// fall back to the sequential path, and every output is the sequential one.
func TestUseThreadsConcurrentSharedBudget(t *testing.T) {
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	g := runtime.NumGoroutine()
	files := []string{"webp-logo-lossy.webp", "webp-logo-lossless.webp", "m4_q75.webp"}
	want := map[string]*image.NRGBA{}
	for _, f := range files {
		want[f] = decode(t, readFile(t, f), nil)
	}
	rounds := 3
	if testing.Short() {
		rounds = 1
	}
	var wg sync.WaitGroup
	for k := 0; k < 6; k++ {
		wg.Go(func() {
			for r := 0; r < rounds; r++ {
				f := files[(k+r)%len(files)]
				got, err := newDec(t, readFile(t, f), &Options{UseThreads: true}).Decode()
				if assert.NoError(t, err, f) {
					assert.Equal(t, want[f], got, f)
				}
			}
		})
	}
	wg.Wait()
	waitGoroutines(t, g)
}
