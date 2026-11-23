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
	"context"
	"fmt"
	"image"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"

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
				if !reflect.DeepEqual(want, got) {
					t.Errorf("%s: Decode differs from the sequential decode", name)
				}

				dst := dirty(4 * want.Rect.Dx() * want.Rect.Dy())
				into, err := newDec(t, data, o).DecodeInto(dst)
				if err != nil {
					t.Fatalf("%s: DecodeInto: %v", name, err)
				}
				if into != dst {
					t.Errorf("%s: DecodeInto returned %p, want dst %p", name, into, dst)
				}
				if !bytes.Equal(want.Pix, into.Pix) {
					t.Errorf("%s: DecodeInto pixels differ from the sequential decode", name)
				}

				ctx, cancel := context.WithCancel(context.Background())
				img, err := newDec(t, data, o).DecodeContext(ctx)
				cancel()
				if err != nil {
					t.Fatalf("%s: DecodeContext: %v", name, err)
				}
				if !reflect.DeepEqual(want, img) {
					t.Errorf("%s: DecodeContext differs from the sequential decode", name)
				}

				if inUse := workers.InUse(); inUse != base {
					t.Errorf("%s: helper slots left taken: workers.InUse() = %d, want %d", name, inUse, base)
				}
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
				// Not readFile and newDec: they call t.Fatal, which must
				// not run on this goroutine.
				data, err := os.ReadFile("../test_data/images/" + f)
				if err != nil {
					t.Error(err)
					return
				}
				o := &Options{UseThreads: true}
				d, err := NewDecoder(bytes.NewReader(data), o)
				if err != nil {
					t.Errorf("%s: NewDecoder(%+v): %v", f, o, err)
					return
				}
				got, err := d.Decode()
				if err != nil {
					t.Errorf("%s: Decode: %v", f, err)
				} else if !reflect.DeepEqual(want[f], got) {
					t.Errorf("%s: Decode differs from the sequential decode", f)
				}
			}
		})
	}
	wg.Wait()
	waitGoroutines(t, g)
}
