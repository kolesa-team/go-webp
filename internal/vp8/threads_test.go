package vp8

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

func init() {
	// Benchmark knob: VP8_MT_BATCH_MBS overrides the UseThreads hand-off
	// batch size, to compare batch sizes with the same test binary.
	if v, err := strconv.Atoi(os.Getenv("VP8_MT_BATCH_MBS")); err == nil && v > 0 {
		mtBatchMBs = v
	}
}

// goid returns the id of the calling goroutine (tests only).
func goid() uint64 {
	var buf [64]byte
	b := bytes.TrimPrefix(buf[:runtime.Stack(buf[:], false)], []byte("goroutine "))
	n, _ := strconv.ParseUint(string(b[:bytes.IndexByte(b, ' ')]), 10, 64)
	return n
}

// setBatch sets the hand-off batch size for the rest of the test.
func setBatch(t testing.TB, mbs int) {
	old := mtBatchMBs
	mtBatchMBs = mbs
	t.Cleanup(func() { mtBatchMBs = old })
}

// setEmit sets mtEmitWaits (when the emit helper starts) for the rest of
// the test.
func setEmit(t testing.TB, waits int) {
	old := mtEmitWaits
	mtEmitWaits = waits
	t.Cleanup(func() { mtEmitWaits = old })
}

// emitModes are the emit helper policies the threaded decoder must be
// insensitive to: never, from the first batch, adaptive (the default).
var emitModes = []struct {
	name  string
	waits int
}{{"emitNever", -1}, {"emitFirst", 0}, {"emitAdaptive", mtEmitWaits}}

// emitModesFor returns the emit policies worth testing with budget b: the
// emit helper needs a second helper slot.
func emitModesFor(limit int, exhausted bool) []struct {
	name  string
	waits int
} {
	if exhausted || limit < 2 {
		return emitModes[2:]
	}
	return emitModes
}

// setBudget sets the helper budget for the rest of the test. With
// exhausted, every slot is taken, so workers.Spawn returns nil.
func setBudget(t testing.TB, limit int, exhausted bool) {
	old := workers.SetLimitForTest(limit)
	taken := 0
	for exhausted && workers.TryAcquire() {
		taken++
	}
	t.Cleanup(func() {
		for ; taken > 0; taken-- {
			workers.Release()
		}
		workers.SetLimitForTest(old)
	})
}

// checkNoLeak fails the test if goroutines started during it are still
// running at its end, or if helper slots are still in use.
func checkNoLeak(t testing.TB) {
	before := runtime.NumGoroutine()
	t.Cleanup(func() {
		deadline := time.Now().Add(5 * time.Second)
		for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond) // a joined helper may still be exiting
		}
		if n := runtime.NumGoroutine(); n > before {
			buf := make([]byte, 1<<16)
			t.Errorf("%d goroutines leaked:\n%s", n-before, buf[:runtime.Stack(buf, true)])
		}
		if n := workers.InUse(); n != 0 {
			t.Errorf("%d helper slots still in use", n)
		}
		if n := workers.HandOffs(); n != 0 {
			t.Errorf("%d UseThreads pipelines still registered", n)
		}
	})
}

// traceSink is a Direct sink that records the Buffer/Commit calls and the
// goroutines that made them, and can fail, panic or stop early.
type traceSink struct {
	img     *image.NRGBA
	need    int
	failAt  int // Commit number (1-based) that returns errSinkFailure
	panicAt int // Commit number that panics
	commits int
	log     []string
	gids    map[uint64]bool
}

var errSinkFailure = errors.New("sink failure")

func newTraceSink(w, h int) *traceSink {
	return &traceSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h, gids: map[uint64]bool{}}
}

func (s *traceSink) Buffer(y, n int) []byte {
	s.log = append(s.log, fmt.Sprintf("B%d+%d", y, n))
	return s.img.Pix[y*s.img.Stride : (y+n)*s.img.Stride]
}
func (s *traceSink) Stride() int { return s.img.Stride }
func (s *traceSink) Commit(y, n int) error {
	s.commits++
	s.gids[goid()] = true
	s.log = append(s.log, fmt.Sprintf("C%d+%d", y, n))
	if s.commits == s.panicAt {
		panic("sink panic")
	}
	if s.commits == s.failAt {
		return errSinkFailure
	}
	return nil
}
func (s *traceSink) NeedRows() int { return s.need }

// traceYUV records DecodeYUV batches of a w pixels wide output.
type traceYUV struct {
	w    int
	rows [][]byte
	log  []string
	gids map[uint64]bool
}

func (s *traceYUV) PutYUV(y, n int, ys []byte, yStride int, us, vs []byte, uvStride int) error {
	s.gids[goid()] = true
	s.log = append(s.log, fmt.Sprintf("P%d+%d", y, n))
	for j := 0; j < n; j++ {
		s.rows = append(s.rows, append([]byte(nil), ys[j*yStride:][:s.w]...))
	}
	for j := 0; j < (n+1)/2; j++ {
		s.rows = append(s.rows, append([]byte(nil), us[j*uvStride:][:(s.w+1)/2]...), append([]byte(nil), vs[j*uvStride:][:(s.w+1)/2]...))
	}
	return nil
}
func (s *traceYUV) NeedRows() int { return 1 << 30 }

// decodeRun is the outcome of one decode.
type decodeRun struct {
	pix  []byte
	log  string
	err  string
	gids map[uint64]bool
}

func runDecode(vp8 []byte, opts Options, crop *image.Rectangle, need, failAt int) decodeRun {
	w, h, err := DecodeHeader(vp8)
	if err != nil {
		return decodeRun{err: err.Error()}
	}
	if crop != nil {
		w, h = crop.Dx(), crop.Dy()
	}
	s := newTraceSink(w, h)
	if need > 0 {
		s.need = need
	}
	s.failAt = failAt
	if crop != nil {
		err = DecodeRegion(vp8, nil, &opts, *crop, s)
	} else {
		err = Decode(vp8, nil, &opts, s)
	}
	r := decodeRun{pix: s.img.Pix, log: fmt.Sprint(s.log), gids: s.gids}
	if err != nil {
		r.err = fmt.Sprintf("%v (code %v)", err, status.CodeOf(err))
	}
	return r
}

func sameRun(a, b decodeRun) string {
	switch {
	case a.err != b.err:
		return fmt.Sprintf("error %q vs %q", a.err, b.err)
	case a.log != b.log:
		return fmt.Sprintf("sink calls %s vs %s", a.log, b.log)
	case !bytes.Equal(a.pix, b.pix):
		return "pixels differ"
	}
	return ""
}

// testVP8Files returns the VP8 payloads of the package test images plus,
// unless -short, the two large repository samples.
func testVP8Files(t *testing.T) map[string][]byte {
	files, _ := filepath.Glob("testdata/*.webp")
	if !testing.Short() {
		files = append(files, "../../test_data/images/m4_q75.webp", "../../test_data/images/webp-logo-lossy.webp")
	}
	out := map[string][]byte{}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		ri, err := extractVP8(data)
		if err != nil {
			t.Fatal(err)
		}
		out[filepath.Base(f)] = ri.vp8
	}
	return out
}

// frameRows returns the number of macroblock rows decode parses for opts
// and crop (brMbY).
func frameRows(t *testing.T, vp8 []byte, opts Options, crop *image.Rectangle) int {
	d := new(decoder)
	if err := d.parseHeaders(vp8); err != nil {
		t.Fatal(err)
	}
	d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
	if crop != nil {
		x, y := crop.Min.X&^1, crop.Min.Y&^1
		d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = x, y, x+crop.Dx(), y+crop.Dy()
	}
	if opts.BypassFiltering {
		d.filterType = 0
	}
	d.enterCritical()
	return d.brMbY
}

// budgets are the helper budgets the threaded decoder must be insensitive
// to: one helper, more helpers than needed, none available.
var budgets = []struct {
	name      string
	limit     int
	exhausted bool
}{{"limit1", 1, false}, {"limit8", 8, false}, {"exhausted", 1, true}}

// TestThreadedMatchesSequential checks that UseThreads gives the same pixels
// and the same sink calls as the sequential decoder, for every option, odd
// sizes (1xN, Nx1, not multiples of 16), crops, batch sizes and helper
// budgets, and that the helper runs exactly when the budget allows it.
func TestThreadedMatchesSequential(t *testing.T) {
	checkNoLeak(t)
	optsList := []Options{{}, {NoFancyUpsampling: true}, {BypassFiltering: true}, {DitheringStrength: 100}}
	for name, vp8 := range testVP8Files(t) {
		w, h, err := DecodeHeader(vp8)
		if err != nil {
			t.Fatal(err)
		}
		crops := []*image.Rectangle{nil}
		if w > 4 && h > 4 {
			c := image.Rect(w/4+1, h/3+1, w-w/5, h-1)
			crops = append(crops, &c)
		}
		for oi, opts := range optsList {
			for _, crop := range crops {
				want := runDecode(vp8, opts, crop, 0, 0)
				if want.err != "" {
					t.Fatalf("%s: %s", name, want.err)
				}
				opts.UseThreads = true
				for _, mbs := range []int{1, 7, 64, 512} {
					for _, b := range budgets {
						if w*h > 1<<20 && (mbs < 64 || b.name == "limit8") {
							continue // keep the large sample fast
						}
						for _, em := range emitModesFor(b.limit, b.exhausted) {
							t.Run(fmt.Sprintf("%s/opts%d/crop%v/mbs%d/%s/%s", name, oi, crop != nil, mbs, b.name, em.name), func(t *testing.T) {
								setBatch(t, mbs)
								setBudget(t, b.limit, b.exhausted)
								setEmit(t, em.waits)
								caller := goid()
								got := runDecode(vp8, opts, crop, 0, 0)
								if msg := sameRun(got, want); msg != "" {
									t.Fatal(msg)
								}
								helper := len(got.gids) > 1 || !got.gids[caller]
								mbW, rows := (w+15)/16, frameRows(t, vp8, opts, crop)
								k := max(1, (mbs+mbW-1)/mbW)
								if wantHelper := !b.exhausted && rows >= 2*k; helper != wantHelper {
									t.Errorf("helper used: %v, want %v (rows %d, batch %d)", helper, wantHelper, rows, k)
								}
							})
						}
					}
				}
			}
		}
	}
}

// TestBatchRows checks the batch schedule: batches of 1..k rows that cover
// the frame exactly, ramping up from one row and down to one row, with only
// a few more hand-offs than full batches would need.
func TestBatchRows(t *testing.T) {
	for k := 1; k <= 40; k++ {
		for rows := 2 * k; rows <= 300; rows++ {
			d := &decoder{brMbY: rows}
			var sizes []int
			for y := 0; y < rows; {
				n := d.mtBatchRows(y, k)
				if n < 1 || n > k || n > rows-y {
					t.Fatalf("k %d rows %d y %d: batch of %d rows", k, rows, y, n)
				}
				sizes = append(sizes, n)
				y += n
			}
			if sizes[0] != 1 || sizes[len(sizes)-1] != 1 {
				t.Fatalf("k %d rows %d: batches %v do not ramp", k, rows, sizes)
			}
			log2 := 0
			for 1<<log2 < k {
				log2++
			}
			if limit := (rows+k-1)/k + 2*log2 + 3; len(sizes) > limit {
				t.Fatalf("k %d rows %d: %d batches (%v), want at most %d", k, rows, len(sizes), sizes, limit)
			}
		}
	}
}

// TestThreadedYUV checks DecodeYUV with UseThreads, whole frame and
// cropped (odd crop origin and size, dithering).
func TestThreadedYUV(t *testing.T) {
	checkNoLeak(t)
	setBatch(t, 1)
	setBudget(t, 2, false)
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	for _, c := range [][2]int{{0, -1}, {0, 0}, {1, -1}, {1, 0}} {
		crop := []*image.Rectangle{nil, {Min: image.Pt(37, 51), Max: image.Pt(200, 180)}}[c[0]]
		setEmit(t, c[1])
		w := 250
		if crop != nil {
			w = crop.Dx()
		}
		run := func(threads bool) *traceYUV {
			s := &traceYUV{w: w, gids: map[uint64]bool{}}
			if err := DecodeYUV(vp8, &Options{UseThreads: threads, DitheringStrength: 50}, crop, s); err != nil {
				t.Fatal(err)
			}
			return s
		}
		want, got := run(false), run(true)
		if fmt.Sprint(want.log) != fmt.Sprint(got.log) || len(want.rows) != len(got.rows) {
			t.Fatalf("crop %v: PutYUV calls %v, want %v", crop, got.log, want.log)
		}
		for i := range want.rows {
			if !bytes.Equal(want.rows[i], got.rows[i]) {
				t.Fatalf("crop %v: row %d differs", crop, i)
			}
		}
		if got.gids[goid()] {
			t.Errorf("crop %v: PutYUV ran on the caller goroutine", crop)
		}
	}
}

// TestThreadedErrorsMatchSequential truncates streams at many points and
// checks that the threaded decoder returns the same error and delivers the
// same rows before it, also when the sink stops early (the sequential
// decoder then never reaches the broken row and succeeds).
func TestThreadedErrorsMatchSequential(t *testing.T) {
	checkNoLeak(t)
	steps := 60
	if testing.Short() {
		steps = 20
	}
	for _, name := range []string{"mix_250x199_o4.webp", "grad_250x199_o9.webp", "mix_129x65_p2s.webp"} {
		vp8 := loadVP8(t, name)
		_, h, _ := DecodeHeader(vp8)
		// Cut inside partition 0 (modes) and inside the token partition,
		// up to the last bytes.
		var cuts []int
		start := 10 + partLen(vp8)
		for i := 0; i <= steps; i++ {
			cuts = append(cuts, start/2+i*(len(vp8)-1-start/2)/steps)
		}
		cuts = append(cuts, start-1, start+1, len(vp8)-2)
		type key struct{ cut, need int }
		want := map[key]decodeRun{}
		stopped, partial := 0, 0
		for _, cut := range cuts {
			for _, need := range []int{0, 17, h / 2} {
				r := runDecode(vp8[:cut], Options{}, nil, need, 0)
				want[key{cut, need}] = r
				if r.err == "" {
					stopped++ // the sink stopped before the broken row
				} else if r.log != "[]" {
					partial++ // rows were delivered before the error
				}
			}
		}
		if stopped == 0 || partial == 0 {
			t.Fatalf("%s: cuts cover %d early stops and %d partial decodes", name, stopped, partial)
		}
		for _, mbs := range []int{1, 32} {
			for _, b := range budgets[:2] {
				for _, em := range emitModesFor(b.limit, b.exhausted) {
					t.Run(fmt.Sprintf("%s/mbs%d/%s/%s", name, mbs, b.name, em.name), func(t *testing.T) {
						setBatch(t, mbs)
						setBudget(t, b.limit, b.exhausted)
						setEmit(t, em.waits)
						for k, w := range want {
							got := runDecode(vp8[:k.cut], Options{UseThreads: true}, nil, k.need, 0)
							if msg := sameRun(got, w); msg != "" {
								t.Fatalf("cut %d need %d: %s", k.cut, k.need, msg)
							}
						}
					})
				}
			}
		}
	}
}

// TestThreadedSinkError checks that a sink error stops a threaded decode
// with that error, after the same rows as a sequential decode, whatever
// the batch size (it used to deadlock waiting for one result too many).
func TestThreadedSinkError(t *testing.T) {
	checkNoLeak(t)
	data, err := os.ReadFile("testdata/grad_250x199_o9.webp")
	if err != nil {
		t.Fatal(err)
	}
	p, err := container.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, mbs := range []int{1, 16, 64} {
		for _, failAt := range []int{1, 2, 5, 9} {
			want := runDecode(p.VP8, Options{}, nil, 0, failAt)
			setBatch(t, mbs)
			setBudget(t, 2, false)
			setEmit(t, []int{-1, 0, 2}[(mbs+failAt)%3])
			done := make(chan decodeRun, 1)
			go func() { done <- runDecode(p.VP8, Options{UseThreads: true}, nil, 0, failAt) }()
			select {
			case got := <-done:
				if msg := sameRun(got, want); msg != "" {
					t.Errorf("mbs %d failAt %d: %s", mbs, failAt, msg)
				}
				if got.err == "" {
					t.Errorf("mbs %d failAt %d: no error", mbs, failAt)
				}
			case <-time.After(10 * time.Second):
				t.Fatalf("mbs %d failAt %d: Decode did not return after a sink error", mbs, failAt)
			}
		}
	}
	// The error value itself is returned.
	img := image.NewNRGBA(image.Rect(0, 0, p.Width, p.Height))
	s := &failingSink{Direct: pipeline.NewDirect(img)}
	if err := Decode(p.VP8, nil, &Options{UseThreads: true}, s); !errors.Is(err, errSinkFailure) {
		t.Errorf("got %v, want the sink error", err)
	}
}

// failingSink returns an error from its second Commit.
type failingSink struct {
	*pipeline.Direct
	n int
}

func (s *failingSink) Commit(y, n int) error {
	s.n++
	if s.n == 2 {
		return errSinkFailure
	}
	return nil
}

// TestThreadedHelperPanic checks that a panic on the helper goroutine (here
// in the sink) neither crashes the process nor deadlocks the parser: Decode
// re-raises it on the calling goroutine and leaves no goroutine behind.
func TestThreadedHelperPanic(t *testing.T) {
	checkNoLeak(t)
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	for _, mbs := range []int{1, 16} {
		for _, panicAt := range []int{1, 2, 6} {
			for _, em := range emitModes {
				setBatch(t, mbs)
				setBudget(t, 2, false)
				setEmit(t, em.waits)
				done := make(chan any, 1)
				go func() {
					defer func() { done <- recover() }()
					s := newTraceSink(250, 199)
					s.panicAt = panicAt
					_ = Decode(vp8, nil, &Options{UseThreads: true}, s)
				}()
				select {
				case r := <-done:
					pe, ok := r.(*workers.PanicError)
					if !ok || pe.Value != "sink panic" {
						t.Fatalf("mbs %d panicAt %d: recovered %v, want a *workers.PanicError of the sink panic", mbs, panicAt, r)
					}
				case <-time.After(10 * time.Second):
					t.Fatalf("mbs %d panicAt %d: Decode deadlocked after a helper panic", mbs, panicAt)
				}
				if n := workers.InUse(); n != 0 {
					t.Fatalf("%d helper slots in use after the panic", n)
				}
				if n := workers.HandOffs(); n != 0 {
					t.Fatalf("%d UseThreads pipelines still registered after the panic", n)
				}
			}
		}
	}
}

// TestThreadedConcurrentDecoders runs threaded decodes of different images
// concurrently with a small helper budget (some get a helper, some decode
// sequentially) and checks every output. Run it with -race: decoders must
// not share their hand-off state.
func TestThreadedConcurrentDecoders(t *testing.T) {
	checkNoLeak(t)
	setBatch(t, 16)
	setBudget(t, 2, false)
	names := []string{"mix_250x199_o4.webp", "grad_250x199_o9.webp", "noise_129x65_o7.webp", "mix_1x300_o6.webp", "mix_129x65_p2s.webp"}
	want := map[string][]byte{}
	data := map[string][]byte{}
	for _, n := range names {
		data[n] = loadVP8(t, n)
		img, err := decodeImage(data[n], nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want[n] = img.Pix
	}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 6; i++ {
				n := names[(g+i)%len(names)]
				img, err := decodeImage(data[n], nil, &Options{UseThreads: true})
				if err != nil {
					errs <- err
					return
				}
				if !bytes.Equal(img.Pix, want[n]) {
					errs <- fmt.Errorf("%s: concurrent threaded decode differs", n)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// TestThreadedAlphaCallback checks Options.Alpha with UseThreads: the
// callback (which runs on the helper goroutine) sees the same row ends as in
// a sequential decode, its plane gives the same pixels, a nil plane makes
// the output opaque, and its error stops the decode after the same rows,
// with or without a crop, whatever the batch size and the helper budget.
func TestThreadedAlphaCallback(t *testing.T) {
	checkNoLeak(t)
	errAlpha := errors.New("alpha failure")
	vp8 := loadVP8(t, "mix_250x199_o4.webp")
	w, h, err := DecodeHeader(vp8)
	if err != nil {
		t.Fatal(err)
	}
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = byte(i*7 + i/w)
	}
	run := func(threads bool, crop *image.Rectangle, failAt int, opaque bool) (string, map[uint64]bool) {
		var calls [][2]int
		var win []byte
		opts := Options{UseThreads: threads, Alpha: func(rowStart, rowEnd int) ([]byte, int, error) {
			calls = append(calls, [2]int{rowStart, rowEnd})
			switch {
			case len(calls) == failAt:
				return nil, 0, errAlpha
			case opaque:
				return nil, 0, nil
			}
			// A window of just the rows asked for; the previous one is
			// overwritten, so reading it after this call shows.
			for i := range win {
				win[i] = 0x5a
			}
			if len(calls)%2 == 0 {
				return plane[:rowEnd*w], 0, nil // a whole plane is a window too
			}
			win = append(win[:0], plane[rowStart*w:rowEnd*w]...)
			return win, rowStart, nil
		}}
		r := runDecode(vp8, opts, crop, 0, 0)
		return fmt.Sprintf("calls %v err %q log %s pix %x", calls, r.err, r.log, sha256.Sum256(r.pix)), r.gids
	}
	crops := []*image.Rectangle{nil, {Min: image.Pt(31, 47), Max: image.Pt(230, 190)}}
	for ci, crop := range crops {
		for _, failAt := range []int{0, 1, 2, 7} {
			for _, opaque := range []bool{false, true} {
				want, _ := run(false, crop, failAt, opaque)
				for _, mbs := range []int{1, 32} {
					for _, b := range budgets {
						for _, em := range emitModesFor(b.limit, b.exhausted) {
							// A subtest, so that the budget (and the slots an
							// exhausted budget takes) is restored after each run.
							t.Run(fmt.Sprintf("crop%d/fail%d/opaque%v/mbs%d/%s/%s", ci, failAt, opaque, mbs, b.name, em.name), func(t *testing.T) {
								setBatch(t, mbs)
								setBudget(t, b.limit, b.exhausted)
								setEmit(t, em.waits)
								got, gids := run(true, crop, failAt, opaque)
								if got != want {
									t.Fatalf("got  %s\nwant %s", got, want)
								}
								// Both regions span all 13 macroblock rows, enough
								// for two batches of either size.
								if helper := len(gids) > 1 || (len(gids) == 1 && !gids[goid()]); helper != !b.exhausted && failAt != 1 {
									t.Errorf("helper used: %v, want %v", helper, !b.exhausted)
								}
							})
						}
					}
				}
			}
		}
	}
}

// TestThreadedParserPanic checks the other panic path of UseThreads: a panic
// in the parser (the calling goroutine), here while earlier batches are
// still being reconstructed, stops and joins the helper before it
// propagates with its own value, releases the helper slot and leaves no
// goroutine behind. No input makes the parser panic, so the test breaks the
// read position of one token partition of a 2-partition stream: the first
// refill of partition p, at the first token of row p, slices out of range
// (for p = 1, while row 0 is with the helper).
func TestThreadedParserPanic(t *testing.T) {
	checkNoLeak(t)
	setBatch(t, 1)
	setBudget(t, 2, false)
	vp8 := loadVP8(t, "mix_64x48_p1.webp")
	want := runDecode(vp8, Options{}, nil, 0, 0)
	for _, part := range []int{0, 1} {
		for round := 0; round < 30; round++ {
			setEmit(t, []int{-1, 0, 2}[round%3])
			d := new(decoder)
			if err := d.parseHeaders(vp8); err != nil {
				t.Fatal(err)
			}
			if d.numPartsMinusOne != 1 {
				t.Fatalf("%d token partitions, want 2", d.numPartsMinusOne+1)
			}
			// The setup of decoder.decode, without alpha or crop.
			s := newTraceSink(d.width, d.height)
			d.cropLeft, d.cropTop, d.cropRight, d.cropBottom = 0, 0, d.width, d.height
			d.outW = d.width
			d.enterCritical()
			d.precomputeFilterStrengths()
			d.initDithering(0)
			d.sink, d.fancy = s, true
			d.initFrame()
			d.parts[part].nbits = -1 // refill at the first token of row part
			d.parts[part].pos = -8
			r := func() (r any) {
				defer func() { r = recover() }()
				_ = d.decodeFrameThreaded()
				return nil
			}()
			if _, ok := r.(runtime.Error); !ok {
				t.Fatalf("partition %d: recovered %v, want the parser's runtime error", part, r)
			}
			// The helper was joined before the panic left decodeFrameThreaded.
			if n := workers.InUse(); n != 0 {
				t.Fatalf("partition %d: %d helper slots in use after the panic", part, n)
			}
			// The helper only emitted rows parsed before the broken one.
			if got := fmt.Sprint(s.log); !strings.HasPrefix(want.log, strings.TrimSuffix(got, "]")) {
				t.Fatalf("partition %d: sink calls %s, not a prefix of %s", part, got, want.log)
			}
			if part == 0 && len(s.log) != 0 {
				t.Fatalf("partition 0: sink calls %v before the first row was parsed", s.log)
			}
		}
	}
}

// TestThreadedModesAhead slows the token parser down so that the idle
// helpers parse the first partition ahead of it, and checks that the
// output, the sink calls and the errors (truncations inside the first
// partition and inside the tokens, early stops) stay those of the
// sequential decoder, for every emit policy.
func TestThreadedModesAhead(t *testing.T) {
	checkNoLeak(t)
	for _, name := range []string{"mix_250x199_o4.webp", "grad_250x199_o9.webp", "mix_64x48_p1.webp"} {
		vp8 := loadVP8(t, name)
		_, h, _ := DecodeHeader(vp8)
		start := 10 + partLen(vp8)
		cuts := []int{len(vp8), len(vp8) - 2, start + (len(vp8)-start)/2, start + 1, start - 1, start * 3 / 4, start / 2}
		for _, em := range emitModes {
			t.Run(name+"/"+em.name, func(t *testing.T) {
				setBatch(t, 1)
				setBudget(t, 2, false)
				setEmit(t, em.waits)
				var ahead atomic.Int64
				old := testHookTokenRow
				testHookTokenRow = func(d *decoder, mbY int) {
					time.Sleep(200 * time.Microsecond)
					// The token parser only parses the modes of the row it
					// needs: those of later rows come from a helper.
					if int(d.mt.modeReady.Load()) > mbY+1 {
						ahead.Add(1)
					}
				}
				t.Cleanup(func() { testHookTokenRow = old })
				for _, cut := range cuts {
					for _, need := range []int{0, h / 2} {
						opts := Options{DitheringStrength: 30}
						want := runDecode(vp8[:cut], opts, nil, need, 0)
						if cut == len(vp8) && need == 0 && want.err != "" {
							t.Fatalf("full stream: %s", want.err)
						}
						opts.UseThreads = true
						got := runDecode(vp8[:cut], opts, nil, need, 0)
						if msg := sameRun(got, want); msg != "" {
							t.Fatalf("cut %d need %d: %s", cut, need, msg)
						}
					}
				}
				if ahead.Load() == 0 {
					t.Errorf("no mode row was parsed ahead by a helper")
				}
			})
		}
	}
}

// TestThreadedDeterministicAcrossBudgets decodes with UseThreads under
// every helper budget from 1 to 8 and several GOMAXPROCS values, with the
// default policies, and checks that every output is the sequential one.
func TestThreadedDeterministicAcrossBudgets(t *testing.T) {
	checkNoLeak(t)
	files := testVP8Files(t)
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	for name, vp8 := range files {
		want := runDecode(vp8, Options{}, nil, 0, 0)
		if want.err != "" {
			t.Fatalf("%s: %s", name, want.err)
		}
		for _, mbs := range []int{16, 512} {
			for limit := 1; limit <= 8; limit++ {
				for _, p := range []int{1, 2, 4, runtime.NumCPU()} {
					t.Run(fmt.Sprintf("%s/mbs%d/limit%d/procs%d", name, mbs, limit, p), func(t *testing.T) {
						runtime.GOMAXPROCS(p)
						setBatch(t, mbs)
						setBudget(t, limit, false)
						got := runDecode(vp8, Options{UseThreads: true}, nil, 0, 0)
						if msg := sameRun(got, want); msg != "" {
							t.Fatal(msg)
						}
					})
				}
			}
		}
	}
}
