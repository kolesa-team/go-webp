package vp8l

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// withAsync runs f with the two-stage decode forced for every image and the
// given job size (0: production sizing), counting the jobs handed over.
func withAsync(jobPixels int, f func(jobs *int)) {
	oldMin, oldJob, oldHook := asyncMinPixels, asyncJobPixels, asyncTestHook
	defer func() { asyncMinPixels, asyncJobPixels, asyncTestHook = oldMin, oldJob, oldHook }()
	jobs := 0
	asyncMinPixels, asyncJobPixels = 0, jobPixels
	asyncTestHook = func(int, int) { jobs++ }
	f(&jobs)
}

// takeBudget occupies every free helper slot until the returned function is
// called.
func takeBudget(t *testing.T) func() {
	t.Helper()
	release := make(chan struct{})
	var tasks []*workers.Task
	for {
		task := workers.Spawn(func() error { <-release; return nil })
		if task == nil {
			break
		}
		tasks = append(tasks, task)
	}
	return func() {
		close(release)
		for _, task := range tasks {
			must(t, task.Wait())
		}
	}
}

// checkNoLeak fails if goroutines started during the test are still
// running shortly after it, or if helper slots are still taken.
func checkNoLeak(t *testing.T) {
	t.Helper()
	before := runtime.NumGoroutine()
	t.Cleanup(func() {
		deadline := time.Now().Add(2 * time.Second)
		for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if n := runtime.NumGoroutine(); n > before {
			t.Errorf("%d goroutines leaked", n-before)
		}
		if n := workers.InUse(); n != 0 {
			t.Errorf("%d helper slots still in use", n)
		}
	})
}

// sinkCase builds a fresh sink for a w x h frame; result returns its
// output for comparison.
type sinkCase struct {
	name string
	make func(w, h int) (pipeline.Sink, func() []byte)
}

func pipelineSink(p func(w, h int) pipeline.Params) func(w, h int) (pipeline.Sink, func() []byte) {
	return func(w, h int) (pipeline.Sink, func() []byte) {
		s, img, err := pipeline.NewSink(p(w, h))
		if err != nil {
			panic(err)
		}
		return s, func() []byte { return img().Pix }
	}
}

var asyncSinks = []sinkCase{
	{"direct", pipelineSink(func(w, h int) pipeline.Params { return pipeline.Params{SrcW: w, SrcH: h} })},
	{"crop", pipelineSink(func(w, h int) pipeline.Params {
		return pipeline.Params{SrcW: w, SrcH: h, Crop: image.Rect(w/5, h/4, w/5+max(w/2, 1), h/4+max(h/2, 1))}
	})},
	{"scale", pipelineSink(func(w, h int) pipeline.Params {
		return pipeline.Params{SrcW: w, SrcH: h, ScaleW: max(w/3, 1), ScaleH: max(h/3, 1), Premultiply: true}
	})},
	{"flip", pipelineSink(func(w, h int) pipeline.Params { return pipeline.Params{SrcW: w, SrcH: h, Flip: true} })},
	{"need-half", func(w, h int) (pipeline.Sink, func() []byte) {
		s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: (h + 1) / 2}
		return s, func() []byte { return append(s.img.Pix, byte(s.next), byte(s.next>>8)) }
	}},
}

// asyncStreams returns the streams of the two-stage tests: the fixtures,
// including images bigger than the LZ77 window (the logo is 3000x2000).
func asyncStreams(t *testing.T) map[string][]byte {
	streams := map[string][]byte{}
	for name := range losslessFixtures {
		if testing.Short() && strings.Contains(name, "webp-logo") {
			continue // 6 Mpx
		}
		streams[name] = findChunk(readFixture(t, "lossless", name), "VP8L")
	}
	return streams
}

// TestAsyncMatchesSequential checks that the two-stage decode gives the
// same output as the sequential one for any helper budget (one helper, more,
// none free) and job size, with every kind of sink.
func TestAsyncMatchesSequential(t *testing.T) {
	checkNoLeak(t)
	streams := asyncStreams(t)
	for name, payload := range streams {
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for _, sc := range asyncSinks {
			if sc.name != "direct" && w*h > 1<<20 && testing.Short() {
				continue
			}
			ref, out := sc.make(w, h)
			must(t, Decode(payload, ref), name, sc.name)
			want := out()
			for _, budget := range []int{1, 4, -1} {
				for _, jobPx := range []int{1, 0, 1 << 30} {
					withAsync(jobPx, func(jobs *int) {
						defer workers.SetLimitForTest(workers.SetLimitForTest(max(budget, 1)))
						release := func() {}
						if budget < 0 {
							release = takeBudget(t)
						}
						s, out := sc.make(w, h)
						err := DecodeWithOptions(payload, s, &Options{UseThreads: true})
						release()
						must(t, err, name, sc.name)
						if !bytes.Equal(want, out()) {
							t.Errorf("%s %s budget %d job %d: output differs from the sequential decode", name, sc.name, budget, jobPx)
						}
						if (budget < 0) != (*jobs == 0) {
							t.Errorf("%s %s budget %d: %d jobs handed over", name, sc.name, budget, *jobs)
						}
					})
				}
			}
		}
	}
}

// TestAsyncTruncated checks that truncated streams fail with the same error,
// after committing the same rows, as the sequential decode.
func TestAsyncTruncated(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	for _, name := range []string{"slide_2000x1000.webp", "pal200_257x129_z8.webp", "alpha_257x129_z6.webp"} {
		payload := findChunk(readFixture(t, "lossless", name), "VP8L")
		w, h, _, err := DecodeHeader(payload)
		must(t, err)
		for n := headerSize; n < len(payload); n += 1 + len(payload)/23 {
			p := payload[:n]
			ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
			wantErr := Decode(p, ref)
			for _, jobPx := range []int{1, 1 << 30} {
				withAsync(jobPx, func(*int) {
					s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
					err := DecodeWithOptions(p, s, &Options{UseThreads: true})
					if fmt.Sprint(err) != fmt.Sprint(wantErr) {
						t.Errorf("%s[:%d] job %d: error %v, want %v", name, n, jobPx, err, wantErr)
					}
					if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
						t.Errorf("%s[:%d] job %d: committed %d rows, want %d", name, n, jobPx, s.next, ref.next)
					}
				})
			}
		}
	}
}

// failSink fails (or panics) on its failAt-th Commit.
type failSink struct {
	rowSink
	failAt int
	panics bool
}

var errSinkFailed = errors.New("sink failed")

func (s *failSink) Commit(y, n int) error {
	if s.rowSink.Commit(y, n); s.calls == s.failAt {
		if s.panics {
			panic("sink panic")
		}
		return errSinkFailed
	}
	return nil
}

// TestAsyncSinkError checks that a sink error stops the helper at the same
// batch as the sequential decode and is returned before any later error.
func TestAsyncSinkError(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	payload := findChunk(readFixture(t, "lossless", "slide_2000x1000.webp"), "VP8L")
	w, h, _, err := DecodeHeader(payload)
	must(t, err)
	for _, failAt := range []int{1, 2, 17, 48, 49, 50, 51, 62} {
		for _, truncate := range []bool{false, true} {
			p := payload
			if truncate {
				if failAt > 30 {
					continue // the bitstream error comes first
				}
				p = payload[:len(payload)*2/3]
			}
			withAsync(1, func(*int) {
				s := &failSink{rowSink: rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}, failAt: failAt}
				err := DecodeWithOptions(p, s, &Options{UseThreads: true})
				if !errors.Is(err, errSinkFailed) {
					t.Errorf("fail at %d, truncated %v: got %v", failAt, truncate, err)
				}
				if s.calls != failAt {
					t.Errorf("fail at %d: %d commits", failAt, s.calls)
				}
			})
		}
	}
}

// TestAsyncHelperPanic checks that a panic on the helper goroutine (here in
// the sink) is re-raised on the caller as a *workers.PanicError, without
// leaking the helper, and that the decoder keeps working.
func TestAsyncHelperPanic(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	payload := findChunk(readFixture(t, "lossless", "slide_2000x1000.webp"), "VP8L")
	w, h, _, err := DecodeHeader(payload)
	must(t, err)
	for _, failAt := range []int{1, 3, 62} {
		withAsync(1, func(*int) {
			s := &failSink{rowSink: rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}, failAt: failAt, panics: true}
			func() {
				defer func() {
					r := recover()
					pe, ok := r.(*workers.PanicError)
					if !ok || pe.Value != "sink panic" {
						t.Errorf("fail at %d: recovered %v, want a *workers.PanicError", failAt, r)
					}
				}()
				_ = DecodeWithOptions(payload, s, &Options{UseThreads: true})
				t.Errorf("fail at %d: no panic", failAt)
			}()
		})
	}
	// The decoder put back into the pool after the panic still works.
	img, err := decodeVP8L(payload)
	must(t, err)
	sum := sha256.Sum256(img.Pix)
	check(t, hex.EncodeToString(sum[:]) == losslessFixtures["slide_2000x1000.webp"], "decode after panic")
}

// TestAsyncProducerPanic checks that a panic of the producer (the caller's
// goroutine) stops and waits for the helper before propagating.
func TestAsyncProducerPanic(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	payload := findChunk(readFixture(t, "lossless", "slide_2000x1000.webp"), "VP8L")
	w, h, _, err := DecodeHeader(payload)
	must(t, err)
	for _, panicAt := range []int{1, 2, 30} {
		withAsync(1, func(jobs *int) {
			asyncTestHook = func(int, int) {
				if *jobs++; *jobs == panicAt {
					panic("producer panic")
				}
			}
			s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
			func() {
				defer func() {
					if r := recover(); r != "producer panic" {
						t.Errorf("panic at %d: recovered %v", panicAt, r)
					}
				}()
				_ = DecodeWithOptions(payload, s, &Options{UseThreads: true})
			}()
			if n := workers.InUse(); n != 0 {
				t.Errorf("panic at %d: helper still running after the panic", panicAt)
			}
		})
	}
}

// TestAsyncGate checks when the helper starts.
func TestAsyncGate(t *testing.T) {
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	payload := findChunk(readFixture(t, "lossless", "slide_2000x1000.webp"), "VP8L")
	small := findChunk(readFixture(t, "lossless", "pal200_257x129_z8.webp"), "VP8L")
	jobs := 0
	defer func(old func(int, int)) { asyncTestHook = old }(asyncTestHook)
	asyncTestHook = func(int, int) { jobs++ }
	for _, c := range []struct {
		name    string
		payload []byte
		opts    *Options
		need    int
		async   bool
	}{
		{"threads", payload, &Options{UseThreads: true}, 1000, true},
		{"no options", payload, nil, 1000, false},
		{"no threads", payload, &Options{}, 1000, false},
		{"small image", small, &Options{UseThreads: true}, 129, false},
		// 2000x100 needed pixels is below asyncMinPixels.
		{"few rows needed", payload, &Options{UseThreads: true}, 100, false},
	} {
		jobs = 0
		w, h, _, err := DecodeHeader(c.payload)
		must(t, err)
		s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: c.need}
		must(t, DecodeWithOptions(c.payload, s, c.opts), c.name)
		check(t, s.next == min(c.need, h), c.name, s.next)
		if (jobs > 0) != c.async {
			t.Errorf("%s: %d jobs handed over", c.name, jobs)
		}
	}
}

func TestJobPixels(t *testing.T) {
	// A 6 Mpx logo coded in 84 KB gets big jobs, a 1.2 Mpx photo coded in
	// 1 MB small ones; there are always at least 16 jobs.
	logo, photo := jobPixels(3000*2000, 84202), jobPixels(960*1280, 1070724)
	check(t, logo > 256*1024 && logo <= 3000*2000/16, logo)
	check(t, photo > 16*1024 && photo < 64*1024, photo)
	check(t, jobPixels(256*1024, 1<<30) >= 1)
	check(t, jobPixels(1, 0) == 1)
}
