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
	"github.com/kolesa-team/go-webp/v2/internal/vp8lenc"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// withAsync runs f with the two-stage decode forced for every image and the
// given job size (0: production sizing), counting the jobs handed over. With
// probePixels > 0 the producer first emits that many pixels itself and then
// always starts the helper (as probeDone does for a picture with enough
// work left); with 0 the helper starts before the first row.
func withAsync(jobPixels, probePixels int, f func(jobs *int)) {
	oldMin, oldProbe, oldWork, oldLLC, oldJob, oldHook := asyncMinPixels, asyncProbePixels, asyncMinWork, asyncSharedLLC, asyncJobPixels, asyncTestHook
	defer func() {
		asyncMinPixels, asyncProbePixels, asyncMinWork, asyncSharedLLC, asyncJobPixels, asyncTestHook = oldMin, oldProbe, oldWork, oldLLC, oldJob, oldHook
	}()
	jobs := 0
	asyncMinPixels, asyncProbePixels, asyncMinWork, asyncJobPixels = 0, probePixels, 0, jobPixels
	asyncSharedLLC = func() bool { return true }
	asyncTestHook = func(int, int) { jobs++ }
	f(&jobs)
}

// threadModes are the ways the tests decode a stream: sequentially, in two
// stages from the first row, and in two stages after a probe of one batch of
// rows (see withAsync).
var threadModes = []struct {
	name    string
	threads bool
	probe   int
}{
	{"sequential", false, 0},
	{"two-stage", true, 0},
	{"probed", true, 1},
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
// none free) and job size, with the helper started at once or after a
// probe, with every kind of sink.
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
					for _, probe := range []int{0, 1} {
						matchesSequential(t, name, sc, payload, want, budget, jobPx, probe)
					}
				}
			}
		}
	}
}

// matchesSequential is one decode of TestAsyncMatchesSequential.
func matchesSequential(t *testing.T, name string, sc sinkCase, payload, want []byte, budget, jobPx, probe int) {
	w, h, _, err := DecodeHeader(payload)
	must(t, err)
	withAsync(jobPx, probe, func(jobs *int) {
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
			t.Errorf("%s %s budget %d job %d probe %d: output differs from the sequential decode", name, sc.name, budget, jobPx, probe)
		}
		// After a probe of one batch, the helper starts if rows are left.
		if started := budget >= 0 && (probe == 0 || h > 2*numCacheRows); started != (*jobs > 0) {
			t.Errorf("%s %s budget %d probe %d: %d jobs handed over", name, sc.name, budget, probe, *jobs)
		}
	})
}

// TestAsyncNarrowSliding checks the two-stage decode of tall, narrow images
// bigger than the LZ77 window (a palette image decoded as index bytes and a
// 32-bit one): a job of 16 rows is then only a few thousand level-0 pixels,
// so many jobs are handed over between window slides, and the slides happen
// with jobs of the previous window still in flight.
func TestAsyncNarrowSliding(t *testing.T) {
	if testing.Short() {
		t.Skip("encodes 1.7 Mpx images")
	}
	checkNoLeak(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(2))
	for _, c := range []struct{ w, h, colors int }{{130, 13000, 40}, {150, 12000, 300}} {
		name := fmt.Sprintf("%dx%d/%d colors", c.w, c.h, c.colors)
		file, err := vp8lenc.Encode(fewColorImage(c.w, c.h, c.colors, int64(c.w)), &vp8lenc.Config{Quality: 75, Method: 4})
		must(t, err, name)
		payload := findChunk(file, "VP8L")
		if payload == nil {
			payload = file
		}
		ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, c.w, c.h)), need: c.h}
		must(t, Decode(payload, ref), name)
		for _, jobPx := range []int{1, 0, 1 << 30} {
			withAsync(jobPx, 0, func(jobs *int) {
				s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, c.w, c.h)), need: c.h}
				must(t, DecodeWithOptions(payload, s, &Options{UseThreads: true}), name, jobPx)
				if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
					t.Errorf("%s job %d: output differs from the sequential decode", name, jobPx)
				}
				if *jobs < 2 {
					t.Errorf("%s job %d: %d jobs handed over", name, jobPx, *jobs)
				}
			})
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
				for _, probe := range []int{0, 1} {
					withAsync(jobPx, probe, func(*int) {
						s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: h}
						err := DecodeWithOptions(p, s, &Options{UseThreads: true})
						if fmt.Sprint(err) != fmt.Sprint(wantErr) {
							t.Errorf("%s[:%d] job %d probe %d: error %v, want %v", name, n, jobPx, probe, err, wantErr)
						}
						if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
							t.Errorf("%s[:%d] job %d probe %d: committed %d rows, want %d", name, n, jobPx, probe, s.next, ref.next)
						}
					})
				}
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
			withAsync(1, 0, func(*int) {
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
		withAsync(1, 0, func(*int) {
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
		withAsync(1, 0, func(jobs *int) {
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
	defer func(old func() bool) { asyncSharedLLC = old }(asyncSharedLLC)
	defer func(old time.Duration) { asyncMinWork = old }(asyncMinWork)
	var projected time.Duration
	probed := false
	defer func(old func(time.Duration)) { asyncProbeHook = old }(asyncProbeHook)
	asyncProbeHook = func(p time.Duration) { projected, probed = p, true }
	for _, c := range []struct {
		name    string
		payload []byte
		opts    *Options
		need    int
		split   bool          // CPUs on several last-level caches
		minWork time.Duration // asyncMinWork
		async   bool
	}{
		{"threads", payload, &Options{UseThreads: true}, 1000, false, 0, true},
		{"no options", payload, nil, 1000, false, asyncMinWork, false},
		{"no threads", payload, &Options{}, 1000, false, asyncMinWork, false},
		{"small image", small, &Options{UseThreads: true}, 129, false, asyncMinWork, false},
		// 2000x100 needed pixels is below asyncMinPixels.
		{"few rows needed", payload, &Options{UseThreads: true}, 100, false, asyncMinWork, false},
		{"several caches", payload, &Options{UseThreads: true}, 1000, true, asyncMinWork, false},
		// The emission of the rest of the picture projects to less work.
		{"little work", payload, &Options{UseThreads: true}, 1000, false, time.Hour, false},
	} {
		jobs, projected, probed = 0, 0, false
		asyncSharedLLC = func() bool { return !c.split }
		asyncMinWork = c.minWork
		w, h, _, err := DecodeHeader(c.payload)
		must(t, err)
		s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: c.need}
		must(t, DecodeWithOptions(c.payload, s, c.opts), c.name)
		check(t, s.next == min(c.need, h), c.name, s.next)
		if (jobs > 0) != c.async {
			t.Errorf("%s: %d jobs handed over (projected emission work %v)", c.name, jobs, projected)
		}
		if c.async && !probed {
			t.Errorf("%s: started without a probe", c.name)
		}
	}
}

// TestSharedLLC checks the last-level cache test on Linux system files:
// an EPYC with two L3 of 4 CPUs, restricted to one of them or not, with
// offline CPUs, and systems that list no L3.
func TestSharedLLC(t *testing.T) {
	epyc := func(allowed string) map[string]string {
		files := map[string]string{
			"/proc/self/status":              "Name:\ttest\nCpus_allowed:\tff\nCpus_allowed_list:\t" + allowed + "\nMems_allowed:\t1\n",
			"/sys/devices/system/cpu/online": "0-7\n",
		}
		for c := 0; c < 8; c++ {
			for i, lv := range []string{"1", "1", "2", "3"} {
				dir := fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cache/index%d/", c, i)
				files[dir+"level"] = lv + "\n"
				switch {
				case lv == "3" && c < 4:
					files[dir+"shared_cpu_list"] = "0-3\n"
				case lv == "3":
					files[dir+"shared_cpu_list"] = "4-7\n"
				default:
					files[dir+"shared_cpu_list"] = fmt.Sprintf("%d\n", c)
				}
			}
		}
		return files
	}
	reader := func(files map[string]string) func(string) ([]byte, error) {
		return func(name string) ([]byte, error) {
			if s, ok := files[name]; ok {
				return []byte(s), nil
			}
			return nil, errors.New("no such file")
		}
	}
	noL3 := epyc("0-7")
	for c := 0; c < 8; c++ {
		delete(noL3, fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cache/index3/level", c))
	}
	offline := epyc("2-5")
	offline["/sys/devices/system/cpu/online"] = "0-3\n"
	for _, c := range []struct {
		name   string
		files  map[string]string
		shared bool
	}{
		{"all CPUs", epyc("0-7"), false},
		{"one L3", epyc("4-7"), true},
		{"one CPU", epyc("5"), true},
		{"across", epyc("3-4"), false},
		{"list", epyc("0,2,6"), false},
		{"offline CPUs", offline, true},
		{"no L3", noL3, true},
		{"no files", map[string]string{}, true},
		{"bad list", epyc("0-x"), true},
	} {
		if got := sharedLLCFrom(reader(c.files)); got != c.shared {
			t.Errorf("%s: shared %v, want %v", c.name, got, c.shared)
		}
	}
	for s, want := range map[string]int{"": 0, "0": 1, "0-3": 4, "0-3,8,10-11\n": 7, " 1 , 3-4": 3} {
		set, ok := parseCPUList(s)
		if !ok || len(set) != want {
			t.Errorf("parseCPUList(%q) = %v, %v; want %d CPUs", s, set, ok, want)
		}
	}
	for _, s := range []string{"x", "3-1", "-1", "0-", "1-2-3"} {
		if _, ok := parseCPUList(s); ok {
			t.Errorf("parseCPUList(%q) accepted", s)
		}
	}
	t.Logf("this machine: shared last-level cache %v", sharedLLC())
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

// FuzzDecodeAsync compares the two-stage decode (forced for every image,
// with small jobs, the helper started at once or after a probe) with the
// sequential one on arbitrary streams: the same error, the same rows
// committed and the same pixels.
func FuzzDecodeAsync(f *testing.F) {
	addSeeds(f, func(p []byte, _, _ int) { f.Add(p, uint16(0), uint8(1)) })
	f.Fuzz(func(t *testing.T, data []byte, need uint16, job uint8) {
		w, h, _, err := DecodeHeader(data)
		if err != nil || w*h > 1<<20 {
			return
		}
		n := h
		if need > 0 {
			n = min(int(need), h)
		}
		ref := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: n}
		wantErr := Decode(data, ref)
		defer workers.SetLimitForTest(workers.SetLimitForTest(2))
		// The helper starts at once, or after a probe of job%4 batches.
		for _, probe := range []int{0, int(job)%4*numCacheRows*w + 1} {
			withAsync(int(job)*w+1, probe, func(*int) {
				s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: n}
				err := DecodeWithOptions(data, s, &Options{UseThreads: true})
				if fmt.Sprint(err) != fmt.Sprint(wantErr) {
					t.Fatalf("probe %d: error %v, want %v", probe, err, wantErr)
				}
				if s.next != ref.next || !bytes.Equal(s.img.Pix, ref.img.Pix) {
					t.Fatalf("probe %d: committed %d rows, want %d (or pixels differ)", probe, s.next, ref.next)
				}
			})
		}
	})
}
