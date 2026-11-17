package vp8lenc

import (
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// ThreadLevel parallelism.
//
// By default an encode runs on the caller's goroutine only. With
// Config.ThreadLevel, independent work items (candidate configurations,
// image stripes, histogram tiles) are spread over the caller and helper
// goroutines started through the process-wide budget of internal/workers:
// when no slot is free, the caller does all the work itself. Items are
// claimed through an atomic counter (no channel operation per item) and
// are coarse (at least about half a millisecond of work each). The
// candidate configurations are spread by parallel; the stages within an
// encoding run on the crew of its scratch, whose helpers are started once
// per encode and serve all its stages. Every result is independent of the
// number of goroutines, so the output is byte-identical to the sequential
// encode.

// parallel runs body on the calling goroutine (worker 0) and on up to
// helpers helper goroutines (workers 1..helpers, never more than items-1),
// each started only while the process-wide budget has a free slot. The
// bodies claim the items 0..items-1 through claim, which hands out every
// item exactly once (in increasing order) and then reports false. parallel
// returns when every body has returned. A panic in a helper is re-raised on
// the caller once all helpers are done; when the caller's body panics, the
// helpers stop claiming items and are waited for before the panic unwinds
// further.
//
// A body must not block waiting for another body (the item it waits for
// may never be claimed when the budget is exhausted): items are
// independent, and per-worker state (e.g. a scratch taken from a pool) is
// acquired and released inside the body, on the goroutine that runs it.
func parallel(items, helpers int, body func(worker int, claim func() (int, bool))) {
	var next atomic.Int64
	const stopped = 1 << 62
	claim := func() (int, bool) {
		i := next.Add(1) - 1
		return int(i), i < int64(items)
	}
	helpers = min(helpers, items-1)
	var tasks []*workers.Task
	finished := false
	defer func() {
		if !finished {
			next.Store(stopped) // the caller panicked: stop the helpers early
		}
		var failed *workers.Task
		for _, t := range tasks {
			if t.Panicked() && failed == nil {
				failed = t
			}
		}
		if finished && failed != nil {
			failed.Wait() // re-raises the helper's panic
		}
	}()
	for k := 1; k <= helpers; k++ {
		t := workers.Spawn(func() error {
			body(k, claim)
			return nil
		})
		if t == nil {
			break
		}
		tasks = append(tasks, t)
	}
	body(0, claim)
	finished = true
}

// crew is the set of helper goroutines of the parallel stages of one
// encoding goroutine (hash chain stripes, predictor batches, histogram
// remap): instead of starting helpers for every stage (the predictor runs
// one stage per batch of tiles, dozens per image), the first stage starts
// them through the process-wide budget and the following ones reuse them,
// woken by a channel send each and joined through one channel; between
// stages they wait, blocked, on their channel. stop ends them: the encode
// that started them calls it before returning (the crew of a scratch is
// stopped before the scratch goes back to its pool), so that no helper
// outlives it.
//
// The zero value is ready to use. A crew is used by one goroutine at a
// time (the one owning its scratch); run must not be called from a body.
type crew struct {
	tasks  []*workers.Task
	start  []chan bool   // per helper: true runs body, false stops it
	done   chan struct{} // one value per helper and job
	failed []bool        // per helper: its body panicked (or exited)

	// The current job, set before the helpers are woken.
	body    func(worker int, claim func() (int, bool))
	next    atomic.Int64
	items   int64
	claimFn func() (int, bool) // c.claim, bound once
}

func (c *crew) claim() (int, bool) {
	i := c.next.Add(1) - 1
	return int(i), i < c.items
}

// run is parallel on the crew: body runs on the calling goroutine (worker
// 0) and on up to helpers helpers of the crew (workers 1..helpers, never
// more than items-1), started if needed (and while the budget allows). It
// returns when every body has returned. A panic in a helper is re-raised on
// the caller (by its Task.Wait) once all bodies are done; when the caller's
// body panics, the helpers stop claiming items and are waited for before
// the panic unwinds further. The same rules as for parallel apply to body.
func (c *crew) run(items, helpers int, body func(worker int, claim func() (int, bool))) {
	if c.claimFn == nil {
		c.claimFn = c.claim
	}
	c.items = int64(items)
	c.next.Store(0)
	n := min(helpers, items-1)
	if n <= 0 {
		body(0, c.claimFn)
		return
	}
	c.grow(n)
	n = min(n, len(c.tasks))
	c.body = body
	sent := 0
	for k := 0; k < n; k++ {
		select {
		case c.start[k] <- true:
			sent++
		case <-c.tasks[k].Done(): // (cannot happen: run re-raises panics)
		}
	}
	finished := false
	defer func() {
		const stopped = 1 << 62
		if !finished {
			c.next.Store(stopped) // the caller panicked: stop the helpers early
		}
		for ; sent > 0; sent-- {
			<-c.done
		}
		c.body = nil
		if finished {
			for k, f := range c.failed {
				if f {
					c.tasks[k].Wait() // re-raises the helper's panic
				}
			}
		}
	}()
	body(0, c.claimFn)
	finished = true
}

// grow starts helpers until the crew has n, while the budget allows.
func (c *crew) grow(n int) {
	if c.done == nil {
		c.done = make(chan struct{}, intraHelpers)
	}
	for k := len(c.tasks); k < n; k++ {
		if k == len(c.start) {
			c.start = append(c.start, make(chan bool, 1))
			c.failed = append(c.failed, false)
		}
		start := c.start[k]
		t := workers.Spawn(func() error {
			for <-start {
				c.work(k + 1)
			}
			return nil
		})
		if t == nil {
			return
		}
		c.tasks = append(c.tasks, t)
	}
}

// work runs the current job as worker w and reports it done, also when
// the body panics (the panic then ends the helper; run re-raises it).
func (c *crew) work(w int) {
	ok := false
	defer func() {
		if !ok {
			c.failed[w-1] = true
		}
		c.done <- struct{}{}
	}()
	c.body(w, c.claimFn)
	ok = true
}

// stop ends the helpers of the crew and waits for them.
func (c *crew) stop() {
	for k, t := range c.tasks {
		select {
		case c.start[k] <- false:
		case <-t.Done():
		}
	}
	for k, t := range c.tasks {
		t.Panicked() // waits for t (a panic was re-raised by run, or another one is unwinding)
		select {
		case <-c.start[k]: // the stop value of a helper that had panicked
		default:
		}
		c.tasks[k] = nil
		c.failed[k] = false
	}
	c.tasks = c.tasks[:0]
}

// intraHelpers is the largest number of helper goroutines one parallel stage
// of an encode (hash chain stripes, predictor tiles, histogram remap) asks
// for; the process-wide budget may grant fewer.
const intraHelpers = 7

// minParallelPixels is the smallest image (in pixels) whose stages are run
// in parallel: below it, the hand-off costs more than it saves. (A variable
// for the tests.)
var minParallelPixels = 1 << 18

// stageHelpers returns the number of helpers for a parallel stage of an
// image of n pixels.
func (p *params) stageHelpers(n int) int {
	if !p.parallel || n < minParallelPixels {
		return 0
	}
	return intraHelpers
}

// remapHelpers returns the number of helpers of the histogram remap (gated
// on its amount of work instead of the image size).
func (p *params) remapHelpers() int {
	if !p.parallel {
		return 0
	}
	return intraHelpers
}

// configHelpers is the number of helper goroutines (besides the caller)
// that encode candidate configurations concurrently. Each encoding
// configuration holds a full working set (15-100 bytes per pixel), so, like
// libwebp's multi-threaded lossless encoder, at most two configurations
// are encoded at a time: at level 9 this halves the time of the sequential
// encode for about twice its peak memory (instead of one working set per
// configuration, up to 8).
const configHelpers = 1

// maxConfigPairPixels is the largest image (in pixels) whose configurations
// are encoded two at a time: a working set takes up to about 90 bytes per
// pixel at level 9 (1 GB for 12 MP), so larger images encode their
// configurations one after the other (each with the parallel stages). A
// variable for the tests.
var maxConfigPairPixels = 8 << 20

// configHelpersFor returns the number of helpers encoding configurations
// of an image of n pixels concurrently with the caller.
func (p *params) configHelpersFor(n int) int {
	if !p.parallel || n > maxConfigPairPixels {
		return 0
	}
	return configHelpers
}

// configCost ranks the configurations by expected encoding time (spatial
// prediction with its mode search and cross-color is the slowest, palette
// encodes the fastest), so that the slowest start first.
func configCost(cc crunchConfig) int {
	switch cc.entropy {
	case modeSpatial, modeSpatialSubGreen:
		return 3
	case modeDirect, modeSubGreen, modePaletteAndSpatial:
		return 2
	}
	return 1
}

// configResult is the outcome of the configurations encoded by one worker:
// its smallest stream (lowest configuration index on ties) and a spare
// output buffer.
type configResult struct {
	best, spare []byte
	idx         int
}

// encodeConfigs encodes argb with every configuration and returns the
// smallest stream (the first one in configuration order on ties) and a
// losing buffer (or spare if unused). Each worker writes each trial into
// the buffer of its previous loser, so it keeps at most two streams.
func encodeConfigs(argb []uint32, w, h int, p *params, configs []crunchConfig, prefix int, sc *scratch, spare []byte, helpers int) (best, spareOut []byte) {
	order := make([]int, len(configs))
	for i := range order {
		order[i] = i
	}
	if helpers > 0 {
		// Longest first (stable), for a better balance.
		for i := 1; i < len(order); i++ {
			for j := i; j > 0 && configCost(configs[order[j]]) > configCost(configs[order[j-1]]); j-- {
				order[j], order[j-1] = order[j-1], order[j]
			}
		}
	}
	res := make([]configResult, 1+min(helpers, len(configs)-1))
	parallel(len(configs), helpers, func(worker int, claim func() (int, bool)) {
		s, buf := sc, spare
		if worker > 0 {
			s, buf = getScratch(w*h), nil
			defer putScratch(s)
			defer s.crew.stop()
		}
		r := &res[worker]
		r.idx = -1
		for {
			k, ok := claim()
			if !ok {
				break
			}
			i := order[k]
			out := encodeStream(argb, false, w, h, p, configs[i], prefix, s, buf)
			if r.best == nil || len(out) < len(r.best) || len(out) == len(r.best) && i < r.idx {
				buf, r.best, r.idx = r.best, out, i
			} else {
				buf = out
			}
		}
		r.spare = buf
	})
	bi := -1
	for k := range res {
		r := &res[k]
		if r.best != nil && (bi < 0 || len(r.best) < len(res[bi].best) || len(r.best) == len(res[bi].best) && r.idx < res[bi].idx) {
			bi = k
		}
	}
	best = res[bi].best
	for k := range res {
		cands := [2][]byte{res[k].spare, res[k].best}
		for c, b := range cands {
			if (k != bi || c == 0) && cap(b) > cap(spareOut) {
				spareOut = b
			}
		}
	}
	return best, spareOut
}
