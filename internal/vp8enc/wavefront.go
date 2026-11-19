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

package vp8enc

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/bufpool"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// Wavefront-parallel coding pass (Config.ThreadLevel).
//
// The calling goroutine and helpers from the process-wide budget of
// internal/workers code the macroblock rows, each worker taking the next
// row when it finishes one. Macroblock (x, r) reads the top context
// (samples, non-zero bits, intra modes and chroma diffusion errors) of
// macroblocks x-1..x+1 of row r-1, so row r codes macroblock x once row r-1
// has finished macroblock x+1. The shared top context arrays (yTop, uvTop,
// nz, topDerr, preds) are then safe: the entries row r-1 still writes are
// beyond those row r reads, row r overwrites an entry only after reading
// it, and row r+1 reads the entries of row r only once row r is past them.
//
// The output is the sequential pass's, byte for byte, whatever the number
// of workers:
//   - The cost tables are refreshed before the same macroblocks. Every
//     period-th macroblock is a refresh point; the worker that reaches one
//     waits until all earlier macroblocks are coded and committed (the
//     rows above), then refreshes the tables and starts a new epoch. A
//     macroblock is coded only in its epoch, so the rows below a refresh
//     point wait for it.
//   - The token statistics and the autofilter SSIM sums are accumulated
//     when a row is committed, in raster order. Workers record each row's
//     tokens into a row buffer, with statistics counted from zero, and the
//     SSIM values into a list. The statistics counters halve when they
//     saturate, so the row's counts are added to the frame's only if no
//     counter reaches that point (mergeStats, almost always); otherwise the
//     row's tokens are replayed one by one. The SSIM values are floating
//     point and are added one by one, in order.
//   - Committing a row appends its tokens to e.tokens and sets its
//     e.mbTokStart entries, so the frame's token buffer and everything that
//     reads it after the pass are unchanged.
//   - The header bit and distortion sums, and the maxEdge maxima, are
//     merged after the pass.
//
// Rows are committed in order by whichever worker finishes the row that
// unblocks them, so committing never makes a worker wait. Row buffers live
// in a ring of a few more slots than workers; a worker starting a row waits
// until the row that used its slot is committed, which bounds the memory
// to a few rows of tokens on top of the frame's.
//
// Every wait blocks on a sync.Cond (one per worker) and is woken only when
// its condition holds: a row's progress is published with an atomic store,
// and the worker of the row below, if it sleeps, is signalled only once the
// row is a full gap (see minWavefrontGap) ahead. Hand-offs are expensive:
// the Go scheduler leaves a goroutine woken by a running one on the
// waker's processor, and another thread steals it only after sleeping
// 3 us, which takes about 50 us on Linux (timer slack) and about 1 ms in
// some virtual machines. So the rows are kept far enough apart that
// workers rarely have to sleep, and only pictures with enough work per
// row and per worker are coded as a wavefront.

const (
	// maxWavefrontWorkers bounds the number of rows coded at once (the
	// refresh points limit the gain beyond).
	maxWavefrontWorkers = 8
	// minWavefrontGap is the smallest distance (in macroblocks) between
	// two rows coded at once: a worker that has to sleep until the row
	// above is far enough ahead is woken only once that row is
	// max(minWavefrontGap, mbW/workers) macroblocks ahead, the spacing
	// rows naturally take, so that each wake-up is followed by a run of
	// work rather than another wait.
	minWavefrontGap = 8
	noWant          = math.MaxInt32

	// The wavefront is used only when the pass is long enough to pay for
	// its hand-offs (each worker sleeps and is woken a few times per
	// refresh point): at least minWavefrontWork of estimated work in all,
	// minWavefrontRowWork per row and wavefrontWorkPerWorker per worker
	// (nanoseconds).
	minWavefrontWork       = 8_000_000
	minWavefrontRowWork    = 250_000
	wavefrontWorkPerWorker = 4_000_000
)

// wavefrontMBCost estimates the coding pass time per macroblock (ns) of
// each method, measured on a photo (Apple M5) with the vector kernels and
// with the Go reference ones.
var wavefrontMBCost = func() [7]int64 {
	if vecKernels {
		return [7]int64{1500, 1500, 2400, 7400, 7800, 9400, 24000}
	}
	return [7]int64{3200, 3500, 4800, 24600, 23600, 24800, 35200}
}()

var (
	// wavefrontPool holds the wfState structs (about 1 KB); their buffers
	// come from the bufpool pools below, which outlive garbage collections
	// (sync.Pool drops its content at every other one).
	wavefrontPool = sync.Pool{New: func() any { return new(wfState) }}
	// lfValuePool holds the autofilter SSIM lists of the row slots.
	lfValuePool bufpool.Pool[float64]
	// wfStatsPool holds the slots' token statistics (4 KB each).
	wfStatsPool bufpool.Pool[[numTypes]statsArray]
	// wfRowPool holds the rows' progress counters.
	wfRowPool bufpool.Pool[wfRow]

	// activeEncodes counts the Encode calls in progress. A wavefront pass
	// takes at most GOMAXPROCS/activeEncodes workers: a server that
	// already codes about one picture per processor gains nothing from
	// helpers, whose rows would then wait on each other's descheduled
	// workers.
	activeEncodes atomic.Int32

	// forceWavefront (tests only) codes every picture of two rows or more
	// as a wavefront, whatever its size.
	forceWavefront bool
	// testHookWavefrontRow, if set (tests only), is called on the worker
	// goroutine before it codes a row of a wavefront pass.
	testHookWavefrontRow func(row int)
)

// wavefrontWorkers returns how many rows the coding pass may code at once
// (1: sequential).
func (e *encoder) wavefrontWorkers() int {
	if !e.cfg.ThreadLevel {
		return 1
	}
	if forceWavefront {
		return max(1, min(maxWavefrontWorkers, e.mbH, e.mbW/2))
	}
	// int64: the estimate of a 16383x16383 picture overflows 32 bits.
	cost := wavefrontMBCost[e.method]
	work := int64(e.mbW) * int64(e.mbH) * cost
	if work < minWavefrontWork || int64(e.mbW)*cost < minWavefrontRowWork {
		return 1
	}
	nw := min(maxWavefrontWorkers, e.mbH, e.mbW/minWavefrontGap, int(min(work/wavefrontWorkPerWorker, maxWavefrontWorkers)))
	return max(1, min(nw, runtime.GOMAXPROCS(0)/max(1, int(activeEncodes.Load()))))
}

// wfWorker is one goroutine of a wavefront pass.
type wfWorker struct {
	it   *iterator
	cond sync.Cond // L is the wfState's mu
	// Under mu: whether the worker sleeps in await, and the condition it
	// waits for (see wfState.ready).
	sleeping                    bool
	above                       int // row whose progress it waits for, or -1
	need, epochNeed, commitNeed int32

	sizeP0, dist uint64
}

// wfSeen holds the last values of the row above's progress and of the
// epoch a worker saw: it reads the shared counters again only when it
// needs more, rather than for every macroblock.
type wfSeen struct{ above, epoch int32 }

// wfRow is the progress of a row, alone in its cache line (128 bytes on
// Apple CPUs): the worker of the row writes prog after every macroblock,
// and only the worker of the row below reads it.
type wfRow struct {
	prog atomic.Int32 // macroblocks coded
	want atomic.Int32 // progress the worker of the row below sleeps for
	_    [120]byte
}

// wfSlot holds the coded but not yet committed part of a row.
type wfSlot struct {
	tokens   tokenBuffer
	stats    *[numTypes]statsArray // statistics of the tokens not committed yet
	mbEnd    []int32               // mbEnd[x+1]: end of macroblock x's tokens; mbEnd[0] = 0
	lf       []float64             // autofilter SSIM values (appendFilterSSIMs)
	lfPos    int                   // values already committed
	commitX  int                   // macroblocks already committed
	finished bool                  // under mu: the row is coded
}

// wfState is the shared state of a wavefront pass.
type wfState struct {
	e        *encoder
	mbW, mbH int
	rdOpt    int
	period   int   // cost tables refresh period, 0: none
	lf       bool  // record the autofilter statistics
	gap      int32 // see minWavefrontGap

	nextRow     atomic.Int32 // next row to take
	epoch       atomic.Int32 // cost table refreshes done
	committed   atomic.Int32 // rows committed
	stopped     atomic.Bool  // interrupted, or a worker panicked
	interrupted atomic.Bool

	rows  []wfRow
	slots []wfSlot
	stats [][numTypes]statsArray // the slots' statistics

	mu      sync.Mutex
	nw      int // registered workers (under mu)
	workers [maxWavefrontWorkers]wfWorker
}

// codePassWavefront is codePass on nw workers: the calling goroutine and
// up to nw-1 helpers. It reports ran == false, without coding anything,
// when the budget has no free slot; the caller then codes the pass itself.
func (e *encoder) codePassWavefront(nw int, isLastPass bool, maxCount int) (sizeP0, distortion uint64, ran bool, err error) {
	if workers.InUse() >= workers.Limit() {
		return 0, 0, false, nil
	}
	wf := wavefrontPool.Get().(*wfState)
	wf.init(e, nw, isLastPass, maxCount)
	w0 := wf.addWorker(&e.it)
	var (
		its   [maxWavefrontWorkers - 1]*iterator
		tasks [maxWavefrontWorkers - 1]*workers.Task
		n     int
	)
	for n+1 < nw {
		sit := sideIteratorPool.Get().(*iterator)
		sit.reset(e)
		w := wf.addWorker(sit)
		t := workers.Spawn(func() error {
			done := false
			defer func() {
				if !done { // panic or Goexit: let the others stop
					wf.stop()
				}
			}()
			wf.run(w)
			done = true
			return nil
		})
		if t == nil { // budget exhausted
			wf.removeWorker()
			sit.enc = nil
			sideIteratorPool.Put(sit)
			break
		}
		its[n], tasks[n] = sit, t
		n++
	}
	joined, interrupted := false, false
	join := func() {
		// Wait for every helper before recycling anything, then re-raise
		// a helper panic, if any.
		for _, t := range tasks[:n] {
			<-t.Done()
		}
		joined = true
		// Read only now: a helper may still poll Interrupt for the last
		// rows after the calling goroutine ran out of rows.
		interrupted = wf.interrupted.Load()
		for k := range wf.nw {
			e.mergeMaxEdge(wf.workers[k].it)
			sizeP0 += wf.workers[k].sizeP0
			distortion += wf.workers[k].dist
		}
		for _, sit := range its[:n] {
			sit.enc = nil
			sideIteratorPool.Put(sit)
		}
		wf.release()
		wavefrontPool.Put(wf)
		for _, t := range tasks[:n] {
			t.Wait()
		}
	}
	defer func() {
		if !joined { // the calling goroutine panicked
			wf.stop()
			join()
		}
	}()
	if n == 0 {
		// No helper: code the pass sequentially instead (the state is
		// recycled by join).
		join()
		return 0, 0, false, nil
	}
	wf.run(w0)
	join()
	if interrupted {
		return 0, 0, true, ErrInterrupted
	}
	return sizeP0, distortion, true, nil
}

func (wf *wfState) init(e *encoder, nw int, isLastPass bool, maxCount int) {
	wf.e = e
	wf.mbW, wf.mbH = e.mbW, e.mbH
	wf.rdOpt = e.rdOptLevel
	wf.period = 0
	if wf.rdOpt > rdOptNone {
		wf.period = maxCount + 1
	}
	wf.lf = isLastPass && e.lfStats != nil
	wf.gap = int32(max(minWavefrontGap, wf.mbW/nw))
	wf.nextRow.Store(0)
	wf.epoch.Store(0)
	wf.committed.Store(0)
	wf.stopped.Store(false)
	wf.interrupted.Store(false)
	wf.rows = wfRowPool.Get(wf.mbH)
	for r := range wf.rows {
		wf.rows[r].prog.Store(0)
		wf.rows[r].want.Store(noWant)
	}
	nslots := min(nw+2, wf.mbH)
	if cap(wf.slots) < nslots {
		wf.slots = make([]wfSlot, nslots)
	}
	wf.slots = wf.slots[:nslots]
	wf.stats = wfStatsPool.Get(nslots)
	for i := range wf.slots {
		s := &wf.slots[i]
		s.stats = &wf.stats[i]
		s.mbEnd = int32Pool.Get(wf.mbW + 1)
		if wf.lf {
			s.lf = lfValuePool.Get(wf.mbW * maxFilterSSIMs)
		}
		s.finished = false
	}
	wf.nw = 0
}

// release returns the slots' buffers to their pools.
func (wf *wfState) release() {
	for i := range wf.slots {
		s := &wf.slots[i]
		s.tokens.release()
		int32Pool.Put(s.mbEnd)
		lfValuePool.Put(s.lf)
		s.mbEnd, s.lf, s.stats = nil, nil, nil
	}
	wfStatsPool.Put(wf.stats)
	wfRowPool.Put(wf.rows)
	wf.stats, wf.rows = nil, nil
	for k := range wf.workers {
		wf.workers[k].it = nil
	}
	wf.e = nil
}

// addWorker registers a worker coding with it.
func (wf *wfState) addWorker(it *iterator) *wfWorker {
	wf.mu.Lock()
	defer wf.mu.Unlock()
	w := &wf.workers[wf.nw]
	w.it = it
	w.cond.L = &wf.mu
	w.sleeping = false
	w.sizeP0, w.dist = 0, 0
	wf.nw++
	return w
}

// removeWorker unregisters the last worker (its helper could not start).
func (wf *wfState) removeWorker() {
	wf.mu.Lock()
	wf.nw--
	wf.workers[wf.nw].it = nil
	wf.mu.Unlock()
}

// run codes rows until there are none left or the pass stopped.
func (wf *wfState) run(w *wfWorker) {
	e := wf.e
	it := w.it
	rd := &it.rd
	nslots := len(wf.slots)
	// The sums stay in registers rather than in w, next to the other
	// workers' fields.
	var sizeP0, dist uint64
	defer func() { w.sizeP0, w.dist = sizeP0, dist }()
	var seen wfSeen
	for {
		r := int(wf.nextRow.Add(1)) - 1
		if r >= wf.mbH || wf.stopped.Load() {
			return
		}
		// Row r reuses the slot of row r-nslots: wait until that row is
		// committed.
		if r >= nslots && !wf.await(w, -1, 0, 0, 0, int32(r-nslots+1)) {
			return
		}
		if e.cfg.Interrupt != nil && e.cfg.Interrupt() {
			wf.interrupted.Store(true)
			wf.stop()
			return
		}
		if testHookWavefrontRow != nil {
			testHookWavefrontRow(r)
		}
		s := &wf.slots[r%nslots]
		s.tokens.reset()
		*s.stats = [numTypes]statsArray{}
		s.mbEnd[0] = 0
		s.lf = s.lf[:0]
		s.lfPos, s.commitX = 0, 0
		seen.above = 0
		it.setRow(r)
		for x := range wf.mbW {
			if !wf.awaitMB(w, r, x, &seen) {
				return
			}
			it.importMB()
			it.decimate(rd, wf.rdOpt)
			it.recordTokens(rd, &s.tokens, s.stats)
			s.mbEnd[x+1] = int32(s.tokens.len())
			sizeP0 += uint64(rd.H)
			dist += uint64(rd.D)
			if wf.lf {
				s.lf = it.appendFilterSSIMs(s.lf)
			}
			it.saveBoundary()
			it.nextInRow()
			wf.publish(r, int32(x+1))
		}
		wf.finishRow(r)
	}
}

// awaitMB waits until macroblock x of row r may be coded: row r-1 has
// coded macroblock x+1 and the cost tables are those of the macroblock's
// epoch. At a refresh point it refreshes them. It returns false if the
// pass stopped.
func (wf *wfState) awaitMB(w *wfWorker, r, x int, seen *wfSeen) bool {
	epoch := int32(0)
	if wf.period > 0 {
		i := r*wf.mbW + x
		epoch = int32((i + 1) / wf.period)
		if (i+1)%wf.period == 0 {
			return wf.refresh(w, r, x, epoch)
		}
	}
	need := int32(min(x+2, wf.mbW))
	if r > 0 && seen.above < need {
		seen.above = wf.rows[r-1].prog.Load()
	}
	if seen.epoch < epoch {
		seen.epoch = wf.epoch.Load()
	}
	if (r == 0 || seen.above >= need) && seen.epoch >= epoch && !wf.stopped.Load() {
		return true
	}
	if !wf.await(w, r-1, need, min(int32(x)+wf.gap, int32(wf.mbW)), epoch, 0) {
		return false
	}
	if r > 0 {
		seen.above = wf.rows[r-1].prog.Load()
	}
	seen.epoch = wf.epoch.Load()
	return true
}

// refresh refreshes the cost tables before macroblock x of row r, a
// refresh point, as the sequential pass does: from the statistics of all
// earlier macroblocks, which it first waits for (the rows above) and
// commits (this row's).
func (wf *wfState) refresh(w *wfWorker, r, x int, epoch int32) bool {
	if !wf.await(w, -1, 0, 0, epoch-1, int32(r)) {
		return false
	}
	wf.commit(r, x)
	p := &wf.e.proba
	p.finalizeTokenProbas()
	p.calculateLevelCosts()
	wf.mu.Lock()
	wf.epoch.Store(epoch)
	wf.wakeReadyLocked()
	wf.mu.Unlock()
	return true
}

// publish records that row r has coded n macroblocks, waking the worker of
// row r+1 if it sleeps until then.
func (wf *wfState) publish(r int, n int32) {
	row := &wf.rows[r]
	row.prog.Store(n)
	if n >= row.want.Load() {
		wf.mu.Lock()
		row.want.Store(noWant)
		wf.wakeReadyLocked()
		wf.mu.Unlock()
	}
}

// finishRow marks row r coded and commits it, with the coded rows below
// it, if the rows above are committed; otherwise the worker committing
// the row above will.
func (wf *wfState) finishRow(r int) {
	nslots := len(wf.slots)
	wf.mu.Lock()
	wf.slots[r%nslots].finished = true
	mine := int(wf.committed.Load()) == r
	wf.mu.Unlock()
	for mine && !wf.stopped.Load() {
		wf.commit(r, wf.mbW)
		wf.mu.Lock()
		wf.slots[r%nslots].finished = false
		r++
		wf.committed.Store(int32(r))
		wf.wakeReadyLocked()
		mine = r < wf.mbH && wf.slots[r%nslots].finished
		wf.mu.Unlock()
	}
}

// commit commits macroblocks [commitX, x1) of row r: it appends their
// tokens to e.tokens, accumulates their token statistics and autofilter
// values. Rows are committed in raster order, as the sequential pass
// records them.
func (wf *wfState) commit(r, x1 int) {
	e := wf.e
	s := &wf.slots[r%len(wf.slots)]
	x0 := s.commitX
	if x0 >= x1 {
		return
	}
	base := r * wf.mbW
	off := int32(e.tokens.len()) - s.mbEnd[x0]
	for x := x0; x < x1; x++ {
		e.mbTokStart[base+x] = s.mbEnd[x] + off
	}
	start, end := int(s.mbEnd[x0]), int(s.mbEnd[x1])
	for i := start; i < end; {
		c := s.tokens.chunk(i, end)
		e.tokens.appendTokens(c)
		i += len(c)
	}
	if !mergeStats(&e.proba.stats, s.stats) {
		s.tokens.recordStatsRange(start, end, &e.proba.stats)
	}
	if x1 < wf.mbW {
		// Count the rest of the row from zero.
		*s.stats = [numTypes]statsArray{}
	}
	if wf.lf {
		for x := x0; x < x1; x++ {
			s.lfPos = e.addFilterSSIMs(&e.mbInfo[base+x], s.lf, s.lfPos)
		}
	}
	s.commitX = x1
}

// ready reports whether w's wait is over (under mu).
func (wf *wfState) ready(w *wfWorker) bool {
	if wf.stopped.Load() {
		return true
	}
	return (w.above < 0 || wf.rows[w.above].prog.Load() >= w.need) &&
		wf.epoch.Load() >= w.epochNeed && wf.committed.Load() >= w.commitNeed
}

// await blocks until row above (if >= 0) has coded need macroblocks, epoch
// epochNeed has begun and commitNeed rows are committed, or the pass
// stopped (it then returns false). Waiting for a row's progress, it sleeps
// until the row has coded target (>= need) macroblocks.
func (wf *wfState) await(w *wfWorker, above int, need, target, epochNeed, commitNeed int32) bool {
	wf.mu.Lock()
	w.above, w.need, w.epochNeed, w.commitNeed = above, need, epochNeed, commitNeed
	if above >= 0 {
		// Stored before ready loads the progress, which publish stores
		// before loading want: either this wait sees the progress or
		// publish sees want.
		wf.rows[above].want.Store(target)
	}
	for !wf.ready(w) {
		w.sleeping = true
		w.cond.Wait()
	}
	w.sleeping = false
	if above >= 0 {
		wf.rows[above].want.Store(noWant)
	}
	wf.mu.Unlock()
	return !wf.stopped.Load()
}

// wakeReadyLocked wakes the sleeping workers whose wait is over (under mu).
func (wf *wfState) wakeReadyLocked() {
	for k := range wf.nw {
		w := &wf.workers[k]
		if w.sleeping && wf.ready(w) {
			w.sleeping = false
			w.cond.Signal()
		}
	}
}

// stop makes every worker return as soon as possible.
func (wf *wfState) stop() {
	wf.stopped.Store(true)
	wf.mu.Lock()
	wf.wakeReadyLocked()
	wf.mu.Unlock()
}

// mergeStats adds the statistics local, counted from zero, to global if
// recording the same events into global one by one gives the same result:
// when none of global's counters reaches the point where recordStats
// halves it (and local's never did). Otherwise it leaves global unchanged
// and returns false.
func mergeStats(global, local *[numTypes]statsArray) bool {
	ok := true
	for t := range local {
		for b := range local[t] {
			for c := range local[t][b] {
				g, l := &global[t][b][c], &local[t][b][c]
				for p := range l {
					// recordStats halves a counter before an event when its
					// total is 0xfffe; a local total of 0x7fff or more may
					// have been halved already.
					n := l[p] >> 16
					ok = ok && n < 0x7fff && g[p]>>16+n <= 0xfffe
				}
			}
		}
	}
	if !ok {
		return false
	}
	for t := range local {
		for b := range local[t] {
			for c := range local[t][b] {
				g, l := &global[t][b][c], &local[t][b][c]
				for p := range l {
					// Totals and counts of ones add separately: the counts
					// never exceed the totals, which stay below 0x10000.
					g[p] += l[p]
				}
			}
		}
	}
	return true
}
