package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): src/dec/frame_dec.c (mt_method 2 worker split).

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// The UseThreads pipeline (decodeFrameThreaded) has up to three stages:
//
//   - the calling goroutine parses the coefficient tokens of batches of
//     macroblock rows;
//   - a reconstruction helper predicts, adds the residuals, filters and
//     dithers each batch, and converts its rows to the output (libwebp's
//     mt_method 2 split);
//   - when the reconstruction helper is the slower stage (the parser has to
//     wait for it), the conversion moves to a second, emit helper, which
//     converts a batch while the next one is reconstructed into the other
//     of two row caches.
//
// The first partition (the per-macroblock modes, which the token parser
// needs) is parsed ahead by whichever helper is idle: with SIMD kernels the
// token parser is the slowest stage, and moving the mode parsing off it
// shortens the critical path by about a fifth. A row whose modes no helper
// has parsed yet is parsed by the token parser itself, so the helpers only
// ever use idle time.
//
// Every stage runs the same operations in the same order as decodeFrame and
// the sink sees the same calls, so the output, and the rows delivered
// before an error or an early stop, are identical. The helpers come from
// the process-wide budget of internal/workers; without a free slot the frame
// is decoded on the calling goroutine (and without a second one, the
// conversion stays on the reconstruction helper).
//
// Every wait is a blocking channel receive. A stage that wakes a blocked
// one hands it the processor while the process has idle ones (see
// parked.handOff).

// mtSlots is the number of macroblock row batches in the UseThreads ring:
// the parser fills one batch while the helper reconstructs the other. It is
// also the number of row caches of the emit stage.
const mtSlots = 2

// mtBatchMBs is the number of macroblocks (rounded up to whole rows) handed
// from the parser to the helper goroutine at a time. Each hand-off may park
// and wake an OS thread (tens of microseconds, much more on some Linux VMs),
// so a batch must carry enough work, about a millisecond, to amortize it.
// Pictures with less than two batches are decoded without a helper.
// Variable for tests.
var mtBatchMBs = 512

// mtEmitWaits is the number of batches for which the parser must wait for
// the reconstruction helper before the conversion moves to an emit helper;
// 0 starts the emit helper with the first batch, a negative value never
// starts it. Variable for tests.
var mtEmitWaits = 2

// mtBatchRows returns the number of rows of the batch that starts at row y,
// for full batches of k rows. The first batches ramp up (1, 1, 2, 4, ...
// rows) so that the helper starts early, and the last ones ramp down (half
// of the remaining rows) so that the parser does not wait for a long last
// batch: the pipeline fills and drains in about one row instead of one
// batch, which matters for pictures of a few batches.
func (d *decoder) mtBatchRows(y, k int) int {
	return min(k, max(1, y), max(1, (d.brMbY-y+1)/2))
}

// mtJob is a batch of parsed macroblock rows [y, y+n) handed to the
// reconstruction helper: row i is blocks[i*mbW:][:mbW]. A non-nil emit
// hands the conversion of this batch and the following ones to the emit
// helper.
type mtJob struct {
	y, n   int
	blocks []mbBlock
	emit   *mtEmitter
}

// mtEmitter is the emit helper: it converts the batches of reconstructed
// rows [y, y+n) that the reconstruction helper leaves in the row cache
// slots[slot].
type mtEmitter struct {
	task   *workers.Task
	jobs   chan mtEmitJob // closed by its owner
	free   chan int       // slots the emit helper has converted
	parked parked         // the emit helper waits for a batch
	// owned is set when the reconstruction helper has received the
	// emitter: it then closes jobs when it returns. Until then the parser
	// does, after the reconstruction helper has returned.
	owned bool
}

type mtEmitJob struct{ y, n, slot int }

// mtCache is one row cache of the emit stage: the extraRows rows the next
// batch's filter still modifies, then 16 rows per macroblock row of the
// batch (same strides as the decoder's own cache).
type mtCache struct{ y, u, v []byte }

// mtState is the state shared by the stages of decodeFrameThreaded.
type mtState struct {
	jobs chan mtJob
	free chan struct{}
	stop atomic.Bool // set when a stage stops early: the others stop too
	// The parser waits for a free batch, the reconstruction helper for a
	// batch or a free row cache.
	parserParked, reconParked parked
	blocks                    []mbBlock // ring of parsed row batches
	caches                    [mtSlots]mtCache

	// Mode parsing: rows [0, modeNext) of the first partition are parsed,
	// into modes (a ring of modeRows rows of macroblock modes; eof tells
	// whether the partition was exhausted after the row). Any goroutine may
	// parse the next row while holding modeMu; modeReady publishes
	// modeNext, modeUsed the number of rows the token parser has consumed.
	modeMu    sync.Mutex
	modeNext  int
	modeReady atomic.Int64
	modeUsed  atomic.Int64
	modeRows  int
	modes     []mbModes
	eof       []bool

	procs int // GOMAXPROCS when the decode started (see parked.handOff)
}

// parked is set while a pipeline stage is blocked waiting for another
// stage.
//
// When a goroutine wakes a blocked one (a channel send) and keeps running,
// the runtime queues the woken goroutine in the waker's own next-to-run
// slot, from which other threads only take it after a back-off: the woken
// stage then starts tens of microseconds late on darwin, and hundreds on
// Linux (measured: a median of 160-440 us in Docker linux/arm64 with 4
// CPUs, against 7-9 us with handOff). The waker therefore yields once
// after waking a parked stage, so that the woken stage runs at once on
// this thread while the waker continues on another one (idle threads take
// it from the global run queue without the back-off). This is no wait:
// Gosched returns at once when nothing else is runnable.
type parked struct{ atomic.Bool }

// wait runs the blocking receive f with p set.
func (p *parked) wait(f func()) {
	p.Store(true)
	f()
	p.Store(false)
}

// handOff is called after a send that may have woken the stage that p
// belongs to. procs is the decode's GOMAXPROCS: when about as many
// pipelines run as there are processors, none is idle to take the waker
// over, and the yield would only add a context switch (see
// workers.SpareProcs).
func (p *parked) handOff(procs int) {
	if p.Load() && workers.SpareProcs(procs) {
		runtime.Gosched()
	}
}

// testHookTokenRow, if set (tests), runs before the token parser parses a
// row with UseThreads.
var testHookTokenRow func(d *decoder, mbY int)

// decodeFrameThreaded is the UseThreads variant of decodeFrame (see the
// comment at the top of the file). When the process-wide helper budget is
// exhausted the frame is decoded by decodeFrame on the calling goroutine.
func (d *decoder) decodeFrameThreaded() error {
	k := max(1, (mtBatchMBs+d.mbW-1)/d.mbW) // rows per batch
	if d.brMbY < 2*k {
		return d.decodeFrame() // nothing to overlap
	}
	if d.mt == nil {
		d.mt = new(mtState)
	}
	m := d.mt
	m.init(d.mbW, k)
	m.procs = runtime.GOMAXPROCS(0)
	workers.BeginHandOffs()
	defer workers.EndHandOffs()
	t := workers.Spawn(d.reconBatches)
	if t == nil {
		return d.decodeFrame()
	}
	var e *mtEmitter // the emit helper, if started
	sendEmit := false
	joined := false
	join := func() {
		close(m.jobs)
		t.Panicked()
		if e != nil {
			if !e.owned {
				close(e.jobs)
			}
			e.task.Panicked()
		}
		joined = true
	}
	defer func() {
		if !joined { // the parser panicked: stop and join the helpers, keep the panic
			m.stop.Store(true)
			join()
		}
	}()

	batch := k * d.mbW
	var perr error
	inFlight, slot, waits := 0, 0, 0
parse:
	for y := 0; y < d.brMbY && perr == nil && !m.stop.Load(); {
		if e == nil && mtEmitWaits >= 0 && waits >= mtEmitWaits {
			// The reconstruction helper is the slower stage: move the
			// conversion to a helper of its own, if the budget allows.
			if e = d.startEmitter(); e != nil {
				sendEmit = true
			} else {
				waits = -1 << 30 // do not try again
			}
		}
		if inFlight == mtSlots {
			// Wait for the oldest batch; the helper may also have returned
			// (error, early stop or panic) without freeing it.
			select {
			case <-m.free:
			default:
				if y >= 2*k {
					// Past the first, short batches, which are mostly
					// waiting for the helper to wake up.
					waits++
				}
				done := false
				m.parserParked.wait(func() {
					select {
					case <-m.free:
					case <-t.Done():
						done = true
					}
				})
				if done {
					break parse
				}
			}
			inFlight--
		}
		blocks := m.blocks[slot*batch:][:batch]
		rows := d.mtBatchRows(y, k)
		n := 0
		for ; n < rows && !m.stop.Load(); n++ {
			if perr = d.parseTokenRow(y+n, blocks[n*d.mbW:][:d.mbW]); perr != nil {
				break
			}
		}
		if n == 0 {
			break
		}
		// The rows parsed before an error are still emitted, as in
		// decodeFrame. inFlight < mtSlots, so the send does not block.
		j := mtJob{y: y, n: n, blocks: blocks}
		if sendEmit {
			j.emit = e
		}
		select {
		case m.jobs <- j:
			sendEmit = false
			m.reconParked.handOff(m.procs)
		case <-t.Done():
			break parse
		}
		inFlight++
		slot = (slot + 1) % mtSlots
		y += n
	}
	join()
	// The reconstruction helper converts the rows before those of the
	// emit helper, so its error (or panic) comes first; either comes from
	// an earlier row than perr.
	if err := t.Wait(); err != nil {
		return err
	}
	if e != nil {
		if err := e.task.Wait(); err != nil {
			return err
		}
	}
	if perr != nil && d.rowsOut < d.sink.NeedRows() {
		// decodeFrame would have stopped before parsing the broken row
		// if the sink needed no more rows.
		return perr
	}
	return nil
}

// init sizes and resets the pipeline state for a frame of mbW macroblock
// columns and batches of k rows.
func (m *mtState) init(mbW, k int) {
	m.jobs = make(chan mtJob, mtSlots)
	m.free = make(chan struct{}, mtSlots)
	m.stop.Store(false)
	if n := mtSlots * k * mbW; cap(m.blocks) < n {
		m.blocks = make([]mbBlock, n)
	} else {
		m.blocks = m.blocks[:n]
	}
	// The token parser consumes about a batch of mode rows while the
	// helpers work on the previous one; two batches let an idle helper
	// keep ahead of it.
	m.modeRows = 2*k + 2
	if n := m.modeRows * mbW; cap(m.modes) < n {
		m.modes = make([]mbModes, n)
	} else {
		m.modes = m.modes[:n]
	}
	if cap(m.eof) < m.modeRows {
		m.eof = make([]bool, m.modeRows)
	}
	m.eof = m.eof[:m.modeRows]
	m.modeNext = 0
	m.modeReady.Store(0)
	m.modeUsed.Store(0)
}

// parseModeRow parses the modes of macroblock row y into the mode ring.
// The caller holds modeMu.
func (d *decoder) parseModeRow(y int) {
	m := d.mt
	r := y % m.modeRows
	modes := m.modes[r*d.mbW:][:d.mbW]
	d.intraL = [4]uint8{}
	for x := range modes {
		d.parseIntraMode(x, &modes[x])
	}
	m.eof[r] = d.br.eof
	m.modeNext = y + 1
	m.modeReady.Store(int64(y + 1))
}

// parseModesAhead parses the modes of the next macroblock row, unless
// another goroutine is parsing modes, the ring is full or the frame is
// done; it reports whether it parsed a row. Idle helpers call it.
func (d *decoder) parseModesAhead() bool {
	m := d.mt
	if !m.modeMu.TryLock() {
		return false
	}
	defer m.modeMu.Unlock()
	y := m.modeNext
	if y >= d.brMbY || y >= int(m.modeUsed.Load())+m.modeRows || m.stop.Load() {
		return false
	}
	d.parseModeRow(y)
	return true
}

// parseTokenRow parses the coefficients of macroblock row mbY into blocks,
// with the modes from the mode ring (parsing them first if no helper has).
func (d *decoder) parseTokenRow(mbY int, blocks []mbBlock) error {
	m := d.mt
	if testHookTokenRow != nil {
		testHookTokenRow(d, mbY)
	}
	if int(m.modeReady.Load()) <= mbY {
		m.modeMu.Lock()
		for m.modeNext <= mbY {
			d.parseModeRow(m.modeNext)
		}
		m.modeMu.Unlock()
	}
	r := mbY % m.modeRows
	modes := m.modes[r*d.mbW:][:d.mbW]
	tokenBR := &d.parts[mbY&d.numPartsMinusOne]
	d.nzL, d.nzDCL = 0, 0
	for mbX := range blocks {
		blk := &blocks[mbX]
		blk.mbModes = modes[mbX]
		if err := d.parseTokens(mbX, blk, tokenBR); err != nil {
			return err
		}
	}
	eof := m.eof[r]
	m.modeUsed.Store(int64(mbY + 1)) // the ring row may be reused
	if eof {
		return status.NotEnoughData("premature end-of-partition0 encountered")
	}
	return nil
}

// nextJob receives the next value from jobs. While none is ready, the
// caller (an idle helper, p) parses modes ahead for the token parser.
func nextJob[T any](d *decoder, jobs <-chan T, p *parked) (j T, ok bool) {
	for {
		select {
		case j, ok = <-jobs:
			return j, ok
		default:
		}
		if !d.parseModesAhead() {
			p.wait(func() { j, ok = <-jobs })
			return j, ok
		}
	}
}

// reconBatches is the reconstruction helper of decodeFrameThreaded. It
// returns at the first error, or when the sink needs no more rows, without
// freeing the batch it was working on; stop tells the parser to stop early
// (and the helper, if the parser panicked).
func (d *decoder) reconBatches() error {
	m := d.mt
	var e *mtEmitter
	cacheY, cacheU, cacheV := d.cacheY, d.cacheU, d.cacheV
	done := false // all batches reconstructed: the emit helper finishes its own
	defer func() {
		if !done {
			m.stop.Store(true)
		}
		if e != nil {
			close(e.jobs)
			d.cacheY, d.cacheU, d.cacheV = cacheY, cacheU, cacheV
		}
	}()
	// carry holds the extraRows rows the next batch's filter still
	// modifies (on top of the last row cache used).
	carry := mtCache{cacheY, cacheU, cacheV}
	for {
		j, ok := nextJob(d, m.jobs, &m.reconParked)
		if !ok {
			done = true
			return nil
		}
		if j.emit != nil {
			e = j.emit
			e.owned = true
		}
		if e == nil {
			for i := 0; i < j.n; i++ {
				if d.rowsOut >= d.sink.NeedRows() || m.stop.Load() {
					return nil
				}
				d.reconRow(j.y+i, j.blocks[i*d.mbW:][:d.mbW])
				if err := d.finishRow(j.y + i); err != nil {
					return err
				}
			}
			m.free <- struct{}{} // never blocks: at most mtSlots batches are in flight
			m.parserParked.handOff(m.procs)
			continue
		}
		s := -1
		m.reconParked.wait(func() {
			select {
			case s = <-e.free:
			case <-e.task.Done():
			}
		})
		if s < 0 {
			return nil // stopped early (stop is set) or panicked
		}
		c := &m.caches[s]
		d.cacheY, d.cacheU, d.cacheV = c.y, c.u, c.v
		extra := d.extraRows
		copy(c.y[:extra*d.cacheYStride], carry.y)
		copy(c.u[:extra/2*d.cacheUVStride], carry.u)
		copy(c.v[:extra/2*d.cacheUVStride], carry.v)
		for i := 0; i < j.n; i++ {
			if m.stop.Load() {
				return nil
			}
			d.cacheYOff = (extra + 16*i) * d.cacheYStride
			d.cacheUVOff = (extra/2 + 8*i) * d.cacheUVStride
			d.reconRow(j.y+i, j.blocks[i*d.mbW:][:d.mbW])
		}
		m.free <- struct{}{}
		m.parserParked.handOff(m.procs)
		carry = mtCache{c.y[16*j.n*d.cacheYStride:], c.u[8*j.n*d.cacheUVStride:], c.v[8*j.n*d.cacheUVStride:]}
		e.jobs <- mtEmitJob{y: j.y, n: j.n, slot: s} // never blocks: the slot was free
		e.parked.handOff(m.procs)
	}
}

// startEmitter starts the emit helper, if the budget allows. It owns the
// output (sink, alpha, rowsOut, upsampling state) from the first batch it
// receives on.
func (d *decoder) startEmitter() *mtEmitter {
	m := d.mt
	k := len(m.blocks) / mtSlots / d.mbW
	extra := d.extraRows
	for i := range m.caches {
		c := &m.caches[i]
		c.y = grow(c.y, (extra+16*k)*d.cacheYStride)
		c.u = grow(c.u, (extra/2+8*k)*d.cacheUVStride)
		c.v = grow(c.v, (extra/2+8*k)*d.cacheUVStride)
	}
	e := &mtEmitter{jobs: make(chan mtEmitJob, mtSlots), free: make(chan int, mtSlots)}
	for i := range mtSlots {
		e.free <- i
	}
	if e.task = workers.Spawn(func() error { return d.emitBatches(e) }); e.task == nil {
		return nil
	}
	return e
}

// emitBatches is the emit helper: it converts the rows of each batch as
// finishRow does.
func (d *decoder) emitBatches(e *mtEmitter) error {
	m := d.mt
	defer m.stop.Store(true)
	for {
		j, ok := nextJob(d, e.jobs, &e.parked)
		if !ok {
			return nil
		}
		c := &m.caches[j.slot]
		for i := 0; i < j.n; i++ {
			if d.rowsOut >= d.sink.NeedRows() || m.stop.Load() {
				return nil
			}
			yo, uvo := 16*i*d.cacheYStride, 8*i*d.cacheUVStride
			if err := d.emitRow(j.y+i, c.y[yo:], c.u[uvo:], c.v[uvo:]); err != nil {
				return err
			}
		}
		e.free <- j.slot // never blocks
		m.reconParked.handOff(m.procs)
	}
}

// parseRow parses the modes and coefficients of one macroblock row into
// blocks (benchmarks).
func (d *decoder) parseRow(mbY int, blocks []mbBlock) error {
	tokenBR := &d.parts[mbY&d.numPartsMinusOne]
	d.startParseRow()
	for mbX := range blocks {
		if err := d.parseMB(mbX, &blocks[mbX], tokenBR); err != nil {
			return err
		}
	}
	if d.br.eof {
		return status.NotEnoughData("premature end-of-partition0 encountered")
	}
	return nil
}

// reconRow reconstructs, filters and dithers a parsed macroblock row into
// the row cache.
func (d *decoder) reconRow(mbY int, blocks []mbBlock) {
	d.initRowEdges(mbY)
	for mbX := range blocks {
		d.reconMB(mbX, mbY, &blocks[mbX])
	}
	if d.dither {
		d.ditherCurrentRow()
	}
}
