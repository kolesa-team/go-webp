package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): src/dec/frame_dec.c (mt_method 2 worker split).

import (
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// mtSlots is the number of macroblock row batches in the UseThreads ring:
// the parser fills one batch while the helper reconstructs the other.
const mtSlots = 2

// mtBatchMBs is the number of macroblocks (rounded up to whole rows) handed
// from the parser to the helper goroutine at a time. Each hand-off may park
// and wake an OS thread (tens of microseconds, much more on some Linux VMs),
// so a batch must carry enough work, about a millisecond, to amortize it.
// Pictures with less than two batches are decoded without a helper.
// Variable for tests.
var mtBatchMBs = 512

// mtBatchRows returns the number of rows of the batch that starts at row y,
// for full batches of k rows. The first batches ramp up (1, 1, 2, 4, ...
// rows) so that the helper starts early, and the last ones ramp down (half
// of the remaining rows) so that the parser does not wait for a long last
// batch: the pipeline fills and drains in about one row instead of one
// batch, which matters for pictures of a few batches.
func (d *decoder) mtBatchRows(y, k int) int {
	return min(k, max(1, y), max(1, (d.brMbY-y+1)/2))
}

// mtJob is a batch of parsed macroblock rows [y, y+n) handed to the helper:
// row i is blocks[i*mbW:][:mbW].
type mtJob struct {
	y, n   int
	blocks []mbBlock
}

// decodeFrameThreaded is the UseThreads variant of decodeFrame (libwebp's
// mt_method 2): the calling goroutine parses batches of macroblock rows
// (modes and coefficients) while a helper goroutine reconstructs, filters,
// dithers and converts the previously parsed batch. Every stage runs in the
// same order as in decodeFrame and the helper makes the same sink calls,
// so the output, and the rows delivered before an error or an early stop,
// are identical. When the process-wide helper budget is exhausted the frame
// is decoded by decodeFrame on the calling goroutine.
func (d *decoder) decodeFrameThreaded() error {
	k := max(1, (mtBatchMBs+d.mbW-1)/d.mbW) // rows per batch
	if d.brMbY < 2*k {
		return d.decodeFrame() // nothing to overlap
	}
	jobs := make(chan mtJob, mtSlots)
	free := make(chan struct{}, mtSlots)
	var stop atomic.Bool
	t := workers.Spawn(func() error { return d.reconBatches(jobs, free, &stop) })
	if t == nil {
		return d.decodeFrame()
	}
	waited := false
	defer func() {
		if !waited { // the parser panicked: stop and join the helper, keep the panic
			stop.Store(true)
			close(jobs)
			t.Panicked()
		}
	}()

	batch := k * d.mbW
	if n := mtSlots * batch; cap(d.mtBlocks) < n {
		d.mtBlocks = make([]mbBlock, n)
	} else {
		d.mtBlocks = d.mtBlocks[:n]
	}
	var perr error
	inFlight, slot := 0, 0
parse:
	for y := 0; y < d.brMbY && perr == nil && !stop.Load(); {
		if inFlight == mtSlots {
			// Wait for the oldest batch; the helper may also have returned
			// (error, early stop or panic) without freeing it.
			select {
			case <-free:
				inFlight--
			case <-t.Done():
				break parse
			}
		}
		blocks := d.mtBlocks[slot*batch:][:batch]
		rows := d.mtBatchRows(y, k)
		n := 0
		for ; n < rows && !stop.Load(); n++ {
			if perr = d.parseRow(y+n, blocks[n*d.mbW:][:d.mbW]); perr != nil {
				break
			}
		}
		if n == 0 {
			break
		}
		// The rows parsed before an error are still emitted, as in
		// decodeFrame. inFlight < mtSlots, so the send does not block.
		select {
		case jobs <- mtJob{y: y, n: n, blocks: blocks}:
		case <-t.Done():
			break parse
		}
		inFlight++
		slot = (slot + 1) % mtSlots
		y += n
	}
	close(jobs)
	waited = true
	if err := t.Wait(); err != nil {
		return err // from an earlier row than perr
	}
	if perr != nil && d.rowsOut < d.sink.NeedRows() {
		// decodeFrame would have stopped before parsing the broken row
		// if the sink needed no more rows.
		return perr
	}
	return nil
}

// reconBatches is the helper goroutine of decodeFrameThreaded. It returns
// at the first error, or when the sink needs no more rows, without freeing
// the batch it was working on; stop tells the parser to stop early (and the
// helper, if the parser panicked).
func (d *decoder) reconBatches(jobs <-chan mtJob, free chan<- struct{}, stop *atomic.Bool) error {
	defer stop.Store(true)
	for j := range jobs {
		for i := 0; i < j.n; i++ {
			if d.rowsOut >= d.sink.NeedRows() || stop.Load() {
				return nil
			}
			if err := d.reconRow(j.y+i, j.blocks[i*d.mbW:][:d.mbW]); err != nil {
				return err
			}
		}
		free <- struct{}{} // never blocks: at most mtSlots batches are in flight
	}
	return nil
}

// parseRow parses one macroblock row into blocks.
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

// reconRow reconstructs, filters and emits a parsed macroblock row.
func (d *decoder) reconRow(mbY int, blocks []mbBlock) error {
	d.initRowEdges(mbY)
	for mbX := range blocks {
		d.reconMB(mbX, mbY, &blocks[mbX])
	}
	if d.dither {
		d.ditherCurrentRow()
	}
	return d.finishRow(mbY)
}
