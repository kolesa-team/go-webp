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

package vp8l

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// Two-stage decoding (Options.UseThreads).
//
// The calling goroutine entropy-decodes the level-0 pixels (decodePixels)
// while a helper goroutine runs the inverse transforms and the output
// conversion (emitRows) on the rows decoded so far, and writes them into the
// sink. Rows are handed over in jobs of about jobPixels image pixels,
// zero-copy: a job is a slice of the level-0 buffer, which the producer does
// not overwrite until the helper has acknowledged the job (see beforeSlide).
//
// The helper processes the jobs in order, in the same row batches as the
// sequential path, so the output does not depend on whether, or when, the
// helper runs. Errors keep their sequential order: a sink error of the
// helper is reported before a later bitstream error of the producer, whose
// pending rows are handed over before stopping (finishAsync).
//
// libwebp ignores use_threads for lossless images; the helper only starts
// when it can pay for itself and the process-wide helper budget
// (internal/workers) has a free slot.
//
// A hand-off costs the producer a channel send and, the helper being idle
// between jobs, a wake-up: on Linux (EPYC 8024P) the helper starts a job
// about 60 microseconds after the send, on darwin (Apple M) 15-25. The
// pipeline saves at most the helper's share of the work, which with the
// assembly kernels is small: about 0.15 ns per pixel for palette images
// (the 3000x2000 logo: 0.8 ms of 3.7, -12% with the helper on one L3 of
// the EPYC, -26% on darwin), 1 ns for photos (-6..-9%, -7..-14% on
// darwin), and almost nothing for pictures made of long LZ77 copies of
// one color (+7..+18% with the helper for such 0.25-1 MP pictures, on
// both systems). So the producer first emits asyncProbePixels pixels
// itself, timing the emission, and starts the helper only if the rest of
// the picture projects to at least asyncMinWork of emission work
// (probeDone): those pictures project to 25-140 us, the smallest gain
// measured (a 2048x1024 part of the logo, -5% on Linux) to 220 us.
//
// And on Linux the helper must share the last-level cache with the
// producer (asyncSharedLLC): the rows it reads and writes were last
// touched by the producer (the level-0 window, the output zeroed by its
// allocation), and with the whole EPYC, two 16 MB L3 of 4 cores each, the
// 3000x2000 logo decoded 10% slower with the helper, with 42% more CPU
// time (while on one L3 it decodes 12-24% faster).
var (
	// asyncMinPixels is the smallest image (in needed pixels) decoded in
	// two stages. A variable for tests.
	asyncMinPixels = 256 * 1024
	// asyncProbePixels is the number of pixels the producer emits, timed,
	// before the helper may start; 0 starts it at once. A variable for
	// tests.
	asyncProbePixels = 32 * 1024
	// asyncMinWork is the least emission work, projected from the probe
	// for the rest of the picture, for which the helper starts. A
	// variable for tests.
	asyncMinWork = 200 * time.Microsecond
	// asyncSharedLLC reports whether the CPUs this process may run on
	// share one last-level cache (sharedLLC, evaluated once). A variable
	// for tests.
	asyncSharedLLC = sync.OnceValue(sharedLLC)
	// asyncJobPixels, when > 0, overrides the job size (jobPixels). A
	// variable for tests.
	asyncJobPixels = 0
	// asyncTestHook, if set, runs on the producer before each job is
	// handed over (tests: producer panics).
	asyncTestHook func(y0, y1 int)
	// asyncProbeHook, if set, receives the projection of each probe
	// (tests).
	asyncProbeHook func(projected time.Duration)
)

// jobPixels returns the number of level-0 pixels per job for an image of
// total needed pixels coded in dataLen bytes. A hand-over costs a channel
// operation and, when the helper waits, a wake-up (tens of microseconds on
// some systems), so a job should be about half a millisecond of work. The
// work per pixel grows with the coded bits per pixel: about 1 ns per pixel
// plus 2 ns per bit (logos and palette images are near 1 ns, photos above
// 10 ns). At least 16 jobs keep the pipeline fill and drain short.
func jobPixels(total, dataLen int) int {
	if asyncJobPixels > 0 {
		return asyncJobPixels
	}
	const targetNs = 500_000
	bits1024 := int(min(int64(dataLen)*8*1024/int64(max(total, 1)), 1<<20)) // bits per 1024 px
	j := targetNs * 1024 / (1024 + 2*bits1024)
	return max(min(j, total/16), 1)
}

// asyncProbe times the emission of the first rows of a picture that may be
// decoded in two stages (see probeDone).
type asyncProbe struct {
	active bool
	pixels int           // pixels emitted while probing
	time   time.Duration // time spent emitting them
}

// asyncWanted reports whether a two-stage decode may be started: the
// option, an RGBA output, a big enough picture, and a helper sharing the
// producer's last-level cache.
func (d *decoder) asyncWanted() bool {
	return d.useThreads && d.out == outRGBA && d.width*d.needRows >= asyncMinPixels && asyncSharedLLC()
}

// probeStart returns the time at which the producer starts emitting rows
// while probing, or the zero time.
func (d *decoder) probeStart() time.Time {
	if !d.probe.active {
		return time.Time{}
	}
	return time.Now()
}

// probeDone ends the timing begun at t0 (probeStart) of the emission of
// the rows up to d.lastRow, from y0. Once asyncProbePixels pixels are
// timed, it starts the helper if the emission of the rest projects to at
// least asyncMinWork.
func (d *decoder) probeDone(t0 time.Time, y0 int) {
	if t0.IsZero() {
		return
	}
	p := &d.probe
	p.pixels += (d.lastRow - y0) * d.width
	p.time += time.Since(t0)
	if p.pixels < asyncProbePixels {
		return
	}
	p.active = false
	rest := (d.needRows - d.lastRow) * d.width
	projected := time.Duration(int64(p.time) * int64(rest) / int64(max(p.pixels, 1)))
	if asyncProbeHook != nil {
		asyncProbeHook(projected)
	}
	if rest > 0 && projected >= asyncMinWork {
		d.startAsync()
	}
}

// maxAsyncJobs bounds the jobs handed to the helper and not yet
// acknowledged.
const maxAsyncJobs = 4

// rowJob is rows [y0, y1) of level-0 pixels, starting at argb[0], or at
// idx[0] for index bytes (idx8.go).
type rowJob struct {
	argb   []uint32
	idx    []byte
	y0, y1 int
}

// asyncEmit is the producer side of a two-stage decode.
type asyncEmit struct {
	task *workers.Task
	jobs chan rowJob
	acks chan struct{}
	// stop asks the helper to return without processing further jobs
	// (abort after a producer panic).
	stop atomic.Bool
	// starts holds the level-0 buffer index of the first pixel of each job
	// in flight, oldest first (a ring of inFlight entries from head).
	starts   [maxAsyncJobs]int
	head     int
	inFlight int
	// pendingRow is the last row boundary passed to processRows: rows
	// [d.lastRow, pendingRow) are decoded but not yet handed over.
	pendingRow int
	// guard, when > 0, is a buffer index the producer must not write at or
	// past until the guardJobs oldest jobs (handed over before the last
	// window slide, and still reading the buffer from guard on) are done.
	guard, guardJobs int
	jobPixels        int
}

// startAsync starts the helper goroutine of a two-stage decode, if the
// image is big enough and the worker budget allows. It returns false when
// the decode must run sequentially.
func (d *decoder) startAsync() bool {
	if !d.asyncWanted() {
		return false
	}
	a := &asyncEmit{
		jobs:      make(chan rowJob, maxAsyncJobs),
		acks:      make(chan struct{}, maxAsyncJobs),
		jobPixels: jobPixels(d.width*d.needRows, len(d.br.data)),
	}
	if d.sliding {
		// Pending rows stay in the window: keep them well below the room
		// a slide leaves (see initBuffer32). The window counts level-0
		// pixels, at most as many as image pixels.
		a.jobPixels = min(a.jobPixels, d.window/4)
	}
	a.task = workers.Spawn(func() error { return d.consume(a) })
	if a.task == nil {
		return false
	}
	d.async = a
	return true
}

// consume is the helper goroutine: it emits the rows of each job in
// numCacheRows batches and acknowledges the job.
func (d *decoder) consume(a *asyncEmit) error {
	w := d.xsize
	for j := range a.jobs {
		if a.stop.Load() {
			return nil
		}
		for y := j.y0; y < j.y1; y += numCacheRows {
			y1 := min(y+numCacheRows, j.y1)
			var err error
			if j.idx != nil {
				err = d.emitIdx8(j.idx[(y-j.y0)*w:(y1-j.y0)*w], y, y1)
			} else {
				err = d.emitRows(j.argb[(y-j.y0)*w:(y1-j.y0)*w], y, y1)
			}
			if err != nil {
				return err
			}
		}
		// Never blocks: at most maxAsyncJobs jobs are unacknowledged.
		a.acks <- struct{}{}
	}
	return nil
}

// sendRows hands rows [d.lastRow, row) of the level-0 buffer (d.buf, or
// d.ib with d.idx8), which starts at pixel base, over to the helper once
// they make a big enough job (or always with flush, or at the last needed
// row).
func (d *decoder) sendRows(base, row int, flush bool) error {
	a := d.async
	y0 := d.lastRow
	if row <= y0 {
		return nil
	}
	w := d.xsize
	if !flush && row < d.needRows && (row-y0)*d.width < a.jobPixels {
		a.pendingRow = row
		return nil
	}
	if a.inFlight == maxAsyncJobs {
		if err := a.waitOne(); err != nil {
			return err
		}
	}
	if asyncTestHook != nil {
		asyncTestHook(y0, row)
	}
	start := y0*w - base
	j := rowJob{y0: y0, y1: row}
	if d.idx8 {
		j.idx = d.ib[start : row*w-base]
	} else {
		j.argb = d.buf[start : row*w-base]
	}
	// Never blocks: there is room for maxAsyncJobs jobs.
	a.jobs <- j
	a.starts[(a.head+a.inFlight)%maxAsyncJobs] = start
	a.inFlight++
	a.pendingRow = row
	d.lastRow = row
	return nil
}

// waitOne waits for the oldest job in flight to be acknowledged. If the
// helper has stopped (it failed), it returns the helper's error.
func (a *asyncEmit) waitOne() error {
	select {
	case <-a.acks:
		a.ack()
		return nil
	case <-a.task.Done():
		return a.helperError()
	}
}

// poll accounts for the jobs acknowledged so far, without blocking.
func (a *asyncEmit) poll() {
	for {
		select {
		case <-a.acks:
			a.ack()
		default:
			return
		}
	}
}

func (a *asyncEmit) ack() {
	a.head = (a.head + 1) % maxAsyncJobs
	a.inFlight--
	if a.guardJobs > 0 {
		if a.guardJobs--; a.guardJobs == 0 {
			a.guard = 0
		}
	}
}

// helperError returns the error the stopped helper returned. A helper that
// stops before its jobs channel is closed has failed; Wait re-raises its
// panic, if any, on this goroutine.
func (a *asyncEmit) helperError() error {
	if err := a.task.Wait(); err != nil {
		return err
	}
	// Unreachable: the helper only returns early with an error or panic.
	return errHelperStopped
}

var errHelperStopped = errors.New("vp8l: helper goroutine stopped")

// waitGuard waits until the producer may write past a.guard.
func (a *asyncEmit) waitGuard() error {
	for a.guardJobs > 0 {
		if err := a.waitOne(); err != nil {
			return err
		}
	}
	return nil
}

// beforeSlide is called before the window slide copies the kept pixels to
// buf[:destEnd]: the copy must not overwrite pixels the helper still reads.
// If the oldest job in flight starts at or past destEnd, the slide can go
// ahead, and the producer must then stop before that start (a.guard) until
// those jobs are done; otherwise beforeSlide waits for them.
func (a *asyncEmit) beforeSlide(destEnd int) error {
	a.poll()
	if a.inFlight == 0 {
		return nil
	}
	if oldest := a.starts[a.head]; destEnd <= oldest {
		a.guard, a.guardJobs = oldest, a.inFlight
		return nil
	}
	for a.inFlight > 0 {
		if err := a.waitOne(); err != nil {
			return err
		}
	}
	return nil
}

// finishAsync ends a two-stage decode whose producer returned err: on error,
// the rows the sequential decoder would have emitted are handed over first;
// then the helper is stopped and waited for. A helper error comes first in
// the sequential order, so it takes precedence over err.
func (d *decoder) finishAsync(err error) error {
	a := d.async
	if err != nil && a.pendingRow > d.lastRow {
		// The rows up to the last batch boundary the producer passed.
		if e := d.sendRows(d.px.base, a.pendingRow, true); e != nil {
			err = e
		}
	}
	d.async = nil
	close(a.jobs)
	if herr := a.task.Wait(); herr != nil {
		return herr
	}
	return err
}

// abortAsync stops the helper after the producer panicked (deferred; a
// no-op once finishAsync ran). It waits for the helper without re-raising
// a helper panic, so that the producer's panic propagates.
func (d *decoder) abortAsync() {
	a := d.async
	if a == nil {
		return
	}
	d.async = nil
	a.stop.Store(true)
	close(a.jobs)
	<-a.task.Done()
}
