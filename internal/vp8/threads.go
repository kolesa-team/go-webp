package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): src/dec/frame_dec.c (mt_method 2 worker split).

import (
	"sync/atomic"

	"github.com/kolesa-team/go-webp/v2/internal/status"
)

// numRowSlots is the number of parsed macroblock rows that can be in flight
// between the parsing goroutine and the reconstruction goroutine.
const numRowSlots = 2

// decodeFrameThreaded is the UseThreads variant of decodeFrame (libwebp's
// mt_method 2): the calling goroutine parses macroblock rows (modes and
// coefficients) while a second goroutine reconstructs, filters, dithers and
// converts the previously parsed row. Every stage runs in the same order as
// in decodeFrame, so the output is identical.
func (d *decoder) decodeFrameThreaded() error {
	for i := range d.rowBlocks {
		if cap(d.rowBlocks[i]) < d.mbW {
			d.rowBlocks[i] = make([]mbBlock, d.mbW)
		}
		d.rowBlocks[i] = d.rowBlocks[i][:d.mbW]
	}
	jobs := make(chan int, numRowSlots)
	done := make(chan error, numRowSlots)
	var stop atomic.Bool
	go func() {
		var err error
		for mbY := range jobs {
			if err == nil && !stop.Load() {
				if d.rowsOut >= d.sink.NeedRows() {
					stop.Store(true)
				} else {
					err = d.reconRow(mbY, d.rowBlocks[mbY%numRowSlots])
				}
			}
			done <- err
		}
	}()

	var err error
	inFlight := 0
	for mbY := 0; mbY < d.brMbY; mbY++ {
		if inFlight == numRowSlots {
			// Account for the received result before checking it, so that
			// the drain loop below waits for exactly the rows still queued.
			err = <-done
			inFlight--
			if err != nil {
				break
			}
		}
		if stop.Load() {
			break
		}
		if err = d.parseRow(mbY, d.rowBlocks[mbY%numRowSlots]); err != nil {
			break
		}
		jobs <- mbY
		inFlight++
	}
	close(jobs)
	for ; inFlight > 0; inFlight-- {
		if e := <-done; err == nil {
			err = e
		}
	}
	return err
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
