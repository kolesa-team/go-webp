package decoder

import (
	"sync"

	"github.com/kolesa-team/go-webp/v2/internal/alpha"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// With UseThreads a lossy+alpha decode moves the alpha plane (lossless
// decoding and unfiltering) to a helper goroutine, which works ahead of the
// rows the VP8 decoder emits: the alpha work then overlaps the VP8 work
// instead of adding to it. The helper decodes in steps of about
// alphaStepPixels pixels and publishes each step (an uncontended lock; the
// requester is only woken up when it waits), up to two bands of
// alphaBandPixels past the last request; it then sleeps until a whole band
// of work is available again, so that hand-offs that wake a goroutine stay
// coarse. A truncated VP8 stream thus costs at most a few bands of alpha
// decoding beyond the rows it emits.
//
// The output and the errors are exactly those of the sequential lazyAlpha:
//
//   - Decoding is deterministic, so the rows the helper publishes are the
//     ones the sequential path decodes.
//   - A failure is only known to lie in the step being decoded. The helper
//     then stops; the rows before the step stay valid, and a request beyond
//     them replays the requests made so far, in order, with a fresh
//     decoder on the requesting goroutine, which reproduces the sequential
//     outcome (including errors that depend on where a request ends, like
//     libwebp's end-of-stream check at the end of each
//     VP8DecompressAlphaRows call).
//   - A successful step means that every sequential request ending at or
//     before it succeeds: errors in the middle of a request depend only on
//     the position of the faulty data, and the end-of-stream condition
//     only grows as more data is read. One exception: a request that
//     reaches the end of the plane succeeds even after running out of data
//     (libwebp only reports that before the last pixel), so the helper
//     never decodes the last row; the request for it continues on the
//     requesting goroutine.
//
// Alpha dithering of quantized planes needs the whole plane at once and
// stays sequential.
//
// With a ring (see lazyAlpha), the helper's look-ahead bounds the rows it
// writes: they stay below the first row of the last request (which the
// requester may still read) plus the ring size, which ringRows makes
// larger than the look-ahead, a step and the rows of a request.

var (
	// alphaBandPixels is the amount of alpha work per wake-up of the
	// helper: the vp8l alpha decoder and the unfilter take 0.5 ns (palette
	// planes) to a few ns per pixel, so a band is at least half a
	// millisecond of work.
	alphaBandPixels = 1 << 20
	// alphaStepPixels is the amount of alpha work published at once.
	alphaStepPixels = 64 << 10
	// minHelperPixels is the smallest frame decoded with a helper: below it,
	// starting the helper costs about as much as the overlap saves.
	minHelperPixels = 256 << 10
)

// alphaHelper is the state shared by the helper goroutine and the
// requesting goroutine (the VP8 decoder's output stage).
type alphaHelper struct {
	task  *workers.Task
	step  int // rows decoded and published at once
	band  int // rows of work the sleeping helper waits for
	ahead int // rows the helper may decode past the last request
	limit int // the helper never decodes this row and the following ones (the last row of the plane, or the crop bottom)

	mu   sync.Mutex
	cond sync.Cond
	// Guarded by mu.
	requested int    // largest rowEnd requested
	published int    // rows [0, published) of pub are final
	pub       []byte // buffer holding the published rows
	idle      bool   // the helper sleeps (published is its row count)
	waiting   bool   // the requester waits for rows
	stop      bool   // set by release: exit after the current step
	failed    bool   // the step after published failed
	exited    bool   // the helper has returned (or panicked)

	// reqs are the distinct rowEnd values requested so far, in order (only
	// touched by the requesting goroutine).
	reqs []int
}

// errHelperPanic aborts a decode whose alpha helper panicked; the panic
// itself is re-raised on the decoding goroutine by release.
var errHelperPanic = status.New(utils.Vp8StatusBitstreamError, "alpha decoder failed")

// useHelper reports whether the alpha plane may be decoded by a helper.
func (a *lazyAlpha) useHelper() bool {
	if a.hdr.PreProcessing == alpha.PreprocessingLevels && a.dithering > 0 {
		return false
	}
	// Only the rows down to the bottom of the output region are decoded.
	return a.w*a.crop.Max.Y >= minHelperPixels && a.h > 1
}

// helperRows returns the rows of a step and of a band of the helper, for
// planes of width w.
func helperRows(w int) (step, band int) {
	band = max((alphaBandPixels+w-1)/w, 1)
	step = min(max((alphaStepPixels+w-1)/w, 1), band)
	return step, band
}

// startHelper starts the helper goroutine, if the worker budget allows;
// a.rows must be 0 (nothing decoded yet).
func (a *lazyAlpha) startHelper() bool {
	step, band := helperRows(a.w)
	// No request goes past the bottom of the output region (crop), so the
	// helper stops there instead of working ahead on rows nobody needs.
	limit := min(a.h-1, a.crop.Max.Y)
	hp := &alphaHelper{step: step, band: band, ahead: 2 * band, limit: limit,
		// The VP8 decoder requests rows once per macroblock row.
		reqs: make([]int, 0, a.h/16+4)}
	hp.cond.L = &hp.mu
	if hp.task = workers.Spawn(func() error { return a.runHelper(hp) }); hp.task == nil {
		return false
	}
	a.hp = hp
	a.async = true
	return true
}

// testHookHelperStep, if set (tests), runs on the helper before each step.
var testHookHelperStep func()

// runHelper is the helper goroutine: it owns the decoding state (dec,
// plane, pp, spare, rows) until it has exited.
func (a *lazyAlpha) runHelper(hp *alphaHelper) error {
	defer func() {
		hp.mu.Lock()
		hp.exited = true
		hp.cond.Broadcast()
		hp.mu.Unlock()
	}()
	// A ring sized for the look-ahead (allocated here, so that a decode
	// whose helper cannot start keeps the smaller sequential ring).
	a.alloc(true)
	for {
		hp.mu.Lock()
		if !hp.stop && a.rows < hp.limit && a.rows >= hp.requested+hp.ahead {
			// Far enough ahead: sleep until a band of work is available.
			hp.idle = true
			for !hp.stop && a.rows+hp.band > hp.requested+hp.ahead {
				hp.cond.Wait()
			}
			hp.idle = false
		}
		if hp.stop || a.rows >= hp.limit {
			hp.mu.Unlock()
			return nil
		}
		hp.mu.Unlock()
		if testHookHelperStep != nil {
			testHookHelperStep()
		}
		err := a.decodeRows(min(a.rows+hp.step, hp.limit))
		hp.mu.Lock()
		if err != nil {
			hp.failed = true
			hp.mu.Unlock()
			return nil
		}
		hp.published, hp.pub = a.rows, a.buf
		if hp.waiting {
			hp.cond.Broadcast()
		}
		hp.mu.Unlock()
	}
}

// getAsync is get while the helper decodes: it waits until the helper has
// published rowEnd rows. When the helper has exited without them (it
// reached its row limit, failed or panicked) the decode continues on the
// calling goroutine.
func (a *lazyAlpha) getAsync(rowStart, rowEnd int) ([]byte, int, error) {
	hp := a.hp
	if n := len(hp.reqs); n == 0 || rowEnd > hp.reqs[n-1] {
		hp.reqs = append(hp.reqs, rowEnd)
	}
	hp.mu.Lock()
	if rowEnd > hp.requested {
		hp.requested = rowEnd
		// A sleeping helper has published all its rows.
		if hp.idle && hp.published+hp.band <= hp.requested+hp.ahead {
			hp.cond.Broadcast()
		}
	}
	for hp.published < rowEnd && !hp.exited {
		hp.waiting = true
		hp.cond.Wait()
	}
	hp.waiting = false
	if hp.published >= rowEnd {
		p := hp.pub
		hp.mu.Unlock()
		// The helper only writes rows from published on, below rowStart
		// plus the ring size: not the rows of the window.
		return a.window(p, rowStart, rowEnd)
	}
	failed := hp.failed
	hp.mu.Unlock()

	// The helper is gone: the decoding state belongs to this goroutine.
	a.async = false
	if hp.task.Panicked() {
		a.err = errHelperPanic
		return nil, 0, a.err
	}
	if failed {
		a.err = a.replay(hp.reqs, rowStart)
	} else {
		a.err = a.decodeRequest(rowStart, rowEnd)
	}
	if a.err != nil {
		return nil, 0, a.err
	}
	return a.window(a.buf, rowStart, rowEnd)
}

// replay redoes the requests reqs from the first row with a fresh decoder,
// as the sequential path would have made them, and returns the first error.
// rowStart is the first row of the last request.
func (a *lazyAlpha) replay(reqs []int, rowStart int) error {
	if a.dec != nil {
		a.dec.Close()
		a.dec = nil
	}
	a.rows = 0
	if err := a.init(); err != nil {
		return err
	}
	for i, r := range reqs {
		// The rows before the last request are not needed any more: any
		// earlier row will do as the first row of a request.
		start := min(rowStart, r)
		if i < len(reqs)-1 {
			start = max(r-1, 0)
		}
		if err := a.decodeRequest(start, r); err != nil {
			return err
		}
	}
	return nil
}

// stopHelper makes the helper exit and waits for it. If it panicked, its
// decoding state is dropped (not pooled) and the panic is re-raised here.
func (a *lazyAlpha) stopHelper() {
	hp := a.hp
	if hp == nil {
		return
	}
	hp.mu.Lock()
	hp.stop = true
	hp.cond.Broadcast()
	hp.mu.Unlock()
	if hp.task.Panicked() {
		a.dec, a.pp, a.spare, a.buf = nil, nil, nil, nil
	}
	a.hp, a.async = nil, false
	_ = hp.task.Wait() // re-raises a panic of the helper
}
