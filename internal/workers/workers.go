// Package workers is the process-wide budget for the helper goroutines the
// codec starts on its own (decoder.Options.UseThreads, encoder ThreadLevel).
//
// Every helper goroutine must be started through Spawn. The budget is
// GOMAXPROCS-1 helpers in total across all concurrent Encode/Decode calls
// (GOMAXPROCS is re-read on every call), so a server that already runs one
// request per core does not get oversubscribed: when no slot is free, Spawn
// returns nil and the caller must do the work itself, on its own goroutine
// (every parallel code path therefore keeps a sequential fallback).
//
// A panic in a helper is recovered, so that it cannot crash the process from
// a goroutine the caller cannot recover in; Task.Wait re-raises it on the
// caller's goroutine as a *PanicError, the same place a panic of the
// sequential code path would surface.
package workers

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync/atomic"
)

var (
	inUse atomic.Int64
	// limit overrides the budget when > 0 (tests).
	limit atomic.Int64
)

// Limit returns the current budget: the number of helper goroutines that may
// run at the same time in the whole process.
func Limit() int {
	if l := limit.Load(); l > 0 {
		return int(l)
	}
	return runtime.GOMAXPROCS(0) - 1
}

// SetLimitForTest overrides the budget (n > 0) or restores the default
// (n <= 0) and returns the previous override. Tests only.
func SetLimitForTest(n int) int {
	if n < 0 {
		n = 0
	}
	return int(limit.Swap(int64(n)))
}

// InUse returns the number of helper goroutines currently running.
func InUse() int { return int(inUse.Load()) }

// TryAcquire reserves one helper slot if the budget allows. A successful call
// must be paired with Release. Prefer Spawn, which pairs them and recovers
// panics.
func TryAcquire() bool {
	max := int64(Limit())
	for {
		n := inUse.Load()
		if n >= max {
			return false
		}
		if inUse.CompareAndSwap(n, n+1) {
			return true
		}
	}
}

// Release returns a slot obtained with TryAcquire.
func Release() { inUse.Add(-1) }

// PanicError carries a panic recovered in a helper goroutine.
type PanicError struct {
	Value any
	Stack []byte
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("webp: panic in helper goroutine: %v\n%s", e.Value, e.Stack)
}

// Unwrap returns the panic value when it is an error.
func (e *PanicError) Unwrap() error {
	err, _ := e.Value.(error)
	return err
}

// Task is a helper goroutine started by Spawn.
type Task struct {
	done chan struct{}
	err  error
	pe   *PanicError
}

// Spawn starts f on a new helper goroutine when the budget has a free slot
// and returns its Task; otherwise it returns nil without running f, and the
// caller must do the work itself.
//
// The slot is released when f returns or panics. The caller must call Wait
// (typically deferred) before returning, so no helper outlives the
// Encode/Decode call that started it.
func Spawn(f func() error) *Task {
	if !TryAcquire() {
		return nil
	}
	t := &Task{done: make(chan struct{})}
	go t.run(f)
	return t
}

// ErrGoexit is the PanicError value of a helper that called runtime.Goexit
// (for example testing.T.FailNow in a callback such as a sink): its work is
// incomplete, so Wait must not report it as a success.
var ErrGoexit = errors.New("runtime.Goexit called in a helper goroutine")

func (t *Task) run(f func() error) {
	defer close(t.done)
	defer Release()
	returned := false
	defer func() {
		if r := recover(); r != nil {
			t.pe = &PanicError{Value: r, Stack: debug.Stack()}
		} else if !returned {
			t.pe = &PanicError{Value: ErrGoexit, Stack: debug.Stack()}
		}
	}()
	t.err = f()
	returned = true
}

// Done is closed when the helper has returned (normally or by panicking).
// Producers blocked on sending to a helper should also select on Done, so
// that a failed helper cannot deadlock them.
func (t *Task) Done() <-chan struct{} { return t.done }

// Wait blocks until the helper has returned and returns f's error. If f
// panicked, or exited through runtime.Goexit (PanicError.Value ErrGoexit),
// Wait panics with the *PanicError on the calling goroutine.
func (t *Task) Wait() error {
	<-t.done
	if t.pe != nil {
		panic(t.pe)
	}
	return t.err
}

// Panicked reports, after Done is closed, whether f panicked or called
// runtime.Goexit (without re-raising it). It lets deferred cleanup run
// before Wait re-panics.
func (t *Task) Panicked() bool {
	<-t.done
	return t.pe != nil
}
