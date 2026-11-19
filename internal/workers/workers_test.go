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

package workers

import (
	"errors"
	"runtime"
	"sync"
	"testing"
)

func TestSpawnRunsAndReturnsError(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(4))
	want := errors.New("boom")
	task := Spawn(func() error { return want })
	if task == nil {
		t.Fatal("Spawn returned nil with free budget")
	}
	if err := task.Wait(); err != want {
		t.Fatalf("Wait = %v, want %v", err, want)
	}
	if n := InUse(); n != 0 {
		t.Fatalf("InUse after Wait = %d, want 0", n)
	}
}

func TestSpawnRespectsBudget(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(2))
	release := make(chan struct{})
	var tasks []*Task
	for i := 0; i < 2; i++ {
		task := Spawn(func() error { <-release; return nil })
		if task == nil {
			t.Fatalf("Spawn %d returned nil within budget", i)
		}
		tasks = append(tasks, task)
	}
	if task := Spawn(func() error { t.Error("ran over budget"); return nil }); task != nil {
		t.Fatal("Spawn over budget returned a task")
	}
	close(release)
	for _, task := range tasks {
		if err := task.Wait(); err != nil {
			t.Fatal(err)
		}
	}
	if task := Spawn(func() error { return nil }); task == nil {
		t.Fatal("slot not released")
	} else {
		task.Wait()
	}
}

func TestSpawnPanicIsRecoveredAndReraised(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(1))
	task := Spawn(func() error { panic("bad input") })
	if task == nil {
		t.Fatal("Spawn returned nil")
	}
	<-task.Done()
	if !task.Panicked() {
		t.Fatal("Panicked = false")
	}
	if n := InUse(); n != 0 {
		t.Fatalf("slot leaked after panic: InUse = %d", n)
	}
	defer func() {
		r := recover()
		pe, ok := r.(*PanicError)
		if !ok || pe.Value != "bad input" || len(pe.Stack) == 0 {
			t.Fatalf("recovered %#v, want *PanicError{bad input}", r)
		}
	}()
	task.Wait()
	t.Fatal("Wait did not re-panic")
}

func TestDefaultBudgetFollowsGOMAXPROCS(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(0)) // default budget
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	// GOMAXPROCS=1 gives a budget of 0: everything runs inline.
	if Limit() != 0 {
		t.Fatalf("Limit = %d with GOMAXPROCS=1, want 0", Limit())
	}
	if task := Spawn(func() error { return nil }); task != nil {
		task.Wait()
		t.Fatal("Spawn started a helper with GOMAXPROCS=1")
	}
	runtime.GOMAXPROCS(4)
	if Limit() != 3 {
		t.Fatalf("Limit = %d with GOMAXPROCS=4, want 3", Limit())
	}
}

func TestConcurrentAcquireNeverExceedsLimit(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(3))
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		peak int
	)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if TryAcquire() {
					mu.Lock()
					if n := InUse(); n > peak {
						peak = n
					}
					mu.Unlock()
					Release()
				}
			}
		}()
	}
	wg.Wait()
	if peak > 3 {
		t.Fatalf("peak %d > limit 3", peak)
	}
	if InUse() != 0 {
		t.Fatalf("InUse = %d after all releases", InUse())
	}
}

// TestSpawnGoexitIsReported: a helper that calls runtime.Goexit (such as
// t.FailNow in a callback) did not finish its work; Wait must re-raise that
// on the caller instead of reporting success.
func TestSpawnGoexitIsReported(t *testing.T) {
	defer SetLimitForTest(SetLimitForTest(4))
	task := Spawn(func() error {
		runtime.Goexit()
		return nil
	})
	if task == nil {
		t.Fatal("Spawn returned nil with free budget")
	}
	if !task.Panicked() {
		t.Fatal("Panicked = false after runtime.Goexit")
	}
	defer func() {
		pe, ok := recover().(*PanicError)
		if !ok || !errors.Is(pe, ErrGoexit) {
			t.Fatalf("Wait re-raised %v, want a *PanicError wrapping ErrGoexit", pe)
		}
		if n := InUse(); n != 0 {
			t.Fatalf("InUse = %d, want 0", n)
		}
	}()
	task.Wait()
	t.Fatal("Wait returned after runtime.Goexit")
}

func TestSpareProcs(t *testing.T) {
	if n := HandOffs(); n != 0 {
		t.Fatalf("HandOffs = %d at start, want 0", n)
	}
	if !SpareProcs(1) || !SpareProcs(4) {
		t.Fatal("SpareProcs = false with no call in progress")
	}
	for range 2 {
		BeginHandOffs()
	}
	if !SpareProcs(4) {
		t.Fatal("SpareProcs(4) = false with 2 calls in progress")
	}
	if SpareProcs(3) {
		t.Fatal("SpareProcs(3) = true with 2 calls in progress")
	}
	for range 2 {
		EndHandOffs()
	}
	if n := HandOffs(); n != 0 {
		t.Fatalf("HandOffs = %d after EndHandOffs, want 0", n)
	}
}
