package vp8

import (
	"errors"
	"image"
	"os"
	"testing"
	"time"

	"github.com/kolesa-team/go-webp/v2/internal/container"
	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

// failingSink returns an error from its second Commit.
type failingSink struct {
	*pipeline.Direct
	n int
}

var errSinkFailure = errors.New("sink failure")

func (s *failingSink) Commit(y, n int) error {
	s.n++
	if s.n == 2 {
		return errSinkFailure
	}
	return nil
}

// TestThreadedSinkError checks that a sink error stops a threaded decode
// (it used to deadlock waiting for one result too many).
func TestThreadedSinkError(t *testing.T) {
	data, err := os.ReadFile("testdata/grad_250x199_o9.webp")
	if err != nil {
		t.Fatal(err)
	}
	p, err := container.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, threads := range []bool{false, true} {
		img := image.NewNRGBA(image.Rect(0, 0, p.Width, p.Height))
		s := &failingSink{Direct: pipeline.NewDirect(img)}
		done := make(chan error, 1)
		go func() { done <- Decode(p.VP8, nil, &Options{UseThreads: threads}, s) }()
		select {
		case err := <-done:
			if !errors.Is(err, errSinkFailure) {
				t.Errorf("threads=%v: got %v, want the sink error", threads, err)
			}
		case <-time.After(10 * time.Second):
			t.Fatalf("threads=%v: Decode did not return after a sink error", threads)
		}
	}
}
