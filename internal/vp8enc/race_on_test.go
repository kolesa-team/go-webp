//go:build race

package vp8enc

import "testing"

// raceEnabled reports whether the race detector is on. Under it the
// heaviest tests check fewer combinations, and tests that start no
// goroutines are skipped (the run without -race covers them): CI runs the
// race detector on 2-4 core runners.
const raceEnabled = true

// skipUnderRace skips a test that starts no goroutines, and so gains
// nothing from the race detector, when the detector is on.
func skipUnderRace(t *testing.T) {
	t.Helper()
	t.Skip("starts no goroutines; covered by the run without -race")
}
