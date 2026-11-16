//go:build race

package vp8

// raceEnabled reports whether the race detector is on (sync.Pool then
// drops items at random, which makes allocation counts meaningless).
const raceEnabled = true
