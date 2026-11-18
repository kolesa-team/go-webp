//go:build !amd64 || purego

package pipeline

import "testing"

// forEachLevel runs f once: only amd64 has more than one kernel level.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	f(t)
}

// eachLevel runs f once.
func eachLevel(f func()) { f() }

// benchLevels returns the single selected level, "kernel".
func benchLevels() ([]string, func(i int)) { return []string{"kernel"}, func(int) {} }

// restoreLevel does nothing: there is a single level.
func restoreLevel() {}
