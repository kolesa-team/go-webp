//go:build amd64 && !purego

package vp8enc

import (
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/cpuinfo"
)

// kernelLevels returns the kernel levels this machine runs (cpuinfo.Go,
// the pure Go reference, first) and their names.
func kernelLevels() []cpuinfo.Level {
	var ls []cpuinfo.Level
	for _, l := range cpuinfo.Levels() {
		if l <= x86MaxLevel {
			ls = append(ls, l)
		}
	}
	return ls
}

// eachLevel calls f with every kernel level selected in turn (x86Level),
// then restores the default level. f gets the level's name for its
// failure messages.
func eachLevel(f func(level string)) {
	defer func() { x86Level = x86MaxLevel }()
	for _, l := range kernelLevels() {
		x86Level = l
		f(l.String())
	}
}

// forEachLevel runs f as a subtest per kernel level.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	defer func() { x86Level = x86MaxLevel }()
	for _, l := range kernelLevels() {
		x86Level = l
		t.Run(l.String(), f)
	}
}

// benchLevels runs f as a sub-benchmark per kernel level.
func benchLevels(b *testing.B, name string, f func()) {
	defer func() { x86Level = x86MaxLevel }()
	for _, l := range kernelLevels() {
		x86Level = l
		b.Run(name+"/"+l.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f()
			}
		})
	}
}
