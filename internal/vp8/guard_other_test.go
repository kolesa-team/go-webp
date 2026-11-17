//go:build !darwin && !linux

package vp8

import "testing"

// guardedBytes returns a plain n-byte slice where guard pages are not
// available.
func guardedBytes(t testing.TB, n int, atEnd bool) []byte {
	return make([]byte, n)
}
