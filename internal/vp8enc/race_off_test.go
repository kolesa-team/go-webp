//go:build !race

package vp8enc

import "testing"

const raceEnabled = false

func skipUnderRace(*testing.T) {}
