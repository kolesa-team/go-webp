//go:build !race

package vp8lenc

import "testing"

const raceEnabled = false

func skipUnderRace(*testing.T) {}
