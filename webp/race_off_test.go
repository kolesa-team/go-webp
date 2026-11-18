//go:build !race

package webp

import "testing"

const raceEnabled = false

func skipUnderRace(*testing.T) {}
