//go:build !unix

package vp8

import "time"

func processCPU() (time.Duration, bool) { return 0, false }
