//go:build !unix

package suite

import "time"

func cpuTime() (time.Duration, bool) { return 0, false }

func peakRSS() int64 { return 0 }
