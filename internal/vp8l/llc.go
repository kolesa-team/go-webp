package vp8l

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// sharedLLC reports whether the CPUs this process may run on share one
// last-level cache (see asyncSharedLLC). It answers true off Linux, and
// when the system does not say otherwise.
func sharedLLC() bool {
	if runtime.GOOS != "linux" {
		return true
	}
	return sharedLLCFrom(os.ReadFile)
}

// sharedLLCFrom is sharedLLC on Linux, reading the files with read: the
// online CPUs of Cpus_allowed_list (/proc/self/status) must all be in the
// shared_cpu_list of the deepest cache (level 3 or more) of the first of
// them (sysfs). Without that information (no L3 listed, as on some arm64
// servers whose system cache sysfs does not show, or files missing) it
// reports true.
func sharedLLCFrom(read func(string) ([]byte, error)) bool {
	status, err := read("/proc/self/status")
	if err != nil {
		return true
	}
	var list string
	for line := range strings.SplitSeq(string(status), "\n") {
		if v, ok := strings.CutPrefix(line, "Cpus_allowed_list:"); ok {
			list = v
			break
		}
	}
	allowed, ok := parseCPUList(list)
	if !ok || len(allowed) == 0 {
		return true
	}
	if b, err := read("/sys/devices/system/cpu/online"); err == nil {
		if online, ok := parseCPUList(string(b)); ok {
			for c := range allowed {
				if !online[c] {
					delete(allowed, c)
				}
			}
		}
	}
	first := -1
	for c := range allowed {
		if first < 0 || c < first {
			first = c
		}
	}
	if first < 0 {
		return true
	}
	level, shared := 0, ""
	for i := 0; i < 16; i++ {
		dir := fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cache/index%d/", first, i)
		b, err := read(dir + "level")
		if err != nil {
			break
		}
		l, err := strconv.Atoi(strings.TrimSpace(string(b)))
		if err != nil || l < 3 || l <= level {
			continue
		}
		if s, err := read(dir + "shared_cpu_list"); err == nil {
			level, shared = l, string(s)
		}
	}
	if level == 0 {
		return true
	}
	llc, ok := parseCPUList(shared)
	if !ok {
		return true
	}
	for c := range allowed {
		if !llc[c] {
			return false
		}
	}
	return true
}

// parseCPUList parses a Linux CPU list such as "0-3,8,10-11\n".
func parseCPUList(s string) (map[int]bool, bool) {
	set := map[int]bool{}
	s = strings.TrimSpace(s)
	if s == "" {
		return set, true
	}
	for part := range strings.SplitSeq(s, ",") {
		lo, hi, isRange := strings.Cut(strings.TrimSpace(part), "-")
		a, err := strconv.Atoi(lo)
		if err != nil || a < 0 {
			return nil, false
		}
		b := a
		if isRange {
			if b, err = strconv.Atoi(hi); err != nil || b < a {
				return nil, false
			}
		}
		if b-a > 1<<16 {
			return nil, false
		}
		for c := a; c <= b; c++ {
			set[c] = true
		}
	}
	return set, true
}
