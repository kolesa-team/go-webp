//go:build !amd64 || purego

package vp8lenc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// testLevels are the kernel levels the tests run: only the Go code here.
func testLevels() []cpuinfo.Level { return []cpuinfo.Level{cpuinfo.Go} }

func setLevel(cpuinfo.Level) (restore func()) { return func() {} }

func setPrefetch(int) (restore func()) { return func() {} }
