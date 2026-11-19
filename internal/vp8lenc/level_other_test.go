//go:build !amd64 || purego

package vp8lenc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// testLevels are the kernel levels the tests run: only the Go code here.
func testLevels() []cpuinfo.Level { return []cpuinfo.Level{cpuinfo.Go} }

// kernelLevels are the levels with kernels of their own: only the Go code.
func kernelLevels() []cpuinfo.Level { return []cpuinfo.Level{cpuinfo.Go} }

func setLevel(cpuinfo.Level) (restore func()) { return func() {} }

func setPrefetch(int) (restore func()) { return func() {} }

func setAddRefsMin(int) (restore func()) { return func() {} }

func setLz77ReachMin(int) (restore func()) { return func() {} }

func setChunks(int) (restore func()) { return func() {} }
