//go:build !arm64 || purego

package vp8

// filterMB applies the in-loop filter to one macroblock; see filterMBGo.
func (d *decoder) filterMB(mbX, mbY int, f fInfo) {
	d.filterMBGo(mbX, mbY, f)
}
