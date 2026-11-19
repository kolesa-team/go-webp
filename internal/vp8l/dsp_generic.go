//go:build (!arm64 && !amd64) || purego

package vp8l

// Without the assembly kernels of dsp_arm64.s and dsp_*amd64.s, the per-row
// kernels are their pure-Go references.

const haveAsm = false

func emitRGBA(src []uint32, dst []byte)                { emitRGBAGo(src, dst) }
func emitAddGreen(src []uint32, dst []byte)            { emitAddGreenGo(src, dst) }
func addGreen(in, out []uint32)                        { addGreenGo(in, out) }
func lookupBytes(pal *[256]byte, in, out []byte)       { lookupBytesGo(pal, in, out) }
func lookupRGBA(pal *[256]uint32, src, dst []byte)     { lookupRGBAGo(pal, src, dst) }
func crossColorRow(mults, src, dst []uint32, bits int) { crossColorRowGo(mults, src, dst, bits) }
func predictRow(modes, in, out, up []uint32, bits int) { predictRowGo(modes, in, out, up, bits) }

func expandBundledRow(a *alphaTables, bits int, src, dst []byte) {
	expandBundledRowGo(a, bits, src, dst)
}

func expandBundledRGBA(pal *[256]uint32, tab []uint64, bits int, src, dst []byte) {
	expandBundledRGBAGo(tab, bits, src, dst)
}
