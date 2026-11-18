//go:build arm64 && !purego

package vp8l

// NEON kernels of the inverse transforms and the output conversion
// (dsp_arm64.s). The wrappers below check every length and pass the
// assembly exact pixel counts: it reads and writes only inside the slices,
// tails included. The pure-Go references (the ...Go functions) are the
// specification; dsp_test.go compares the two.

const haveAsm = true

//go:noescape
func emitRGBAAsm(src *uint32, dst *byte, n int)

//go:noescape
func emitAddGreenAsm(src *uint32, dst *byte, n int)

//go:noescape
func addGreenAsm(in, out *uint32, n int)

//go:noescape
func lookupBytesAsm(pal *[256]byte, in, out *byte, n int)

//go:noescape
func crossColorRowAsm(mults, src, dst *uint32, width, bits int)

//go:noescape
func predictRowAsm(in, out, up, modes *uint32, x, width, bits int) int

//go:noescape
func expandBundled8Asm(pal *[256]byte, src, dst *byte, n, bits int)

//go:noescape
func expandBundledRGBAAsm(pal *[256]uint32, src, dst *byte, n, bits int)

// emitRGBA converts the ARGB pixels src to RGBA bytes in dst (as many
// pixels as fit).
func emitRGBA(src []uint32, dst []byte) {
	if n := min(len(src), len(dst)/4); n > 0 {
		emitRGBAAsm(&src[0], &dst[0], n)
	}
}

// emitAddGreen fuses the inverse subtract-green transform with emitRGBA.
func emitAddGreen(src []uint32, dst []byte) {
	if n := min(len(src), len(dst)/4); n > 0 {
		emitAddGreenAsm(&src[0], &dst[0], n)
	}
}

// addGreen undoes the subtract-green transform. in may alias out.
func addGreen(in, out []uint32) {
	out = out[:len(in)]
	if len(in) > 0 {
		addGreenAsm(&in[0], &out[0], len(in))
	}
}

// lookupBytes sets out[i] = pal[in[i]] (len(out) >= len(in)), 16 bytes at
// a time in the assembly, the last len(in)%16 in Go.
func lookupBytes(pal *[256]byte, in, out []byte) {
	out = out[:len(in)]
	n := len(in) &^ 15
	if n > 0 {
		lookupBytesAsm(pal, &in[0], &out[0], n)
	}
	for i := n; i < len(in); i++ {
		out[i] = pal[in[i]]
	}
}

// crossColorRow undoes the cross-color transform on one row src into dst,
// with the multipliers of each tile of 1<<bits pixels in mults. src may
// alias dst.
func crossColorRow(mults, src, dst []uint32, bits int) {
	width := len(src)
	if width == 0 {
		return
	}
	dst = dst[:width]
	if bits < 0 || bits > 30 {
		panic("vp8l: invalid cross-color tile bits")
	}
	mults = mults[:subSampleSize(width, bits)]
	crossColorRowAsm(&mults[0], &src[0], &dst[0], width, bits)
}

// predictRow undoes the predictor transform for the pixels 1 to
// len(out)-1 of a row whose pixel 0 is already done, with the mode of each
// tile of 1<<bits pixels in modes. up is the previous output row extended
// by out[0] (see predictRun). in may alias out.
//
// The assembly handles every mode but 12 and 13 (rare, and no faster in
// NEON than the channel-parallel Go): it returns at the first tile of
// those, which predictRun then does before the assembly goes on.
func predictRow(modes, in, out, up []uint32, bits int) {
	width := len(out)
	if width < 2 {
		return
	}
	if bits < 0 || bits > 30 {
		panic("vp8l: invalid predictor tile bits")
	}
	in = in[:width]
	up = up[:width+1]
	modes = modes[:subSampleSize(width, bits)]
	x := 1
	for {
		x = predictRowAsm(&in[0], &out[0], &up[0], &modes[0], x, width, bits)
		if x >= width {
			return
		}
		xEnd := min((x>>bits+1)<<bits, width)
		predictRun(modes[x>>bits]>>8&0xf, in, out, up, x, xEnd)
		x = xEnd
	}
}

// expandBundledRow expands one row of bundled indices (each byte of src
// holds 1<<bits of them, bits 1 to 3) into the output bytes dst, from the
// palette a.pal8 (at most 16 entries). The assembly does whole blocks of
// 16 packed bytes, the Go reference the rest of the row.
func expandBundledRow(a *alphaTables, bits int, src, dst []byte) {
	n := min(len(dst)>>bits, len(src)) &^ 15
	if n > 0 && bits >= 1 && bits <= 3 {
		expandBundled8Asm(&a.pal8, &src[0], &dst[0], n, bits)
		src, dst = src[n:], dst[n<<bits:]
	}
	expandBundledRowGo(a, bits, src, dst)
}

// lookupRGBA writes the RGBA words pal[src[i]] to dst (as many pixels as
// fit): the Go reference, since a 256-color palette does not fit NEON
// tables.
func lookupRGBA(pal *[256]uint32, src, dst []byte) { lookupRGBAGo(pal, src, dst) }

// expandBundledRGBA expands one row of bundled palette indices (each byte
// of src holds 1<<bits of them, bits 1 to 3) into the RGBA bytes dst (as
// many pixels as fit), from the RGBA palette pal (at most 16 entries). For
// bits 1 and 2 the assembly does whole blocks of 16 packed bytes; the Go
// reference (with tab, see bundleTable) does the rest of the row, and
// 1-bit indices, where the assembly gains little.
func expandBundledRGBA(pal *[256]uint32, tab []uint64, bits int, src, dst []byte) {
	n := min(len(dst)/(4<<bits), len(src)) &^ 15
	if n > 0 && (bits == 1 || bits == 2) {
		expandBundledRGBAAsm(pal, &src[0], &dst[0], n, bits)
		src, dst = src[n:], dst[n*(4<<bits):]
	}
	expandBundledRGBAGo(tab, bits, src, dst)
}
