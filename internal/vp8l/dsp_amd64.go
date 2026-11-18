//go:build amd64 && !purego

package vp8l

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

// SSE4.1, AVX2 and AVX-512 kernels of the inverse transforms and the output
// conversion (emit_amd64.s, predict_amd64.s, bundled_amd64.s,
// lookup_amd64.s, palette_amd64.s), selected by x86Level (and x86VBMI).
// The wrappers below check every length and pass the assembly exact pixel
// counts: it reads and writes only inside the slices, tails included. The
// pure-Go references (the ...Go functions) are the specification;
// dsp_test.go compares them with every level.
//
// Levels without a kernel of their own use the next lower one: AVX-512
// adds only the VBMI palette lookups (the other kernels are bound by
// loads and stores, or by serial dependencies, at 256 bits already).

const haveAsm = true

//go:noescape
func emitRGBASSE41(src *uint32, dst *byte, n int)

//go:noescape
func emitRGBAAVX2(src *uint32, dst *byte, n int)

//go:noescape
func emitAddGreenSSE41(src *uint32, dst *byte, n int)

//go:noescape
func emitAddGreenAVX2(src *uint32, dst *byte, n int)

//go:noescape
func addGreenSSE41(in, out *uint32, n int)

//go:noescape
func addGreenAVX2(in, out *uint32, n int)

//go:noescape
func crossColorRowSSE41(mults, src, dst *uint32, width, bits int)

//go:noescape
func crossColorRowAVX2(mults, src, dst *uint32, width, bits int)

//go:noescape
func predictRowSSE41(in, out, up, modes *uint32, x, width, bits int) int

//go:noescape
func lookupRGBAVBMI(pal *[256]uint32, src, dst *byte, n int)

//go:noescape
func lookupBytesVBMI(pal *[256]byte, in, out *byte, n int)

//go:noescape
func lookupBytesSSE41(pal *[256]byte, in, out *byte, n int)

//go:noescape
func lookupBytesAVX2(pal *[256]byte, in, out *byte, n int)

//go:noescape
func expandBundled8SSE41(pal *[256]byte, src, dst *byte, n, bits int)

//go:noescape
func expandBundled8AVX2(pal *[256]byte, src, dst *byte, n, bits int)

//go:noescape
func expandBundledRGBAAVX2(pal *[256]uint32, src, dst *byte, n, bits int)

// emitRGBA converts the ARGB pixels src to RGBA bytes in dst (as many
// pixels as fit).
func emitRGBA(src []uint32, dst []byte) {
	n := min(len(src), len(dst)/4)
	switch {
	case n == 0 || x86Level < cpuinfo.SSE41:
		emitRGBAGo(src, dst)
	case x86Level >= cpuinfo.AVX2:
		emitRGBAAVX2(&src[0], &dst[0], n)
	default:
		emitRGBASSE41(&src[0], &dst[0], n)
	}
}

// emitAddGreen fuses the inverse subtract-green transform with emitRGBA.
func emitAddGreen(src []uint32, dst []byte) {
	n := min(len(src), len(dst)/4)
	switch {
	case n == 0 || x86Level < cpuinfo.SSE41:
		emitAddGreenGo(src, dst)
	case x86Level >= cpuinfo.AVX2:
		emitAddGreenAVX2(&src[0], &dst[0], n)
	default:
		emitAddGreenSSE41(&src[0], &dst[0], n)
	}
}

// addGreen undoes the subtract-green transform. in may alias out.
func addGreen(in, out []uint32) {
	out = out[:len(in)]
	switch {
	case len(in) == 0 || x86Level < cpuinfo.SSE41:
		addGreenGo(in, out)
	case x86Level >= cpuinfo.AVX2:
		addGreenAVX2(&in[0], &out[0], len(in))
	default:
		addGreenSSE41(&in[0], &out[0], len(in))
	}
}

// lookupBytes sets out[i] = pal[in[i]] (len(out) >= len(in)).
func lookupBytes(pal *[256]byte, in, out []byte) {
	out = out[:len(in)]
	n := len(in) &^ 15
	switch {
	case len(in) == 0:
		return
	case x86Level >= cpuinfo.AVX512 && x86VBMI:
		lookupBytesVBMI(pal, &in[0], &out[0], len(in))
		return
	case n == 0:
	case x86Level >= cpuinfo.AVX2:
		lookupBytesAVX2(pal, &in[0], &out[0], n)
	case x86Level >= cpuinfo.SSE41:
		lookupBytesSSE41(pal, &in[0], &out[0], n)
	default:
		n = 0
	}
	lookupBytesGo(pal, in[n:], out[n:])
}

// lookupRGBAMinVBMI is the shortest row for lookupRGBAVBMI, which first
// transposes the palette (about as long as 40 pixels of the Go loop).
const lookupRGBAMinVBMI = 64

// lookupRGBA writes the RGBA words pal[src[i]] to dst (as many pixels as
// fit).
func lookupRGBA(pal *[256]uint32, src, dst []byte) {
	n := min(len(src), len(dst)/4)
	if n >= lookupRGBAMinVBMI && x86Level >= cpuinfo.AVX512 && x86VBMI {
		lookupRGBAVBMI(pal, &src[0], &dst[0], n)
		return
	}
	lookupRGBAGo(pal, src, dst)
}

// crossColorRow undoes the cross-color transform on one row src into dst,
// with the multipliers of each tile of 1<<bits pixels in mults. src may
// alias dst.
func crossColorRow(mults, src, dst []uint32, bits int) {
	width := len(src)
	if width == 0 || x86Level < cpuinfo.SSE41 {
		crossColorRowGo(mults, src, dst, bits)
		return
	}
	dst = dst[:width]
	if bits < 0 || bits > 30 {
		panic("vp8l: invalid cross-color tile bits")
	}
	mults = mults[:subSampleSize(width, bits)]
	if x86Level >= cpuinfo.AVX2 {
		crossColorRowAVX2(&mults[0], &src[0], &dst[0], width, bits)
	} else {
		crossColorRowSSE41(&mults[0], &src[0], &dst[0], width, bits)
	}
}

// predictRow undoes the predictor transform for the pixels 1 to
// len(out)-1 of a row whose pixel 0 is already done, with the mode of each
// tile of 1<<bits pixels in modes. up is the previous output row extended
// by out[0] (see predictRun). in may alias out.
//
// The assembly handles every mode but 12 and 13 (rare, and serial per
// pixel): it returns at the first tile of those, which predictRun then
// does before the assembly goes on.
func predictRow(modes, in, out, up []uint32, bits int) {
	width := len(out)
	if width < 2 || x86Level < cpuinfo.SSE41 {
		predictRowGo(modes, in, out, up, bits)
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
		x = predictRowSSE41(&in[0], &out[0], &up[0], &modes[0], x, width, bits)
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
	if n > 0 && bits >= 1 && bits <= 3 && x86Level >= cpuinfo.SSE41 {
		if x86Level >= cpuinfo.AVX2 {
			expandBundled8AVX2(&a.pal8, &src[0], &dst[0], n, bits)
		} else {
			expandBundled8SSE41(&a.pal8, &src[0], &dst[0], n, bits)
		}
		src, dst = src[n:], dst[n<<bits:]
	}
	expandBundledRowGo(a, bits, src, dst)
}

// expandBundledRGBA expands one row of bundled palette indices (each byte
// of src holds 1<<bits of them, bits 1 to 3) into the RGBA bytes dst (as
// many pixels as fit), from the RGBA palette pal (at most 16 entries).
// The assembly does whole groups of 8 pixels, the Go reference (with tab,
// see bundleTable) the rest of the row.
func expandBundledRGBA(pal *[256]uint32, tab []uint64, bits int, src, dst []byte) {
	if x86Level >= cpuinfo.AVX2 && bits >= 1 && bits <= 3 {
		group := 8 >> bits // packed bytes of 8 pixels
		n := min(len(dst)/(4<<bits), len(src)) &^ (group - 1)
		if n > 0 {
			expandBundledRGBAAVX2(pal, &src[0], &dst[0], n, bits)
			src, dst = src[n:], dst[n*(4<<bits):]
		}
	}
	expandBundledRGBAGo(tab, bits, src, dst)
}
