// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//go:build (!arm64 && !amd64) || purego

package resample

// archState holds no kernel state without assembly; kind is always 0 (the
// Go kernels).
type archState struct{ kind uint8 }

func archTableSize(dstWidth, numChannels int) int { return 0 }

func (r *Rescaler) initArch(tab []uint32) {}

func (r *Rescaler) shrinkRow(src []byte, acc []uint32) { r.shrinkRowRef(src, acc) }

func (r *Rescaler) expandRow(src []byte, acc []uint32) { r.expandRowRef(src, acc) }

func exportExpand(dst []byte, frow []uint32, fy uint32) { exportExpandGo(dst, frow, fy) }

func exportExpandMix(dst []byte, frow, irow []uint32, a, b, fy uint32) {
	exportExpandMixGo(dst, frow, irow, a, b, fy)
}

func exportShrink(dst []byte, irow, frow []uint32, yscale, fxy uint32) {
	exportShrinkGo(dst, irow, frow, yscale, fxy)
}

func exportShrinkZero(dst []byte, irow []uint32, fxy uint32) { exportShrinkZeroGo(dst, irow, fxy) }

func premultiplyRow(row []byte) { multRGBA(row, &premulScale) }

func unpremultiplyRow(row []byte) { multRGBA(row, &unpremulScale) }
