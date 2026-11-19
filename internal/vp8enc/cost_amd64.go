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

//go:build amd64 && !purego

package vp8enc

import "github.com/kolesa-team/go-webp/v2/internal/cpuinfo"

//go:noescape
func residualCostSSE41(costs *[numBands][numCtx]levelCostTable, prob *[numBands][numCtx][numProbas]uint8,
	coeffs *[16]int16, first, last, ctx0 int) int

// residualCost returns the bit-cost of coding the residual with context
// ctx0: residualCostGo (cost.go) computed by residualCostSSE41
// (cost_amd64.s), which takes no data-dependent branch. Residuals the
// encoder never builds (first > 1, which the Go code reads as a band
// index, last > 15, ctx0 > 2) take the Go code.
func (p *encProba) residualCost(ctx0 int, res *residual) int {
	if x86Level < cpuinfo.SSE41 || uint(res.first) > 1 || res.last > 15 || uint(ctx0) > 2 {
		return p.residualCostGo(ctx0, res)
	}
	return residualCostSSE41(&p.levelCost[res.ctype], &p.coeffs[res.ctype], res.coeffs, res.first, res.last, ctx0)
}
