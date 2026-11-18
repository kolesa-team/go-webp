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
