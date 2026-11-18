//go:build !amd64 || purego

package vp8enc

// residualCost returns the bit-cost of coding the residual with context
// ctx0 (the Go code of cost.go).
func (p *encProba) residualCost(ctx0 int, res *residual) int {
	return p.residualCostGo(ctx0, res)
}
