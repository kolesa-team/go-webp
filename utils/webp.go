package utils

type FormatType int

//noinspection GoUnusedConst
const (
	FormatUndefined FormatType = iota
	FormatLossy
	FormatLossless
)

type BitstreamFeatures struct {
	Width        int
	Height       int
	HasAlpha     bool
	HasAnimation bool
	Format       FormatType
}
