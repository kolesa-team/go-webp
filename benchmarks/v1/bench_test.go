package v1

import (
	"testing"

	"github.com/kolesa-team/go-webp/benchmarks/suite"
)

func BenchmarkDecode(b *testing.B)    { suite.Decode(b, Codec{}) }
func BenchmarkEncode(b *testing.B)    { suite.Encode(b, Codec{}) }
func BenchmarkSatDecode(b *testing.B) { suite.SatDecode(b, Codec{}) }
func BenchmarkSatEncode(b *testing.B) { suite.SatEncode(b, Codec{}) }
