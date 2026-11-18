// Command rss measures the peak resident memory of one v2 workload; see
// suite.RSSMain.
package main

import (
	"github.com/kolesa-team/go-webp/benchmarks/suite"
	v2 "github.com/kolesa-team/go-webp/benchmarks/v2"
)

func main() { suite.RSSMain(v2.Codec{}) }
