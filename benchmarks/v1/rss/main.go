// Command rss measures the peak resident memory of one v1 (libwebp)
// workload; see suite.RSSMain.
package main

import (
	"github.com/kolesa-team/go-webp/benchmarks/suite"
	v1 "github.com/kolesa-team/go-webp/benchmarks/v1"
)

func main() { suite.RSSMain(v1.Codec{}) }
