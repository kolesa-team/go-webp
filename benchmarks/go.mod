// Benchmarks comparing v1 (cgo, libwebp) with v2 (this repository). A
// separate module, so that the library itself does not depend on v1 or
// need cgo. See run.sh and the Benchmarks section of ../README.md.
module github.com/kolesa-team/go-webp/benchmarks

go 1.27

require (
	github.com/kolesa-team/go-webp v1.0.5
	github.com/kolesa-team/go-webp/v2 v2.0.0
)

replace github.com/kolesa-team/go-webp/v2 => ../
