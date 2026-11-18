.PHONY: test race vet bench bench-compare interop fuzz

GO ?= go

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

vet:
	$(GO) vet ./...

bench:
	$(GO) test -run '^$$' -bench . -benchmem ./webp

# v1 (cgo, libwebp) vs v2 pure Go vs v2 as built by default, the tables of
# the README's Benchmarks section (needs libwebp and benchstat; about an
# hour, see benchmarks/run.sh for the knobs).
bench-compare:
	./benchmarks/run.sh

# Conformance tests against libwebp's cwebp/dwebp (must be in PATH).
interop:
	$(GO) test -tags interop -run Interop ./...

fuzz:
	$(GO) test -run '^$$' -fuzz FuzzDecode -fuzztime 60s ./decoder
