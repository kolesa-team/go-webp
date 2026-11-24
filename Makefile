.PHONY: test race vet cover bench bench-compare interop fuzz

GO ?= go

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

vet:
	$(GO) vet ./...

# Coverage of the whole module, as CI measures it for the README's badge:
# code that any package's tests run counts. Browse it with
# go tool cover -html=coverage.txt.
cover:
	$(GO) test -coverpkg=./... -coverprofile=coverage.txt ./...
	$(GO) tool cover -func=coverage.txt | tail -1

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
