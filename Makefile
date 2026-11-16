.PHONY: test race vet bench interop fuzz

GO ?= go

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

vet:
	$(GO) vet ./...

bench:
	$(GO) test -run '^$$' -bench . -benchmem ./webp

# Conformance tests against libwebp's cwebp/dwebp (must be in PATH).
interop:
	$(GO) test -tags interop -run Interop ./...

fuzz:
	$(GO) test -run '^$$' -fuzz FuzzDecode -fuzztime 60s ./decoder
