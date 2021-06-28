.PHONY: vendor test

GO ?= $(which go)

test:
	@GO test ./... -v

vendor:
	@GO mod vendor

vet:
	@GO vet ./... -v

bench:
	@GO test -bench=. ./webp

libwebp-mac:
	brew install webp

libwebp-ubuntu:
	sudo apt-get install libwebp-dev
