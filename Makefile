SHELL := /bin/bash
.PHONY: build release lint test clean

all: build

build:
	cargo build

release:
	cargo build --release

lint:
	cargo clippy -- -D warnings

test:
	cargo test

clean:
	cargo clean
