SHELL := /bin/bash
BINARY = litmus
OUT_DIR = out

.PHONY: build release tarball rollout-bastille lint test clean

all: build

build:
	cargo build

release: $(OUT_DIR)
	cargo build --release
	cp target/release/$(BINARY) $(OUT_DIR)/

tarball: release
	tar -czf $(OUT_DIR)/litmus.tgz -C $(OUT_DIR) litmus
	@echo "Tarball: $(OUT_DIR)/litmus.tgz"

rollout-bastille:
	@[ -n "$(BUILD)" ] && [ -n "$(RUN)" ] || { echo "Usage: make rollout-bastille BUILD=<build-jail> RUN=<run-jail>"; exit 1; }
	./hacks/rollout-bastille.sh "$(BUILD)" "$(RUN)"

lint:
	cargo clippy -- -D warnings

test:
	cargo test

clean:
	cargo clean
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)
