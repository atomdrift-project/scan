SHELL := /bin/sh
BINARY = litmus
OUT_DIR = out
BUILD ?= build
RUN   ?= litmus
DATASET ?= slow
BENCHMARK_ROOT ?= /Users/t/data/benchmark
BENCHMARK_PATH ?= $(BENCHMARK_ROOT)/$(DATASET)
SCAN_THREADS ?=
SLOW_RULE_MS ?= 200

.PHONY: build release install check-cargo tarball deploy rollout-bastille rollout-debian rollout-ubuntu rollout-openbsd rollout-alpine rollout-macos benchmark profile-slow lint test clean

all: build

build:
	cargo build

check-cargo:
	@command -v cargo >/dev/null 2>&1 || { \
		echo "Error: cargo not found. Install Rust via:"; \
		case "$$(uname -s)" in \
			Darwin)  echo "  brew install rust   # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;; \
			FreeBSD) echo "  pkg install rust" ;; \
			OpenBSD) echo "  pkg_add rust" ;; \
			NetBSD)  echo "  pkgin install rust   # or: pkg_add rust" ;; \
			SunOS)   echo "  pkgin install rust" ;; \
			Linux) \
				if command -v apt-get >/dev/null 2>&1; then \
					echo "  apt-get install cargo   # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
				elif command -v dnf >/dev/null 2>&1; then \
					echo "  dnf install cargo   # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
				elif command -v pacman >/dev/null 2>&1; then \
					echo "  pacman -S rust   # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
				elif command -v apk >/dev/null 2>&1; then \
					echo "  apk add cargo   # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
				else \
					echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
				fi ;; \
			*) echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;; \
		esac; \
		exit 1; \
	}

release: check-cargo $(OUT_DIR)
	cargo build --release
	cp target/release/$(BINARY) $(OUT_DIR)/

install: release
	@set -e; \
	if echo "$$PATH" | tr ':' '\n' | grep -qx "$$HOME/.cargo/bin" && [ -d "$$HOME/.cargo/bin" ]; then \
		dest="$$HOME/.cargo/bin/$(BINARY)"; \
	elif [ -d "$$HOME/bin" ] && [ -w "$$HOME/bin" ]; then \
		dest="$$HOME/bin/$(BINARY)"; \
	elif [ -d "$$HOME/.local/bin" ] && [ -w "$$HOME/.local/bin" ]; then \
		dest="$$HOME/.local/bin/$(BINARY)"; \
	elif [ -w /usr/local/bin ]; then \
		dest="/usr/local/bin/$(BINARY)"; \
	else \
		mkdir -p "$$HOME/.cargo/bin"; \
		dest="$$HOME/.cargo/bin/$(BINARY)"; \
	fi; \
	install -m 755 $(OUT_DIR)/$(BINARY) "$$dest.new" && mv -f "$$dest.new" "$$dest"; \
	echo "✓ Installed to $$dest"

tarball: release
	tar -czf $(OUT_DIR)/litmus.tgz -C $(OUT_DIR) litmus
	@echo "Tarball: $(OUT_DIR)/litmus.tgz"

deploy:
	git pull
	@case "$$(uname -s)" in \
		Darwin)  ./hacks/rollout-macos.sh ;; \
		FreeBSD) ./hacks/rollout-bastille.sh "$(BUILD)" "$(RUN)" ;; \
		Linux)   if [ -f /etc/alpine-release ]; then \
		           ./hacks/rollout-alpine.sh; \
		         elif grep -q '^ID=ubuntu$$' /etc/os-release 2>/dev/null; then \
		           ./hacks/rollout-ubuntu.sh; \
		         else \
		           [ -n "$(BUILD)" ] && [ -n "$(RUN)" ] || \
		             { echo "Usage: make deploy BUILD=<build-host> RUN=<run-host>"; exit 1; }; \
		           ./hacks/rollout-debian.sh "$(BUILD)" "$(RUN)"; \
		         fi ;; \
		OpenBSD) ./hacks/rollout-openbsd.sh ;; \
		*) echo "error: no deploy target for $$(uname -s)"; exit 1 ;; \
	esac

rollout-bastille:
	./hacks/rollout-bastille.sh "$(BUILD)" "$(RUN)"

rollout-macos:
	./hacks/rollout-macos.sh

rollout-debian:
	@[ -n "$(BUILD)" ] && [ -n "$(RUN)" ] || { echo "Usage: make rollout-debian BUILD=<build-host> RUN=<run-host>"; exit 1; }
	./hacks/rollout-debian.sh "$(BUILD)" "$(RUN)"

rollout-ubuntu:
	@grep -q '^ID=ubuntu$$' /etc/os-release 2>/dev/null || { echo "error: rollout-ubuntu requires Ubuntu"; exit 1; }
	./hacks/rollout-ubuntu.sh

rollout-openbsd:
	./hacks/rollout-openbsd.sh

rollout-alpine:
	./hacks/rollout-alpine.sh

benchmark: release
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)" >/dev/null

profile-slow: release
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	samply record --save-only --duration 20 -o /tmp/litmus-$(DATASET)-profile.json.gz -- \
		env CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)"

lint:
	cargo clippy -- -D warnings

test:
	cargo test

clean:
	cargo clean
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)
