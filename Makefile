SHELL := /bin/sh
BINARY = litmus
OUT_DIR = out
BUILD ?= build
SERVER_RUN ?= litmus
WORKER_RUN ?= litworker
DATASET ?= slow
BENCHMARK_ROOT ?= /Users/t/data/benchmark
BENCHMARK_PATH ?= $(BENCHMARK_ROOT)/$(DATASET)
SCAN_THREADS ?=
SLOW_RULE_MS ?= 200
MAX_JOBS ?= 25
WORKERS  ?=

.PHONY: build release install check-cargo tarball deploy-server deploy-worker deploy-worker-nodes uninstall-server uninstall-server-nodes uninstall-worker uninstall-worker-nodes rollout-bastille benchmark benchmark-worker profile-worker profile-slow lint test clean

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
	cp target/release/$(BINARY) $(OUT_DIR)/$(BINARY).new && mv -f $(OUT_DIR)/$(BINARY).new $(OUT_DIR)/$(BINARY)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		codesign --force --sign - $(OUT_DIR)/$(BINARY); \
	fi

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

deploy-server:
	git pull
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/server/rollout-bastille.sh "$(BUILD)" "$(SERVER_RUN)" ;; \
		*) echo "error: server deployments are bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

deploy-worker:
	@[ -n "$(URL)" ] || { echo "Usage: make deploy-worker URL=<url> [BUILD=<host>] [WORKER_RUN=<host>]"; exit 1; }
	git stash
	git pull
	@case "$$(uname -s)" in \
		Darwin)  ./scripts/worker/worker-macos.sh "$(URL)" ;; \
		FreeBSD) ./scripts/worker/worker-bastille.sh "$(BUILD)" "$(WORKER_RUN)" "$(URL)" ;; \
		Linux)   if [ -f /etc/alpine-release ]; then \
		           ./scripts/worker/worker-alpine.sh "$(URL)"; \
		         elif grep -q '^ID=ubuntu$$' /etc/os-release 2>/dev/null; then \
		           ./scripts/worker/worker-ubuntu.sh "$(URL)"; \
		         else \
		           [ -n "$(BUILD)" ] && [ -n "$(WORKER_RUN)" ] || \
		             { echo "Usage: make deploy-worker BUILD=<build-host> WORKER_RUN=<run-host> URL=<url>"; exit 1; }; \
		           ./scripts/worker/worker-debian.sh "$(BUILD)" "$(WORKER_RUN)" "$(URL)"; \
		         fi ;; \
		OpenBSD) ./scripts/worker/worker-openbsd.sh "$(URL)" ;; \
		*) echo "error: no deploy-worker target for $$(uname -s)"; exit 1 ;; \
	esac

uninstall-server:
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/server/uninstall-bastille.sh "$(SERVER_RUN)" ;; \
		*) echo "error: server deployments are bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

uninstall-worker:
	@case "$$(uname -s)" in \
		Darwin)  ./scripts/worker/uninstall-macos.sh ;; \
		FreeBSD) ./scripts/worker/uninstall-bastille.sh "$(WORKER_RUN)" ;; \
		Linux)   if [ -f /etc/alpine-release ]; then \
		           ./scripts/worker/uninstall-alpine.sh; \
		         elif grep -q '^ID=ubuntu$$' /etc/os-release 2>/dev/null; then \
		           ./scripts/worker/uninstall-ubuntu.sh; \
		         else \
		           [ -n "$(WORKER_RUN)" ] || { echo "Usage: make uninstall-worker WORKER_RUN=<run-host>"; exit 1; }; \
		           ./scripts/worker/uninstall-debian.sh "$(WORKER_RUN)"; \
		         fi ;; \
		OpenBSD) ./scripts/worker/uninstall-openbsd.sh ;; \
		*) echo "error: no uninstall-worker target for $$(uname -s)"; exit 1 ;; \
	esac

deploy-worker-nodes:
	@[ -n "$(URL)" ] || { echo "Usage: make deploy-worker-nodes URL=<url> NODES=\"node1 node2\""; exit 1; }
	@[ -n "$(NODES)" ] || { echo "Usage: make deploy-worker-nodes URL=<url> NODES=\"node1 node2\""; exit 1; }
	./scripts/worker/update-nodes.sh "$(URL)" $(NODES)

uninstall-server-nodes:
	@[ -n "$(NODES)" ] || { echo "Usage: make uninstall-server-nodes NODES=\"node1 node2\""; exit 1; }
	./scripts/server/uninstall-nodes.sh $(NODES)

uninstall-worker-nodes:
	@[ -n "$(NODES)" ] || { echo "Usage: make uninstall-worker-nodes NODES=\"node1 node2\""; exit 1; }
	./scripts/worker/uninstall-nodes.sh $(NODES)

rollout-bastille:
	./scripts/server/rollout-bastille.sh "$(BUILD)" "$(SERVER_RUN)"

benchmark: release
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)" >/dev/null

benchmark-worker: release
	@[ -n "$(URL)" ] || { echo "Usage: make benchmark-worker URL=<hopper-url>"; exit 1; }
	./out/$(BINARY) --verbose worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

profile-worker:
	cargo build --profile profiling
	@[ -n "$(URL)" ] || { echo "Usage: make profile-worker URL=<hopper-url>"; exit 1; }
	samply record -o /tmp/litmus-worker-profile.json.gz -- \
		./target/profiling/$(BINARY) --verbose worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

profile-slow:
	cargo build --profile profiling
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	samply record --save-only --duration 20 -o /tmp/litmus-$(DATASET)-profile.json.gz -- \
		env CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./target/profiling/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)"

lint:
	cargo clippy -- -D warnings

test:
	cargo test

clean:
	cargo clean
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)
