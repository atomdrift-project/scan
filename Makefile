SHELL := /bin/sh
BINARY = litmus
OUT_DIR = out
BUILD ?= build
SERVER_RUN ?= litmus
WORKER_RUN ?= litworker
DATASET ?= slow

# Truthy ("1") iff the user explicitly passed BUILD or WORKER_RUN on the make
# command line or via the environment, rather than inheriting the defaults
# above. Used to opt into the SSH-orchestrated worker deploy on Linux; with no
# override the Linux dispatch installs locally via systemd.
WORKER_REMOTE := $(if $(strip \
    $(filter command line environment,$(origin BUILD)) \
    $(filter command line environment,$(origin WORKER_RUN))),1,)
BENCHMARK_ROOT ?= /Users/t/data/benchmark
BENCHMARK_PATH ?= $(BENCHMARK_ROOT)/$(DATASET)
SCAN_THREADS ?=
SLOW_RULE_MS ?= 200
MAX_JOBS ?= 25
WORKERS  ?=

# Scrub GNU make's jobserver from cargo's environment. Without this, build
# scripts that spawn their own `make` (e.g. tikv-jemalloc-sys) inherit a
# malformed MAKEFLAGS and fail with "No rule to make target '-j'".
CARGO = env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS cargo

.PHONY: build release install check-cargo tarball deploy deploy-server deploy-worker deploy-worker-nodes uninstall-server uninstall-server-nodes uninstall-worker uninstall-worker-nodes rollout-bastille benchmark benchmark-worker worker profile-worker profile-slow bench-build sampled-benchmark heap-build heap-benchmark tuna tuna-once lint test clean wolfi wolfi-bootstrap wolfi-build wolfi-test wolfi-shell wolfi-clean wolfi-nuke docker-login docker-publish

all: build

build:
	$(CARGO) build

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
	$(CARGO) build --release
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

# Clear MAKEFLAGS for deploy recipes: GNU Make would otherwise inject the
# outer invocation's `-j`/`--jobserver-*` flags plus command-line `URL=` into
# the env, which tikv-jemalloc-sys's build.rs re-passes to its bundled `make`,
# producing `*** No rule to make target '-j'` inside the jemalloc build.
deploy-server deploy-worker deploy-worker-nodes rollout-bastille: export MAKEFLAGS :=

deploy: deploy-server

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
		Linux)   if [ -n "$(WORKER_REMOTE)" ]; then \
		           ./scripts/worker/worker-debian.sh "$(BUILD)" "$(WORKER_RUN)" "$(URL)"; \
		         elif [ -f /etc/alpine-release ]; then \
		           ./scripts/worker/worker-alpine.sh "$(URL)"; \
		         elif command -v systemctl >/dev/null 2>&1; then \
		           ./scripts/worker/worker-linux.sh "$(URL)"; \
		         else \
		           echo "error: unsupported Linux (no systemd; not Alpine; pass BUILD+WORKER_RUN for SSH deploy)"; exit 1; \
		         fi ;; \
		OpenBSD) ./scripts/worker/worker-openbsd.sh "$(URL)" ;; \
		SunOS)   ./scripts/worker/worker-omnios.sh "$(URL)" ;; \
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
		Linux)   if [ -n "$(WORKER_REMOTE)" ]; then \
		           ./scripts/worker/uninstall-debian.sh "$(WORKER_RUN)"; \
		         elif [ -f /etc/alpine-release ]; then \
		           ./scripts/worker/uninstall-alpine.sh; \
		         elif command -v systemctl >/dev/null 2>&1; then \
		           ./scripts/worker/uninstall-linux.sh; \
		         else \
		           echo "error: unsupported Linux"; exit 1; \
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
	CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)" > /tmp/litmus-benchmark-$(DATASET).json

benchmark-worker: release
	@[ -n "$(URL)" ] || { echo "Usage: make benchmark-worker URL=<hopper-url>"; exit 1; }
	./out/$(BINARY) --verbose worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

# Run a worker in the foreground for interactive use. The worker self-nices
# to 10 by default; pass NICE=0 to disable.
worker: release
	@[ -n "$(URL)" ] || { echo "Usage: make worker URL=<hopper-url> [WORKERS=<n>] [NICE=<int>]"; exit 1; }
	./out/$(BINARY) worker --url "$(URL)" \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(NICE),--nice $(NICE),)

profile-worker:
	$(CARGO) build --profile profiling
	@[ -n "$(URL)" ] || { echo "Usage: make profile-worker URL=<hopper-url>"; exit 1; }
	samply record -o /tmp/litmus-worker-profile.json.gz -- \
		./target/profiling/$(BINARY) --verbose worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

profile-slow:
	$(CARGO) build --profile profiling
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	samply record --save-only --duration 20 -o /tmp/litmus-$(DATASET)-profile.json.gz -- \
		env CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./target/profiling/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)"

# ----- cleave-tuna integration --------------------------------------------
# Standardized targets that cleave-tuna drives. The naming mirrors cleave's
# Makefile so one tuna binary can tune both repos. See ../cleave-tuna/README.md.
#
# Honor CARGO_TARGET_DIR if set (cleave-tuna sets it to share the cargo
# cache across worktrees). Falls back to the cargo default `target` otherwise.
CARGO_TARGET ?= $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),target)

TUNA_DATASET     ?= 200MB
TUNA_BENCH_PATH  ?= $(BENCHMARK_ROOT)/$(TUNA_DATASET)

bench-build: $(OUT_DIR) ## Build benchmark binary (profiling profile, release + debug syms)
	$(CARGO) build --profile profiling
	cp $(CARGO_TARGET)/profiling/$(BINARY) $(OUT_DIR)/$(BINARY).bench
	@if [ "$$(uname)" = "Darwin" ]; then codesign -s - -f $(OUT_DIR)/$(BINARY).bench; fi
	@echo "✓ Benchmark binary: $(OUT_DIR)/$(BINARY).bench"

sampled-benchmark: bench-build ## Benchmark with samply CPU profiling
	@command -v samply >/dev/null 2>&1 || { echo "Error: samply not installed. Run: cargo install samply"; exit 1; }
	@[ -e "$(TUNA_BENCH_PATH)" ] || { echo "error: benchmark path not found: $(TUNA_BENCH_PATH)"; exit 1; }
	CLEAVE_SKIP_CACHE=1 samply record --save-only -o $(OUT_DIR)/bench.profile.json.gz -- \
		$(OUT_DIR)/$(BINARY).bench --slow-rule-ms $(SLOW_RULE_MS) -f json $(TUNA_BENCH_PATH) \
		>$(OUT_DIR)/bench.out 2>$(OUT_DIR)/bench.err
	@echo "✓ Profile: $(OUT_DIR)/bench.profile.json.gz  Logs: $(OUT_DIR)/bench.err"

heap-build: $(OUT_DIR) ## Build with jemalloc heap profiling support
	$(CARGO) build --profile profiling --features jemalloc-prof
	cp $(CARGO_TARGET)/profiling/$(BINARY) $(OUT_DIR)/$(BINARY).heap
	@if [ "$$(uname)" = "Darwin" ]; then codesign -s - -f $(OUT_DIR)/$(BINARY).heap; fi
	@echo "✓ Heap-profiling binary: $(OUT_DIR)/$(BINARY).heap"

heap-benchmark: heap-build ## Benchmark with jemalloc heap profiling
	@[ -e "$(TUNA_BENCH_PATH)" ] || { echo "error: benchmark path not found: $(TUNA_BENCH_PATH)"; exit 1; }
	@rm -rf $(OUT_DIR)/heap && mkdir -p $(OUT_DIR)/heap
	CLEAVE_SKIP_CACHE=1 _RJEM_MALLOC_CONF="prof:true,prof_active:true,prof_final:true,lg_prof_interval:28,prof_prefix:$(OUT_DIR)/heap/jeprof" \
		$(OUT_DIR)/$(BINARY).heap --slow-rule-ms $(SLOW_RULE_MS) -f json $(TUNA_BENCH_PATH) \
		>$(OUT_DIR)/bench.out 2>$(OUT_DIR)/bench.err
	@echo "✓ Heap profiles: $(OUT_DIR)/heap/jeprof.*.heap"
	@echo "  Analyze with: jeprof --text $(OUT_DIR)/$(BINARY).heap $(OUT_DIR)/heap/jeprof.*.heap"

# cleave-tuna: LLM-driven CPU+memory autoresearch loop. See ../cleave-tuna/README.md.
TUNA_REPO            ?= ../cleave-tuna
TUNA_BIN             ?= $(TUNA_REPO)/out/cleave-tuna
TUNA_EXPERIMENTS     ?= 6
TUNA_SCREEN_SAMPLES  ?= 1
TUNA_CONFIRM_SAMPLES ?= 3
TUNA_PROVIDER        ?= gemini,codex,claude
TUNA_MODE            ?=
TUNA_INTERVAL        ?= 30

tuna: ## Run cleave-tuna in a loop, alternating memory/cpu; cherry-picks wins (Ctrl-C to stop)
	@test -x $(TUNA_BIN) || { echo "build cleave-tuna first: (cd $(TUNA_REPO) && make build)"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree must be clean before starting tuna"; exit 1; }
	@echo "tuna: looping forever, alternating memory/cpu (Ctrl-C to stop). settings: dataset=$(TUNA_DATASET) experiments=$(TUNA_EXPERIMENTS) screen-samples=$(TUNA_SCREEN_SAMPLES) confirm-samples=$(TUNA_CONFIRM_SAMPLES) provider=$(TUNA_PROVIDER)"
	@mode=memory; \
	while true; do \
		echo "tuna: starting cycle in $$mode mode"; \
		$(MAKE) tuna-once TUNA_MODE=$$mode || exit $$?; \
		if [ "$$mode" = "memory" ]; then mode=cpu; else mode=memory; fi; \
		echo "tuna: sleeping $(TUNA_INTERVAL)s before next cycle ($$mode) — Ctrl-C to stop"; \
		sleep $(TUNA_INTERVAL); \
	done

tuna-once: ## One cleave-tuna cycle, then cherry-pick accepted experiments
	@test -x $(TUNA_BIN) || { echo "build cleave-tuna first: (cd $(TUNA_REPO) && make build)"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree must be clean before tuna-once"; exit 1; }
	@before=$$(git rev-parse HEAD); \
	$(TUNA_BIN) --source $(CURDIR) --root $(TUNA_REPO) --dataset $(TUNA_DATASET) \
		--name litmus \
		--bench-arg --slow-rule-ms --bench-arg $(SLOW_RULE_MS) --bench-arg -f --bench-arg json \
		--bench-env CLEAVE_SKIP_CACHE=1 \
		--deny vendor/ --deny packaging/ --deny scripts/ --deny testdata/ \
		--experiments $(TUNA_EXPERIMENTS) \
		--screen-samples $(TUNA_SCREEN_SAMPLES) --confirm-samples $(TUNA_CONFIRM_SAMPLES) \
		--provider $(TUNA_PROVIDER) $(if $(TUNA_MODE),--$(TUNA_MODE),) \
		|| { echo "tuna: cleave-tuna exited non-zero; not cherry-picking"; exit 1; }; \
	branch=$$(git for-each-ref --sort=-committerdate --format='%(refname:short)' 'refs/heads/tuna/*' | head -1); \
	if [ -z "$$branch" ]; then echo "tuna: no tuna/* branch found"; exit 0; fi; \
	ahead=$$(git rev-list --count $$before..$$branch); \
	if [ "$$ahead" = "0" ]; then \
		echo "tuna: no accepted experiments on $$branch — nothing to cherry-pick"; \
		exit 0; \
	fi; \
	echo "tuna: cherry-picking $$ahead commit(s) from $$branch"; \
	git cherry-pick $$branch~$$ahead..$$branch

# --------------------------------------------------------------------------

lint:
	$(CARGO) clippy -- -D warnings

test:
	$(CARGO) test

clean:
	$(CARGO) clean
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ----- Wolfi packaging ----------------------------------------------------
# Build a Wolfi-based OCI image for litmus via melange + apko. Mirrors
# the cleave/packaging/wolfi flow; depends on a sibling cleave checkout
# at ../cleave (the local build stages cleave + filefacts + litmus and
# overrides the cleave git dep via a Cargo [patch] block). On macOS the
# build runs inside a dedicated Lima VM (`litmus-wolfi`). See
# packaging/wolfi/README.md.
WOLFI_DIR = packaging/wolfi
WOLFI_OUT = $(OUT_DIR)/wolfi
WOLFI_ARCH ?=

wolfi: wolfi-bootstrap wolfi-build wolfi-test

wolfi-bootstrap:
	@$(WOLFI_DIR)/scripts/bootstrap-lima.sh

wolfi-build:
	@WOLFI_ARCH="$(WOLFI_ARCH)" $(WOLFI_DIR)/scripts/build.sh
	@echo "✓ Wolfi image: $(WOLFI_OUT)/litmus.tar"

wolfi-test:
	@$(WOLFI_DIR)/scripts/smoke-test.sh

wolfi-shell:
	@[ -f $(WOLFI_OUT)/litmus.tar ] || { echo "error: run 'make wolfi-build' first"; exit 1; }
	@case "$$(uname -s)" in \
		Darwin) limactl shell --workdir / litmus-wolfi nerdctl run --rm -it --entrypoint /bin/sh litmus:smoke ;; \
		Linux)  for r in nerdctl docker podman; do command -v $$r >/dev/null 2>&1 && { exec $$r run --rm -it --entrypoint /bin/sh litmus:smoke; }; done; echo "no container runtime"; exit 1 ;; \
	esac

wolfi-clean:
	rm -rf $(WOLFI_OUT)
	@echo "✓ Wolfi output cleaned"

wolfi-nuke: wolfi-clean
	@case "$$(uname -s)" in \
		Darwin) limactl delete --force litmus-wolfi 2>/dev/null || true ;; \
	esac
	rm -rf $$HOME/.cache/litmus-wolfi
	@echo "✓ Wolfi VM and cache removed"

# ----- Publish -----------------------------------------------------------
# Push the multi-arch litmus image to a registry and sign it keyless with
# cosign via Google OIDC. Override REGISTRY / ORG / ARCHS via env. See
# packaging/wolfi/README.md for prerequisites.
REGISTRY ?= docker.io
ORG      ?= atomdrift

docker-login: wolfi-bootstrap ## Log the lima VM's runtime into REGISTRY (interactive)
	@case "$$(uname -s)" in \
		Darwin) limactl shell --workdir / litmus-wolfi nerdctl login $(REGISTRY) ;; \
		Linux)  for r in nerdctl docker podman; do command -v $$r >/dev/null 2>&1 && { exec $$r login $(REGISTRY); }; done; echo "no container runtime"; exit 1 ;; \
	esac

docker-publish: wolfi-bootstrap ## Build multi-arch + push + cosign sign (set DRY_RUN=1 to skip the push)
	@REGISTRY="$(REGISTRY)" ORG="$(ORG)" \
		$(WOLFI_DIR)/scripts/publish.sh
