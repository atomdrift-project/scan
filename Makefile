SHELL := /bin/sh
BINARY = scan
# Canonical name for the locally-installed binary. The build artifact is `scan`
# (see Cargo.toml), but `make install` lands it as `atomscan` and adds a `scan`
# symlink only when PATH has no `scan` already (avast ships /usr/bin/scan).
INSTALL_NAME = atomscan
OUT_DIR = out
BUILD ?= build
SERVER_RUN ?= scan
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
MAX_RSS_GB ?=
# LLM second-opinion pass for `make worker` (matches the deploy scripts'
# defaults). LLM is exported as SCAN_LLM; INTERPRET_MIN_PROB gates which samples
# are sent. The benchmark/profile targets deliberately omit interpret so LLM
# round-trips don't distort wall/RSS measurements.
LLM ?= http://10.9.8.149:8000/v1
INTERPRET_MIN_PROB ?= 0.15

# Scrub GNU make's jobserver from cargo's environment. Without this, build
# scripts that spawn their own `make` (e.g. tikv-jemalloc-sys) inherit a
# malformed MAKEFLAGS and fail with "No rule to make target '-j'".
CARGO = env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS cargo

.PHONY: build release release-lto install uninstall check-cargo tarball deploy deploy-server deploy-worker deploy-jail-worker deploy-bloomer uninstall-bloomer deploy-worker-nodes deploy-workers deploy-workers-tmux uninstall-server uninstall-server-nodes stop-worker uninstall-worker uninstall-jail-worker uninstall-worker-nodes rollout-bastille benchmark benchmark-worker worker-benchmark worker profile-worker profile-slow bench-build sampled-benchmark heap-build heap-benchmark tuna tuna-once lint fix test test-unit install-precommit clean wolfi wolfi-bootstrap wolfi-build wolfi-test wolfi-shell wolfi-clean wolfi-nuke docker-login docker-publish

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

# --- Model publishing (R2-backed, via cleave's manifest-gen) -----------------
# Reuses the proven cleave manifest-gen with litmus/azoth params: model version =
# azoth commit, oracle = `litmus validate --skip-traits` (feature/ABI compat),
# bundle = git archive azoth@commit | zstd. Layout: <remote>/litmus/versions.toml
# + <remote>/litmus/azoth/<date>-<commit>.tar.zst.
MANIFEST_GEN ?= ../cleave/tools/manifest-gen
TRAITS ?= ../azoth
R2_REMOTE ?= atomdrift-updates:atomdrift-updates
# Published model-distribution channel path. Intentionally NOT rebranded:
# existing release bundles and clients already resolve models under this
# prefix, and `--engine-bin litmus,scan` below compat-tests historical release
# binaries by name (`litmus` for older releases, `scan` going forward).
# Renaming the channel is a separate, coordinated migration (re-upload + client
# default change), not part of the CLI rebrand.
R2_LITMUS ?= litmus
ISSUER ?= https://accounts.google.com
DIST ?= dist
VERSIONS ?= 3
publish-models: release ## Compat-test azoth vs last (VERSIONS-1) litmus releases → sign → verify → upload to R2 (IDENTITY=<signer>)
	@[ -n "$(IDENTITY)" ] || { echo "publish-models: IDENTITY=<signer> required (e.g. releaser@<project>.iam.gserviceaccount.com)"; exit 1; }
	@command -v rclone >/dev/null || { echo "publish-models: rclone not found"; exit 1; }
	@command -v cosign >/dev/null || { echo "publish-models: cosign not found"; exit 1; }
	# manifest-gen reads the traits repo's LOCAL git log (no fetch), so the newest
	# azoth commit must already be checked out. Fast-forward to the remote tip and
	# abort the publish if it can't (diverged/dirty/offline) — never publish stale.
	git -C $(TRAITS) pull --ff-only
	cd $(MANIFEST_GEN) && GOWORK=off go build -o manifest-gen .
	$(MANIFEST_GEN)/manifest-gen \
	  --traits $(TRAITS) --repo . --out "$(DIST)" \
	  --engine-bin litmus,scan --traits-env SCAN_MODELS_DIR --validate-args "validate --skip-traits" \
	  --head-engine ./target/release/$(BINARY) \
	  --releases $(shell expr $(VERSIONS) - 1) --commits 100 --soak-days 0 \
	  --channels stable --artifact-prefix "azoth/" \
	  --sign --identity "$(IDENTITY)"
	python3 $(MANIFEST_GEN)/check-manifest.py "$(DIST)"
	@echo "→ verifying signature"
	cosign verify-blob --new-bundle-format \
	  --bundle "$(DIST)/versions.toml.sigstore.json" \
	  --certificate-identity "$(IDENTITY)" --certificate-oidc-issuer "$(ISSUER)" \
	  "$(DIST)/versions.toml" && echo "✓ signature verifies for $(IDENTITY)"
	@echo "→ uploading to R2"
	rclone copy "$(DIST)" "$(R2_REMOTE)/$(R2_LITMUS)/azoth/" --include "*.tar.zst" \
	  --header-upload "Cache-Control: public, max-age=31536000, immutable" --progress
	rclone copyto "$(DIST)/versions.toml" "$(R2_REMOTE)/$(R2_LITMUS)/versions.toml" \
	  --header-upload "Cache-Control: public, max-age=60"
	rclone copyto "$(DIST)/versions.toml.sigstore.json" "$(R2_REMOTE)/$(R2_LITMUS)/versions.toml.sigstore.json" \
	  --header-upload "Cache-Control: public, max-age=60"
	@echo "✓ publish-models complete: compat-tested HEAD + last $(shell expr $(VERSIONS) - 1) release(s), signed, verified, uploaded"

# --- Bloom filter publishing (R2-backed; outputs revision-controlled in ../bloom)
# Builds the known-good/known-bad filters from hopper's labelled samples into the
# BLOOM_REPO git repo (the source of truth, like ../azoth for models) and uploads
# them to R2. The pool defaults to the local hopper replica (same DSN collimator
# trains from; auth via ~/.pgpass) — see scripts/bloom_pool.sql for the tier
# predicates. Override with POOL=<file.ndjson> to build from a captured export.
# Filters and their manifest go to a format-versioned prefix (litmus/bloom/v<N>/),
# both short-cached: the download flow compares per-file sha256 to decide what to pull,
# so artifacts are overwritten in place rather than parked under dated, immutable
# paths. A format bump (FORMAT_VERSION) moves the whole prefix, leaving the old
# one intact for clients still on it. No signing: the filters carry no
# commit/authenticity claim, only a versioned layout and per-file sha256.
BLOOM_REPO ?= ../bloom
BLOOM_DB ?= postgres://hopper@localhost:5432/hopper
BLOOM_POOL_SQL ?= scripts/bloom_pool.sql
BLOOM_DATE ?= $(shell date -u +%F)
# Only bless/flag samples analyzed within this many days, so the tables reflect
# the current model/ruleset. 90 is a fair default; use 7 for quick test builds.
BLOOM_MAX_AGE_DAYS ?= 90
# Window for the unattended hourly cycle (make bloom-cron / scan-bloomer.timer).
# Wider than the interactive default so the published filters cover a full year.
BLOOM_CRON_MAX_AGE_DAYS ?= 365
.PHONY: build-bloom publish-bloom bloom-cron
build-bloom: ## Build bloom filters from the local hopper replica (or POOL=<ndjson>) into $(BLOOM_REPO) — no upload
	$(CARGO) build --release --bin scan-bloom-build
	mkdir -p "$(BLOOM_REPO)"
	@if [ -n "$(POOL)" ]; then \
	  echo "→ building filters from POOL=$(POOL)"; \
	  ./target/release/scan-bloom-build --in "$(POOL)" --out "$(BLOOM_REPO)" --date "$(BLOOM_DATE)"; \
	else \
	  command -v psql >/dev/null || { echo "build-bloom: psql not found (need it for BLOOM_DB=$(BLOOM_DB), or pass POOL=<ndjson>)"; exit 1; }; \
	  echo "→ exporting pool from replica $(BLOOM_DB) (analyzed within $(BLOOM_MAX_AGE_DAYS)d)"; \
	  psql "$(BLOOM_DB)" -X -q -v ON_ERROR_STOP=1 -v max_age_days=$(BLOOM_MAX_AGE_DAYS) -f "$(BLOOM_POOL_SQL)" -o "$(BLOOM_REPO)/pool.ndjson" && \
	  ./target/release/scan-bloom-build --in "$(BLOOM_REPO)/pool.ndjson" --out "$(BLOOM_REPO)" --date "$(BLOOM_DATE)" && \
	  rm -f "$(BLOOM_REPO)/pool.ndjson"; \
	fi
	@echo "✓ filters written to $(BLOOM_REPO)"
	@echo "  → review & commit $(BLOOM_REPO), then 'make publish-bloom' to release"

# Pure upload — no build, no database. Reads format_version from the manifest so
# the artifacts land under the same litmus/v<N>/ prefix the client fetches.
publish-bloom: ## Upload the already-built filters in $(BLOOM_REPO) to R2 under litmus/v<N>/ (run build-bloom first)
	@command -v rclone >/dev/null || { echo "publish-bloom: rclone not found"; exit 1; }
	@[ -f "$(BLOOM_REPO)/bloom.toml" ] || { echo "publish-bloom: no $(BLOOM_REPO)/bloom.toml — run 'make build-bloom' first"; exit 1; }
	@built=$$(awk -F'"' '/^built/{print $$2}' "$(BLOOM_REPO)/bloom.toml"); \
	ver=$$(awk -F'= *' '/format_version/{print $$2; exit}' "$(BLOOM_REPO)/bloom.toml"); \
	[ -n "$$built" ] || { echo "publish-bloom: no 'built' date in bloom.toml"; exit 1; }; \
	[ -n "$$ver" ] || { echo "publish-bloom: no format_version in bloom.toml"; exit 1; }; \
	echo "→ uploading filters built $$built to R2 (litmus/bloom/v$$ver/, short cache)"; \
	rclone copy "$(BLOOM_REPO)" "$(R2_REMOTE)/$(R2_LITMUS)/bloom/v$$ver/" --include "*.adbl" \
	  --header-upload "Cache-Control: public, max-age=300" --progress; \
	echo "→ publishing manifest (short cache)"; \
	rclone copyto "$(BLOOM_REPO)/bloom.toml" "$(R2_REMOTE)/$(R2_LITMUS)/bloom/v$$ver/bloom.toml" \
	  --header-upload "Cache-Control: public, max-age=60"; \
	echo "✓ publish-bloom complete: built $$built → litmus/bloom/v$$ver/"

# Unattended cycle for the hourly systemd timer (scripts/worker/bloomer-linux.sh):
# rebuild a fresh 365-day filter set, commit+push the source-of-truth $(BLOOM_REPO),
# then upload to R2. Commit is skipped when the rebuild is byte-identical, and the
# push fires whenever the local branch is ahead of origin (so a transient push
# failure self-heals next cycle). The R2 upload always runs and is idempotent
# (rclone skips unchanged objects), so a no-op hour only costs the rebuild plus a
# manifest-pointer refresh. A push failure is logged but does not block the upload.
# Safe to run by hand, too.
bloom-cron: ## Hourly-timer cycle: build (BLOOM_CRON_MAX_AGE_DAYS, default 365) -> commit+push $(BLOOM_REPO) -> upload to R2
	@git -C "$(BLOOM_REPO)" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
	  || { echo "bloom-cron: $(BLOOM_REPO) is not a git checkout — clone the bloom repo there first (don't let the timer build into a bare dir)"; exit 1; }
	$(MAKE) build-bloom BLOOM_MAX_AGE_DAYS=$(BLOOM_CRON_MAX_AGE_DAYS)
	@cd "$(BLOOM_REPO)" && { \
	  if [ -n "$$(git status --porcelain)" ]; then \
	    built=$$(awk -F'"' '/^built/{print $$2}' bloom.toml); \
	    echo "→ committing $(BLOOM_REPO) (built $$built)"; \
	    git add -A && git commit -q -m "bloom: $$built ($(BLOOM_CRON_MAX_AGE_DAYS)d window)" \
	      || { echo "bloom-cron: commit failed — refusing to publish unrecorded filters (is user.name/email set in $(BLOOM_REPO)?)"; exit 1; }; \
	  else \
	    echo "→ $(BLOOM_REPO) unchanged since last cycle; nothing to commit"; \
	  fi; \
	  if [ -n "$$(git rev-list @{u}.. 2>/dev/null)" ]; then \
	    echo "→ pushing $(BLOOM_REPO) to origin"; \
	    git push -q || echo "WARN: git push failed; local commit retained, will retry next cycle"; \
	  fi; \
	}
	$(MAKE) publish-bloom

# Fat LTO + single codegen unit. Multi-minute link, marginal runtime win
# over the default release profile. Use for container/tarball builds.
release-lto: check-cargo $(OUT_DIR)
	$(CARGO) build --profile release-lto
	cp target/release-lto/$(BINARY) $(OUT_DIR)/$(BINARY).new && mv -f $(OUT_DIR)/$(BINARY).new $(OUT_DIR)/$(BINARY)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		codesign --force --sign - $(OUT_DIR)/$(BINARY); \
	fi

install: release
	@set -e; \
	if echo "$$PATH" | tr ':' '\n' | grep -qx "$$HOME/.cargo/bin" && [ -d "$$HOME/.cargo/bin" ]; then \
		bindir="$$HOME/.cargo/bin"; \
	elif [ -d "$$HOME/bin" ] && [ -w "$$HOME/bin" ]; then \
		bindir="$$HOME/bin"; \
	elif [ -d "$$HOME/.local/bin" ] && [ -w "$$HOME/.local/bin" ]; then \
		bindir="$$HOME/.local/bin"; \
	elif [ -w /usr/local/bin ]; then \
		bindir="/usr/local/bin"; \
	else \
		mkdir -p "$$HOME/.cargo/bin"; \
		bindir="$$HOME/.cargo/bin"; \
	fi; \
	dest="$$bindir/$(INSTALL_NAME)"; \
	install -m 755 $(OUT_DIR)/$(BINARY) "$$dest.new" && mv -f "$$dest.new" "$$dest"; \
	echo "✓ Installed to $$dest"; \
	link="$$bindir/scan"; \
	if [ -e "$$link" ] || [ -L "$$link" ]; then \
		echo "• Left existing $$link untouched (not linking scan -> $(INSTALL_NAME))"; \
	else \
		ln -s "$(INSTALL_NAME)" "$$link" && echo "✓ Linked $$link -> $(INSTALL_NAME)"; \
	fi; \
	legacy="$$bindir/ascan"; \
	if [ -f "$$legacy" ] && [ ! -L "$$legacy" ]; then \
		rm -f "$$legacy" && ln -s "$(INSTALL_NAME)" "$$legacy" \
			&& echo "✓ Migrated legacy $$legacy (old binary) -> $(INSTALL_NAME)"; \
	fi

# Remove a local `make install`: the `atomscan` binary plus the `scan`/`ascan`
# symlinks, but ONLY when those point at `atomscan` — a foreign `scan` (avast)
# or a real `ascan` binary is left untouched.
uninstall:
	@set -e; \
	found=0; \
	for bindir in "$$HOME/.cargo/bin" "$$HOME/bin" "$$HOME/.local/bin" /usr/local/bin; do \
		[ -d "$$bindir" ] || continue; \
		target="$$bindir/$(INSTALL_NAME)"; \
		{ [ -e "$$target" ] || [ -L "$$target" ]; } || continue; \
		found=1; \
		rm -f "$$target" && echo "✓ Removed $$target"; \
		for alias in scan ascan; do \
			link="$$bindir/$$alias"; \
			if [ -L "$$link" ] && [ "$$(readlink "$$link")" = "$(INSTALL_NAME)" ]; then \
				rm -f "$$link" && echo "✓ Removed $$link -> $(INSTALL_NAME)"; \
			elif [ -e "$$link" ] || [ -L "$$link" ]; then \
				echo "• Left $$link untouched (not our symlink)"; \
			fi; \
		done; \
	done; \
	[ "$$found" = 1 ] || echo "Nothing to uninstall (no $(INSTALL_NAME) in known bindirs)"

tarball: release
	tar -czf $(OUT_DIR)/scan.tgz -C $(OUT_DIR) $(BINARY)
	@echo "Tarball: $(OUT_DIR)/scan.tgz"

# Clear MAKEFLAGS for deploy recipes: GNU Make would otherwise inject the
# outer invocation's `-j`/`--jobserver-*` flags plus command-line `URL=` into
# the env, which tikv-jemalloc-sys's build.rs re-passes to its bundled `make`,
# producing `*** No rule to make target '-j'` inside the jemalloc build.
deploy-server deploy-worker deploy-jail-worker deploy-bloomer bloom-cron deploy-worker-nodes rollout-bastille: export MAKEFLAGS :=

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
		FreeBSD) ./scripts/worker/worker-freebsd.sh "$(URL)" ;; \
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

# Install the hourly bloom build+publish timer (scan-bloomer.timer). Runs as a
# dedicated `bloom` system user (isolated from the worker's `scan` user), from a
# provisioned checkout under /var/lib/bloom. The install script sets up the
# checkouts + Rust toolchain and tells you which secrets to drop in (codeberg
# push key, rclone R2 remote, ~/.pgpass for hopper). Systemd Linux only.
deploy-bloomer:
	@case "$$(uname -s)" in \
		Linux) if command -v systemctl >/dev/null 2>&1; then \
		           ./scripts/worker/bloomer-linux.sh; \
		         else \
		           echo "error: deploy-bloomer needs a systemd Linux host"; exit 1; \
		         fi ;; \
		*) echo "error: no deploy-bloomer target for $$(uname -s) (systemd Linux only)"; exit 1 ;; \
	esac

# Jailed worker deploy: builds in a Bastille build jail and runs the rc.d
# service inside a separate run jail. Use this instead of deploy-worker when
# isolating the worker in a jail; deploy-worker installs natively on the host.
deploy-jail-worker:
	@[ -n "$(URL)" ] || { echo "Usage: make deploy-jail-worker URL=<url> [BUILD=<jail>] [WORKER_RUN=<jail>]"; exit 1; }
	git stash
	git pull
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/worker/worker-bastille.sh "$(BUILD)" "$(WORKER_RUN)" "$(URL)" ;; \
		*) echo "error: jail worker deployments are FreeBSD/bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

uninstall-server:
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/server/uninstall-bastille.sh "$(SERVER_RUN)" ;; \
		*) echo "error: server deployments are bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

# Hard-stop the running worker without uninstalling it: stops the systemd unit
# (so Restart= can't respawn) then escalates SIGTERM -> SIGKILL. Used before a
# redeploy so the old process frees its RAM before the rebuild. Idempotent.
stop-worker:
	./scripts/worker/stop-worker.sh

uninstall-worker:
	@case "$$(uname -s)" in \
		Darwin)  ./scripts/worker/uninstall-macos.sh ;; \
		FreeBSD) ./scripts/worker/uninstall-freebsd.sh ;; \
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

# Stop and remove the hourly bloom timer (counterpart to deploy-bloomer). Leaves
# the `bloom` user and its checkouts/credentials under /var/lib/bloom in place.
uninstall-bloomer:
	@case "$$(uname -s)" in \
		Linux) ./scripts/worker/uninstall-bloomer-linux.sh ;; \
		*) echo "error: no uninstall-bloomer target for $$(uname -s) (systemd Linux only)"; exit 1 ;; \
	esac

# Remove the jailed worker service (counterpart to deploy-jail-worker).
uninstall-jail-worker:
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/worker/uninstall-bastille.sh "$(WORKER_RUN)" ;; \
		*) echo "error: jail worker deployments are FreeBSD/bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

deploy-worker-nodes:
	@[ -n "$(URL)" ] || { echo "Usage: make deploy-worker-nodes URL=<url> NODES=\"node1 node2\""; exit 1; }
	@[ -n "$(NODES)" ] || { echo "Usage: make deploy-worker-nodes URL=<url> NODES=\"node1 node2\""; exit 1; }
	./scripts/worker/update-nodes.sh "$(URL)" $(NODES)

# Roll the standing worker pool, one node at a time (YubiKey-friendly), then
# redeploy hopper. Each worker runs `git pull && make stop-worker &&
# make deploy-worker` over SSH. Override defaults with URL=, WORKER_NODES=,
# HOPPER_NODE= (see scripts/worker/deploy-workers.sh).
deploy-workers:
	URL="$(URL)" WORKER_NODES="$(WORKER_NODES)" HOPPER_NODE="$(HOPPER_NODE)" \
		./scripts/worker/deploy-workers.sh

# Same roll, but fanned out into one tmux window per node (builds run in
# parallel). Launches are staggered (STAGGER=5s) so the YubiKey faces one SSH
# touch prompt at a time. Override URL=, WORKER_NODES=, HOPPER_NODE=, STAGGER=.
deploy-workers-tmux:
	URL="$(URL)" WORKER_NODES="$(WORKER_NODES)" HOPPER_NODE="$(HOPPER_NODE)" STAGGER="$(STAGGER)" \
		./scripts/worker/deploy-workers-tmux.sh

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
	./out/$(BINARY) worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

# Run a worker in the foreground for interactive use. The worker self-nices
# to 10 by default; pass NICE=0 to disable.
worker: release
	@[ -n "$(URL)" ] || { echo "Usage: make worker URL=<hopper-url> [WORKERS=<n>] [NICE=<int>] [LLM=<endpoint>]"; exit 1; }
	SCAN_LLM="$(LLM)" ./out/$(BINARY) worker --url "$(URL)" \
		--interpret --interpret-min-prob $(INTERPRET_MIN_PROB) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(NICE),--nice $(NICE),)

# Self-contained worker benchmark: start the bundled mock hopper on a local
# dataset, point a real worker at it, and measure the worker's wall + maxrss
# plus per-job latency (claim → result, measured hopper-side). Iterate with
# realworld-small; reserve realworld for final verification. Override the
# dataset, slot count, and job handout order, e.g.:
#   make worker-benchmark WORKER_BENCH_DATASET=realworld-small WORKERS=12 ORDER=big-first
# ORDER: fifo (default), shuffle[:seed] (reproducible realistic mix),
# big-first (small-job-starvation worst case), small-first.
WORKER_BENCH_DATASET ?= $(DATASET)
WORKER_BENCH_PATH    ?= $(BENCHMARK_ROOT)/$(WORKER_BENCH_DATASET)
HEARTBEAT_SECS       ?= 5
ORDER                ?= fifo
BENCH_SUMMARY        ?= /tmp/litmus-worker-bench-summary.json
worker-benchmark: release ## Benchmark the worker model over a local dataset via the bundled mock hopper
	@[ -e "$(WORKER_BENCH_PATH)" ] || { echo "error: dataset not found: $(WORKER_BENCH_PATH)"; exit 1; }
	@echo "worker-benchmark: dataset=$(WORKER_BENCH_PATH) workers=$(if $(WORKERS),$(WORKERS),default) order=$(ORDER) max_rss_gb=$(if $(MAX_RSS_GB),$(MAX_RSS_GB),auto)"
	@out=$$(mktemp); results=/tmp/litmus-worker-results.jsonl; rm -f $$results "$(BENCH_SUMMARY)"; \
	./target/release/scan-bench-hopper --dataset "$(WORKER_BENCH_PATH)" --port 0 --dump $$results \
		--order "$(ORDER)" --summary "$(BENCH_SUMMARY)" \
		>$$out 2>/tmp/scan-bench-hopper.err & \
	hp=$$!; \
	trap 'kill $$hp 2>/dev/null' EXIT INT TERM; \
	port=; jobs=; \
	for i in $$(seq 1 100); do \
		port=$$(sed -n 's/^PORT=//p' $$out); \
		jobs=$$(sed -n 's/^JOBS=//p' $$out); \
		[ -n "$$port" ] && [ -n "$$jobs" ] && break; \
		sleep 0.1; \
	done; \
	[ -n "$$port" ] || { echo "error: mock hopper did not start"; cat $$out; exit 1; }; \
	echo "hopper: port=$$port jobs=$$jobs"; \
	tflag=$$( [ "$$(uname -s)" = "Darwin" ] && echo -l || echo -v ); \
	CLEAVE_SKIP_CACHE=1 SCAN_HEARTBEAT_SECS=$(HEARTBEAT_SECS) \
	$(if $(GRID),SCAN_PER_SLOT_POOLS=1,) \
	/usr/bin/time $$tflag ./out/$(BINARY) worker \
		--url "http://127.0.0.1:$$port" \
		--data-dir "$(WORKER_BENCH_PATH)" \
		--exit-if-empty --nice 0 --no-update --no-validate \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(MAX_RSS_GB),--max-rss-gb $(MAX_RSS_GB),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log; \
	sleep 0.5; \
	received=$$(sort -u $$results 2>/dev/null | grep -c .); \
	echo "✓ log: /tmp/litmus-worker-benchmark.log"; \
	echo "--- thread budget (from log) ---"; \
	grep -E "thread budget|oversubscribes|rayon pool ready" /tmp/litmus-worker-benchmark.log | head; \
	echo "--- completeness (SERVER-side): hopper received $$received / $$jobs results ---"; \
	[ "$$received" = "$$jobs" ] && echo "✓ COMPLETE: all $$jobs results reached the hopper" \
		|| echo "❌ INCOMPLETE: $$received/$$jobs — worker dropped results (see /tmp/scan-bench-hopper.err)"; \
	echo "--- latency summary (hopper-side, claim → result) ---"; \
	cat "$(BENCH_SUMMARY)" 2>/dev/null || echo "(no summary written)"; \
	echo "✓ summary: $(BENCH_SUMMARY)"

profile-worker:
	$(CARGO) build --profile profiling
	@[ -n "$(URL)" ] || { echo "Usage: make profile-worker URL=<hopper-url>"; exit 1; }
	samply record -o /tmp/litmus-worker-profile.json.gz -- \
		./target/profiling/$(BINARY) worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

profile-slow:
	$(CARGO) build --profile profiling
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	samply record --save-only --duration 20 -o /tmp/litmus-$(DATASET)-profile.json.gz -- \
		env CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" CLEAVE_SKIP_YARA_CACHE=0 ./target/profiling/$(BINARY) --slow-rule-ms "$(SLOW_RULE_MS)" -f json "$(BENCHMARK_PATH)"

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
	CLEAVE_SKIP_CACHE=1 CLEAVE_SKIP_YARA_CACHE=0 samply record --save-only -o $(OUT_DIR)/bench.profile.json.gz -- \
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
		--name scan \
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

# Auto-fix what clippy and rustfmt can fix on their own. Run fmt last so it
# tidies any code clippy rewrote. Mirrors what `lint` checks.
fix:
	$(CARGO) clippy --fix --allow-dirty --allow-staged
	$(CARGO) fmt

test:
	@set -e; \
	if command -v cargo-nextest >/dev/null 2>&1; then \
		$(CARGO) test --lib bench_hopper::tests --quiet -- --test-threads=1; \
		$(CARGO) test --lib worker::tests::prefetcher_fills_to_target_backpressures_and_refills --quiet; \
		$(CARGO) nextest run --workspace -- \
			--skip bench_hopper::tests \
			--skip worker::tests::prefetcher_fills_to_target_backpressures_and_refills; \
		$(CARGO) test --doc --quiet; \
	else \
		$(CARGO) test --workspace; \
	fi

test-unit:
	@set -e; \
	if command -v cargo-nextest >/dev/null 2>&1; then \
		$(CARGO) test --lib bench_hopper::tests --quiet -- --test-threads=1; \
		$(CARGO) test --lib worker::tests::prefetcher_fills_to_target_backpressures_and_refills --quiet; \
		$(CARGO) nextest run --lib -- \
			--skip bench_hopper::tests \
			--skip worker::tests::prefetcher_fills_to_target_backpressures_and_refills; \
	else \
		$(CARGO) test --lib; \
	fi

# Install the pre-commit gate (no local Cargo.toml path overrides + make lint +
# make test). Bypass an individual commit with `git commit --no-verify`.
install-precommit:
	cp scripts/pre-commit "$$(git rev-parse --git-dir)/hooks/pre-commit"
	chmod +x "$$(git rev-parse --git-dir)/hooks/pre-commit"
	@echo "✓ Pre-commit hook installed."

clean:
	$(CARGO) clean
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ----- Wolfi packaging ----------------------------------------------------
# Build a Wolfi-based OCI image for scan via melange + apko. Mirrors
# the cleave/packaging/wolfi flow; depends on a sibling cleave checkout
# at ../cleave (the local build stages cleave + filefacts + the scan
# source repo and overrides the cleave git dep via a Cargo [patch] block).
# On macOS the build runs inside a dedicated Lima VM (`scan-wolfi`). See
# packaging/wolfi/README.md.
WOLFI_DIR = packaging/wolfi
WOLFI_OUT = $(OUT_DIR)/wolfi
WOLFI_ARCH ?=

wolfi: wolfi-bootstrap wolfi-build wolfi-test

wolfi-bootstrap:
	@$(WOLFI_DIR)/scripts/bootstrap-lima.sh

wolfi-build:
	@WOLFI_ARCH="$(WOLFI_ARCH)" $(WOLFI_DIR)/scripts/build.sh
	@echo "✓ Wolfi image: $(WOLFI_OUT)/scan.tar"

wolfi-test:
	@$(WOLFI_DIR)/scripts/smoke-test.sh

wolfi-shell:
	@[ -f $(WOLFI_OUT)/scan.tar ] || { echo "error: run 'make wolfi-build' first"; exit 1; }
	@case "$$(uname -s)" in \
		Darwin) limactl shell --workdir / scan-wolfi nerdctl run --rm -it --entrypoint /bin/sh scan:smoke ;; \
		Linux)  for r in nerdctl docker podman; do command -v $$r >/dev/null 2>&1 && { exec $$r run --rm -it --entrypoint /bin/sh scan:smoke; }; done; echo "no container runtime"; exit 1 ;; \
	esac

wolfi-clean:
	rm -rf $(WOLFI_OUT)
	@echo "✓ Wolfi output cleaned"

wolfi-nuke: wolfi-clean
	@case "$$(uname -s)" in \
		Darwin) limactl delete --force scan-wolfi 2>/dev/null || true ;; \
	esac
	rm -rf $$HOME/.cache/scan-wolfi
	@echo "✓ Wolfi VM and cache removed"

# ----- Publish -----------------------------------------------------------
# Push the multi-arch scan image to a registry and sign it keyless with
# cosign via Google OIDC. Override REGISTRY / ORG / ARCHS via env. See
# packaging/wolfi/README.md for prerequisites.
REGISTRY ?= docker.io
ORG      ?= atomdrift

docker-login: wolfi-bootstrap ## Log the lima VM's runtime into REGISTRY (interactive)
	@case "$$(uname -s)" in \
		Darwin) limactl shell --workdir / scan-wolfi nerdctl login $(REGISTRY) ;; \
		Linux)  for r in nerdctl docker podman; do command -v $$r >/dev/null 2>&1 && { exec $$r login $(REGISTRY); }; done; echo "no container runtime"; exit 1 ;; \
	esac

docker-publish: wolfi-bootstrap ## Build multi-arch + push + cosign sign (set DRY_RUN=1 to skip the push)
	@REGISTRY="$(REGISTRY)" ORG="$(ORG)" \
		$(WOLFI_DIR)/scripts/publish.sh
