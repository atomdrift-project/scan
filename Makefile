SHELL := /bin/sh

# --- Windows bootstrap ------------------------------------------------------
# GNU Make for Windows resolves `/bin/sh` (used for every recipe below, e.g.
# `check-hopper-token`'s POSIX `if`/`then`) by searching PATH for `sh.exe`; a
# fresh shell has no reason to have Git's `usr/bin` (sh, awk, grep, ...) on
# PATH, so that search fails and make silently falls back to cmd.exe, which
# cannot parse this file's recipes -- every one then breaks, with
# "CreateProcess ... failed" or a cmd syntax error as the only clue. Rather
# than requiring a PATH/HOME already set up by hand (a PowerShell-profile
# edit, easy to forget or to make in the wrong profile -- Windows PowerShell
# 5.1 and PowerShell 7 do not share one), find Git for Windows here and put
# its `usr/bin` on PATH ourselves, so `/bin/sh`'s own PATH search succeeds
# and `make` just works from a stock shell. (Leave SHELL itself alone: an
# absolute override here once made a `$(shell ...)` call's own argument
# splitting misbehave -- see the PACKAGE comment below.)
ifeq ($(OS),Windows_NT)
  nullstring :=
  space := $(nullstring) $(nullstring)
  # The two real-world install roots: the admin-installed default, and the
  # per-user one the installer uses without admin rights. `$(wildcard)`
  # splits its argument on whitespace, so the embedded space in "Program
  # Files" must be backslash-escaped for the existence check; the value kept
  # in WIN_GIT_ROOT has the real space back, since SHELL/PATH are plain
  # strings and are never re-split by make.
  WIN_GIT_CAND1_RAW := C:/Program Files/Git
  WIN_GIT_CAND1_ESC := $(subst $(space),\$(space),$(WIN_GIT_CAND1_RAW))
  WIN_GIT_CAND2_RAW := $(subst \,/,$(LOCALAPPDATA))/Programs/Git
  WIN_GIT_CAND2_ESC := $(subst $(space),\$(space),$(WIN_GIT_CAND2_RAW))
  WIN_GIT_ROOT_2 := $(if $(wildcard $(WIN_GIT_CAND2_ESC)/usr/bin/sh.exe),$(WIN_GIT_CAND2_RAW),)
  WIN_GIT_ROOT := $(if $(wildcard $(WIN_GIT_CAND1_ESC)/usr/bin/sh.exe),$(WIN_GIT_CAND1_RAW),$(WIN_GIT_ROOT_2))
  ifeq ($(WIN_GIT_ROOT),)
    $(error Git for Windows not found at "$(WIN_GIT_CAND1_RAW)" or "$(WIN_GIT_CAND2_RAW)" -- install it (https://git-scm.com/download/win), or if it lives elsewhere, add its usr/bin to PATH yourself)
  endif
  export PATH := $(WIN_GIT_ROOT)/usr/bin;$(PATH)
  # `?=` only: an already-set HOME (Git Bash sets its own) is never overridden.
  HOME ?= $(USERPROFILE)
  export HOME
endif

# The CLI command is `atomscan` — the cargo bin, the build artifact, and the
# installed binary all share this name. `scan` is not a safe global command name
# (avast ships its own /usr/bin/scan), so we no longer install one. The product
# name stays "Atomdrift Scan"; only the invocation is `atomscan`.
BINARY = atomscan
# Cargo package name, which is not always the binary name (scan's package is
# `atomdrift-scan` but ships `atomscan`). Read from Cargo.toml so `cut-release`
# passes the right `-p` without a second place to keep in sync.
#
# Windows needs a different invocation: GNU Make's Windows port runs a
# "simple" `$(shell ...)` command through its own naive, double-quote-only
# command-line splitter instead of a real shell, so the single quotes below
# (needed for awk's `-F'"'` and the `{print $2; exit}` script) get shredded
# into separate, garbled arguments -- a harmless but alarming
# "awk: cmd. line:1: = / ^ syntax error" on every invocation. Wrapping the
# whole thing as one double-quoted argument to `sh -c "..."` makes that
# splitter treat it as a single opaque string (it doesn't do variable
# expansion, just quote-grouping), so real Git-Bash `sh` receives the
# original, unmangled script and parses it correctly.
#
# This exact wrapper must stay Windows-only: routed through a *real* shell
# (any actual POSIX system, or `$(shell)` on a non-Windows make) it is wrong
# -- a real shell expands `$2` inside the outer double quotes before the
# inner `sh -c` ever sees it, printing the whole matched line instead of the
# second field.
ifeq ($(OS),Windows_NT)
  PACKAGE := $(shell sh -c "awk -F'\"' '/^name = /{print $$2; exit}' Cargo.toml")
else
  PACKAGE := $(shell awk -F'"' '/^name = /{print $$2; exit}' Cargo.toml)
endif

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
# $(HOME)/data/benchmark on every platform — the hardcoded /Users/t path meant
# every benchmark/profile target aborted with "benchmark path not found" on the
# Linux box the nightly gauntlet actually runs on.
BENCHMARK_ROOT ?= $(HOME)/data/benchmark
BENCHMARK_PATH ?= $(BENCHMARK_ROOT)/$(DATASET)
SCAN_THREADS ?=
MAX_JOBS ?= 25
WORKERS  ?=
MAX_RSS_GB ?=
URL ?= http://10.9.8.10:8081/
# Hopper's API token. Hopper requires `Authorization: Bearer <token>` on every
# route and does not exempt loopback, so the pull worker and every
# `--hopper`/`--upload` run need it. `atomscan` reads it from `~/.tok/hopper`
# in its own home (`$HOPPER_TOKEN` overrides); the deploy scripts copy this
# file into the service account's home, since a supervised service does not
# share the operator's. Exported so a command-line override reaches them.
HOPPER_TOKEN_FILE ?= $(HOME)/.tok/hopper
export HOPPER_TOKEN_FILE

# The LLM endpoint's bearer token. Our vLLM requires one on every /v1 route, so
# a deploy without it drops that endpoint from the failover chain and grades on
# whatever is left (OpenRouter, or nothing). `atomscan` reads it from
# `~/.tok/llm` in its own home (`SCAN_LLM_KEY` overrides); the deploy scripts
# copy this file into the service account's home, since a supervised service
# does not share the operator's. Exported so a command-line override reaches
# them.
LLM_TOKEN_FILE ?= $(HOME)/.tok/llm
export LLM_TOKEN_FILE

# --- `make deploy` / `make deploy-server` knobs ------------------------------
# Read by scripts/server/server-linux.sh (Linux), server-freebsd.sh (FreeBSD)
# and rollout-bastille.sh (`make deploy-jail`); see docs/SERVER_API.md.
#
#   make deploy HOPPER=https://hops.isotope13.ai
#
# HOPPER is the hopper the server renews results on (`serve --hopper`). Setting
# it is also what makes the deploy install HOPPER_TOKEN_FILE for the service
# account — hopper rejects an unauthenticated renewal with 401.
#
# It may name the same corpus twice, replica first and primary behind it, so a
# replica outage costs a retry rather than a lost verdict:
#
#   make deploy HOPPER=https://hops-ro.isotope13.ai,https://hops.isotope13.ai
#
# Lookups and renewals walk that list. The idle worker does not: it claims from
# the primary alone, because a replica refuses worker routes with a 403.
#
# It is REQUIRED: `deploy-server` refuses to install a server that files
# nothing. Pass HOPPER=none to opt out deliberately.
HOPPER ?=
BIND ?=
ALLOWED_DIRS ?=
MEMORY_MAX ?=

# IDLE caps the embedded idle worker: analysis slots `serve` may spend on
# hopper queue work while no request is in flight. Unset keeps the server's own
# default (half of --workers, rounded down). 0 turns background claiming off
# entirely, so the host only ever works on interactive requests:
#
#   make deploy IDLE=0
#
# The cap is half the slots either way — the server clamps a larger IDLE — and
# the idle worker is off regardless when HOPPER is unset, since there would be
# nothing to claim from.
IDLE ?=

# HOPPER=none is the deliberate opt-out. It collapses to an empty HOPPER here,
# before the export, so the deploy scripts see "unset" and emit no --hopper —
# they must never receive the literal string as a URL.
ifeq ($(strip $(HOPPER)),none)
override HOPPER :=
HOPPER_OPTOUT := 1
endif

export HOPPER BIND ALLOWED_DIRS MEMORY_MAX IDLE
# ALLOW_CIDR and TOKEN_SRC are deliberately NOT declared here. They default
# with `${VAR-default}` (unset) rather than `${VAR:-default}` (unset or empty),
# because empty is a meaningful value: no CIDR allow-list, and no
# authentication at all. Declaring them would export an empty value on every
# deploy and silently strip both. Pass them on the command line instead — make
# exports command-line variables, so they still reach the script:
#   make deploy ALLOW_CIDR=192.168.0.0/16
#   make deploy TOKEN_SRC=            # deliberately unauthenticated

# LLM second-opinion pass for `make worker` (matches the deploy scripts'
# defaults). LLM / LLM_URL is exported as SCAN_LLM (`local`, `openrouter`, or a
# base URL). LLM_MODEL is SCAN_LLM_MODEL (required for OpenRouter). The endpoint
# requires a bearer token; atomscan reads it from ~/.tok/llm, and the deploy
# scripts install that file for the service account (LLM_TOKEN_FILE overrides). The
# benchmark/profile targets deliberately omit interpret so LLM
# round-trips don't distort wall/RSS measurements.
# Comma-separated is a failover chain, tried in order: our own vLLM first, the
# billed public API only when it cannot answer. LLM_MODEL pairs positionally
# with it — an empty first slot asks our endpoint what it serves, while
# OpenRouter's catalog is never auto-selected and so must be named.
LLM ?= https://llm.isotope13.ai/v1,openrouter
LLM_MODEL ?= ,qwen/qwen3.8-27b

# Scrub GNU make's jobserver from cargo's environment. Without this, build
# scripts that spawn their own `make` (e.g. tikv-jemalloc-sys) inherit a
# malformed MAKEFLAGS and fail with "No rule to make target '-j'".
CARGO = env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS cargo

.PHONY: pgo-train ensure-llm-token bench-archive bench-archive-scaling profile-archive bench-typed bench-typed-extract bench-typed-goal baseline-typed-detection check-typed-detection build release release-lto install uninstall check-cargo check-hopper-token check-hopper-url tarball deploy deploy-server deploy-jail deploy-worker deploy-jail-worker deploy-worker-nodes deploy-workers deploy-workers-tmux uninstall-server uninstall-jail uninstall-server-nodes stop-worker kill-scan uninstall-worker uninstall-jail-worker uninstall-worker-nodes rollout-bastille benchmark benchmark-worker worker-benchmark server-benchmark server-heap-benchmark worker profile-worker profile-slow bench-build sampled-benchmark heap-build heap-benchmark tuna tuna-once lint fix test test-unit install-precommit clean wolfi wolfi-bootstrap wolfi-build wolfi-test wolfi-shell wolfi-clean wolfi-nuke docker-login docker-publish cut-release

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

# Profile-guided optimization. When the local profile exists (`make pgo-train`
# writes it; it is untracked — ~100 MB and host-generated), release builds use
# it: measured 2026-08-31 on the poppy worker benchmark, wall −8% / CPU −11%
# with byte-identical output. Without the file the build is exactly as before.
# rustc prints "no profile data available" warnings for cold functions under
# profile-use; they are expected and harmless.
PGO_PROFDATA ?= build/pgo/atomscan.profdata
PGO_FLAGS = $(if $(wildcard $(PGO_PROFDATA)),-C profile-use=$(abspath $(PGO_PROFDATA)),)

release: check-cargo $(OUT_DIR)
	@if [ -n "$(PGO_FLAGS)" ]; then echo "release: PGO enabled ($(PGO_PROFDATA))"; else echo "release: no PGO profile at $(PGO_PROFDATA) — run 'make pgo-train' to create one"; fi
	env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS RUSTFLAGS="$$RUSTFLAGS $(PGO_FLAGS)" cargo build --release
	cp target/release/$(BINARY) $(OUT_DIR)/$(BINARY).new && mv -f $(OUT_DIR)/$(BINARY).new $(OUT_DIR)/$(BINARY)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		codesign --force --sign - $(OUT_DIR)/$(BINARY); \
	fi

# Kill any locally-running atomscan processes before a build so a busy worker
# (or server) doesn't steal CPU from the compile. Wired into `worker` and
# `deploy-worker`. Portable across Linux/macOS/FreeBSD via pkill/pgrep, matching
# the exact process name (-x) so cargo/make/grep aren't caught. Friendly SIGTERM
# first, then SIGKILL for anything still alive after ~5s. A missing pkill or no
# matching process is not an error.
kill-scan:
	@if command -v pkill >/dev/null 2>&1; then \
		if pkill -x $(BINARY) 2>/dev/null; then \
			echo "kill-scan: sent SIGTERM to running $(BINARY); waiting for graceful exit..."; \
			i=0; while pgrep -x $(BINARY) >/dev/null 2>&1 && [ $$i -lt 5 ]; do sleep 1; i=$$((i+1)); done; \
			if pgrep -x $(BINARY) >/dev/null 2>&1; then \
				echo "kill-scan: process still alive, sending SIGKILL"; \
				pkill -9 -x $(BINARY) 2>/dev/null || true; \
			fi; \
			echo "kill-scan: cleared existing $(BINARY) process(es)"; \
		else \
			echo "kill-scan: no running $(BINARY) to stop"; \
		fi; \
	else \
		echo "kill-scan: pkill not available; skipping"; \
	fi


# Fat LTO + single codegen unit. Multi-minute link, marginal runtime win
# over the default release profile. Use for container/tarball builds.
release-lto: check-cargo $(OUT_DIR)
	@if [ -n "$(PGO_FLAGS)" ]; then echo "release-lto: PGO enabled ($(PGO_PROFDATA))"; fi
	env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS RUSTFLAGS="$$RUSTFLAGS $(PGO_FLAGS)" cargo build --profile release-lto
	cp target/release-lto/$(BINARY) $(OUT_DIR)/$(BINARY).new && mv -f $(OUT_DIR)/$(BINARY).new $(OUT_DIR)/$(BINARY)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		codesign --force --sign - $(OUT_DIR)/$(BINARY); \
	fi

# Regenerate the PGO profile: build an instrumented release, scan the training
# corpus with it (analysis pipeline only, fetch off), merge the counters. The
# training set defaults to the quarter-size worker benchmark corpus; any
# representative directory works — PGO only needs the hot paths exercised.
# Requires rustup's llvm-tools (`rustup component add llvm-tools`).
PGO_TRAIN_PATH ?= $(BENCHMARK_ROOT)/poppy-q
pgo-train: check-cargo
	@[ -e "$(PGO_TRAIN_PATH)" ] || { echo "error: training corpus not found: $(PGO_TRAIN_PATH) (set PGO_TRAIN_PATH)"; exit 1; }
	@lp=$$(find ~/.rustup -name llvm-profdata | head -1); [ -n "$$lp" ] || { echo "error: llvm-profdata not found — run: rustup component add llvm-tools"; exit 1; }
	rm -rf /tmp/scan-pgo-train && mkdir -p /tmp/scan-pgo-train build/pgo
	env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS RUSTFLAGS="-C profile-generate=/tmp/scan-pgo-train" cargo build --release
	@echo "pgo-train: scanning $(PGO_TRAIN_PATH) with the instrumented binary (slow: ~2x)"
	SCAN_FETCH=none SCAN_NO_ANALYSIS_CACHE=1 CLEAVE_SKIP_CACHE=1 target/release/$(BINARY) -f json "$(PGO_TRAIN_PATH)" > /dev/null || true
	$$(find ~/.rustup -name llvm-profdata | head -1) merge -o $(PGO_PROFDATA) /tmp/scan-pgo-train/*.profraw
	@echo "✓ PGO profile: $(PGO_PROFDATA) — subsequent 'make release' builds use it"

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
	dest="$$bindir/$(BINARY)"; \
	install -m 755 $(OUT_DIR)/$(BINARY) "$$dest.new" && mv -f "$$dest.new" "$$dest"; \
	echo "✓ Installed to $$dest"

# Remove a local `make install`: the `atomscan` binary, plus any stale `scan`/
# `ascan` symlinks left by older installs that still point at `atomscan` — a
# foreign `scan` (avast) or a real `ascan` binary is left untouched.
uninstall:
	@set -e; \
	found=0; \
	for bindir in "$$HOME/.cargo/bin" "$$HOME/bin" "$$HOME/.local/bin" /usr/local/bin; do \
		[ -d "$$bindir" ] || continue; \
		target="$$bindir/$(BINARY)"; \
		{ [ -e "$$target" ] || [ -L "$$target" ]; } || continue; \
		found=1; \
		rm -f "$$target" && echo "✓ Removed $$target"; \
		for alias in scan ascan; do \
			link="$$bindir/$$alias"; \
			if [ -L "$$link" ] && [ "$$(readlink "$$link")" = "$(BINARY)" ]; then \
				rm -f "$$link" && echo "✓ Removed stale $$link -> $(BINARY)"; \
			fi; \
		done; \
	done; \
	[ "$$found" = 1 ] || echo "Nothing to uninstall (no $(BINARY) in known bindirs)"

tarball: release
	tar -czf $(OUT_DIR)/$(BINARY).tgz -C $(OUT_DIR) $(BINARY)
	@echo "Tarball: $(OUT_DIR)/$(BINARY).tgz"

# Clear MAKEFLAGS for deploy recipes: GNU Make would otherwise inject the
# outer invocation's `-j`/`--jobserver-*` flags plus command-line `URL=` into
# the env, which tikv-jemalloc-sys's build.rs re-passes to its bundled `make`,
# producing `*** No rule to make target '-j'` inside the jemalloc build.
deploy-server deploy-jail deploy-worker deploy-jail-worker deploy-worker-nodes rollout-bastille: export MAKEFLAGS :=

deploy: deploy-server

# Cloudflare Tunnel for the server. "auto" installs and supervises a connector
# only when CF_TUNNEL_TOKEN is passed or a token from an earlier deploy is on
# disk, so a server reached over the LAN needs no extra flags; 0 skips it, 1
# requires it. CF_TUNNEL_TOKEN is read from the environment, never from here,
# so the token stays out of the repository and out of ps(1) on the deploy host.
CLOUDFLARED ?= auto

# Long-lived `atomscan serve`, installed natively on this host: a FreeBSD rc.d
# service (`service scan`) or a systemd unit (`scan.service`). Override BIND=,
# ALLOW_CIDR=, LLM= / LLM_URL=, LLM_MODEL=, MEMORY_MAX= (Linux), MAX_RSS_GB=
# (FreeBSD), CLOUDFLARED= — see scripts/server/server-freebsd.sh and
# scripts/server/server-linux.sh. For a jailed FreeBSD server use `make
# deploy-jail`.
# Put the LLM key where the deploy scripts look for it, if it is not there
# already. They copy $(LLM_TOKEN_FILE) into the service account's home; this
# only covers the case where the operator has the key in the environment and no
# file yet, which is how a fresh host gets one. Never overwrites: a file already
# on disk is the operator's, and clobbering it would silently rotate the fleet's
# credential on an unrelated deploy.
#
# Deliberately not fatal, unlike check-hopper-token. A worker without a hopper
# token is a silent no-op; a scan without an LLM key still scans — it loses the
# second opinion on that endpoint and falls through to the rest of the chain.
ensure-llm-token:
	@if [ -s "$(LLM_TOKEN_FILE)" ]; then exit 0; fi; \
	if [ -n "$${SCAN_LLM_KEY:-}" ]; then \
		umask 077; mkdir -p "$$(dirname "$(LLM_TOKEN_FILE)")"; \
		printf '%s\n' "$${SCAN_LLM_KEY}" > "$(LLM_TOKEN_FILE)"; \
		chmod 600 "$(LLM_TOKEN_FILE)"; \
		echo "installed the LLM endpoint token at $(LLM_TOKEN_FILE)"; \
	else \
		echo "warning: no LLM token at $(LLM_TOKEN_FILE) and SCAN_LLM_KEY unset."; \
		echo "         An endpoint that requires one will refuse this deploy's"; \
		echo "         second-opinion pass with 401; the chain falls through to"; \
		echo "         whatever else \$$LLM names. Write the token to that file"; \
		echo "         or pass SCAN_LLM_KEY=<token> to install it."; \
	fi

deploy-server: export CLOUDFLARED := $(CLOUDFLARED)
deploy-server: check-hopper-url ensure-llm-token
	git pull
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/server/server-freebsd.sh ;; \
		Linux)   if command -v systemctl >/dev/null 2>&1; then \
		           ./scripts/server/server-linux.sh; \
		         else \
		           echo "error: unsupported Linux (systemd required for server deploy)"; exit 1; \
		         fi ;; \
		*) echo "error: no deploy-server target for $$(uname -s) (FreeBSD/rc.d or Linux/systemd)"; exit 1 ;; \
	esac

# Refuse to start a worker without a hopper credential. Hopper requires
# `Authorization: Bearer <token>` on every route and does not exempt loopback,
# so a worker without one 401s on every poll and retries forever behind a
# heartbeat that still looks healthy — a silent no-op that used to cost a whole
# run. Mirrors `resolve_credential` in src/upload.rs: `$HOPPER_TOKEN` wins,
# otherwise the file must hold a non-blank line. Deliberately not wired into
# `deploy-worker-nodes` / `deploy-workers`, which use each remote account's own
# token and never ship one from here.
check-hopper-token:
	@if [ -n "$${HOPPER_TOKEN:-}" ] && printf '%s' "$${HOPPER_TOKEN}" | grep -q '[^[:space:]]'; then exit 0; fi; \
	if [ ! -f "$(HOPPER_TOKEN_FILE)" ]; then \
		echo "error: no hopper API token at $(HOPPER_TOKEN_FILE)"; \
		echo "       an authenticated hopper rejects every poll from this worker with 401."; \
		echo "       install the token there, or pass HOPPER_TOKEN_FILE=<path> / HOPPER_TOKEN=<token>."; \
		exit 1; \
	fi; \
	if ! grep -q '[^[:space:]]' "$(HOPPER_TOKEN_FILE)" 2>/dev/null; then \
		echo "error: hopper API token at $(HOPPER_TOKEN_FILE) is empty"; \
		echo "       an authenticated hopper rejects every poll from this worker with 401."; \
		echo "       write the token to that file, or pass HOPPER_TOKEN_FILE=<path> / HOPPER_TOKEN=<token>."; \
		exit 1; \
	fi

# Refuse to install a server that files nothing. Without `serve --hopper` the
# server answers every analysis and stores none of them: the caller caches the
# verdict, so the same PURL is never asked again, and hopper never receives the
# artifact. Nothing fails at deploy time and nothing fails at request time —
# the gap only surfaces much later, as a sample hopper should have had and
# does not. That is the failure this check exists to prevent; a scan server
# with no hopper to feed is not a configuration anyone wants by accident.
#
# Mirrors check-hopper-token: HOPPER=none is the explicit escape hatch, the
# same shape as `TOKEN_SRC=` for a deliberately unauthenticated server.
check-hopper-url:
	@if [ -n "$(HOPPER_OPTOUT)" ]; then \
		echo "warning: HOPPER=none — this server will analyze and file nothing on hopper"; \
		exit 0; \
	fi; \
	if [ -n "$(strip $(HOPPER))" ]; then exit 0; fi; \
	echo "error: HOPPER is unset, so the server would run without --hopper."; \
	echo "       It would answer every analysis and file none of them: the caller"; \
	echo "       caches the verdict and hopper never sees the artifact, so the loss"; \
	echo "       is silent until something asks hopper for a sample it should hold."; \
	echo; \
	echo "       make deploy HOPPER=https://hops.isotope13.ai"; \
	echo "       make deploy HOPPER=none    # deliberately file nothing"; \
	exit 1

deploy-worker: check-hopper-token ensure-llm-token kill-scan
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
		MINGW*|MSYS*) \
		         if command -v pwsh >/dev/null 2>&1; then ps=pwsh; else ps=powershell; fi; \
		         "$$ps" -NoProfile -ExecutionPolicy Bypass -File scripts/worker/worker-windows.ps1 -Url "$(URL)" ;; \
		*) echo "error: no deploy-worker target for $$(uname -s)"; exit 1 ;; \
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
		FreeBSD) ./scripts/server/uninstall-freebsd.sh ;; \
		Linux)   if command -v systemctl >/dev/null 2>&1; then \
		           ./scripts/server/uninstall-linux.sh; \
		         else \
		           echo "error: unsupported Linux (systemd required)"; exit 1; \
		         fi ;; \
		*) echo "error: no uninstall-server target for $$(uname -s)"; exit 1 ;; \
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
		MINGW*|MSYS*) \
		         if command -v pwsh >/dev/null 2>&1; then ps=pwsh; else ps=powershell; fi; \
		         "$$ps" -NoProfile -ExecutionPolicy Bypass -File scripts/worker/uninstall-windows.ps1 ;; \
		*) echo "error: no uninstall-worker target for $$(uname -s)"; exit 1 ;; \
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

# Jailed server deploy: builds in a Bastille build jail and runs the rc.d
# service inside a separate run jail. Use this instead of deploy-server when
# isolating the server in a jail; deploy-server installs natively on the host.
deploy-jail: check-hopper-url
	./scripts/server/rollout-bastille.sh "$(BUILD)" "$(SERVER_RUN)"

# Remove the jailed server service (counterpart to deploy-jail).
uninstall-jail:
	@case "$$(uname -s)" in \
		FreeBSD) ./scripts/server/uninstall-bastille.sh "$(SERVER_RUN)" ;; \
		*) echo "error: jail server deployments are FreeBSD/bastille-only; run from a FreeBSD host"; exit 1 ;; \
	esac

# Historical name for deploy-jail; kept so existing muscle memory and scripts
# keep working.
rollout-bastille: deploy-jail

benchmark: release
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	@# Exit 1/2 mean hostile/suspicious found — expected on a malware corpus;
	@# only scan errors (3+) fail the benchmark.
	CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) -f json "$(BENCHMARK_PATH)" > /tmp/litmus-benchmark-$(DATASET).json; \
	status=$$?; [ $$status -le 2 ] || exit $$status

benchmark-worker: release
	@[ -n "$(URL)" ] || { echo "Usage: make benchmark-worker URL=<hopper-url>"; exit 1; }
	./out/$(BINARY) worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

# Run a worker in the foreground for interactive use. The worker self-nices
# to 10 by default; pass NICE=0 to disable.
worker: check-hopper-token kill-scan release
	@[ -n "$(URL)" ] || { echo "Usage: make worker URL=<hopper-url> [WORKERS=<n>] [NICE=<int>] [LLM=<endpoint>] [LLM_MODEL=<name>]"; exit 1; }
	@# Runs as the invoking user, so atomscan finds $(HOPPER_TOKEN_FILE) on its
	@# own; `check-hopper-token` has already established that it is there.
	SCAN_LLM="$(LLM)" SCAN_LLM_MODEL="$(LLM_MODEL)" ./out/$(BINARY) worker --url "$(URL)" \
		--interpret \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(NICE),--nice $(NICE),)

# Self-contained worker benchmark: start the bundled mock hopper on a local
# dataset, point a real worker at it, and measure the worker's wall + maxrss
# plus per-job latency (claim → result, measured hopper-side). Completeness
# reads the summary's `done` (dataset jobs matched by sha): with a fetch
# policy on, mirrored dependency verdicts also post to /api/result, so raw
# post counts overshoot the job count. Iterate with
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
BENCH_BODIES         ?= /tmp/litmus-worker-bodies
# Disable persisted analysis-result reuse while retaining compiled mapper/YARA
# caches. Long-lived workers pay compilation once, so forcing those caches cold
# would measure artificial startup work rather than sustained throughput.
worker-benchmark: release ## Benchmark the worker model over a local dataset via the bundled mock hopper
	@[ -e "$(WORKER_BENCH_PATH)" ] || { echo "error: dataset not found: $(WORKER_BENCH_PATH)"; exit 1; }
	@echo "worker-benchmark: dataset=$(WORKER_BENCH_PATH) workers=$(if $(WORKERS),$(WORKERS),default) order=$(ORDER) max_rss_gb=$(if $(MAX_RSS_GB),$(MAX_RSS_GB),auto)"
	@out=$$(mktemp); results=/tmp/litmus-worker-results.jsonl; rm -f $$results "$(BENCH_SUMMARY)"; \
	mkdir -p "$(BENCH_BODIES)" && rm -f "$(BENCH_BODIES)"/*.json; \
	./target/release/scan-bench-hopper --dataset "$(WORKER_BENCH_PATH)" --port 0 --dump $$results \
		--dump-bodies "$(BENCH_BODIES)" --order "$(ORDER)" --summary "$(BENCH_SUMMARY)" \
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
	SCAN_NO_ANALYSIS_CACHE=1 SCAN_HEARTBEAT_SECS=$(HEARTBEAT_SECS) \
	$(if $(GRID),SCAN_PER_SLOT_POOLS=1,) \
	/usr/bin/time $$tflag ./out/$(BINARY) worker \
		--url "http://127.0.0.1:$$port" \
		--data-dir "$(WORKER_BENCH_PATH)" \
		--exit-if-empty --nice 0 --no-update --no-validate \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(MAX_RSS_GB),--max-rss-gb $(MAX_RSS_GB),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log; \
	sleep 0.5; \
	received=$$(python3 -c 'import json;print(json.load(open("$(BENCH_SUMMARY)"))["done"])' 2>/dev/null || echo 0); \
	echo "✓ log: /tmp/litmus-worker-benchmark.log"; \
	echo "--- thread budget (from log) ---"; \
	grep -E "thread budget|oversubscribes|rayon pool ready" /tmp/litmus-worker-benchmark.log | head; \
	echo "--- completeness (SERVER-side): hopper matched $$received / $$jobs job results ---"; \
	[ "$$received" = "$$jobs" ] && echo "✓ COMPLETE: all $$jobs results reached the hopper" \
		|| echo "❌ INCOMPLETE: $$received/$$jobs — worker dropped results (see /tmp/scan-bench-hopper.err)"; \
	echo "--- latency summary (hopper-side, claim → result) ---"; \
	cat "$(BENCH_SUMMARY)" 2>/dev/null || echo "(no summary written)"; \
	echo "✓ summary: $(BENCH_SUMMARY)"

profile-worker:
	$(CARGO) build --profile profiling --bin $(BINARY)
	@[ -n "$(URL)" ] || { echo "Usage: make profile-worker URL=<hopper-url>"; exit 1; }
	samply record -o /tmp/litmus-worker-profile.json.gz -- \
		./target/profiling/$(BINARY) worker --url "$(URL)" --max-jobs $(MAX_JOBS) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		2>&1 | tee /tmp/litmus-worker-benchmark.log

profile-slow:
	$(CARGO) build --profile profiling --bin $(BINARY)
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	samply record --save-only --duration 20 -o /tmp/litmus-$(DATASET)-profile.json.gz -- \
		env CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" CLEAVE_SKIP_YARA_CACHE=0 ./target/profiling/$(BINARY) -f json "$(BENCHMARK_PATH)"

# ----- archive-mode benchmark ---------------------------------------------
# An archive scan folds every inner file into ONE aggregated result, and that
# aggregation — not per-file analysis — is what makes a large archive
# pathological. Measured 2026-07-23 on definitelytyped (63,107 files): 2,865 s
# as an archive vs 337 s for the identical files as a loose directory tree,
# 8.5x, with 19.5 GB peak RSS and a 420 MiB output record. That is how a single
# sample blows through the nightly gauntlet's 20-minute per-scanner cap.
#
# ARCHIVE_DATASET picks the size. typed-4k / typed-16k are file-count subsets of
# the same corpus, for the edit-measure loop; typed is the full regression case
# and takes tens of minutes. Because the subsets share a corpus and file mix,
# comparing them isolates how cost grows with inner-file count.
ARCHIVE_DATASET ?= typed-4k
ARCHIVE_PATH    ?= $(BENCHMARK_ROOT)/$(ARCHIVE_DATASET)
# Appended to, never truncated: a before/after row per run, tagged with the
# cleave rev, is the whole point — a single number can't show a regression.
ARCHIVE_RESULTS ?= $(OUT_DIR)/bench-archive.tsv
# Rows are labeled with the cleave rev under test (plus -dirty for uncommitted
# src edits) — the whole point of the history is before/after across cleave
# changes, and an unlabeled row can't be attributed to anything.
ARCHIVE_LABEL   ?= $(shell git -C ../cleave rev-parse --short HEAD 2>/dev/null || echo local)$(shell git -C ../cleave diff --quiet -- src crates 2>/dev/null || echo -dirty)
# Best-of-N. This box runs hopper/cyclotron/forager alongside, and wall time
# under that load varies 4x run to run; contention only ever ADDS time, so the
# minimum is the closest thing to an uncontended measurement available here.
ARCHIVE_RUNS    ?= 3
# CLEAVE_SKIP_CACHE=1 forces the full per-file analysis every run — without it a
# warm analysis cache turns a 400 s scan into 14 s and the benchmark silently
# measures cache hits. CLEAVE_SKIP_YARA_CACHE=0 keeps the compiled rule set (it
# is not what we're measuring, and recompiling costs 4-18 s of pure noise).
# SCAN_FETCH=none keeps the scan offline: fetching defaults to ON, and a
# benchmark that fetches measures the network (and drifts run to run — the
# fetched-node set differed by ~400 entries between two otherwise identical
# runs, which broke detection comparison). Matches sampled-benchmark/profile-slow.
ARCHIVE_ENV = CLEAVE_SKIP_CACHE=1 CLEAVE_SKIP_YARA_CACHE=0 SCAN_FETCH=none CLEAVE_SCAN_THREADS="$(SCAN_THREADS)"

bench-archive: release ## Time+RSS+output an archive-mode scan, best of ARCHIVE_RUNS (ARCHIVE_DATASET=typed-4k|typed-16k|typed)
	@[ -e "$(ARCHIVE_PATH)" ] || { echo "error: archive dataset not found: $(ARCHIVE_PATH)"; exit 1; }
	@mkdir -p $(OUT_DIR)
	@[ -s "$(ARCHIVE_RESULTS)" ] || printf 'dataset\tlabel\twall_s\tmaxrss_mib\tout_mib\tuser_s\n' > $(ARCHIVE_RESULTS)
	@rm -f $(OUT_DIR)/.bench-archive.runs
	@i=1; while [ $$i -le $(ARCHIVE_RUNS) ]; do \
		printf '  run %d/%d ... ' $$i $(ARCHIVE_RUNS); \
		/usr/bin/time -f '%e\t%M\t%U' -a -o $(OUT_DIR)/.bench-archive.runs \
			env $(ARCHIVE_ENV) \
			./out/$(BINARY) -f json --show all "$(ARCHIVE_PATH)" > $(OUT_DIR)/bench-archive-$(ARCHIVE_DATASET).json; \
		status=$$?; \
		: '# exit 1/2 = hostile/suspicious found, expected on a real corpus; 3+ is a scan error'; \
		[ $$status -le 2 ] || { echo "scan failed: exit $$status"; exit $$status; }; \
		tail -1 $(OUT_DIR)/.bench-archive.runs | cut -f1 | sed 's/$$/s/'; \
		i=$$((i+1)); \
	done
	@awk -v d='$(ARCHIVE_DATASET)' -v l='$(ARCHIVE_LABEL)' \
		-v o="$$(wc -c < $(OUT_DIR)/bench-archive-$(ARCHIVE_DATASET).json)" \
		-F'\t' 'NR==1||$$1<w{w=$$1;r=$$2;u=$$3} END{printf "%s\t%s\t%.1f\t%.0f\t%.1f\t%.0f\n", d, l, w, r/1024, o/1048576, u}' \
		$(OUT_DIR)/.bench-archive.runs | tee -a $(ARCHIVE_RESULTS)
	@echo "✓ history: $(ARCHIVE_RESULTS)  (column 3 is best-of-$(ARCHIVE_RUNS) wall seconds; lower is better)"

bench-archive-scaling: ## Run bench-archive across every typed-* subset, to show how cost grows with file count
	@for d in typed-4k typed-16k; do $(MAKE) --no-print-directory bench-archive ARCHIVE_DATASET=$$d; done
	@column -t $(ARCHIVE_RESULTS)

# ----- the typed parity goal (2026-07-23) ----------------------------------
# typed = the full definitelytyped zip (63,107 inner files) scanned as an
# archive; typed-extract = the identical files as a loose tree. Baseline:
# 2,865 s vs 337 s — the archive fold, not per-file analysis, is the gap, and
# it is how one sample blows the gauntlet's 20-minute cap. The goal:
#   typed wall  <= 1.05x typed-extract wall
#   typed RSS   <= 1.15x its own baseline; detection unchanged
#   typed-extract wall/RSS/detection not regressed
bench-typed: ## Benchmark the full typed archive (one run; it's the slow case)
	@$(MAKE) --no-print-directory bench-archive ARCHIVE_DATASET=typed ARCHIVE_RUNS=$(or $(RUNS),1)

bench-typed-extract: ## Benchmark the same corpus as loose files (the parity target)
	@$(MAKE) --no-print-directory bench-archive ARCHIVE_DATASET=typed-extract ARCHIVE_RUNS=$(or $(RUNS),2)

bench-typed-goal: ## Judge the newest typed/typed-extract rows against the 5% parity goal
	@awk -F'\t' '$$1=="typed"{t=$$3;tm=$$4} $$1=="typed-extract"{e=$$3;em=$$4} END{ \
		if (!t || !e) { print "bench-typed-goal: need a typed and a typed-extract row in $(ARCHIVE_RESULTS)"; exit 2 }; \
		printf "typed=%.0fs (%.0f MiB)  typed-extract=%.0fs (%.0f MiB)  ratio=%.2fx  goal<=1.05x  %s\n", \
			t, tm, e, em, t/e, (t <= 1.05*e) ? "PASS" : "FAIL"; \
		exit (t <= 1.05*e) ? 0 : 1 }' $(ARCHIVE_RESULTS)

baseline-typed-detection: ## Snapshot current verdicts as the detection baseline
	@for d in typed typed-extract; do \
		[ -s $(OUT_DIR)/bench-archive-$$d.json ] || { echo "run bench-typed / bench-typed-extract first ($$d missing)"; exit 1; }; \
		cp $(OUT_DIR)/bench-archive-$$d.json $(OUT_DIR)/baseline-$$d.json; \
		echo "✓ baseline: $(OUT_DIR)/baseline-$$d.json"; \
	done

check-typed-detection: ## Diff current verdicts against the snapshot (a change is a failure)
	@for d in typed typed-extract; do \
		echo "--- $$d ---"; \
		python3 scripts/detect-diff.py $(OUT_DIR)/baseline-$$d.json $(OUT_DIR)/bench-archive-$$d.json || exit 1; \
	done

profile-archive: ## perf-record an archive-mode scan and print the top symbols
	@command -v perf >/dev/null 2>&1 || { echo "Error: perf not installed"; exit 1; }
	$(CARGO) build --profile profiling --bin $(BINARY)
	@[ -e "$(ARCHIVE_PATH)" ] || { echo "error: archive dataset not found: $(ARCHIVE_PATH)"; exit 1; }
	@mkdir -p $(OUT_DIR)
	@# Flat sampling, no call graph: these are release builds without frame
	@# pointers, so dwarf unwinding yields empty stacks anyway (verified) while
	@# costing ~25x the perf.data size. The flat symbol list is what identifies
	@# the hot leaf, which is what this target is for.
	perf record -F 299 -o $(OUT_DIR)/archive.perf.data -- \
		env $(ARCHIVE_ENV) \
		./target/profiling/$(BINARY) -f json --show all "$(ARCHIVE_PATH)" > /dev/null
	@echo "--- top symbols ($(ARCHIVE_DATASET)) ---"
	@perf report -i $(OUT_DIR)/archive.perf.data --stdio --no-children -F overhead,sym 2>/dev/null \
		| grep -E '^ +[0-9]+\.[0-9]+%' | head -30
	@echo "✓ profile: $(OUT_DIR)/archive.perf.data"

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
	$(CARGO) build --profile profiling --bin $(BINARY)
	cp $(CARGO_TARGET)/profiling/$(BINARY) $(OUT_DIR)/$(BINARY).bench
	@if [ "$$(uname)" = "Darwin" ]; then codesign -s - -f $(OUT_DIR)/$(BINARY).bench; fi
	@echo "✓ Benchmark binary: $(OUT_DIR)/$(BINARY).bench"

sampled-benchmark: bench-build ## Benchmark with samply CPU profiling
	@command -v samply >/dev/null 2>&1 || { echo "Error: samply not installed. Run: cargo install samply"; exit 1; }
	@[ -e "$(TUNA_BENCH_PATH)" ] || { echo "error: benchmark path not found: $(TUNA_BENCH_PATH)"; exit 1; }
	CLEAVE_SKIP_CACHE=1 CLEAVE_SKIP_YARA_CACHE=0 samply record --save-only -o $(OUT_DIR)/bench.profile.json.gz -- \
		$(OUT_DIR)/$(BINARY).bench -f json $(TUNA_BENCH_PATH) \
		>$(OUT_DIR)/bench.out 2>$(OUT_DIR)/bench.err
	@echo "✓ Profile: $(OUT_DIR)/bench.profile.json.gz  Logs: $(OUT_DIR)/bench.err"

heap-build: $(OUT_DIR) ## Build with jemalloc heap profiling support
	$(CARGO) build --profile profiling --features jemalloc-prof --bin $(BINARY)
	cp $(CARGO_TARGET)/profiling/$(BINARY) $(OUT_DIR)/$(BINARY).heap
	@if [ "$$(uname)" = "Darwin" ]; then codesign -s - -f $(OUT_DIR)/$(BINARY).heap; fi
	@echo "✓ Heap-profiling binary: $(OUT_DIR)/$(BINARY).heap"

heap-benchmark: heap-build ## Benchmark with jemalloc heap profiling
	@[ -e "$(TUNA_BENCH_PATH)" ] || { echo "error: benchmark path not found: $(TUNA_BENCH_PATH)"; exit 1; }
	@rm -rf $(OUT_DIR)/heap && mkdir -p $(OUT_DIR)/heap
	CLEAVE_SKIP_CACHE=1 _RJEM_MALLOC_CONF="prof:true,prof_active:true,prof_final:true,lg_prof_interval:28,prof_prefix:$(OUT_DIR)/heap/jeprof" \
		$(OUT_DIR)/$(BINARY).heap -f json $(TUNA_BENCH_PATH) \
		>$(OUT_DIR)/bench.out 2>$(OUT_DIR)/bench.err
	@echo "✓ Heap profiles: $(OUT_DIR)/heap/jeprof.*.heap"
	@echo "  Analyze with: jeprof --text $(OUT_DIR)/$(BINARY).heap $(OUT_DIR)/heap/jeprof.*.heap"

# Long-lived server benchmark. Unlike the CLI profiling targets above, this
# exercises the production /analyze path, fans the whole dataset out at once,
# enables every fetch kind, and leaves the process alive long enough to sample
# both VmHWM and post-batch RSS. Analysis caches are disabled across scan,
# cleave, filefacts, and stng; fetch/registry blob caches stay warm so this
# measures analysis rather than the network. Outputs include one response per
# sample plus a deterministic ML fingerprint for parity checks.
SERVER_BENCH_DATASET ?= realworld-small
SERVER_BENCH_PATH    ?= $(BENCHMARK_ROOT)/$(SERVER_BENCH_DATASET)
SERVER_BENCH_OUT     ?= $(OUT_DIR)/server-bench
SERVER_BENCH_PORT    ?= 49997
SERVER_BENCH_WORKERS ?= 20

server-benchmark: bench-build ## Benchmark real /analyze requests in server mode with --fetch=all
	@[ -e "$(SERVER_BENCH_PATH)" ] || { echo "error: dataset not found: $(SERVER_BENCH_PATH)"; exit 1; }
	SCAN_NO_ANALYSIS_CACHE=1 SCAN_FETCH=all SCAN_NO_UPDATE_CHECK=1 \
		scripts/bench-serve.sh "$(OUT_DIR)/$(BINARY).bench" "$(SERVER_BENCH_PATH)" \
		"$(SERVER_BENCH_OUT)" "$(SERVER_BENCH_PORT)" "$(SERVER_BENCH_WORKERS)"

server-heap-benchmark: heap-build ## Heap-profile the server-mode --fetch=all benchmark
	@[ -e "$(SERVER_BENCH_PATH)" ] || { echo "error: dataset not found: $(SERVER_BENCH_PATH)"; exit 1; }
	@rm -rf "$(OUT_DIR)/server-heap" && mkdir -p "$(OUT_DIR)/server-heap"
	SCAN_NO_ANALYSIS_CACHE=1 SCAN_FETCH=all SCAN_NO_UPDATE_CHECK=1 \
	_RJEM_MALLOC_CONF="prof:true,prof_active:true,prof_final:true,lg_prof_interval:30,prof_prefix:$(OUT_DIR)/server-heap/jeprof" \
		scripts/bench-serve.sh "$(OUT_DIR)/$(BINARY).heap" "$(SERVER_BENCH_PATH)" \
		"$(SERVER_BENCH_OUT)" "$(SERVER_BENCH_PORT)" "$(SERVER_BENCH_WORKERS)"
	@echo "✓ Heap profiles: $(OUT_DIR)/server-heap/jeprof.*.heap"
	@echo "  Analyze with: jeprof --text $(OUT_DIR)/$(BINARY).heap <profile.heap>"

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
		--bench-arg -f --bench-arg json \
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

# Cut a release: set the version everywhere it is recorded, prove the result
# builds the way CI will, and commit + tag it as one unit.
#
#     make cut-release VERSION=1.2.3
#
# The version lives in three places that must agree — Cargo.toml, Cargo.lock,
# and the tag — and release.yml rejects the build when any pair disagrees.
# Doing it by hand cost four failed release runs in one day: a tag ahead of
# Cargo.toml, then Cargo.toml ahead of Cargo.lock, each discovered ~40 minutes
# into a matrix that `cargo check --locked` disproves in seconds.
#
# Pushing stays manual on purpose. That is the step that spends an hour of CI
# and publishes artifacts people download, so it gets a human; everything this
# target does is local and revertible with `git reset --hard HEAD~1` plus
# `git tag -d`.
cut-release: ## Bump version + lockfile, verify, commit and tag (VERSION=x.y.z)
	@test -n "$(VERSION)" || { echo "usage: make cut-release VERSION=x.y.z" >&2; exit 1; }
	@printf '%s\n' "$(VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$$' \
		|| { echo "VERSION must look like 1.2.3 (got '$(VERSION)')" >&2; exit 1; }
	@test -z "$$(git status --porcelain)" \
		|| { echo "working tree is dirty — the tag must capture exactly what was tested:" >&2; \
		     git status --short >&2; exit 1; }
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
		echo "tag v$(VERSION) already exists" >&2; exit 1; fi
	@# Rewrite only the first `version =`, which is the one in [package].
	@awk -v v="$(VERSION)" 'BEGIN{d=0} /^version = "/ && !d {print "version = \"" v "\""; d=1; next} {print}' \
		Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml
	$(CARGO) update -p $(PACKAGE) --offline
	@# The exact gate release.yml applies, minus the hour of linking.
	$(CARGO) check --locked --all-targets
	git add Cargo.toml Cargo.lock
	git commit -m "v$(VERSION)"
	git tag -a "v$(VERSION)" -m "$(BINARY) $(VERSION)"
	@echo
	@echo "tagged v$(VERSION). to release:"
	@echo "    git push origin $$(git rev-parse --abbrev-ref HEAD) && git push origin v$(VERSION)"

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
