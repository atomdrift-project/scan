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
DATASET ?= slow
BENCHMARK_ROOT ?= $(HOME)/data/benchmark
BENCHMARK_PATH ?= $(BENCHMARK_ROOT)/$(DATASET)
SCAN_THREADS ?=

# Scrub GNU make's jobserver from cargo's environment. Without this, build
# scripts that spawn their own `make` (e.g. tikv-jemalloc-sys) inherit a
# malformed MAKEFLAGS and fail with "No rule to make target '-j'".
CARGO = env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS cargo

# `make lint LOCKED=--locked` (CI) refuses a Cargo.lock that would change.
# Empty by default so a local `[patch]` experiment still lints and tests.
LOCKED ?=

.PHONY: pgo-train bench-archive bench-archive-scaling profile-archive bench-typed bench-typed-extract bench-typed-goal baseline-typed-detection check-typed-detection build release release-lto install uninstall check-cargo tarball benchmark profile-slow bench-build sampled-benchmark heap-build heap-benchmark tuna tuna-once lint fix test test-unit install-precommit clean wolfi wolfi-bootstrap wolfi-build wolfi-test wolfi-shell wolfi-clean wolfi-nuke docker-login docker-publish cut-release

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
	env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS RUSTFLAGS="$$RUSTFLAGS $(PGO_FLAGS)" cargo build --locked --release
	cp target/release/$(BINARY) $(OUT_DIR)/$(BINARY).new && mv -f $(OUT_DIR)/$(BINARY).new $(OUT_DIR)/$(BINARY)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		codesign --force --sign - $(OUT_DIR)/$(BINARY); \
	fi

# Fat LTO + single codegen unit. Multi-minute link, marginal runtime win
# over the default release profile. Use for container/tarball builds.
release-lto: check-cargo $(OUT_DIR)
	@if [ -n "$(PGO_FLAGS)" ]; then echo "release-lto: PGO enabled ($(PGO_PROFDATA))"; fi
	env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS RUSTFLAGS="$$RUSTFLAGS $(PGO_FLAGS)" cargo build --locked --profile release-lto
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
	SCAN_FOLLOW=none SCAN_NO_ANALYSIS_CACHE=1 CLEAVE_SKIP_CACHE=1 target/release/$(BINARY) -f json "$(PGO_TRAIN_PATH)" > /dev/null || true
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

benchmark: release
	@[ -e "$(BENCHMARK_PATH)" ] || { echo "error: benchmark path not found: $(BENCHMARK_PATH)"; exit 1; }
	@# Exit 1/2 mean hostile/suspicious found — expected on a malware corpus;
	@# only scan errors (3+) fail the benchmark.
	CLEAVE_SCAN_THREADS="$(SCAN_THREADS)" ./out/$(BINARY) -f json "$(BENCHMARK_PATH)" > /tmp/litmus-benchmark-$(DATASET).json; \
	status=$$?; [ $$status -le 2 ] || exit $$status

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
# SCAN_FOLLOW=none keeps the scan offline: fetching defaults to ON, and a
# benchmark that fetches measures the network (and drifts run to run — the
# fetched-node set differed by ~400 entries between two otherwise identical
# runs, which broke detection comparison). Matches sampled-benchmark/profile-slow.
ARCHIVE_ENV = SCAN_NO_ANALYSIS_CACHE=1 CLEAVE_SKIP_CACHE=1 CLEAVE_SKIP_YARA_CACHE=0 SCAN_FOLLOW=none CLEAVE_RAYON_THREADS="$(SCAN_THREADS)"

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

# Model loading only, independent of traits, bloom filters, and sample scans.
# EAGER=1 includes every specialist; default measures normal lazy startup.
MODEL_LOAD_DIR ?= ../azoth
MODEL_LOAD_RUNS ?= 20
MODEL_LOAD_MAX_MS ?= 300
.PHONY: bench-model-load profile-model-load
bench-model-load: ## Time model startup (MODEL_LOAD_DIR=../azoth EAGER=1 for all specialists)
	$(CARGO) build --release --example model-load
	$(CARGO_TARGET)/release/examples/model-load "$(MODEL_LOAD_DIR)" "$(MODEL_LOAD_RUNS)" --max-ms "$(MODEL_LOAD_MAX_MS)" $(if $(filter 1,$(EAGER)),--eager,)

profile-model-load: $(OUT_DIR) ## CPU-profile model loading independently of file analysis
	$(CARGO) build --profile profiling --example model-load
	perf record -F 999 -o $(OUT_DIR)/model-load.perf.data -- \
		$(CARGO_TARGET)/profiling/examples/model-load "$(MODEL_LOAD_DIR)" "$(MODEL_LOAD_RUNS)" $(if $(filter 1,$(EAGER)),--eager,)
	perf report -i $(OUT_DIR)/model-load.perf.data --stdio --no-children -F overhead,sym

# Full local analysis, including all archive members. Keep compiled rules,
# but disable analysis-result caches at every layer (not just cleave).
# Reference downloads are outside this benchmark's measured corpus.
BENCHMARK_RUNS ?= 1
BENCHMARK_BINARY ?= $(OUT_DIR)/$(BINARY).bench
BENCHMARK_RIZIN ?=
BENCHMARK_NATIVE_CPUS ?=
.PHONY: balanced-benchmark balanced-benchmark-existing
balanced-benchmark: bench-build
balanced-benchmark balanced-benchmark-existing: ## Cache-disabled balanced analysis; -existing uses BENCHMARK_BINARY without rebuilding
	CLEAVE_RAYON_THREADS="$(SCAN_THREADS)" python3 scripts/bench-balanced.py --binary "$(BENCHMARK_BINARY)" \
		--output $(OUT_DIR)/balanced-benchmark --runs $(BENCHMARK_RUNS) $(if $(BENCHMARK_LIMIT),--limit $(BENCHMARK_LIMIT),) \
		$(if $(BENCHMARK_RIZIN),--rizin "$(BENCHMARK_RIZIN)",) $(if $(BENCHMARK_NATIVE_CPUS),--native-cpus "$(BENCHMARK_NATIVE_CPUS)",) \
		"$(BENCHMARK_PATH)"

sampled-benchmark: bench-build ## Benchmark with samply CPU profiling
	@command -v samply >/dev/null 2>&1 || { echo "Error: samply not installed. Run: cargo install samply"; exit 1; }
	@[ -e "$(TUNA_BENCH_PATH)" ] || { echo "error: benchmark path not found: $(TUNA_BENCH_PATH)"; exit 1; }
	$(ARCHIVE_ENV) SCAN_NO_UPDATE_CHECK=1 samply record --save-only -o $(OUT_DIR)/bench.profile.json.gz -- \
		$(OUT_DIR)/$(BINARY).bench --mode balanced -f json $(TUNA_BENCH_PATH) \
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
	$(CARGO) fmt --all --check
	$(CARGO) clippy $(LOCKED) --workspace --all-targets -- -D warnings
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc $(LOCKED) --no-deps --lib --quiet

# Auto-fix what clippy and rustfmt can fix on their own. Run fmt last so it
# tidies any code clippy rewrote. Mirrors what `lint` checks.
fix:
	$(CARGO) clippy --workspace --all-targets --fix --allow-dirty --allow-staged
	$(CARGO) fmt --all

test:
	@set -e; \
	if command -v cargo-nextest >/dev/null 2>&1; then \
		$(CARGO) nextest run $(LOCKED) --workspace; \
		$(CARGO) test $(LOCKED) --doc --quiet; \
	else \
		$(CARGO) test $(LOCKED) --workspace; \
	fi

.PHONY: test-cli
test-cli:
	$(CARGO) test --locked --test cli_smoke

test-unit:
	@set -e; \
	if command -v cargo-nextest >/dev/null 2>&1; then \
		$(CARGO) nextest run --lib; \
	else \
		$(CARGO) test --lib worker::tests::prefetcher_fills_to_target_backpressures_and_refills --quiet; \
		$(CARGO) test --lib -- \
			--skip worker::tests::prefetcher_fills_to_target_backpressures_and_refills; \
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
