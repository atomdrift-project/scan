# Wolfi packaging for litmus

This directory builds [Wolfi](https://wolfi.dev) apks for `litmus` and a
`litmus-models` data subpackage, then assembles them into a minimal OCI
image alongside the cleave + cleave-traits apks built by
`cleave/packaging/wolfi/`. The `melange.yaml` is shaped to be drop-in
copyable into [wolfi-dev/os](https://github.com/wolfi-dev/os) for
upstream submission; the local Makefile targets exist so contributors
can iterate without a Wolfi-dev clone.

## Layout

```
packaging/wolfi/
  melange.yaml              # litmus + litmus-models subpackages (UPSTREAMABLE shape)
  apko.yaml                 # local OCI app image (NOT upstreamed)
  lima.yaml                 # Ubuntu 26.04 LTS sandbox for macOS builds
  scripts/
    bootstrap-lima.sh       # idempotent VM + runtime setup
    build.sh                # stages → 2× melange → apko, with stamp-skip
    smoke-test.sh           # asserts the image runs and uses bundled models
```

## Local build (macOS or Linux)

```sh
make wolfi          # bootstrap + build + smoke-test
make wolfi-build    # just (re)build the image
make wolfi-test     # just run smoke tests against an existing image
make wolfi-shell    # interactive shell in the built image
make wolfi-clean    # remove out/wolfi/ (keeps the Lima VM)
make wolfi-nuke     # also deletes the Lima VM (slow to recreate)
```

`make wolfi` defaults to building apks for **both** `aarch64` and
`x86_64` (so the apks are publish-ready). Cross-arch cargo builds run
through QEMU and are slow — for fast single-arch local iteration:

```sh
WOLFI_ARCH=$(uname -m | sed 's/arm64/aarch64/;s/amd64/x86_64/') \
  make wolfi-build
```

The local smoke-test image (`out/wolfi/litmus.tar`) is built host-arch
only regardless of `WOLFI_ARCH`, since `nerdctl run` on your host can
only execute its own arch.

Idempotent + per-component caching:

| Stamp file                  | Inputs                                                                | Skips                             |
| --------------------------- | --------------------------------------------------------------------- | --------------------------------- |
| `out/wolfi/.build.stamp`    | image_hash (everything)                                                | the entire pipeline               |
| `out/wolfi/.litmus.stamp`   | litmus src + azoth + litmus melange.yaml + cleave_hash                 | the litmus + litmus-models melange step |
| `out/wolfi/.cleave.stamp`   | cleave src + filefacts src + cleave melange.yaml                       | the cleave + cleave-traits melange step |

Common cases:

- **Touch nothing, re-run** → 0s (image stamp hits).
- **Edit only litmus source** → cleave melange skipped (~11 min saved); litmus melange runs (~11 min); apko + smoke.
- **Edit only cleave source** → both melange steps run; full rebuild.
- **Edit only apko.yaml** → both melange steps skipped (apks intact); apko + smoke only (~30s).

Cargo's own `target/` cache is NOT preserved between melange runs
(melange's bubblewrap chroot is fresh each invocation), so within a
single melange step the cargo build is always cold. The biggest remaining
speedup would be wiring sccache + a host-mounted cache dir; melange's
bubblewrap runner doesn't expose host bind-mounts directly, so that's
a follow-up worth a small spike.

Per-arch build:

```sh
WOLFI_ARCH=aarch64 make wolfi-build      # default is the host arch
WOLFI_ARCH=x86_64  make wolfi-build      # cross via QEMU; significantly slower
```

## Required sibling checkouts

The local build needs both repos checked out next to litmus:

```
.../atomdrift/
  cleave/       (codeberg.org/atomdrift/cleave)
  filefacts/    (codeberg.org/atomdrift/filefacts)
  litmus/       (this repo)
```

`scripts/bootstrap-lima.sh` errors out if `../cleave` is missing.
filefacts is required because cleave's `Cargo.toml` references it via a
path dep.

## How the local build differs from upstream Wolfi

`melange.yaml` declares a `git-checkout` step that pulls litmus from
codeberg at a pinned tag — what upstream CI runs. Locally, `build.sh`:

1. **Strips the git-checkout step** (via `LOCAL_BUILD_STRIP_BEGIN/END`
   markers) so melange's `--source-dir` wins.
2. **Stages cleave + filefacts + litmus** into `out/wolfi/source/` with
   `target/` / `.git/` / `out/` excluded.
3. **Vendors filefacts** as `cleave/_filefacts/` and rewrites cleave's
   `Cargo.toml` to point there (same recipe as cleave's own packaging).
4. **Appends a `[patch]` block** to litmus's `Cargo.toml` so cargo uses
   the staged cleave instead of fetching from codeberg:
   ```toml
   [patch."https://codeberg.org/atomdrift/cleave.git"]
   cleave = { path = "../cleave" }
   ```
5. **Substitutes `cd litmus`** for the `# LOCAL_BUILD_CD_HERE` marker
   in the cargo build step, because the staged workspace at
   `/home/build` contains both `cleave/` and `litmus/` subdirs.
6. **Drops `--locked`** because the appended `[patch]` block forces
   cargo to refresh `Cargo.lock`.
7. **Builds cleave's apks too** (by invoking cleave's own melange.yaml
   against the staged cleave tree) so apko has the full local repo
   when assembling the image.

The upstream-shape yaml is untouched by all of this — same file works
in wolfi-dev/os CI once the upstream blockers below are resolved.

## Upstream blockers

This package inherits cleave's blocker plus its own:

1. **filefacts path dep in cleave** — see `cleave/packaging/wolfi/README.md`.
   Until cleave resolves this, neither cleave nor litmus can land in
   wolfi-dev/os.
2. **litmus's git dep on cleave** — Wolfi prefers package-to-package
   deps. Once cleave is in wolfi-dev/os, litmus's Cargo.toml should
   switch from `cleave = { git = "…" }` to a versioned crates.io dep
   (or stay git-pinned, but the apk dep `cleave` in `melange.yaml`
   already declares the runtime relationship — Wolfi reviewers will
   notice and ask).

The local build sidesteps both via staging + `[patch]`. The upstream PR
needs both fixed first.

## Submitting upstream

1. Resolve both blockers above.
2. Tag a litmus release (`vX.Y.Z` on codeberg).
3. Update `melange.yaml`:
   - `package.version` → new version (drop the `v` prefix).
   - `expected-commit` in the first `git-checkout` → the tag's commit SHA.
   - `expected-commit` in `litmus-models` subpackage → the azoth commit
     you want shipped with this release.
4. Validate locally: `make wolfi` should still pass after upstream
   changes are merged into litmus.
5. In a wolfi-dev/os clone:
   ```sh
   cp .../litmus/packaging/wolfi/melange.yaml packages/litmus.yaml
   make package/litmus
   ```

## Improvements vs the cleave packaging

- **`[patch]` over Cargo.toml dep rewriting** — litmus's cleave
  override is a single appended block at the bottom of `Cargo.toml`,
  not an in-line rewrite of the dep line. Cleaner diff, less
  brittle if the dep declaration changes shape.
- **Single staging dir for the workspace** — cleave + filefacts +
  litmus all sit under one staged tree; melange's `--source-dir`
  points at the parent and the build step `cd`s into `litmus/`.
- **Smoke test asserts litmus's documented exit codes** — accepts
  `0/1/2` from a real scan (clean/hostile/suspicious), fails only on
  `3+` which means scanning itself broke.

## Models

`litmus-models` bundles a pinned snapshot of [azoth](https://codeberg.org/atomdrift/azoth)
into the apk at build time, with `LITMUS_MODELS_DIR` set in the apko
config so the image never tries to clone at runtime. If you want fresh
models, bump the `expected-commit` in `melange.yaml` and rebuild — the
hash stamp will see the change and trigger a rebuild.

For a lazy-clone variant (smaller image, requires git + network at
runtime), drop the `litmus-models` subpackage from `apko.yaml`, add
`git` and `ca-certificates-bundle` to the image, and remove the
`LITMUS_MODELS_DIR` env so litmus falls back to its auto-clone path.

## Publishing to a registry

```sh
make docker-login                     # one-time: log lima VM into docker.io as atomdrift
make docker-publish                   # build multi-arch, push, cosign sign keyless
DRY_RUN=1 make docker-publish         # build everything but skip push + sign
REGISTRY=ghcr.io ORG=foo make docker-publish   # override target
```

This pushes:

- `docker.io/atomdrift/litmus:<VERSION>` (from `melange.yaml`)
- `docker.io/atomdrift/litmus:latest`

…both as multi-arch manifest lists with `aarch64` and `x86_64` platform
manifests inside.

### Signing

Each tag is signed keyless with [cosign](https://github.com/sigstore/cosign)
via Google OIDC:

```
cosign sign --yes --oidc-issuer=https://accounts.google.com <image>
```

cosign opens a browser the first time for the OIDC flow (re-uses the
short-lived Fulcio cert for subsequent signs in the same session). The
signature is stored alongside the image in the registry; the
transparency-log record goes to the public-good Rekor instance.

For CI, set `COSIGN_IDENTITY_TOKEN` to a Google service-account JWT and
cosign skips the browser flow.

### Verifying

```sh
cosign verify \
  --certificate-identity-regexp 'YOUR_EMAIL@DOMAIN' \
  --certificate-oidc-issuer https://accounts.google.com \
  docker.io/atomdrift/litmus:1.2.1
```

Replace `--certificate-identity-regexp` with whatever Google identity
actually signed the image (your email, or a service-account address for
CI builds). Verification checks the Fulcio cert chain, queries Rekor for
the inclusion proof, and confirms the signature matches the image digest.

### Prerequisites

- `cosign` on PATH — `brew install cosign`
- lima VM logged into the target registry — `make docker-login`
- A Google account with push permission on the target org/registry
  (interactive browser OIDC flow), or `COSIGN_IDENTITY_TOKEN` set
- For cross-arch builds (default): QEMU support inside lima (already
  present in the Ubuntu 26.04 base via `binfmt_misc`)

## Troubleshooting

- **`error: missing .../cleave`** — `git clone ssh://git@codeberg.org/atomdrift/cleave.git ../cleave`.
- **First build is very slow (~13 min)** — expected; cargo cold-builds
  cleave + filefacts + litmus inside the sandbox with fat LTO. Stamp
  skip makes subsequent runs instant.
- **OOM during link** — bump `memory:` in `lima.yaml` and recreate the
  VM with `make wolfi-nuke && make wolfi-bootstrap`.
