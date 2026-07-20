# Dependencies as first-class artifacts

Status: proposal. Cross-repo — `scan`, `hopper`, `prism`, `promoter`.

A fetched dependency is an artifact someone else's package told us to go get.
It has its own bytes, its own identity, and its own verdict. Today it is
second-class in three independent places, for three unrelated reasons. This
document explains the reasons, states one principle that resolves most of
them, and sequences the work.

## The principle

**A dependency is not a kind of scan. It is a scan plus an edge.**

Two orthogonal primitives that compose, rather than one fused case kept in
sync by comment. Everything below is a consequence.

Corollaries:

- *In scan.* Scanning an artifact produces a `ScanResult`. Following a
  reference produces an edge. A dependency is both; it is not a third thing
  with its own degraded result type.
- *In hopper.* `samples` describes artifacts. `sample_locations` describes
  edges. Any column on `samples` that summarizes edges is **derived**, never
  asserted by a producer.
- *Everywhere.* Package-vs-URL is not a branch. `RefLocator` already unifies
  the two upstream; `purl_base` present-or-empty distinguishes them
  downstream. No code should ask "is this a package dependency or a URL
  dependency?" — it should ask whether there is a PURL.

## Why it is broken today

Three defects, independent causes, one symptom.

### 1. scan produces a degraded envelope

`ScanResult::into_envelope` (`engine.rs:4571`) is the real envelope builder.
`dep_envelope` (`engine.rs:2165`) is a second implementation for dependencies,
carrying this claim:

> the same shape a first-hand scan of those bytes would post, so hopper
> records the dependency exactly as if it had been scanned directly

It is not the same shape. It differs in six fields: `model_scores` and
`skipped_models` are empty, `llm` is `None`, `v` is hardcoded `"7"`,
`version`/`analyzed_at` are borrowed from the parent run, and `build_ml_files`
receives `&MemberEvals::new()` — an empty per-member eval table where the real
path passes `&self.embedded_files`.

Equivalence asserted in prose instead of guaranteed by construction. It has
already drifted.

`DepResult` is likewise `ScanResult` minus exactly the fields that make a
result first-class.

Two bugs follow directly:

- **Shared classification budget.** `process_report` passes `Some(100)` to
  `classify_report` (`engine.rs:3975`) — one budget covering the root's
  members *and* every dependency, in report order. Dependencies are appended
  last (`fetch.rs:1913`).
- **Fabricated benign.** A dependency the embedded pass never reached defaults
  to `Classification::Benign, level: -1` (`engine.rs:3345`) and is uploaded as
  authoritative. On a package with >100 embedded files, an unevaluated hostile
  dependency is recorded as confidently benign.

The second is the sharpest item in this document. Once dependencies feed bloom
filters, a fabricated benign becomes a global known-good bless that suppresses
all future fetching of that package (`fetch.rs:588`).

### 2. hopper models a dependency as an archive member

`memberSamplesFromEnvelope` (`hopper.go:1000`) parses `rel` and documents it
correctly:

> "fetched" for content retrieved from a reference the parent declares (never
> inside it) … so consumers can tell "found in this archive" from "referenced
> by this sample"

It then never branches on it. Every child is built the same way
(`hopper.go:1105`): `Parent: e.parent.SHA256`, `Label: e.parent.Label`.

Consequences, all confirmed by reading:

- **`parent` is a race.** It is absent from `sampleConflictUpdatePG`'s
  `DO UPDATE SET` (`pg.go:1658`), so it is written only at INSERT — first
  writer wins. Two writers race for every dependency: scan's upload
  (`parent=''`) and the parent's explode (`parent=<archive sha>`). The
  conflict clause is additionally guarded by `WHERE EXCLUDED.parent = ''`
  (`pg.go:1710`), so an explode-first row can never be repaired by the later
  upload carrying the real verdict.
- **Borrowed label.** Explode inherits `e.parent.Label`. A benign dependency
  of a hostile package is stored as `bad`.
- **False path.** Explode assigns `<parent path>!!<name>` (`hopper.go:1094`).
  A `!!` path asserts the bytes are extractable from that archive. For
  referenced bytes that was never true, so the row claims a retrievability it
  does not have. Nothing can reassemble it, and triage cannot relocate it.
- **Clobbered report.** `RefreshStaleMemberAnalysis` (`hopper.go:1158`) pushes
  the parent's per-member slice onto existing rows when the parent's analysis
  is newer. That slice is a single-file stub (`hopper.go:1074`). A dependency
  that is itself an archive — every npm tarball — has its standalone
  multi-file report replaced by a one-node stub, and `max_crit` /
  `suspicious_count` are re-derived from the stub.
- **Missing bytes.** `/api/known` answers from row existence alone
  (`hopper.go:1998`: `SELECT sha256 FROM samples WHERE sha256 = ANY($1)`). An
  explode-created row makes hopper claim it has bytes it has never held, so
  scan takes the provenance-only branch (`upload.rs:541`) and the bytes are
  never sent.

The last item is the only case where scan genuinely fails to store dependency
bytes. Otherwise it does: `upload_scan_result` (`engine.rs:2136`) sends the
scanned file's bytes, each dependency's bytes and provenance, then the verdict.

Note the tell: prism's feed query is
`parent = '' AND NOT EXISTS (SELECT 1 FROM sample_locations …)`
(`prism/main.go:2598` → `hopper/pg.go:4967`). Someone already discovered that
`parent` lies and patched around it. `samples.parent` is a scalar cache of a
many-valued relation the ledger already owns — hopper's own comment says
*"children may have multiple parents; the locations ledger is the authority."*

### 3. nothing can label a dependency `good`

Bloom filters are built from a pool of *curated* labels
(`scripts/bloom_pool.sql`), not from scan verdicts. Two labelers exist:

- **promoter** never sees scan uploads. Its discovery query requires
  `starts_with(path, 'unknown/foraged/')` (`hopper/pg.go:2821`,
  `promoter/cmd/promoter/main.go:50`); hopper files uploads under
  `unknown/uploads/<aa>/<bb>/` (`hopper/cmd/hopper/api.go:2532`) and says an
  upload never controls its own path. Also `uploadSample` never sets `mtime`
  (`api.go:2611`), independently failing the `mtime IS NOT NULL` gate.
- **cyclotron** has no path predicate and *can* rule on a dependency — but its
  `new` queue requires `suspicious_count >= 1` (`hopper/pg.go:3053`). A
  dependency that scans clean has zero, so no queue takes it.

So the bad half has a route and the good half has none. This is inverted from
what is wanted: known-good is the filter doing the performance work, since a
known-good PURL is never fetched at all (`fetch.rs:588`).

Nothing *semantic* excludes dependencies from promotion. Confirmed
non-gates: `source`, `feed`, `label_source`, empty `purl_base`, missing
download URL (`promoter/rules.go:156` — "an absence of confirmation, not
evidence of malice"), unmapped ecosystem. Promoter has no notion of
`rel='fetched'` at all. The only barrier is where hopper filed the bytes.

## The design

### scan: one path from reference to result

`process_report` (`engine.rs:3955`) is already the right boundary — report in,
`Result<ScanResult>` out. `record_file_result` is a sink wrapped around it
(render, upload, tally, returns `()`), and `ScanSummary` is only counters.

The obstacle is nesting direction. Today fetching happens *inside*
classification, which is *inside* `process_report`:

    process_report(root)
    └─ classify_report(budget=100, fetch_policy)
       └─ fetch::orchestrate
          └─ per ref: analyze_bytes → capture_dependency → DepResult

Invert it, so fetch orchestration recurses through the shared entry point:

    process_report(artifact, depth) -> ScanResult
    ├─ classify_report(budget=100)            // its own budget, per artifact
    ├─ per ref, if depth < max:
    │     process_report(ref, depth+1) -> ScanResult
    │     + record edge (declaring file, locator, rel, resolved url)
    └─ graft child report into the parent's *display* report only

Mutual recursion, bounded by the existing `DEFAULT_FETCH_DEPTH`
(`fetch.rs:52`) — the recursion `--fetch-depth` already documents. It happens
today inside `fetch.rs`'s own loop rather than through the shared entry.

What this deletes:

- `dep_envelope` — a dependency's envelope comes from
  `ScanResult::into_envelope`. Equivalence becomes a type, not a claim.
- `DepResult` as a degraded shadow — it becomes an edge plus a real result.
- The empty `MemberEvals` — a dependency's members are evaluated by the
  dependency's own scan.
- The shared 100-node cap — each artifact gets its own budget by construction.
- The fabricated benign at `engine.rs:3345` — there is no "unevaluated
  dependency" state left to invent a verdict for.

The graft into the parent's report becomes display-only, which is what
`engine.rs:3205` already claims it is ("Display/attribution only") and is not.

**Cost, stated honestly.** This is surgery on `classify_report`, the central
function of a 4,600-line file: fetch has to move out from under
classification, and depth/budget become explicit threaded context instead of
`fetch.rs` loop state. A child scan needs a quiet mode — there is precedent,
since `record_file_result` already takes `progress: Option<&Progress>` and
`uploader: Option<&Uploader>`. The upload ordering dependencies rely on
(bytes and provenance before verdict, `upload.rs:490`) must survive the move.

### hopper: stop storing an edge on the artifact

Make the containment summary **derived by trigger from `sample_locations`**,
counting only containment rels. Hopper already does trigger-derived columns
(`samples_derive_cleave_cols`, `samples_derive_litmus_score`), so this is an
existing idiom rather than a new mechanism.

Then producers *cannot* write it: the race cannot exist rather than being
arbitrated, the repair migration for frozen values disappears, and
explode-vs-upload ordering stops mattering — which removes a whole class of
bug instead of sequencing around it.

Name the distinction once. A `hopper.Rel` type with `IsContainment()`,
mirroring cleave's `Rel` and absorbing prism's existing
`ParentArchive.containsChild()` (`prism/main.go:635`), which is already this
predicate written a second time.

Then:

- **Explode records edges, not rows, for reference rels.** Scan already
  uploads referenced artifacts as first-class rows with better data — its own
  standalone report, its own provenance, its own PURL. One writer per fact. If
  no producer uploaded the artifact, the edge stands with no row; prism
  already renders that honestly as an unlinked chip.
- **`RefreshStaleMemberAnalysis` skips reference rows.** Never overwrite a
  standalone report with a single-file slice.
- **`/api/known` means "bytes retrievable"**, not "row exists" — a sha counts
  as known only when it has a real on-disk path or is reassemblable from a
  containment parent. One predicate change, and it fixes byte coverage for
  every producer.

  It does *not* fully repair already-damaged rows on its own. The bytes now
  upload, but `parent` stays frozen at the archive sha (it is absent from the
  `DO UPDATE SET`), so `handleFile` still routes retrieval through
  `serveArchiveMember` and tries to extract bytes that were never in that
  archive. Phase 1 completes the repair; Phase 0 stops the gap from growing.

### blooms: nothing to write

`bloom_pool.sql` already emits a versioned PURL when `purl_base` and `version`
are known and always emits `sha256`. Dependency rows already carry `purl_base`
(`api.go:2641`, from `prov.Package.PURL`). A URL dependency simply has none,
so it keys by content.

"Packages key by PURL, URL references key by SHA256" therefore needs **no new
code**. The only change is a comment that stops saying "top-level" and starts
saying "containment", which is what it always meant.

### prism

- `TopLevelOnly` becomes rel-aware, so dependencies get index rows.
- `feedRow` gains a fan-in count ("pulled in by N packages", via the existing
  `idx_sl_sha256_parents`). The flood concern that motivated `TopLevelOnly` is
  then answered by aggregation rather than suppression.
- `FallbackTraits` (`main.go:1003`) stops returning nil whenever an LLM `Why`
  exists, which currently drops dependency chips on the best-analyzed rows.

### display (scan)

Independent of the above and parallelizable.

Root cause of the format inconsistency: `payload_name` (`fetch.rs:1978`) gives
a dependency a bare basename, so Terminal and Tiny render its traits
anonymously — indistinguishable from an archive member. `engine.rs:3566`
admits this. And `fetch/dependency-verdict` is injected into `report_json`
only, never into the typed `report` the text renderers consume.

- Give a dependency its locator as identity at the source. This fixes
  Terminal/Tiny anonymity *and* the appendix's basename collision
  (`engine.rs:3640`, where two dependencies both named `index.js` merge their
  findings) in one change.
- Inject the dependency-verdict trait into the typed `report`.
- Exempt reference roots from cross-file dedup (`cleave/output.rs:748`), which
  can currently delete a hostile dependency trait because the parent declared
  the same one.
- The `== FETCHED DEPENDENCIES ==` appendix (`engine.rs:3579`) is currently
  stranded on the `llm_view` branch, so `--format tiny` and `--format
  interpret` produce identical main renders and only one explains itself. Once
  dependencies are self-identifying the appendix becomes a summary rather than
  the only source of truth, and should be available to every text format.

## Sequence

**Phase 0 — scan safety.** Unevaluated is not benign (`engine.rs:3345`); do
not upload a verdict that was not computed. Pair with the `/api/known`
predicate fix, which repairs byte coverage independently of everything else.
Blocking: nothing that feeds blooms may land before this.

**Phase 1 — the two unifications.** scan's `process_report` inversion and
hopper's derived containment column. Each deletes a parallel implementation.
Best on separate branches with existing dependency tests green.

**Phase 2 — promotion policy.** File scan pushes where promoter can discover
them, add that prefix to `promoteSrcRoots` (`posttriage.go:83`) so
`rulingPlan` preserves the subpath instead of flattening to basename, and set
`mtime` on upload. **This is a decision, not a bug fix** — it makes every scan
push promotable, not only dependencies.

**Phase 3 — prism.** Index rows, fan-in lineage, chip suppression.

**Phase 4 — display.** Parallelizable with 2 and 3.

## Open questions

- **Phase 2 is a policy choice** and is the only route to a `good` label.
  Recommendation: route scan pushes to promoter rather than loosening
  cyclotron. Promoter's rule is conservative and evidence-based — zero signals
  across independent families, benign class, age-ramped probability cap, a
  30-day floor that means "nobody reported this in 30 days". Cyclotron's
  good/bad decision is an LLM touching a marker file (`cyclotron/analyze.go:299`)
  with no confidence floor on the write path, and on an `unknown` row every
  judgement becomes a live ruling (`analyze.go:328`). That is tolerable when a
  label is only a label; it is not when a label becomes a global
  fetch-suppression bless.
- **Dependency bytes are best-effort** in a way foraged artifacts are not:
  they come from fletch's blob cache (`upload.rs:416`), and an eviction means
  the row exists with no bytes and a logged warning. Decide whether that
  warrants retention policy or a repair path.
- **Explode-first frequency is unmeasured.** The mechanism is certain; the
  rate is not. Worth counting rows with `parent <> ''` whose location edges
  are all reference rels before committing to a migration.

## Verification

- A dependency's stored envelope should be byte-identical in shape to the same
  artifact scanned directly via `scan purl <locator>`. This is the acceptance
  test for Phase 1, and it is exactly the equivalence `dep_envelope`'s comment
  claims today without delivering.
- A dependency that is itself an archive keeps its multi-file report after its
  parent is re-analyzed.
- A hostile dependency past the 100th embedded file is recorded hostile.
- A dependency appears once on the prism index with a fan-in count, and never
  as a member of an archive that did not contain it.
- A benign package dependency contributes its versioned PURL to the known-good
  pool; a benign URL dependency contributes its sha256.
