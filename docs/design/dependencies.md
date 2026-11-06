# Dependencies as independent artifacts

Status: design; implementation deferred. Updated 2026-10-20.
Scope: scan owns dependency discovery, resolution, and assessment; Hopper stores
artifacts and relationships. Cleave supplies declarations and evidence. Prism
consumes the stored graph. No dependency-storage migration is authorized by this
proposal alone.

## Existing foundation

Scan already captures and grades fetched dependencies as standalone artifacts
(`src/fetch.rs`, `src/engine/pipeline.rs`) and uploads their bytes, provenance,
and results (`src/engine/hopper.rs`). A dependency can itself be an archive and
must retain its complete standalone analysis and member evaluations.

Hopper distinguishes containment (`rel=''` or `unpacked`) from references
(`fetched` or `registry`). Reference observations do not establish an extractable
archive path or inherit the declaring package's training label. Its
`RefreshStaleMemberAnalysis` skips reference observations so a parent report's
single-file slice cannot replace a dependency's standalone archive report.

Cleave's compact `refs` describe declarations with locators, kinds, source
offsets, and optional report-local target IDs. These are useful input facts,
not a durable resolution ledger.

An earlier version of this document catalogued defects in dependency grading,
containment, upload, and promotion. Several have since been fixed. Its old line
numbers and "broken today" descriptions are not a current audit. This design
addresses the remaining relationship and assessment-versioning problem.

## Principle

An artifact has bytes, an identity, and independent analysis. A dependency is a
relationship to an artifact, possibly not yet resolved. Store those separately.

- `samples.sha256` identifies bytes. PURLs, versions, and URLs are lookup and
  provenance information; they do not replace content identity.
- A file can declare dependencies whether or not it is a package or archive.
- One artifact can be used by many owners and in multiple occurrences.
- A declaration is not proof of installation, reachability, or execution.
- Archive aggregate grading and grading of every contained file remain intact.
  Fetched dependencies retain independent grading and influence the owner's
  decision through scan's explicit dependency policy.

## Storage model

Keep `samples` and archive containment unchanged. Add `dependency_edges` for
logical declarations and their observed resolutions. Do not force unresolved
references into `sample_locations`: that table requires a target SHA and a
nonempty path and identifies observations by `(sha256, path)`.

Each edge observation records:

| Field | Meaning |
| --- | --- |
| Owner SHA | Package or standalone artifact whose graph is being evaluated |
| Declaring file SHA and occurrence | File containing the declaration, including its location within the owner |
| Declaration evidence | Reference kind, original locator/specifier, offset when known |
| Scope | Runtime, development, build, optional, or unknown; retain ecosystem evidence |
| Resolution context | Platform, relevant lockfile/manifest identity, and resolver policy/version |
| Resolved locator | Exact selected package coordinate or URL, when available |
| Target SHA | Actual artifact identity, nullable until bytes are identified |
| Resolution status | Resolved, unresolved, skipped, failed, or a recognized non-artifact target |
| Observation revision | Scan/resolution revision and observation time |

The declaring occurrence is necessary: identical manifest bytes in two packages
can resolve differently because of lockfiles, layout, or platform. Report-local
integer IDs must be translated into durable owner/file/occurrence identities
when ingesting the envelope. They are not global database IDs.

Use a stable declaration key within an owner and an immutable resolution
observation when the outcome changes. A current projection can select the latest
accepted observation. Repeating an unchanged observation must be idempotent; it
must not rewrite artifact JSON or mint an unbounded history of identical checks.
Exact SQL keys and revision representation remain implementation decisions.

Keep declarations separate from the observations that resolve them. Preserve the
original constraint (`^1.2`, a tag, an unversioned URL) alongside the resolved
version and SHA. A resolved coordinate may have multiple artifact hashes across
platforms or publication events. Do not assume PURL-to-SHA is one-to-one.

Store direct edges. Compute transitive reachability with a visited set and
explicit limits rather than storing every transitive pair. Keep repeated direct
occurrences even when their target analysis is shared.

## Scan envelope and responsibilities

Add an explicit dependency-edge section to scan's envelope. It carries the
relationship independently of any display graft into `raw.files`. Upload a
standalone artifact result once, and reference it from all applicable edges.
Edges may exist without a fetched artifact or target sample row.

- **Cleave:** extract declarations, references, identity claims, and evidence.
- **Scan:** resolve in context, fetch/analyze artifacts, record outcomes and
  coverage, and apply dependency escalation policy.
- **Hopper:** validate/store observations, serve forward and reverse queries,
  and track assessment freshness. Do not implement package resolution here.
- **Prism:** render declared/resolved/bundled relationships and their assessment
  freshness without implying execution from a declaration.

Represent discovery coverage explicitly. A scan that stops at a timeout, depth
limit, fetch policy, or budget has partial coverage. An incomplete scan cannot
replace a complete edge set with an apparently dependency-free one. Record
failures and skips as such; neither is a benign assessment.

## Independent updates and reproducibility

Distinguish two operations:

1. **Reassess existing bytes.** Same target SHA; new analysis, model, policy, or
   registry-evidence revision. Update the target assessment once. The owner's
   declaration and resolved target do not change.
2. **Resolve again.** A declaration may now select a different version or SHA.
   Record a new resolution observation. Never silently retarget an earlier scan.

Preserve the owner's original scan result with the exact dependency assessment
revisions it consumed. Expose a separate current dependency assessment, or mark
it stale until scan recomputes it. Its identity includes the edge-resolution
snapshot, target assessment revisions, relevant registry evidence, and policy.
Do not overwrite the artifact's own static analysis when external evidence
changes. Dependency-aware composite evaluation may require scan to rerun that
assessment stage; a database join alone must not invent a new verdict.

Training and evaluation snapshots must pin these revisions. Joining live latest
dependency assessments during a historical experiment would change its inputs
and can introduce future information into the evaluation.

## Efficient invalidation

Index direct edges by owner and by resolved target SHA; add locator lookup only
for a measured consumer. Refresh one dependency independently, then invalidate
or enqueue affected owner assessments using reverse edges. Coalesce repeated
requests and bound each processing batch, with a resumable cursor so popular
dependencies do not permanently lose parents past a fixed limit.

Handle both increases and decreases in severity, model changes, registry evidence
changes, and resolution changes. Hopper's existing bounded parent requeue on a
severity increase is a useful precedent, not a complete freshness mechanism.
Do not update every parent's large JSON report synchronously inside the target's
write transaction. Cycles must terminate and cannot cause endless mutual requeues.

An unchanged observation should cause no artifact/report write. Preserve Hopper's
existing avoidance of redundant JSONB/TOAST updates and row-lock contention.

## Example: SFRA's bundled fs package

Preserve three separate facts:

- The VSIX physically contains a security-holder package manifest: containment.
- That manifest claims a package identity: identity evidence.
- Source code imports `fs`: an import reference whose runtime resolution may be
  Node's built-in module.

The third fact does not prove the bundled placeholder executes. The placeholder
can still be useful security evidence. A bundled file is not automatically the
same artifact as a registry tarball; only record a content identity supported by
bytes or explicit resolution evidence.

## Deferred implementation sequence

1. Define scan's edge/coverage envelope and fixtures, including unresolved edges.
2. Add Hopper persistence and forward/reverse APIs, with idempotent ingestion.
3. Wire independent dependency refresh and versioned owner assessments.
4. Expose current versus observed assessments to readers; pin training snapshots.
5. Backfill only relationships supported by retained evidence. Leave missing
   declaration/resolution context unknown rather than guessing from paths.

No wholesale archive-storage rewrite or new Cleave JSON generation is required.

## Acceptance tests

- One dependency shared by two owners updates once; both become stale, while
  their recorded historical assessments remain reproducible.
- Identical manifests in different owner contexts can resolve differently.
- A package and a standalone source file can each declare an unresolved edge.
- An immutable target SHA remains pinned when a newer package release appears.
- Dependency archives retain complete standalone reports after owner rescans.
- Bundled members and fetched references retain distinct containment semantics.
- Partial scans preserve known relationships and report incomplete coverage.
- Built-in imports, skipped fetches, failed resolutions, and unevaluated targets
  never become fabricated benign verdicts or execution claims.
- Cycles terminate; popular-dependency invalidation resumes across bounded batches.
- Improved and worsened assessments both invalidate the appropriate consumers.
- Repeated identical uploads do not rewrite artifact reports or duplicate edges.
- Fresh standalone and dependency scans produce equivalent artifact assessments
  under the same bytes, configuration, and external-evidence snapshot.
