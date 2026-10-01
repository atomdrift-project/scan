//! What a stored report keeps, and the listing-only members `--show=all` adds.

use std::collections::HashSet;

use cleave::Criticality;
use cleave::types::{CompactFile, CompactReport};

use super::{ARCHIVE_DELIMITER, compact_crit};

/// A minimal archive-member record captured before `finalize()` discards
/// `archive_contents`. Carries only what a `--show=all` listing needs.
pub(super) struct ArchiveMemberStub {
    /// Archive-relative path (single `!` between nesting levels, as cleave
    /// records it in `archive_contents`).
    pub(super) path: String,
    /// Detected file type (e.g. "markdown", "png").
    pub(super) file_type: String,
    /// SHA256 of the member's contents — the key used to skip members that were
    /// already analyzed and therefore already present in `files[]`.
    pub(super) sha256: String,
    /// Uncompressed size in bytes.
    pub(super) size_bytes: u64,
}

/// Ceiling on identity-only survivors, the one retention rule an attacker can
/// drive per-member: `ident` comes from the sample itself (a PE version
/// resource, a bundle manifest), so an archive of a million near-identical
/// tiny signed-looking members — which compresses to almost nothing — would
/// otherwise turn a rubric meant to *shed* stored bytes into an amplifier.
/// Every other rule is bounded by interestingness (top-3, cited-by-finding,
/// notable trait). Far above any real installer; only a hostile archive or a
/// pathological corpus sample reaches it.
const MAX_IDENTITY_ONLY_NODES: usize = 4096;
/// Quiet diagnostic rows are useful but also attacker-controlled. Keep their
/// output bounded, and mark any elision on the root explicitly.
const MAX_DIAGNOSTIC_ONLY_NODES: usize = 4096;

/// Post-classification retention rubric (operator policy, 2026-08-01): a
/// stored report keeps a member node only when someone will plausibly read it
/// again. A node survives when it is
///
/// 1. the root (depth 0 / id 0),
/// 2. among the top 3 nodes by `risk` in this report,
/// 3. carrying its own crit ≥ 3 (notable+) trait,
/// 4. cited by any crit ≥ 3 finding's `from` list (composite contributors), or
/// 5. provenance skeleton: a `registry` record or a fetch-edge placeholder
///    (empty type) — dependency provenance sidecars reference these by
///    `file_id`, so they must not dangle.
///
/// A node failing every rule but carrying identity claims (`ident`: a PE
/// version resource, a bundle manifest, a signer) is stripped, not dropped:
/// its analysis payload — `facts`/`ctx`/`traits`, the bytes retention exists
/// to shed — is cleared, and the listing skeleton plus `ident` survives in a
/// few hundred bytes. hopper projects those claims into queryable identity
/// columns, and a member dropped here never becomes a hopper row at all.
/// Capped at [`MAX_IDENTITY_ONLY_NODES`] — it is the one rule whose survivor
/// count an attacker controls directly.
///
/// Everything else is removed outright — no stub rows. On the benchmark
/// corpus that is ~77% of nodes and ~56% of stored report bytes, almost all
/// of it `facts`/`ctx` that only trait evaluation and featurization (both
/// already complete) ever consume. Collimator's retraining featurizer reads
/// `facts` from the nodes that remain. `SCAN_KEEP_ALL_MEMBERS=1` makes the
/// caller skip the rubric, for debugging or corpus captures.
pub(crate) fn apply_report_retention(report: &mut CompactReport) {
    let notable = |f: &CompactFile| {
        f.findings
            .iter()
            .any(|t| compact_crit(t) >= Criticality::Notable)
    };
    let files = &mut report.files;

    // Contributors cited by any notable+ finding, across all nodes.
    let cited: HashSet<u32> = files
        .iter()
        .flat_map(|f| &f.findings)
        .filter(|t| compact_crit(t) >= Criticality::Notable)
        .flat_map(|t| t.from.iter().map(|r| r.file))
        .collect();
    // Top 3 by risk (ties keep earlier nodes — stable sort on a stable list).
    let mut by_risk: Vec<(u32, i64)> = files.iter().map(|f| (f.id, f.risk)).collect();
    by_risk.sort_by_key(|&(_, risk)| std::cmp::Reverse(risk));
    let top3: HashSet<u32> = by_risk.iter().take(3).map(|&(id, _)| id).collect();

    let mut identity_only = 0usize;
    let mut diagnostic_only = 0usize;
    files.retain_mut(|f| {
        if f.depth == 0
            || f.id == 0
            || top3.contains(&f.id)
            || cited.contains(&f.id)
            || f.file_type.is_empty()
            || f.file_type == "registry"
            || notable(f)
        {
            return true;
        }
        // A diagnostic-only listing needs its path, identity and gaps, not
        // quiet finding payloads or metrics already used for scoring.
        let budget = if !f.analysis_gaps.is_empty() {
            diagnostic_only += 1;
            diagnostic_only <= MAX_DIAGNOSTIC_ONLY_NODES
        } else if f.identity.as_ref().is_some_and(|i| !i.is_empty()) {
            identity_only += 1;
            identity_only <= MAX_IDENTITY_ONLY_NODES
        } else {
            false
        };
        if budget {
            f.formula = None;
            f.findings = Vec::new();
            f.refs = Vec::new();
            f.context = Vec::new();
            f.facts = cleave::types::CompactFacts::default();
        }
        budget
    });
    if diagnostic_only > MAX_DIAGNOSTIC_ONLY_NODES
        && let Some(root) = files.first()
    {
        root.analysis_gaps
            .record(cleave::types::AnalysisGap::ReportRetentionLimited);
    }
    if identity_only > MAX_IDENTITY_ONLY_NODES {
        tracing::warn!(
            kept = MAX_IDENTITY_ONLY_NODES,
            dropped = identity_only - MAX_IDENTITY_ONLY_NODES,
            "report retention: identity-only members over cap; report truncated"
        );
    }
}

/// Sentinel `risk` for an archive member scan listed but never analyzed. Keeps
/// an unanalyzed listing (-1) distinguishable from an analyzed member that
/// simply produced no traits (0), so no consumer mistakes silence for a clean
/// result.
pub(crate) const UNANALYZED_MEMBER_RISK: i64 = -1;

/// Append the archive members cleave never analyzed to `report.files` as
/// listing-only entries (`id`/`path`/`type`/`sha`/`size`/`depth`). Members that
/// were analyzed — matched by sha256 — are already present and skipped. Each
/// appended entry carries the [`UNANALYZED_MEMBER_RISK`] sentinel, which keeps
/// it out of `ml.files` while it stays in the raw manifest.
pub(super) fn append_unanalyzed_members(report: &mut CompactReport, members: &[ArchiveMemberStub]) {
    let listed: HashSet<&str> = members.iter().map(|m| m.path.as_str()).collect();
    let mut seen: HashSet<&str> = report.files.iter().map(|f| f.sha.as_str()).collect();
    // Listing paths are archive-relative; re-root them under the root file's
    // path with the report's own delimiter so they match their analyzed siblings.
    let root_path = report.files.first().map_or("", |f| f.path.as_str());
    let mut next_id = report.files.iter().map(|f| f.id).max().map_or(0, |m| m + 1);
    let mut listing = Vec::new();
    for m in members {
        if m.sha256.is_empty() || !seen.insert(&m.sha256) {
            continue;
        }
        let layers = listing_layers(&m.path, &listed);
        listing.push(CompactFile {
            id: next_id,
            path: format!(
                "{root_path}{ARCHIVE_DELIMITER}{}",
                layers.join(ARCHIVE_DELIMITER)
            ),
            file_type: m.file_type.clone(),
            sha: m.sha256.clone(),
            size: m.size_bytes,
            risk: UNANALYZED_MEMBER_RISK,
            depth: u32::try_from(layers.len()).unwrap_or(u32::MAX),
            ..CompactFile::default()
        });
        next_id += 1;
    }
    report.files.extend(listing);
}

/// The archive layers of an `archive_contents` path.
///
/// cleave joins nesting levels there with a single `!` (`inner.tar!lib/a.so`),
/// and a member's own name may contain `!` too. A `!` is a layer boundary only
/// where the text before it is itself a listed entry: cleave lists every nested
/// archive before its contents.
fn listing_layers<'a>(path: &'a str, listed: &HashSet<&str>) -> Vec<&'a str> {
    let mut layers = Vec::new();
    let mut start = 0;
    for (bang, _) in path.match_indices('!') {
        // `bang` indexes an ASCII '!', so both slices fall on char boundaries.
        if let (Some(container), Some(layer)) = (path.get(..bang), path.get(start..bang))
            && listed.contains(container)
        {
            layers.push(layer);
            start = bang + 1;
        }
    }
    layers.push(path.get(start..).unwrap_or(path));
    layers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a wire-shaped fixture into the typed report the pipeline carries,
    /// so these tests exercise the same reader production does.
    fn report(v: serde_json::Value) -> cleave::types::CompactReport {
        serde_json::from_value(v).unwrap()
    }

    fn stub(path: &str, ty: &str, sha: &str, size: u64) -> ArchiveMemberStub {
        ArchiveMemberStub {
            path: path.to_string(),
            file_type: ty.to_string(),
            sha256: sha.to_string(),
            size_bytes: size,
        }
    }

    #[test]
    fn appends_unanalyzed_members_skipping_already_analyzed() {
        // files[0] is the root archive; files[1] an analyzed member with a sha.
        let mut report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "sha": "root", "size": 4096},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "sha": "aaa", "size": 200, "risk": 9},
            ]
        }));
        let members = vec![
            // Already analyzed (matches sha "aaa") — must be skipped, no duplicate.
            stub("evil.sh", "shell", "aaa", 200),
            // Never analyzed — must be appended as a listing-only entry.
            stub("README.md", "markdown", "bbb", 1024),
            // A nested archive is listed before its contents; a member inside
            // it is re-rooted with `!!`, and depth counts the layers.
            stub("inner.tar", "tar", "ddd", 9000),
            stub("inner.tar!logo.png", "png", "ccc", 8192),
        ];
        append_unanalyzed_members(&mut report, &members);

        let files = &report.files;
        assert_eq!(files.len(), 5, "three listing-only members appended");

        let readme = &files[2];
        assert_eq!(readme.id, 2);
        assert_eq!(readme.path, "app.zip!!README.md");
        assert_eq!(readme.file_type, "markdown");
        assert_eq!(readme.size, 1024);
        assert_eq!(readme.depth, 1);
        assert_eq!(
            readme.risk, UNANALYZED_MEMBER_RISK,
            "sentinel marks the member unanalyzed"
        );

        let logo = &files[4];
        assert_eq!(logo.path, "app.zip!!inner.tar!!logo.png");
        assert_eq!(logo.depth, 2);
        assert_eq!(logo.risk, UNANALYZED_MEMBER_RISK);
    }

    /// A `!` in a member's own name is not a layer boundary: only a `!` after
    /// a listed entry (a nested archive) is.
    #[test]
    fn a_bang_in_a_member_name_is_not_a_layer() {
        let mut report = report(serde_json::json!({
            "files": [{"id": 0, "path": "app.zip", "type": "zip", "sha": "root", "size": 1}]
        }));
        let members = vec![
            stub("docs/Hello!.txt", "text", "a", 1),
            stub("x.tar", "tar", "b", 1),
            stub("x.tar!wow!.md", "markdown", "c", 1),
        ];
        append_unanalyzed_members(&mut report, &members);
        let listed: Vec<(&str, u32)> = report.files[1..]
            .iter()
            .map(|f| (f.path.as_str(), f.depth))
            .collect();
        assert_eq!(
            listed,
            vec![
                ("app.zip!!docs/Hello!.txt", 1),
                ("app.zip!!x.tar", 1),
                ("app.zip!!x.tar!!wow!.md", 2),
            ]
        );
    }

    /// Retention decides what ships in the stored report, so this pins the
    /// rubric directly: root kept, top-3-by-risk kept, notable+ kept, and an
    /// ordinary quiet member dropped.
    #[test]
    fn retention_keeps_only_readable_nodes_from_a_real_report() {
        use cleave::types::{Criticality, FindingKind};

        let mut files = Vec::new();
        for (id, depth, risk, crit) in [
            (0u32, 0u32, 0u32, Criticality::Baseline),
            (1, 1, 99, Criticality::Baseline),
            (2, 1, 0, Criticality::Baseline),
            (3, 1, 0, Criticality::Notable),
            (4, 1, 0, Criticality::Baseline),
        ] {
            let mut fa = cleave::FileAnalysis {
                id,
                path: format!("m{id}.py"),
                file_type: "python".to_string(),
                sha256: format!("{id:064}"),
                size: 100,
                depth,
                score: risk,
                ..Default::default()
            };
            let mut f = cleave::types::Finding::new(
                "objectives/execution/shell::sh".to_string(),
                FindingKind::Capability,
                String::new(),
                0.9,
            );
            f.crit = crit;
            fa.findings.push(f);
            files.push(fa);
        }
        let mut report = cleave::types::compact_from_files(&files);
        apply_report_retention(&mut report);

        let kept: Vec<u32> = report.files.iter().map(|f| f.id).collect();
        // 0: root. 1: top-3 by risk. 3: carries a notable finding.
        // 2 and 4 are quiet, low-risk members — nobody reads them again.
        assert_eq!(kept, vec![0, 1, 2, 3], "retention kept the wrong set");
    }

    #[test]
    fn diagnostic_only_retention_is_bounded_and_explicit() {
        let mut files = Vec::new();
        for id in 0..MAX_DIAGNOSTIC_ONLY_NODES + 10 {
            files.push(serde_json::json!({"id":id,"path":format!("p.tar!!{id}.js"),"sha":"x","size":1,"type":"javascript","depth":1,"analysis_gaps":["flow-query-incomplete"]}));
        }
        let mut report: cleave::types::CompactReport =
            serde_json::from_value(serde_json::json!({"files": files})).unwrap();
        apply_report_retention(&mut report);
        assert_eq!(report.files.len(), MAX_DIAGNOSTIC_ONLY_NODES + 3);
        assert!(
            report.files[0]
                .analysis_gaps
                .iter()
                .any(|g| g == cleave::types::AnalysisGap::ReportRetentionLimited)
        );
    }

    #[test]
    fn retention_keeps_incomplete_quiet_members() {
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({"files": [
            {"id":0,"path":"p.tar","sha":"0","size":1,"type":"tar"},
            {"id":1,"path":"p.tar!!a","sha":"1","size":1,"type":"data","depth":1},
            {"id":2,"path":"p.tar!!b","sha":"2","size":1,"type":"data","depth":1},
            {"id":3,"path":"p.tar!!incomplete.js","sha":"3","size":1,"type":"javascript","depth":1,"analysis_gaps":["flow-query-incomplete"]},
            {"id":4,"path":"p.tar!!quiet.js","sha":"4","size":1,"type":"javascript","depth":1}
        ]})).unwrap();
        apply_report_retention(&mut report);
        assert!(
            report
                .files
                .iter()
                .any(|f| f.id == 3 && !f.analysis_gaps.is_empty())
        );
        assert!(!report.files.iter().any(|f| f.id == 4));
    }

    #[test]
    fn retention_rubric_keeps_only_readable_nodes() {
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({"files": [
            // Root: kept (depth 0 / id 0) even with no traits.
            {"id": 0, "path": "p.tgz", "sha": "0", "size": 1, "depth": 0, "type": "npm", "risk": 1},
            // Own notable trait: kept.
            {"id": 1, "path": "p.tgz!!a.js", "sha": "1", "size": 1, "depth": 1, "type": "javascript",
             "traits": [{"id": "a", "crit": 3}]},
            // Cited by a notable composite on another node: kept.
            {"id": 2, "path": "p.tgz!!b.md", "sha": "2", "size": 1, "depth": 1, "type": "markdown",
             "traits": [{"id": "b", "crit": 1}]},
            // The citing node (notable, with from): kept.
            {"id": 3, "path": "p.tgz!!c.js", "sha": "3", "size": 1, "depth": 1, "type": "javascript",
             "traits": [{"id": "c", "crit": 4, "from": [{"file": 2}]}]},
            // Provenance skeleton: kept.
            {"id": 4, "path": "p.tgz!!reg", "sha": "4", "size": 1, "depth": 2, "type": "registry"},
            {"id": 5, "path": "p.tgz!!edge", "sha": "5", "size": 1, "depth": 2, "type": "", "rel": "fetched"},
            // High risk: kept via top-3 (risks 9 > root's 1).
            {"id": 6, "path": "p.tgz!!e.txt", "sha": "6", "size": 1, "depth": 1, "type": "text", "risk": 9},
            // Sub-notable, uncited, low-risk: dropped.
            {"id": 7, "path": "p.tgz!!f.md", "sha": "7", "size": 1, "depth": 1, "type": "markdown",
             "traits": [{"id": "d", "crit": 2}]},
            {"id": 8, "path": "p.tgz!!g.txt", "sha": "8", "size": 1, "depth": 2, "type": "text"},
            // Equally quiet, but carrying identity claims: stripped, not dropped.
            {"id": 9, "path": "p.tgz!!tool.exe", "sha": "9", "size": 1, "depth": 1, "type": "pe",
             "mol": "C2H4", "traits": [{"id": "e", "crit": 2}],
             "ident": {"name": {"value": "tool", "source": "pe.version.product_name", "verified": false},
                       "version": {"value": "1.2.3", "source": "pe.version.file_version", "verified": false},
                       "trust": "unsigned"}},
        ]}))
        .unwrap();
        apply_report_retention(&mut report);
        let ids: Vec<u32> = report.files.iter().map(|f| f.id).collect();
        assert!(
            ids.contains(&0) && ids.contains(&1) && ids.contains(&2) && ids.contains(&3),
            "root, notable, cited contributor, and citing node survive: {ids:?}"
        );
        assert!(
            ids.contains(&4) && ids.contains(&5),
            "registry and fetch-placeholder provenance skeleton survive: {ids:?}"
        );
        assert!(ids.contains(&6), "top-risk node survives: {ids:?}");
        let exe = report
            .files
            .iter()
            .find(|f| f.id == 9)
            .expect("identity-bearing member survives as a listing entry");
        assert!(
            exe.findings.is_empty() && exe.formula.is_none(),
            "stripped member sheds its analysis payload"
        );
        assert!(
            exe.identity.as_ref().is_some_and(|i| !i.is_empty()),
            "stripped member keeps its identity claims"
        );
        assert!(
            !ids.contains(&7) && !ids.contains(&8),
            "sub-notable uncited nodes are dropped outright: {ids:?}"
        );
    }

    /// A hostile archive can carry unlimited quiet members that each claim an
    /// identity, so the one attacker-driven retention rule has to stop.
    #[test]
    fn identity_only_retention_is_capped() {
        let members: Vec<serde_json::Value> = (1..=super::MAX_IDENTITY_ONLY_NODES + 500)
            .map(|i| {
                serde_json::json!({
                    "id": i, "path": format!("p.tgz!!m{i}.exe"), "sha": format!("{i}"),
                    "size": 1, "depth": 1, "type": "pe",
                    "ident": {
                        "name": {"value": "tool", "source": "pe.version.product_name",
                                 "verified": false},
                        "trust": "unsigned"
                    }
                })
            })
            .collect();
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": std::iter::once(serde_json::json!(
                {"id": 0, "path": "p.tgz", "sha": "0", "size": 1, "depth": 0, "type": "zip"}))
                .chain(members)
                .collect::<Vec<_>>(),
        }))
        .unwrap();

        apply_report_retention(&mut report);

        // Three nodes survive on rules the identity budget never sees: the root
        // (rule 1) and ids 1-2, which win top-3-by-risk because every risk ties
        // and the sort is stable. The other 4096 are the identity budget spent
        // in full — the remaining 500 members are dropped.
        assert_eq!(
            report.files.len(),
            super::MAX_IDENTITY_ONLY_NODES + 3,
            "identity-only survivors stop at the cap"
        );
        assert!(
            report.files.iter().any(|f| f.id == 0),
            "the root is kept by rule 1, not by the identity budget"
        );
        let highest_kept =
            u32::try_from(super::MAX_IDENTITY_ONLY_NODES + 2).expect("cap fits in an id");
        assert!(
            report.files.iter().all(|f| f.id <= highest_kept),
            "the cap keeps a deterministic prefix, not an arbitrary subset"
        );
    }
}
