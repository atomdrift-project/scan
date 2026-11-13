//! What a stored report, and an upload, keep of a scan's members.

use std::collections::{HashMap, HashSet};

use cleave::Criticality;
use cleave::types::{CompactFile, CompactReport, Rel, Role};

use super::compact_crit;
use crate::model::Classification;

/// The most members one artifact keeps for their findings. The containers on
/// the way to a kept member do not count against it.
pub(super) const MAX_KEPT_MEMBERS: usize = 256;

/// Quiet diagnostic rows are useful but also attacker-controlled. Keep their
/// output bounded, and mark any elision on the root explicitly.
const MAX_DIAGNOSTIC_ONLY_NODES: usize = 4096;

/// Each node's parent, by index. A member cleave recorded no parent for is
/// taken to sit in the root.
pub(super) fn parents(nodes: &[CompactFile]) -> Vec<Option<usize>> {
    let index: HashMap<u32, usize> = nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| match node.parent {
            Some(pid) => index.get(&pid).copied().filter(|&p| p != i),
            None => (i != 0).then_some(0),
        })
        .collect()
}

/// Which nodes are the artifact's own: the root, and every node reached from
/// it through containment or decoding. A fetched dependency, a registry
/// record and anything beneath either are not.
pub(super) fn own_nodes(nodes: &[CompactFile], parents: &[Option<usize>]) -> Vec<bool> {
    let mut order: Vec<usize> = (1..nodes.len()).collect();
    order.sort_by_key(|&i| nodes[i].depth);
    let mut own = vec![false; nodes.len()];
    if let Some(first) = own.first_mut() {
        *first = true;
    }
    for i in order {
        let node = &nodes[i];
        let contained = matches!(node.rel, Rel::Member | Rel::Unpacked)
            && node.role == Role::Content
            && node.file_type != "registry";
        own[i] = contained && parents[i].is_some_and(|parent| own[parent]);
    }
    own
}

/// The artifact's own members that a stored report and an upload keep, by
/// node id.
///
/// A member is a candidate when it is among the three riskiest, carries a
/// notable+ (crit ≥ 3) trait of its own, or is cited by one. At most
/// [`MAX_KEPT_MEMBERS`] candidates are kept: gravest verdict first
/// (`class_of`, `None` for a member nobody evaluated), then the most cited,
/// then the riskiest, then by sha256, so the same scan always keeps the same
/// set. Every container on the path from the root to a kept member is kept
/// too, so it has somewhere to sit.
///
/// The cut hides nothing from a verdict: every member is folded into its
/// containers before this runs.
pub(super) fn kept_members(
    report: &CompactReport,
    class_of: impl Fn(u32) -> Option<Classification>,
) -> HashSet<u32> {
    let nodes = &report.files;
    let parents = parents(nodes);
    let own = own_nodes(nodes, &parents);
    let mut cited: HashMap<u32, usize> = HashMap::new();
    for source in nodes
        .iter()
        .flat_map(|f| &f.findings)
        .filter(|t| compact_crit(t) >= Criticality::Notable)
        .flat_map(|t| &t.from)
    {
        *cited.entry(source.file).or_default() += 1;
    }
    let members: Vec<usize> = (1..nodes.len()).filter(|&i| own[i]).collect();
    let mut by_risk = members.clone();
    by_risk.sort_by_key(|&i| std::cmp::Reverse(nodes[i].risk));
    let top3: HashSet<usize> = by_risk.into_iter().take(3).collect();
    let notable = |f: &CompactFile| {
        f.findings
            .iter()
            .any(|t| compact_crit(t) >= Criticality::Notable)
    };
    let citations = |i: usize| cited.get(&nodes[i].id).copied().unwrap_or(0);
    let mut chosen: Vec<usize> = members
        .into_iter()
        .filter(|&i| top3.contains(&i) || notable(&nodes[i]) || citations(i) > 0)
        .collect();
    chosen.sort_by(|&a, &b| {
        class_of(nodes[b].id)
            .cmp(&class_of(nodes[a].id))
            .then_with(|| citations(b).cmp(&citations(a)))
            .then_with(|| nodes[b].risk.cmp(&nodes[a].risk))
            .then_with(|| nodes[a].sha.cmp(&nodes[b].sha))
    });
    if chosen.len() > MAX_KEPT_MEMBERS {
        tracing::info!(
            sha256 = %nodes.first().map_or("", |f| f.sha.as_str()),
            candidates = chosen.len(),
            kept = MAX_KEPT_MEMBERS,
            "report retention: notable members over the cap; the least grave are left out",
        );
        chosen.truncate(MAX_KEPT_MEMBERS);
    }
    let mut kept = HashSet::new();
    for mut at in chosen {
        // Climb to the root, stopping where an earlier member's path joined.
        while at != 0 && kept.insert(nodes[at].id) {
            let Some(parent) = parents[at] else { break };
            at = parent;
        }
    }
    kept
}

/// Post-classification retention rubric (operator policy): a
/// stored report keeps a member node only when someone will plausibly read it
/// again. A node survives when it is
///
/// 1. the root (depth 0 / id 0),
/// 2. one of the artifact's own members in `kept` (see [`kept_members`]),
/// 3. a node of a fetched dependency's subtree that is among the top 3 nodes
///    by `risk`, carries its own crit ≥ 3 (notable+) trait, or is cited by
///    one, or
/// 4. provenance skeleton: a `registry` record or a fetch-edge placeholder
///    (empty type) — dependency provenance sidecars reference these by
///    `file_id`, so they must not dangle.
///
/// An own member outside `kept` that records an analysis gap is stripped, not
/// dropped: its listing skeleton and gaps survive so a reader can see the
/// analysis was incomplete, capped at [`MAX_DIAGNOSTIC_ONLY_NODES`].
///
/// Everything else is removed outright — no stub rows. That is most of a
/// typical report's nodes and bytes, almost all of it `facts`/`ctx` that only
/// trait evaluation and featurization (both already complete) ever consume. A
/// retraining featurizer reads `facts` from the nodes that remain.
pub(crate) fn apply_report_retention(report: &mut CompactReport, kept: &HashSet<u32>) {
    let notable = |f: &CompactFile| {
        f.findings
            .iter()
            .any(|t| compact_crit(t) >= Criticality::Notable)
    };
    let own: HashSet<u32> = {
        let parents = parents(&report.files);
        let own = own_nodes(&report.files, &parents);
        report
            .files
            .iter()
            .zip(own)
            .filter_map(|(f, own)| own.then_some(f.id))
            .collect()
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

    let mut diagnostic_only = 0usize;
    files.retain_mut(|f| {
        let readable = if own.contains(&f.id) {
            kept.contains(&f.id)
        } else {
            top3.contains(&f.id) || cited.contains(&f.id) || notable(f)
        };
        if f.depth == 0
            || f.id == 0
            || readable
            || f.file_type.is_empty()
            || f.file_type == "registry"
        {
            return true;
        }
        // A diagnostic-only listing needs its path, identity and gaps, not
        // quiet finding payloads or metrics already used for scoring.
        if f.analysis_gaps.is_empty() {
            return false;
        }
        diagnostic_only += 1;
        let budget = diagnostic_only <= MAX_DIAGNOSTIC_ONLY_NODES;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Retention as the pipeline applies it, with no member evaluated.
    fn retain(report: &mut CompactReport) {
        let kept = kept_members(report, |_| None);
        apply_report_retention(report, &kept);
    }

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
        let mut report = cleave::types::compact_from_files(&files, cleave::types::Facts::Full);
        retain(&mut report);

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
        retain(&mut report);
        // The root, the three riskiest members, then the diagnostic budget.
        assert_eq!(report.files.len(), MAX_DIAGNOSTIC_ONLY_NODES + 4);
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
        retain(&mut report);
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
            // Equally quiet, but carrying identity claims: dropped all the same.
            {"id": 9, "path": "p.tgz!!tool.exe", "sha": "9", "size": 1, "depth": 1, "type": "pe",
             "mol": "C2H4", "traits": [{"id": "e", "crit": 2}],
             "ident": {"name": {"value": "tool", "source": "pe.version.product_name", "verified": false},
                       "version": {"value": "1.2.3", "source": "pe.version.file_version", "verified": false},
                       "trust": "unsigned"}},
        ]}))
        .unwrap();
        retain(&mut report);
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
        assert!(
            !ids.contains(&9),
            "a member is kept for its findings, not its identity claims: {ids:?}"
        );
        assert!(
            !ids.contains(&7) && !ids.contains(&8),
            "sub-notable uncited nodes are dropped outright: {ids:?}"
        );
    }

    /// An archive with more notable members than the cap keeps the gravest,
    /// the same ones every time, and the containers on the way to them.
    #[test]
    fn kept_members_are_capped_gravest_first_with_their_containers() {
        let extra = 44;
        let members: Vec<serde_json::Value> = (1..=MAX_KEPT_MEMBERS + extra)
            .map(|i| {
                serde_json::json!({
                    "id": i + 1, "pid": 1, "path": format!("p.tgz!!inner.zip!!m{i:04}.js"),
                    "sha": format!("{i:064}"), "size": 1, "depth": 2, "type": "javascript",
                    "traits": [{"id": "x", "crit": 3}]
                })
            })
            .collect();
        let files: Vec<serde_json::Value> = [
            serde_json::json!({"id": 0, "path": "p.tgz", "sha": "0", "size": 1, "type": "npm"}),
            serde_json::json!({"id": 1, "pid": 0, "path": "p.tgz!!inner.zip", "sha": "1", "size": 1, "depth": 1, "type": "zip"}),
        ]
        .into_iter()
        .chain(members)
        .collect();
        let report: CompactReport =
            serde_json::from_value(serde_json::json!({ "files": files })).unwrap();
        // The last ids are the graver ones.
        let class_of = |id: u32| {
            let threshold = u32::try_from(MAX_KEPT_MEMBERS + extra + 1 - 10).unwrap();
            Some(if id > threshold {
                Classification::Hostile
            } else {
                Classification::Benign
            })
        };
        let kept = kept_members(&report, class_of);
        assert_eq!(
            kept,
            kept_members(&report, class_of),
            "the same set every time"
        );
        assert!(kept.contains(&1), "the container on the way is kept");
        assert_eq!(
            kept.len(),
            MAX_KEPT_MEMBERS + 1,
            "the container does not count against the cap"
        );
        let last = u32::try_from(MAX_KEPT_MEMBERS + extra + 1).unwrap();
        assert!(
            (last - 9..=last).all(|id| kept.contains(&id)),
            "the hostile members are kept ahead of the rest"
        );
        let mut retained = report.clone();
        apply_report_retention(&mut retained, &kept);
        assert_eq!(retained.files.len(), MAX_KEPT_MEMBERS + 2);
    }
}
