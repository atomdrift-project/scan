//! What a file says it is: the header line, and the signer, product and
//! trust claims beneath it.

use cleave::output::sample_text;
use cleave::types::{Criticality, FileAnalysis};

use super::{format_size, tiny_field, tiny_path};

/// The `path\ttype size score[\theadline]` line that opens a file's block.
pub(super) fn header(out: &mut String, file: &FileAnalysis) {
    out.push_str(&tiny_field(&tiny_path(&file.path)));
    out.push('\t');
    out.push_str(&tiny_field(&file.file_type));
    out.push(' ');
    out.push_str(&format_size(file.size));
    out.push(' ');
    out.push_str(&file.score.to_string());
    if let Some(headline) = file.identity.as_ref().and_then(identity_headline) {
        out.push('\t');
        out.push_str(&headline);
    }
    out.push('\n');
}

/// The identity claims the headline consumed, by value.
///
/// Mirrors [`identity_headline`]'s selection exactly so [`render_identity`]
/// can show everything it *didn't* pick without repeating what it did. Kept
/// beside the headline rather than derived from its rendered string: the
/// headline truncates and joins, and matching on that text would silently
/// start duplicating the moment either changes.
fn headline_claims(id: &filefacts::Identity) -> Vec<String> {
    use filefacts::Trust;
    let mut used = Vec::new();
    // what — the subject, in the same order the headline tries them.
    if let Some(c) = &id.identifier {
        used.push(c.value.clone());
    } else if let Some(c) = &id.title {
        used.push(c.value.clone());
    } else if let Some(c) = &id.name {
        used.push(c.value.clone());
        // The version is folded into the subject only when the name is used.
        if let Some(v) = &id.version {
            used.push(v.value.clone());
        }
    }
    // who — the responsible party, most authoritative first.
    if let Some(s) = id
        .signer
        .as_ref()
        .and_then(|s| s.organization.clone().or_else(|| s.common_name.clone()))
    {
        used.push(s);
    } else if let Some(a) = id
        .authors
        .first()
        .and_then(|a| a.name.clone().or_else(|| a.email.clone()))
    {
        used.push(a);
    } else if let Some(c) = &id.organization {
        used.push(c.value.clone());
    } else if let Some(c) = &id.team_id {
        used.push(c.value.clone());
    }
    // provenance — the producing tool stands in only when nothing signed it.
    if matches!(id.trust, Trust::Unsigned | Trust::AdHoc)
        && let Some(c) = &id.producer
    {
        used.push(c.value.clone());
    }
    used
}

/// Supplementary identity lines: every claim the artifact makes about itself
/// that the headline had no room for, plus the forensic fields that never
/// belonged in a headline (the build path — which often leaks the dev
/// account — contact emails, stable unique ids).
///
/// These are rendered as labelled pairs rather than folded into prose because
/// the *disagreements* are the point. A PE whose version resource names one
/// company while its Authenticode chain names another, a document whose
/// author is not the account that produced it, a Mach-O whose bundle id does
/// not match its team — each is a supply-chain signal that only exists when
/// both halves are on the page. The headline shows one of each pair by
/// design; this shows the rest, and lets the reader notice.
///
/// Emitted only when present, so most files render nothing here.
pub(super) fn render_identity(out: &mut String, id: &filefacts::Identity) {
    let used = headline_claims(id);
    let fresh = |v: &str| !v.is_empty() && !used.iter().any(|u| u == v);

    // What it claims to be, beyond the one the headline picked.
    let mut claims: Vec<String> = Vec::new();
    {
        let mut push_claim = |key: &str, claim: &Option<filefacts::Claim>, max: usize| {
            if let Some(c) = claim
                && fresh(&c.value)
            {
                claims.push(format!("{key}=\"{}\"", truncate_end(&c.value, max)));
            }
        };
        push_claim("product", &id.project, 48);
        push_claim("company", &id.organization, 48);
        push_claim("file", &id.name, 48);
        push_claim("title", &id.title, 48);
        push_claim("id", &id.identifier, 64);
        push_claim("version", &id.version, 24);
        push_claim("team", &id.team_id, 32);
        push_claim("producer", &id.producer, 32);
        // The rights holder, unless it only restates the company or the signer
        // (`© Microsoft Corporation. All rights reserved.` beside a Microsoft
        // signature says nothing new). A holder that differs from both is a
        // disagreement worth a token.
        if let Some(c) = &id.copyright {
            let holder = copyright_holder(&c.value);
            let restates = [
                id.organization.as_ref().map(|o| o.value.as_str()),
                id.signer.as_ref().and_then(|s| s.organization.as_deref()),
            ]
            .into_iter()
            .flatten()
            .any(|company| copyright_holder(company) == holder);
            if !holder.is_empty() && !restates && fresh(&c.value) {
                claims.push(format!("copyright=\"{}\"", truncate_end(&c.value, 48)));
            }
        }
    }
    if !claims.is_empty() {
        out.push_str(&format!("  {}  {}\n", "claims", claims.join(" ")));
    }

    // What it says it is for, in the publisher's words. Prose rather than a
    // token, so it gets its own line. It is the claim the behaviour is held
    // against: a package calling itself a "CLI core binary for Windows x64"
    // is expected to ship executables.
    if let Some(c) = &id.description
        && fresh(&c.value)
        && ![&id.name, &id.project, &id.title, &id.identifier]
            .into_iter()
            .flatten()
            .any(|other| other.value.trim().eq_ignore_ascii_case(c.value.trim()))
    {
        out.push_str(&format!(
            "  {}  \"{}\"\n",
            "about",
            truncate_end(&c.value, 100)
        ));
    }

    // Who signed it, in the signature's own words. The headline reduces a
    // signer to one name; a chain that a reader can compare against the
    // claims above needs the issuer and the subject too.
    if let Some(signer) = &id.signer {
        let mut parts: Vec<String> = Vec::new();
        for (key, value) in [
            ("CN", &signer.common_name),
            ("O", &signer.organization),
            ("subject", &signer.subject),
            ("issuer", &signer.issuer),
            ("at", &signer.signed_at),
        ] {
            if let Some(v) = value
                && fresh(v)
            {
                parts.push(format!("{key}=\"{}\"", truncate_end(v, 64)));
            }
        }
        if !parts.is_empty() {
            out.push_str(&format!("  {}  {}\n", "signer", parts.join(" ")));
        }
    }

    // Every named party, not just the one the headline chose.
    let authors: Vec<String> = id
        .authors
        .iter()
        .filter_map(|a| {
            let name = a.name.as_deref().or(a.email.as_deref())?;
            if !fresh(name) {
                return None;
            }
            let mut text = truncate_end(name, 48);
            if let Some(email) = a.email.as_deref().filter(|e| Some(*e) != a.name.as_deref()) {
                text.push_str(&format!(" <{}>", truncate_end(email, 48)));
            }
            text.push_str(&format!(" ({})", a.role));
            Some(text)
        })
        .collect();
    if !authors.is_empty() {
        out.push_str(&format!("  {}  {}\n", "authors", authors.join(", ")));
    }

    // Where it says it lives. A repository or homepage that disagrees with the
    // package name is the classic clone-and-rename tell.
    let urls: Vec<String> = id
        .urls
        .iter()
        .map(|u| format!("{:?}={}", u.kind, truncate_end(&u.url, 72)).to_lowercase())
        .collect();
    if !urls.is_empty() {
        out.push_str(&format!("  {}  {}\n", "links", urls.join(" ")));
    }

    if let Some(bp) = &id.build_path {
        out.push_str(&format!("  {}  {}\n", "build", truncate_mid(&bp.value, 72)));
    }
    if !id.emails.is_empty() {
        out.push_str(&format!("  {}  {}\n", "contact", id.emails.join(", ")));
    }
    if !id.unique_ids.is_empty() {
        let ids: Vec<String> = id
            .unique_ids
            .iter()
            .map(|(k, v)| format!("{k}={}", truncate_end(v, 16)))
            .collect();
        out.push_str(&format!("  {}  {}\n", "ids", ids.join(" ")));
    }
}

/// The concise identity headline: the 2-3 facts a security engineer
/// actually needs — *what it claims to be*, *who is responsible*, and
/// *how it's signed* — deduplicated, never a 100-field dump.
///
/// Shape: `<what> — <who> · <provenance>`.
/// - **what**: the canonical id / title / name (version folded in for
///   packages). For binaries this is the bundle id; for documents the
///   title; for packages `name version`.
/// - **who**: the responsible party, picked in order of authority —
///   verified signer org, then named author/maintainer, then a claimed
///   company, then a publisher namespace, then an explicit `ad hoc`.
/// - **provenance**: the trust tier when signed (System/Developer-ID/…),
///   otherwise the producing tool (Word/Excel/…) for documents.
///
/// Each slot is dropped when absent or when it would merely echo another.
/// Returns `None` when there is nothing worth showing.
pub(super) fn identity_headline(id: &filefacts::Identity) -> Option<String> {
    use filefacts::Trust;

    // what — canonical identity; fold the version in for package-style names.
    let what = id
        .identifier
        .as_ref()
        .map(|c| c.value.clone())
        .or_else(|| id.title.as_ref().map(|c| c.value.clone()))
        .or_else(|| {
            id.name.as_ref().map(|c| match &id.version {
                Some(v) => format!("{} {}", c.value, v.value),
                None => c.value.clone(),
            })
        })
        .map(|w| truncate_end(&w, 48));

    // who — responsible party, most authoritative first. `ad hoc` stands
    // in only when nothing is named.
    let signer_org = id
        .signer
        .as_ref()
        .and_then(|s| s.organization.clone().or_else(|| s.common_name.clone()));
    let author = id
        .authors
        .first()
        .and_then(|a| a.name.clone().or_else(|| a.email.clone()));
    let who = signer_org
        .or(author)
        .or_else(|| id.organization.as_ref().map(|c| c.value.clone()))
        .or_else(|| id.team_id.as_ref().map(|c| c.value.clone()))
        .or_else(|| matches!(id.trust, Trust::AdHoc).then(|| "ad hoc".to_string()))
        .map(|w| truncate_end(&w, 28))
        // Dropped when it just echoes `what`.
        .filter(|w| Some(w.as_str()) != what.as_deref());

    // provenance — trust tier when signed, else the producing tool.
    let third = if !matches!(id.trust, Trust::Unsigned | Trust::AdHoc) {
        Some(trust_text(id))
    } else {
        id.producer
            .as_ref()
            .map(|c| truncate_end(&c.value, 28))
            .filter(|p| Some(p.as_str()) != who.as_deref())
    };

    // Assemble at most three parts. `—` introduces the party; `·` the rest.
    let mut parts: Vec<(&str, String)> = Vec::new();
    if let Some(w) = what {
        parts.push(("·", w));
    }
    if let Some(w) = who {
        parts.push(("—", w));
    }
    if let Some(t) = third {
        parts.push(("·", t));
    }
    if parts.is_empty() {
        return None;
    }
    let mut s = String::new();
    for (i, (sym, text)) in parts.iter().enumerate() {
        if i > 0 {
            s.push_str(&format!(" {sym} "));
        }
        s.push_str(text);
    }
    Some(s)
}

/// Whether a file is signed by the platform vendor with a signature that
/// verified one of its claims, and raised nothing above notable.
///
/// Such a file's header and identity lines already say what it is (a
/// Microsoft-signed C runtime, an Apple system library). Listing its
/// capabilities, the file and memory APIs every runtime imports, spends the
/// LLM view's budget on facts that argue neither way and pushes the
/// artifact's own code further down the page.
pub(super) fn vouched_and_quiet(file: &FileAnalysis) -> bool {
    use filefacts::Trust;
    let Some(id) = &file.identity else {
        return false;
    };
    let vouched = matches!(id.trust, Trust::Platform | Trust::System)
        && [
            &id.name,
            &id.identifier,
            &id.organization,
            &id.team_id,
            &id.version,
            &id.project,
        ]
        .into_iter()
        .flatten()
        .any(|c| c.verified);
    vouched
        && !file
            .findings
            .iter()
            .any(|f| f.src.is_none() && f.crit >= Criticality::Suspicious)
}

/// The trust tier word, suffixed with `✓` when a signature
/// cryptographically verified one of the identity claims.
fn trust_text(id: &filefacts::Identity) -> String {
    use filefacts::Trust;
    let word = match id.trust {
        Trust::System => "system",
        Trust::Platform => "platform",
        Trust::DeveloperId => "developer-id",
        Trust::CaSigned => "ca-signed",
        Trust::SelfSigned => "self-signed",
        Trust::AdHoc => "ad-hoc",
        Trust::Unverified => "signed-unverified",
        Trust::Unsigned => "unsigned",
        _ => "unknown",
    };
    let verified = [
        &id.name,
        &id.identifier,
        &id.organization,
        &id.team_id,
        &id.version,
        &id.project,
    ]
    .into_iter()
    .flatten()
    .any(|c| c.verified);
    if verified {
        format!("{word} ✓")
    } else {
        word.to_string()
    }
}

/// Truncate keeping head and tail (for paths): `/home/user/…/main.rs`.
fn truncate_mid(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let head = max.saturating_sub(1) / 2;
    let tail = max.saturating_sub(1) - head;
    let h: String = chars[..head].iter().collect();
    let t: String = chars[chars.len() - tail..].iter().collect();
    format!("{h}…{t}")
}

/// The rights holder a copyright notice names, folded for comparison: the
/// letters left once the `©` / `(c)` / `Copyright` / `All rights reserved`
/// boilerplate, years and punctuation are gone. Applied to a company name too,
/// so `© 2024 Tencent. All rights reserved.` and `Tencent` compare equal.
fn copyright_holder(s: &str) -> String {
    let lower = s.to_lowercase();
    let stripped = ["all rights reserved", "copyright", "(c)"]
        .iter()
        .fold(lower, |acc, boilerplate| acc.replace(boilerplate, " "));
    stripped.chars().filter(|c| c.is_alphabetic()).collect()
}

/// A claim value cut to `max` characters, with the sample's controls and
/// invisible characters made visible first (see [`sample_text`]).
fn truncate_end(s: &str, max: usize) -> String {
    let s = sample_text(s);
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.into_owned();
    }
    let h: String = chars[..max].iter().collect();
    format!("{h}…")
}
