//! Preparing a rendered report for the prompt: sanitized, readable, and
//! free of text addressed to the analyzer.

/// Prepare a cleave tiny render for the LLM by stripping any ANSI escapes (tiny
/// must never carry color) and normalizing the archive delimiter. Path hygiene —
/// basenaming the root and showing archive members archive-relative, so a corpus
/// directory can't leak a ground-truth label — is done by cleave's `tiny()` view
/// (`basename_root`), so we must not re-strip here: that would mangle a
/// `a.zip/member` header into a bare `member` when the container section is
/// omitted.
///
/// cleave's `tiny()` rewrites `!!` to `/` only in the file *header* path; virtual
/// paths on finding/evidence lines (`doc.pdf!!pdf/object5.js`, `!!vba/Module1`,
/// embedded-extraction sub-views) still carry the raw `!!`. We collapse those too
/// so the model sees one consistent path syntax — a normal `/` separator it reads
/// reliably — rather than a `!!`/`/` mix it stumbles over.
#[must_use]
pub fn sanitize_context(rendered: &str) -> String {
    let mut out = String::with_capacity(rendered.len());
    for raw in rendered.lines() {
        out.push_str(&strip_ansi(raw).replace("!!", "/"));
        out.push('\n');
    }
    out
}

/// Heuristic for a rendered context line that is escaped binary rather than
/// readable source: cleave renders non-printable bytes as `\xNN`/`\0`, so two or
/// more such escapes is decisive; otherwise fall back to a low printable ratio.
pub(super) fn is_binary_render(line: &str) -> bool {
    let s = line.trim();
    if s.len() < 16 {
        return false;
    }
    if s.matches("\\x").count() + s.matches("\\0").count() >= 2 {
        return true;
    }
    let printable = s
        .chars()
        .filter(|&c| (' '..='~').contains(&c) || c == '\t')
        .count();
    printable * 100 < s.chars().count() * 75
}

/// Whether the render is mostly readable source rather than escaped binary bytes
/// — fewer than half of its context lines (non-annotation, non-blank) render as
/// escaped bytes. A render with no context lines counts as readable (there is
/// nothing opaque to distrust). Gates the blend's content-aware safety valve.
pub(super) fn render_mostly_readable(rendered: &str) -> bool {
    // Judged per member, not over the whole render.
    //
    // Counting lines across the render lets one member outvote every other: a
    // wheel of readable Python that also ships a DLL renders as thousands of hex
    // rows beside a few hundred source lines, so the whole archive reads as
    // opaque and the LLM's clear is discarded. `PyAutoIt`, a legitimate AutoIt
    // wrapper the model correctly clears, is that case: a whole-render count
    // would refuse it. A package is opaque when *most of its members* are,
    // which is the question this gate was always asking.
    //
    // Note what it does not claim: that the LLM read the flagged bytes. A clear
    // can rest on recognizing the package — an AutoIt wrapper shipping AutoIt
    // DLLs — and that is a judgment about a mostly-readable archive, not about
    // hex it cannot see. A sample that really is one packed binary still has a
    // single opaque member and is still refused.
    let mut readable_members = 0usize;
    let mut total_members = 0usize;
    let mut binary = 0usize;
    let mut lines = 0usize;
    let close = |binary: usize, lines: usize, readable: &mut usize, total: &mut usize| {
        if lines == 0 {
            return;
        }
        *total += 1;
        if binary * 2 < lines {
            *readable += 1;
        }
    };
    for line in rendered.lines() {
        if is_member_header(line) {
            close(binary, lines, &mut readable_members, &mut total_members);
            binary = 0;
            lines = 0;
            continue;
        }
        if line.trim().is_empty() || crate::engine::split_annotation(line).is_some() {
            continue;
        }
        lines += 1;
        if is_binary_render(line) {
            binary += 1;
        }
    }
    close(binary, lines, &mut readable_members, &mut total_members);
    total_members == 0 || readable_members * 2 >= total_members
}

/// A member header line: `path\ttype size count`, the row cleave draws above each
/// file's context. Matched on the shape of the second field — a bare type word
/// followed by a size like `19KB` — because a source line may well contain a tab
/// but will not have that after one.
pub(super) fn is_member_header(line: &str) -> bool {
    let Some((_, rest)) = line.trim_end().split_once('\t') else {
        return false;
    };
    let mut fields = rest.split_whitespace();
    let Some(kind) = fields.next() else {
        return false;
    };
    if kind.is_empty()
        || !kind
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
    {
        return false;
    }
    fields.next().is_some_and(|size| {
        size.starts_with(|c: char| c.is_ascii_digit())
            && size.ends_with('B')
            && size.chars().all(|c| c.is_ascii_alphanumeric() || c == '.')
    })
}

/// Text that addresses the grader instead of describing the program. Matched
/// case-insensitively as substrings of the sanitized render.
///
/// Kept short and specific rather than broad: each entry is either a stock
/// instruction-override opener, a chat-template control token, a verbatim
/// fragment of our own prompt or reply schema (a sample echoing those is
/// mirroring the harness, not doing its job), or a direct assertion of
/// innocence. Deliberately *not* included are soft phrases like "not malicious"
/// or "safe to run", which appear in ordinary documentation and comments.
///
/// Note the asymmetry that makes a loose match cheap: a hit only revokes the
/// LLM's ability to *lower* a verdict. A false positive costs one un-cleared ML
/// false positive — a human triage minute — while a miss costs a cleared
/// malware sample.
pub(super) const ANALYZER_DIRECTED: &[&str] = &[
    "ignore previous instruction",
    "ignore all previous",
    "ignore the above",
    "disregard previous",
    "disregard all previous",
    "disregard the above",
    "prior instructions",
    "system prompt",
    "you are a helpful",
    "as an ai language model",
    "<|im_start|>",
    "<|im_end|>",
    "<system>",
    "</system>",
    // Fragments of this module's own prompt and reply schema.
    "benign|suspicious|hostile",
    "\"grade\":\"benign\"",
    "\"grade\": \"benign\"",
    // Direct assertions of innocence, in the forms that read as addressed to a
    // reader rather than as prose about a subject.
    "is not malware",
    "is not a virus",
    "not malware.",
];

/// Whether the render carries text aimed at the grader rather than at a human
/// reading the program — the prompt-injection tell. Benign software does not
/// instruct its analyzer, so a hit is a reason to distrust an LLM verdict that
/// happens to favor the author (see [`blend`]).
///
/// Deliberately a plain substring scan over the sanitized render: deterministic,
/// dependency-free, and independent of the model's cooperation, which is exactly
/// what a prompt-level defense cannot offer. It does not attempt to see through
/// obfuscation — an escaped or encoded payload the scanner misses is also one the
/// model is unlikely to read as an instruction.
pub(super) fn addresses_the_analyzer(rendered: &str) -> bool {
    let haystack = rendered.to_ascii_lowercase();
    ANALYZER_DIRECTED.iter().any(|p| haystack.contains(p))
}

/// Remove ANSI CSI escape sequences (`ESC [ … <final>`).
pub(super) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        // Skip a CSI sequence: ESC '[' params... final byte in '@'..='~'.
        if chars.clone().next() == Some('[') {
            chars.next();
            for n in chars.by_ref() {
                if ('@'..='~').contains(&n) {
                    break;
                }
            }
        }
    }
    out
}
