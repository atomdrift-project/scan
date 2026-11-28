//! Drawing a file's planned windows: source rows for text, printable-byte
//! rows for binaries, each finding announced on a comment line above the row
//! it matched in.

use std::collections::HashSet;
use std::fmt::Write as _;

use cleave::types::{ContextLine, Criticality, FileAnalysis, Note};

use super::select::Planned;
use super::{CONTEXT_ROWS, SOURCE_WIDTH, annotate_desc, display_description};

/// Draw the windows `plan` keeps for `file`.
pub(super) fn render_windows(
    out: &mut String,
    file: &FileAnalysis,
    plan: &[Planned<'_>],
    owed: &HashSet<&str>,
) {
    if is_binary_file_type(&file.file_type) {
        render_ascii_context(out, file, plan);
    } else {
        render_text_chunks(out, file, plan, owed);
    }
}

/// Render binary context: each run of matched bytes as
/// ASCII-forward rows — printable bytes verbatim, non-printables as C-string
/// escapes — preceded by one `{marker} SEV OFFSET desc` line per detection in
/// that run, so the detections describe the bytes that follow. Every row opens
/// with an xxd-style gutter (the absolute byte offset of its first byte in
/// unpadded hex), `:` when a detection matched inside the row and `-` when it is
/// surrounding context, so any byte is addressable without the terminal view's
/// hex column and a clipped run needs no `…` marker.
///
/// A captured window is split into one block per contiguous run its own
/// annotations cover. Capture merges overlapping windows, and low-tier traits
/// this view does not list bridge them: on `/bin/ls` sixteen listed detections
/// arrived as a single 1.5 KB window, so the block stopped describing the bytes
/// it held. Splitting on the gaps restores the property that makes the view
/// readable — every block is contiguous data that its own annotations account
/// for.
fn render_ascii_context(out: &mut String, file: &FileAnalysis, plan: &[Planned<'_>]) {
    let marker = comment_marker(&file.file_type);
    let mut emitted = false;
    for (line, w) in file.context.iter().zip(plan) {
        let Some(ids) = &w.ids else {
            continue;
        };
        // Distinct planned findings on this window. (Placement already capped
        // locations per trait; merging can pool several here.)
        let mut seen = HashSet::new();
        let notes: Vec<&Note> = w
            .notes
            .iter()
            .filter(|n| ids.contains(n.id.as_str()) && seen.insert(n.id.as_str()))
            .collect();
        if notes.is_empty() {
            continue;
        }

        let blocks = ascii_blocks(line, notes);

        for (start, end, mut block_notes) in blocks {
            // A blank line sets each block apart — from the file-level
            // annotations above the first one, and from the preceding block.
            if emitted || (!out.is_empty() && !out.ends_with("\n\n")) {
                out.push('\n');
            }
            emitted = true;

            // Strongest first: the block leads with the worst thing in it.
            block_notes.sort_by(|a, b| b.crit.cmp(&a.crit).then_with(|| a.off.cmp(&b.off)));
            // Each annotation carries the hex offset of the bytes it fired on,
            // in the same base and form as the row gutter below, so a detection
            // can be pointed at its bytes even when a block holds several. The
            // text view has always shown `LINE:COL` here.
            for note in &block_notes {
                let desc = annotate_desc(&display_description(&note.desc), &note.id);
                let _ = writeln!(out, "{marker} {} {:x} {desc}", note.crit.letter(), note.off);
            }

            let mut row = start;
            while row < end {
                let row_end = (row + ASCII_ROW).min(end);
                let row_lo = line.loc + row as u64;
                let row_hi = line.loc + row_end as u64;
                // `:` marks a row a detection actually matched in, `-` a row of
                // surrounding context — the same distinction the terminal
                // view's gutter draws.
                let hit = block_notes
                    .iter()
                    .any(|n| n.off < row_hi && n.off + u64::from(n.len.max(1)) > row_lo);
                let _ = writeln!(
                    out,
                    "{row_lo:x}{} {}",
                    if hit { ':' } else { '-' },
                    ascii_printable(&line.data[row..row_end])
                );
                row = row_end;
            }
        }
    }
}

/// Split a binary window into contiguous blocks around its selected notes:
/// each opens on the row its match starts in and runs one row past the match.
/// Leading rows were machine code and padding a model reads nothing from.
fn ascii_blocks<'a>(
    line: &'a ContextLine,
    mut notes: Vec<&'a Note>,
) -> Vec<(usize, usize, Vec<&'a Note>)> {
    let len = line.data.len();
    let within = |off: u64| {
        usize::try_from(off.saturating_sub(line.loc))
            .unwrap_or(usize::MAX)
            .min(len)
    };
    notes.sort_by_key(|n| n.off);
    let mut blocks: Vec<(usize, usize, Vec<&Note>)> = Vec::new();
    for note in notes {
        let lo = within(note.off);
        let hi = within(note.off.saturating_add(u64::from(note.len.max(1))))
            .saturating_add(ASCII_ROW)
            .min(len);
        match blocks.last_mut() {
            Some((_, end, group)) if lo <= *end => {
                *end = (*end).max(hi);
                group.push(note);
            }
            _ => blocks.push((lo, hi, vec![note])),
        }
    }
    blocks
}

/// Bytes per gutter-prefixed row in the ASCII-forward view. Wider than xxd's 16
/// so the escaped stream stays token-cheap for a model, while every row still
/// carries an addressable offset.
const ASCII_ROW: usize = 64;

/// Minimum `<printable>\0` pairs to treat a run as UTF-16LE ASCII and collapse
/// it to text. PE version-info / Authenticode strings are UTF-16LE and sit at
/// mid-file offsets (no BOM for `cleave::file_io::normalize_text_encoding` to
/// catch), so without this `ProductName` renders as `P<00>r<00>o<00>…`.
const UTF16_RUN: usize = 4;

/// The printable projection of a byte window: printable ASCII verbatim, a
/// UTF-16LE ASCII run collapsed to its text, and every other byte — NUL,
/// control, high-bit, machine code — a single space, runs of them one space.
///
/// What the model reads in a window is the strings: a path, a method name, a
/// key, an import. Machine-code bytes between them are not evidence it can
/// use, but rendered as `\xNN` escapes they cost three to four tokens each,
/// and a 54 MB ELF's 107 windows came to 40k tokens. Measured on
/// 32 benchmark renders graded by a model, five runs each: this
/// projection cut the prompts 21% overall (a binary-heavy render 41.5k →
/// 27.6k tokens) with every grade unchanged, where dropping the context rows
/// or collapsing runs to a `.{n}` marker each flipped a known-bad .NET
/// sample. The string boundaries stay visible (each non-printable run is a
/// space), which is what kept `get_Threshold set_Threshold …` readable as
/// separate names.
pub(super) fn ascii_printable(data: &[u8]) -> String {
    let printable = |b: u8| (0x20..=0x7e).contains(&b);
    let mut s = String::with_capacity(data.len());
    let mut i = 0;
    let mut gap = false;
    while i < data.len() {
        let mut run = 0;
        while i + run * 2 + 1 < data.len()
            && printable(data[i + run * 2])
            && data[i + run * 2 + 1] == 0
        {
            run += 1;
        }
        if run >= UTF16_RUN {
            if gap {
                s.push(' ');
                gap = false;
            }
            for k in 0..run {
                s.push(data[i + k * 2] as char);
            }
            i += run * 2;
            continue;
        }
        let b = data[i];
        if printable(b) {
            if gap {
                s.push(' ');
                gap = false;
            }
            s.push(b as char);
        } else {
            gap = true;
        }
        i += 1;
    }
    if gap {
        s.push(' ');
    }
    s
}

/// A physical line within a textual chunk: its absolute byte range in the file
/// and the 1-based source position of its first content byte. Splitting a
/// byte-addressed chunk into these rows is what lets one renderer serve both a
/// normal multi-line window (numbered lines) and a slice of an ultra-long line
/// (a single pseudo-line labelled `line:col`).
struct ChunkRow {
    /// Absolute byte offset of the row's first content byte.
    off: u64,
    /// 1-based source line.
    line: u64,
    /// 1-based column of the first content byte (>1 for a mid-line slice).
    col: u64,
    /// Byte range `[start, end)` of the row's content in `chunk.data`, excluding
    /// any trailing `\r`/`\n`.
    start: usize,
    end: usize,
    /// Absolute offset where the next row begins — the exclusive upper bound for
    /// assigning a note to this row.
    cover_end: u64,
}

/// Split a textual chunk into physical rows. The first row starts at the chunk's
/// own `line`/`col` (which may be mid-line, when the chunk slices an ultra-long
/// line); each subsequent row begins at column 1 on the next line.
fn chunk_rows(chunk: &ContextLine) -> Vec<ChunkRow> {
    let base = chunk.loc;
    let mut line = chunk.line.unwrap_or(0);
    let mut col = chunk.col.unwrap_or(1);
    let data = &chunk.data;
    let mut rows = Vec::new();
    let mut start = 0usize;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            // Drop a CRLF's `\r` so the row doesn't render a trailing carriage return.
            let end = if i > start && data[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            rows.push(ChunkRow {
                off: base + start as u64,
                line,
                col,
                start,
                end,
                cover_end: base + i as u64 + 1,
            });
            start = i + 1;
            line += 1;
            col = 1;
        }
    }
    rows.push(ChunkRow {
        off: base + start as u64,
        line,
        col,
        start,
        end: data.len(),
        cover_end: base + data.len() as u64,
    });
    rows
}

/// Render textual context: every stored chunk is an already-sized byte window,
/// split into physical rows. A row carrying a finding is preceded by its
/// `{marker} SEV line:col desc` annotation; a chunk that is a slice of an
/// ultra-long line has a single row, clipped to [`SOURCE_WIDTH`] with `...`
/// marking the trimmed sides.
fn render_text_chunks(
    out: &mut String,
    file: &FileAnalysis,
    plan: &[Planned<'_>],
    owed: &HashSet<&str>,
) {
    // Chunks the plan draws, each with the finding ids it annotates.
    let shown: Vec<(&ContextLine, &[Note], &HashSet<&str>)> = file
        .context
        .iter()
        .zip(plan)
        .filter_map(|(c, w)| w.ids.as_ref().map(|ids| (c, w.notes, ids)))
        .collect();
    if shown.is_empty() {
        return;
    }
    let layout = SourceLayout {
        owed,
        marker: comment_marker(&file.file_type),
        content_width: SOURCE_WIDTH,
    };

    // Set the context off from the identity lines above it.
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }

    let mut emitted = false;
    for (chunk, chunk_notes, ids) in shown {
        // A blank line sets each window apart as its own paragraph.
        if emitted && !out.ends_with("\n\n") {
            out.push('\n');
        }
        emitted = true;

        // The window ends mid-line (its last row is a clipped tail) when the chunk
        // stops before a newline and more file follows — then that row earns a
        // trailing `...`.
        let chunk_end = chunk.loc + chunk.data.len() as u64;
        let cut_right = chunk.data.last() != Some(&b'\n') && chunk_end < file.size;
        let rows = chunk_rows(chunk);
        let last = rows.len().saturating_sub(1);
        // Index of the last row actually rendered, so a jump over dropped rows
        // can be marked. A chunk is one contiguous byte range, but the loop
        // below renders only rows that carry a match or real surrounding code —
        // so consecutive output lines are routinely not consecutive source
        // lines. Run together they read as one block of code that never existed.
        let mut last_rendered: Option<usize> = None;
        // Rows accumulate here so a finished block can be trimmed before it is
        // appended — a block's first or last line carrying nothing visible is
        // padding, and against a blank separator it reads as a gap twice as big
        // as the one that is really there.
        let mut block = String::new();
        let row_notes: Vec<Vec<&Note>> = rows
            .iter()
            .map(|row| {
                chunk_notes
                    .iter()
                    .filter(|n| {
                        ids.contains(n.id.as_str()) && n.off >= row.off && n.off < row.cover_end
                    })
                    .collect()
            })
            .collect();
        // Only the rows next to a finding are kept: the grader reads the
        // matched row and its neighbours, and wider context cost bytes
        // without changing a grade. A file's leading comment is the
        // exception — it says what the code is for — and so is a window
        // with no finding, which is that comment alone.
        let hits: Vec<usize> = row_notes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.is_empty())
            .map(|(i, _)| i)
            .collect();
        let leading = leading_comment_rows(chunk, &rows, layout.marker);
        let near_hit = |idx: usize| {
            hits.is_empty() || leading[idx] || hits.iter().any(|&h| idx.abs_diff(h) <= CONTEXT_ROWS)
        };
        for (idx, row) in rows.iter().enumerate() {
            let bytes = &chunk.data[row.start..row.end];
            let notes = &row_notes[idx];
            if !near_hit(idx) {
                continue;
            }
            // A context row (no match of its own) earns a line only when it carries
            // real code — skip near-empty ones like a lone `}` that add height
            // without context. A hit row always renders.
            let real_code = bytes.iter().filter(|b| !b.is_ascii_whitespace()).count() >= 6;
            if notes.is_empty() && !real_code {
                continue;
            }
            // A blank line where rows were dropped: each run of consecutive
            // source lines becomes its own block.
            if last_rendered.is_some_and(|prev| prev + 1 != idx) {
                push_source_block(out, &mut block);
            }
            last_rendered = Some(idx);

            let comment = notes
                .iter()
                .copied()
                .max_by_key(|n| (n.crit, std::cmp::Reverse(n.off)));
            render_source_line(
                &mut block,
                &SourceRow {
                    data: bytes,
                    byte_base: row.off,
                    line_no: row.line,
                    col_base: row.col,
                    cut_right: idx == last && cut_right,
                    notes,
                    comment,
                },
                &layout,
            );
        }
        push_source_block(out, &mut block);
    }
}

/// Which of `chunk`'s rows belong to the file's leading comment: the chunk
/// opens the file, the row is a comment or docstring row, and nothing but
/// such rows, blank rows and a shebang come before it. Comment syntax is the
/// file's `marker` family.
fn leading_comment_rows(chunk: &ContextLine, rows: &[ChunkRow], marker: &str) -> Vec<bool> {
    let mut leading = vec![false; rows.len()];
    if chunk.loc != 0 {
        return leading;
    }
    let contains = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
    let mut open: Option<&[u8]> = None;
    for (i, row) in rows.iter().enumerate() {
        let t = chunk.data[row.start..row.end].trim_ascii();
        if let Some(closer) = open {
            if contains(t, closer) {
                open = None;
            }
        } else if t.is_empty() || (i == 0 && t.starts_with(b"#!")) {
            continue;
        } else if marker == "//" && t.starts_with(b"/*") {
            open = (!contains(t.get(2..).unwrap_or_default(), b"*/")).then_some(b"*/");
        } else if let Some(quote) = [&b"\"\"\""[..], b"'''"]
            .into_iter()
            .find(|q| marker == "#" && t.starts_with(q))
        {
            open = (!contains(t.get(3..).unwrap_or_default(), quote)).then_some(quote);
        } else if !(t.starts_with(marker.as_bytes()) || (marker == "//" && t.starts_with(b"*"))) {
            break;
        }
        leading[i] = true;
    }
    leading
}

/// Append one block of rendered source rows, set off by a blank line, with any
/// leading or trailing rows that carry no visible text dropped.
///
/// A match can land on an empty source line, and a block that opens or closes on
/// one reads — next to the blank line separating blocks — as a gap twice the
/// size of the real one. Trimming is by rendered text, so a terminal row that is
/// blank apart from its line-number gutter is content and stays. A block that is
/// nothing but blank rows appends nothing, so it can never leave a stray
/// separator behind.
fn push_source_block(out: &mut String, block: &mut String) {
    let mut lines: Vec<&str> = block.lines().collect();
    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    if !lines.is_empty() {
        if !out.is_empty() && !out.ends_with("\n\n") {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
        out.push_str(&lines.join("\n"));
        out.push('\n');
    }
    block.clear();
}

/// True for file types whose context renders as a hex+ascii dump.
/// Mirrors `filefacts::FileType::is_binary()` without requiring the enum.
pub(super) fn is_binary_file_type(file_type: &str) -> bool {
    matches!(
        file_type,
        "elf"
            | "pe"
            | "macho"
            | "java_class"
            | "python_bytecode"
            | "beam"
            | "wasm"
            | "dos_com"
            | "shellcode"
    )
}

/// Line-comment marker for the sample's language; `//` for binary and anything
/// C-family, so the annotation reads like a comment the analyst wrote.
pub(super) fn comment_marker(file_type: &str) -> &'static str {
    match file_type {
        "python" | "shell" | "bash" | "ruby" | "perl" | "r" | "powershell" | "yaml" | "toml"
        | "makefile" | "dockerfile" | "systemd_service" | "desktop_entry" | "text" | "ini"
        | "properties" => "#",
        "lua" | "sql" | "haskell" | "applescript" | "vbs" | "vba" => "--",
        _ => "//",
    }
}

/// Pick the byte window `[b0, b1)` of `text` to show within `content_width`
/// display columns, keeping the primary match visible: end-clip (drop the tail,
/// trailing `…`) when the match starts within the width, else begin-clip with a
/// leading `…` so at least 16 chars precede the match. Returns the window plus
/// whether each end is elided. Assumes `text` exceeds `content_width`.
fn source_window(
    text: &str,
    content_width: usize,
    match_byte: Option<usize>,
) -> (usize, usize, bool, bool) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    if n <= content_width {
        return (0, text.len(), false, false);
    }
    let byte_at = |ci: usize| chars.get(ci).map_or(text.len(), |&(bi, _)| bi);
    // The primary match's char index (first char at or after its byte offset).
    let mc = match_byte
        .and_then(|mb| chars.iter().position(|&(bi, _)| bi >= mb))
        .unwrap_or(0);

    // End-clip keeps the match when it starts within the width (1 col for `…`).
    if mc < content_width.saturating_sub(1) {
        return (0, byte_at(content_width - 1), false, true);
    }
    // Begin-clip: show ≥16 chars of lead before the match.
    let w0 = mc.saturating_sub(16);
    let lead = w0 > 0;
    let mut w1 = (w0 + content_width.saturating_sub(usize::from(lead) + 1)).min(n);
    let trail = w1 < n;
    if !trail {
        w1 = (w0 + content_width.saturating_sub(usize::from(lead))).min(n);
    }
    (byte_at(w0), byte_at(w1), lead, trail)
}

/// One source row to render: its bytes, where they sit in the file, and the
/// notes whose match starts on it.
#[derive(Clone, Copy)]
struct SourceRow<'a> {
    data: &'a [u8],
    /// Absolute offset of the row's first byte.
    byte_base: u64,
    /// 1-based source line of the row's first byte.
    line_no: u64,
    /// 1-based source column of the row's first byte.
    col_base: u64,
    /// The source line continues past the captured window.
    cut_right: bool,
    notes: &'a [&'a Note],
    /// The strongest note on the row, which leads its annotations.
    comment: Option<&'a Note>,
}

/// How every row of one file's source context is laid out.
#[derive(Clone, Copy)]
struct SourceLayout<'a> {
    owed: &'a HashSet<&'a str>,
    marker: &'a str,
    content_width: usize,
}

/// Render one physical source row (a slice of a textual chunk): the code
/// verbatim, windowed to a fixed width with `...` marking any trim, preceded —
/// when it carries a finding — by a `{marker} SEV LINE:COL DESC` annotation
/// line. Keeping the code clear of annotation columns lets the model read it
/// as the file's own source; the preceding comment line marks the detection
/// as the analyzer's, not text from the file. COL is the 1-based column of
/// the match, offset by the row's own starting column (a mid-line slice of a
/// long line starts past column 1). Context and component rows carry no
/// annotation and render as bare source.
fn render_source_line(out: &mut String, row: &SourceRow<'_>, layout: &SourceLayout<'_>) {
    let SourceRow {
        data,
        byte_base,
        line_no,
        col_base,
        cut_right,
        notes,
        comment,
    } = *row;
    let SourceLayout {
        owed,
        marker,
        content_width,
    } = *layout;
    let raw = String::from_utf8_lossy(data);
    let loc_str = line_no.to_string();
    // A mid-line row (the chunk sliced into an ultra-long line) lost its left
    // side; a `cut_right` row loses its right — mark each with a `...` even
    // when the content itself fits the width.
    let cut_left = col_base > 1;
    let (display, map) = expand_tabs(&raw);
    let to_disp = |off: u64| -> Option<usize> {
        map.get(usize::try_from(off.checked_sub(byte_base)?).ok()?)
            .copied()
    };
    let match_disp = comment.and_then(|n| to_disp(n.off));

    // The strongest note leads; every other owed note on the row — a
    // suspicious/hostile atom, or the leg of a withheld composite (see
    // `evidence_legs`), whatever its grade — follows on its own line. Keeping
    // only the strongest hid the rest of the row's evidence from the reader.
    let owes = |n: &Note| owed.contains(n.id.as_str());
    let mut announced: Vec<&Note> = comment
        .filter(|&n| n.crit >= Criticality::Notable || owes(n))
        .into_iter()
        .collect();
    for &n in notes {
        if owes(n) && announced.iter().all(|a| a.id != n.id) {
            announced.push(n);
        }
    }
    let announced_none = announced.is_empty();
    for n in announced {
        out.push_str(marker);
        out.push(' ');
        out.push(n.crit.letter());
        out.push(' ');
        out.push_str(&loc_str);
        if let Some(col) = to_disp(n.off)
            .map(|b| col_base + display.get(..b).map_or(0, |s| s.chars().count() as u64))
        {
            out.push(':');
            out.push_str(&col.to_string());
        }
        out.push(' ');
        out.push_str(&annotate_desc(&display_description(&n.desc), &n.id));
        out.push('\n');
    }

    let (b0, b1, win_lead, win_trail) = source_window(&display, content_width, match_disp);
    let (lead, trail) = (win_lead || cut_left, win_trail || cut_right);
    // A bare context row that starts mid-line (a slice of an ultra-long line)
    // is a fragment with neither a finding nor the line's start; dropping
    // these cut the LLM prompt ~12% with every fixture grade unchanged.
    if lead && announced_none {
        return;
    }
    let slice = display.get(b0..b1).unwrap_or("");
    if lead {
        out.push_str("...");
    }
    out.push_str(slice);
    if trail {
        out.push_str("...");
    }
    out.push('\n');
}

/// Spaces a tab expands to in a source row.
const TAB_WIDTH: usize = 2;

/// Expand tabs to a fixed display width and escape control characters,
/// returning the display form plus a map from every input byte offset to the
/// corresponding display byte offset.
///
/// Control characters (except the tab handled above) are escaped as
/// `\u{XX}` — sample-derived source lines can carry terminal escape
/// sequences, and this is the single funnel both the colorized and plain
/// renderers draw their text from. The map points every byte of an escaped
/// character at the start of its escape sequence, so a match highlight
/// whose offset lands on one covers the whole escape.
pub(super) fn expand_tabs(line: &str) -> (String, Vec<usize>) {
    let mut display = String::with_capacity(line.len());
    let mut map = vec![0usize; line.len() + 1];
    let mut opos = 0usize;
    for ch in line.chars() {
        let clen = ch.len_utf8();
        for slot in map.iter_mut().skip(opos).take(clen) {
            *slot = display.len();
        }
        opos += clen;
        if ch == '\t' {
            for _ in 0..TAB_WIDTH {
                display.push(' ');
            }
        } else if ch.is_control() {
            // Infallible on String; see escape_control_chars.
            let _ = write!(display, "\\u{{{:x}}}", u32::from(ch));
        } else {
            display.push(ch);
        }
    }
    map[line.len()] = display.len();
    (display, map)
}
