//! Locally-synced bloom filters for known-good / known-bad lookups.
//!
//! A scan can consult two filters before doing expensive work:
//!
//! - **known-good** — a hit means "we have already blessed this exact key; the
//!   full scan can be skipped". Bloom filters have no false negatives, so a
//!   *miss* is authoritative ("never seen → scan it"); only a *hit* is
//!   probabilistic. The false-positive rate is therefore the rate at which an
//!   unknown key is wrongly skipped, so it is sized conservatively.
//! - **known-bad** — a hit means "definitely worth a full scan". This vetoes a
//!   known-good skip: `good.contains(k) && !bad.contains(k)` is the skip test.
//!   It doubles as a fast revocation channel — a small filter that can ship more
//!   often than the large good filter and immediately overrides a stale bless.
//!
//! The on-disk form is a self-describing header followed by a raw bit array.
//! [`FORMAT_VERSION`] is checked on load and **fails closed** on a mismatch: an
//! unrecognized layout is treated as "no filter", never reinterpreted. Content
//! freshness (which day's data) is tracked out of band by the distribution
//! manifest, orthogonal to this layout version.
//!
//! Hashing is uniform across key kinds: a key is reduced to a 32-byte SHA-256
//! digest and `k` bit indices are derived from it by double hashing
//! (Kirsch–Mitzenmacher). A SHA-256 *is already* a uniform digest, so a
//! [`Kind::Sha256`] key is sliced directly with no rehash. SHA-256 is identical
//! across languages, so a producer in another language agrees bit-for-bit as
//! long as it canonicalizes keys the same way (see [`canonical_purl`]).

use std::collections::HashSet;
use std::fmt;

use sha2::{Digest, Sha256};

/// On-disk layout version. Bump on any change to the header or bit-derivation
/// scheme; readers reject a filter whose version they do not understand.
pub const FORMAT_VERSION: u16 = 1;

/// Magic bytes identifying an Atomdrift bloom file (`"ADBL"`).
const MAGIC: [u8; 4] = *b"ADBL";

/// Fixed header length in bytes. Layout (all little-endian):
/// `magic[4] version:u16 kind:u8 tier:u8 k:u32 m_bits:u64 n:u64 seed:u64`.
const HEADER_LEN: usize = 4 + 2 + 1 + 1 + 4 + 8 + 8 + 8;

/// The category of key a filter holds. Stored in the header so a reader can
/// reject a filter loaded into the wrong slot (fail closed, never guess).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Canonical package URLs (`pkg:npm/left-pad@1.3.0`).
    Purl,
    /// Raw fetch URLs.
    Url,
    /// 32-byte SHA-256 artifact digests (sliced directly, no rehash).
    Sha256,
}

/// The trust tier a filter encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Affirmatively blessed; a hit is skip-eligible.
    Good,
    /// Catalogued bad; a hit forces a full scan and vetoes a good hit.
    Bad,
    /// Provisionally clean (the scanner found nothing); a hint, not a skip.
    UnknownClean,
}

impl Kind {
    const fn to_u8(self) -> u8 {
        match self {
            Self::Purl => 1,
            Self::Url => 2,
            Self::Sha256 => 3,
        }
    }

    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Purl),
            2 => Some(Self::Url),
            3 => Some(Self::Sha256),
            _ => None,
        }
    }

    /// Lowercase slug used in artifact file names.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Purl => "purl",
            Self::Url => "url",
            Self::Sha256 => "sha256",
        }
    }
}

impl Tier {
    const fn to_u8(self) -> u8 {
        match self {
            Self::Good => 1,
            Self::Bad => 2,
            Self::UnknownClean => 3,
        }
    }

    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Good),
            2 => Some(Self::Bad),
            3 => Some(Self::UnknownClean),
            _ => None,
        }
    }

    /// Lowercase slug used in artifact file names.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Bad => "bad",
            Self::UnknownClean => "unknown-clean",
        }
    }
}

/// Why a serialized filter could not be loaded. Every variant maps to the same
/// caller response: ignore the filter and fall back to a full scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// Buffer shorter than a complete header (or its declared bit array).
    Truncated,
    /// Leading magic bytes are not `"ADBL"`.
    BadMagic,
    /// Layout version differs from [`FORMAT_VERSION`].
    UnsupportedVersion(u16),
    /// Header `kind`/`tier`/`k`/`m_bits` field is structurally invalid.
    Corrupt(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("bloom filter truncated"),
            Self::BadMagic => f.write_str("bloom filter magic mismatch"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported bloom format version {v} (expected {FORMAT_VERSION})"
                )
            }
            Self::Corrupt(what) => write!(f, "corrupt bloom header: {what}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// An immutable, queryable bloom filter.
#[derive(Debug, Clone)]
pub struct Filter {
    kind: Kind,
    tier: Tier,
    k: u32,
    /// Number of bits; always a power of two so indexing is a mask, not a `%`.
    m_bits: u64,
    /// Element count at build time (informational; not used for queries).
    n: u64,
    /// Placement seed, XORed into the first hash lane. Lets the producer
    /// decorrelate two filters without a format change.
    seed: u64,
    bits: Vec<u8>,
}

impl Filter {
    /// The key category this filter holds.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// The trust tier this filter encodes.
    #[must_use]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    /// Number of elements inserted at build time.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.n
    }

    /// Canonical artifact stem, e.g. `purl-good` or `sha256-bad`. Producer and
    /// scanner agree on filter file names through this single function.
    #[must_use]
    pub fn artifact_stem(&self) -> String {
        format!("{}-{}", self.kind.as_str(), self.tier.as_str())
    }

    /// True when no elements were inserted.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Test a variable-length key (PURL, URL) for possible membership.
    ///
    /// `false` is authoritative (definitely absent); `true` is probabilistic.
    #[must_use]
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.contains_digest(&sha256(key))
    }

    /// Test a 32-byte SHA-256 digest directly, skipping the rehash.
    #[must_use]
    pub fn contains_digest(&self, digest: &[u8; 32]) -> bool {
        let mut hit = true;
        each_index(self.k, self.m_bits, self.seed, digest, |idx| {
            hit = hit && self.bit(idx);
        });
        hit
    }

    #[inline]
    fn bit(&self, idx: usize) -> bool {
        let byte = idx >> 3;
        // byte < m_bits/8 == bits.len() by construction; guard rather than trust.
        self.bits
            .get(byte)
            .is_some_and(|b| b & (1u8 << (idx & 7)) != 0)
    }

    /// Serialize to the on-disk form: [`HEADER_LEN`]-byte header + bit array.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.bits.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.push(self.kind.to_u8());
        out.push(self.tier.to_u8());
        out.extend_from_slice(&self.k.to_le_bytes());
        out.extend_from_slice(&self.m_bits.to_le_bytes());
        out.extend_from_slice(&self.n.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.bits);
        out
    }

    /// Parse a filter from its on-disk form. Fails closed on any mismatch.
    ///
    /// # Errors
    /// Returns [`LoadError`] if the buffer is truncated, the magic or version
    /// does not match, or a header field is structurally invalid.
    pub fn load(buf: &[u8]) -> Result<Self, LoadError> {
        if buf.len() < HEADER_LEN {
            return Err(LoadError::Truncated);
        }
        if buf.get(0..4) != Some(&MAGIC) {
            return Err(LoadError::BadMagic);
        }
        let version = rd_u16(buf, 4);
        if version != FORMAT_VERSION {
            return Err(LoadError::UnsupportedVersion(version));
        }
        let kind = Kind::from_u8(buf[6]).ok_or(LoadError::Corrupt("kind"))?;
        let tier = Tier::from_u8(buf[7]).ok_or(LoadError::Corrupt("tier"))?;
        let k = rd_u32(buf, 8);
        let m_bits = rd_u64(buf, 12);
        let n = rd_u64(buf, 20);
        let seed = rd_u64(buf, 28);

        if k == 0 || k > MAX_K {
            return Err(LoadError::Corrupt("k out of range"));
        }
        if m_bits == 0 || !m_bits.is_power_of_two() {
            return Err(LoadError::Corrupt("m_bits not a power of two"));
        }
        let want_bytes = (m_bits / 8) as usize;
        let bits = buf.get(HEADER_LEN..).unwrap_or_default();
        if bits.len() != want_bytes {
            return Err(LoadError::Truncated);
        }
        Ok(Self {
            kind,
            tier,
            k,
            m_bits,
            n,
            seed,
            bits: bits.to_vec(),
        })
    }
}

/// Upper bound on hash functions. Keeps `k` sane and bounds the build loop.
const MAX_K: u32 = 64;

/// Accumulates keys and produces a [`Filter`] sized for a target false-positive
/// rate. Build is a full rebuild from source each cycle — bloom filters cannot
/// delete, so revocation is "rebuild without the key", never an in-place edit.
#[derive(Debug)]
pub struct Builder {
    kind: Kind,
    tier: Tier,
    k: u32,
    m_bits: u64,
    seed: u64,
    n: u64,
    bits: Vec<u8>,
}

impl Builder {
    /// Size a builder for `expected` elements at `target_fp` false-positive
    /// rate. `seed` decorrelates placement across filters (use `0` by default).
    ///
    /// `target_fp` is clamped to `(0, 1)`; out-of-range values fall back to a
    /// sane `1e-9`. `expected` is floored at 1 so the filter is never zero-bit.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    // Build-time sizing: inputs are small positive counts/probabilities and the
    // result is bounded by `next_power_of_two`, so the f64↔u64 conversions here
    // cannot truncate or lose sign in practice.
    pub fn sized_for(kind: Kind, tier: Tier, expected: u64, target_fp: f64, seed: u64) -> Self {
        let n = expected.max(1);
        let p = if target_fp.is_finite() && target_fp > 0.0 && target_fp < 1.0 {
            target_fp
        } else {
            1e-9
        };
        const LN2: f64 = std::f64::consts::LN_2;
        // m = -n·ln p / (ln 2)², rounded up to a power of two for mask indexing.
        let raw_bits = -(n as f64) * p.ln() / (LN2 * LN2);
        let m_bits = (raw_bits.ceil() as u64).max(64).next_power_of_two();
        // k = (m/n)·ln 2, the count minimizing the false-positive rate.
        let k = (((m_bits as f64) / (n as f64) * LN2).round() as u32).clamp(1, MAX_K);
        Self {
            kind,
            tier,
            k,
            m_bits,
            seed,
            n: 0,
            bits: vec![0u8; (m_bits / 8) as usize],
        }
    }

    /// Insert a variable-length key (PURL, URL).
    pub fn insert_key(&mut self, key: &[u8]) {
        self.insert_digest(&sha256(key));
    }

    /// Insert a 32-byte SHA-256 digest directly.
    pub fn insert_digest(&mut self, digest: &[u8; 32]) {
        each_index(self.k, self.m_bits, self.seed, digest, |idx| {
            if let Some(byte) = self.bits.get_mut(idx >> 3) {
                *byte |= 1u8 << (idx & 7);
            }
        });
        self.n += 1;
    }

    /// Finalize into an immutable [`Filter`].
    #[must_use]
    pub fn build(self) -> Filter {
        Filter {
            kind: self.kind,
            tier: self.tier,
            k: self.k,
            m_bits: self.m_bits,
            n: self.n,
            seed: self.seed,
            bits: self.bits,
        }
    }
}

/// Derive the `k` bit indices for a digest and pass each to `f`.
///
/// Double hashing: `idx_i = (h1 + i·h2) mod m`, with `h1`/`h2` taken from the
/// first two 64-bit lanes of the digest. `h2` is forced odd so it is coprime
/// with the power-of-two modulus and the indices cycle through all residues.
#[inline]
#[allow(clippy::cast_possible_truncation)]
// `idx < m_bits`, which is built to fit `usize` on the 64-bit targets we ship.
fn each_index(k: u32, m_bits: u64, seed: u64, digest: &[u8; 32], mut f: impl FnMut(usize)) {
    let mask = m_bits - 1;
    let h2 = lane(digest, 1) | 1;
    let mut h = lane(digest, 0) ^ seed;
    for _ in 0..k {
        f((h & mask) as usize);
        h = h.wrapping_add(h2);
    }
}

/// Read 64-bit lane `i` (`i < 4`) from a digest, little-endian.
#[inline]
const fn lane(d: &[u8; 32], i: usize) -> u64 {
    let s = i * 8;
    u64::from_le_bytes([
        d[s],
        d[s + 1],
        d[s + 2],
        d[s + 3],
        d[s + 4],
        d[s + 5],
        d[s + 6],
        d[s + 7],
    ])
}

fn rd_u16(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2)
        .map_or(0, |s| u16::from_le_bytes([s[0], s[1]]))
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u64(b: &[u8], at: usize) -> u64 {
    b.get(at..at + 8).map_or(0, |s| {
        u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
    })
}

/// SHA-256 a variable-length key into the 32-byte digest bit indices derive from.
fn sha256(key: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(key));
    out
}

/// Normalize a PURL to the canonical string used as a filter key.
///
/// This is the shared contract between the producer (which builds the filter)
/// and the scanner (which queries it): both **must** key on this exact form or
/// lookups silently miss. The `pkg` scheme and package *type* are
/// case-insensitive per the PURL spec, so they are lowercased; the remainder is
/// left untouched, since case significance is type-specific.
///
/// It also folds the non-spec PURL spellings this project emitted before onto the
/// spec/common-practice form the producer now generates, so an old and a new
/// spelling of the same package compare equal: `pkg:chrome`→`chrome-extension`,
/// `pkg:vscode`/`pkg:openvsx`→`vscode-extension` (Open VSX keeping its
/// `repository_url` qualifier), and the bare distro types
/// `pkg:debian`/`arch`/`fedora`/… → `deb`/`rpm`/`apk`/`alpm` with the distro as
/// namespace. This mirrors `hopper`'s `pkgparse.CanonicalizePURL`; the two must
/// stay in lockstep. The extension types are case-insensitive per spec, so their
/// bodies are lowercased too; every other type keeps its body case.
#[must_use]
pub fn canonical_purl(raw: &str) -> String {
    let s = raw.trim();
    let Some((scheme_type, rest)) = s.split_once('/') else {
        return s.to_ascii_lowercase();
    };
    // The `pkg` scheme and type are case-insensitive.
    let typ = scheme_type
        .to_ascii_lowercase()
        .strip_prefix("pkg:")
        .unwrap_or_default()
        .to_string();
    // Split the remainder into the coordinate path and the @version/?qualifier
    // tail so the type can be re-keyed without disturbing either.
    let (path, tail) = match rest.find(['@', '?']) {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };

    match typ.as_str() {
        // Browser / editor extensions: case-insensitive bodies, ratified types.
        "chrome" | "chrome-extension" => {
            format!(
                "pkg:chrome-extension/{}{tail}",
                last_purl_segment(path).to_ascii_lowercase()
            )
        }
        "vscode" | "vscode-extension" => {
            format!("pkg:vscode-extension/{}{tail}", path.to_ascii_lowercase())
        }
        "openvsx" => format!(
            "pkg:vscode-extension/{}{}",
            path.to_ascii_lowercase(),
            add_purl_qualifier(tail, "repository_url=https://open-vsx.org")
        ),
        other => {
            if let Some((spec, ns)) = distro_purl_spec(other) {
                format!("pkg:{spec}/{ns}/{}{tail}", last_purl_segment(path))
            } else {
                // Language/registry and unrecognized types: canonical type, body
                // case preserved (significance is type-specific).
                format!("pkg:{typ}/{rest}")
            }
        }
    }
}

/// The final `/`-separated segment (the app-store and distro types drop any
/// vendor path, matching `fletch`'s resolver and `hopper`'s builder).
fn last_purl_segment(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Map a legacy bare-distro PURL type onto the spec type and namespace.
fn distro_purl_spec(typ: &str) -> Option<(&'static str, &'static str)> {
    Some(match typ {
        "debian" => ("deb", "debian"),
        "ubuntu" => ("deb", "ubuntu"),
        "fedora" => ("rpm", "fedora"),
        "opensuse" => ("rpm", "opensuse"),
        "rpmfusion" => ("rpm", "rpmfusion"),
        "arch" => ("alpm", "arch"),
        "aur" => ("alpm", "aur"),
        "alpine" => ("apk", "alpine"),
        "wolfi" => ("apk", "wolfi"),
        _ => return None,
    })
}

/// Merge one qualifier into a PURL `@version`/`?qualifiers` tail, leaving an
/// already-present key untouched.
fn add_purl_qualifier(tail: &str, qualifier: &str) -> String {
    let key = qualifier.split('=').next().unwrap_or(qualifier);
    match tail.split_once('?') {
        None => format!("{tail}?{qualifier}"),
        Some((ver, quals)) => {
            if quals.split('&').any(|q| {
                q.split('=')
                    .next()
                    .is_some_and(|k| k.eq_ignore_ascii_case(key))
            }) {
                tail.to_string()
            } else {
                format!("{ver}?{quals}&{qualifier}")
            }
        }
    }
}

/// One artifact drawn from the good/bad pool. Either field may be absent: a
/// fetched package carries both its PURL and tarball digest; a bare file upload
/// carries only a digest. The pool adapter applies trust policy (what counts as
/// good vs. bad) *before* building records — this layer only does set algebra.
#[derive(Debug, Clone, Default)]
pub struct Record {
    /// Canonical or raw PURL; normalized via [`canonical_purl`] on ingest.
    pub purl: Option<String>,
    /// 32-byte artifact digest.
    pub sha256: Option<[u8; 32]>,
}

/// The trust tier a pool record carries into [`KeySets::insert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    /// Skip-eligible (`good` filters).
    Good,
    /// Catalogued bad (`bad` filters; also subtracted from good).
    Bad,
}

/// Streaming accumulator for a pool: keys are inserted one record at a time and
/// the four filters are produced once at the end. This is the memory floor for
/// an exact build — only the deduplicated key *sets* are held, never the records
/// themselves — so a 25M-row pool costs the SHA-256 set (~1 GB), not a second
/// copy of every record. Feed it from any source (NDJSON, a DB row stream).
///
/// PURLs and SHA-256s differ only in how a key becomes a digest; both kinds flow
/// through the same insert path.
#[derive(Debug, Default)]
pub struct KeySets {
    good_purl: HashSet<String>,
    bad_purl: HashSet<String>,
    good_sha: HashSet<[u8; 32]>,
    bad_sha: HashSet<[u8; 32]>,
}

impl KeySets {
    /// Insert one record under `label`. A record may carry a PURL, a digest, or
    /// both; each present key joins its set. Duplicates collapse (it's a set).
    pub fn insert(&mut self, label: Label, record: Record) {
        let (purls, shas) = match label {
            Label::Good => (&mut self.good_purl, &mut self.good_sha),
            Label::Bad => (&mut self.bad_purl, &mut self.bad_sha),
        };
        if let Some(p) = record.purl {
            purls.insert(canonical_purl(&p));
        }
        if let Some(h) = record.sha256 {
            shas.insert(h);
        }
    }

    /// Deduplicated `(purl, sha256)` good-key counts (before subtraction).
    #[must_use]
    pub fn good_counts(&self) -> (usize, usize) {
        (self.good_purl.len(), self.good_sha.len())
    }

    /// Deduplicated `(purl, sha256)` bad-key counts.
    #[must_use]
    pub fn bad_counts(&self) -> (usize, usize) {
        (self.bad_purl.len(), self.bad_sha.len())
    }

    /// Apply `good − bad` and build one self-describing [`Filter`] per
    /// `(kind, tier)`. Subtracting bad from good drops a later-revoked sample
    /// from the skip-eligible set on every rebuild. The caller writes each
    /// filter to `<stem>.adbl` (see [`Filter::artifact_stem`]).
    #[must_use]
    pub fn into_filters(mut self, target_fp: f64) -> Vec<Filter> {
        for k in &self.bad_purl {
            self.good_purl.remove(k);
        }
        for k in &self.bad_sha {
            self.good_sha.remove(k);
        }
        vec![
            from_keys(Kind::Purl, Tier::Good, &self.good_purl, target_fp),
            from_keys(Kind::Purl, Tier::Bad, &self.bad_purl, target_fp),
            from_digests(Kind::Sha256, Tier::Good, &self.good_sha, target_fp),
            from_digests(Kind::Sha256, Tier::Bad, &self.bad_sha, target_fp),
        ]
    }
}

/// Build the good/bad filters from in-memory good/bad records.
///
/// A convenience over [`KeySets`] for callers that already hold the records;
/// large or streaming producers should feed [`KeySets::insert`] directly to
/// avoid materializing the records. Returns one [`Filter`] per `(kind, tier)`.
#[must_use]
pub fn generate(
    good: impl IntoIterator<Item = Record>,
    bad: impl IntoIterator<Item = Record>,
    target_fp: f64,
) -> Vec<Filter> {
    let mut sets = KeySets::default();
    for r in good {
        sets.insert(Label::Good, r);
    }
    for r in bad {
        sets.insert(Label::Bad, r);
    }
    sets.into_filters(target_fp)
}

/// Build a string-keyed filter (PURL, URL) from a set of canonical keys.
fn from_keys(kind: Kind, tier: Tier, keys: &std::collections::HashSet<String>, fp: f64) -> Filter {
    let mut b = Builder::sized_for(kind, tier, keys.len() as u64, fp, 0);
    for k in keys {
        b.insert_key(k.as_bytes());
    }
    b.build()
}

/// Build a digest-keyed filter (SHA-256) directly from 32-byte digests.
fn from_digests(
    kind: Kind,
    tier: Tier,
    keys: &std::collections::HashSet<[u8; 32]>,
    fp: f64,
) -> Filter {
    let mut b = Builder::sized_for(kind, tier, keys.len() as u64, fp, 0);
    for k in keys {
        b.insert_digest(k);
    }
    b.build()
}

/// Parse a 64-character hex SHA-256 into its 32 bytes. Returns `None` for any
/// wrong length or non-hex character, so a malformed pool row is skipped, never
/// silently hashed as text.
#[must_use]
pub fn parse_sha256_hex(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (hex_nibble(bytes[i * 2])? << 4) | hex_nibble(bytes[i * 2 + 1])?;
    }
    Some(out)
}

const fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn good_purl_filter(purls: &[&str], fp: f64) -> Filter {
        let mut b = Builder::sized_for(Kind::Purl, Tier::Good, purls.len() as u64, fp, 0);
        for p in purls {
            b.insert_key(canonical_purl(p).as_bytes());
        }
        b.build()
    }

    #[test]
    fn no_false_negatives() {
        let purls = [
            "pkg:npm/left-pad@1.3.0",
            "pkg:pypi/requests@2.31.0",
            "pkg:gem/rails@7.0.0",
        ];
        let f = good_purl_filter(&purls, 1e-6);
        for p in purls {
            assert!(
                f.contains_key(canonical_purl(p).as_bytes()),
                "{p} must be present"
            );
        }
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn canonicalization_lowercases_scheme_and_type_only() {
        assert_eq!(
            canonical_purl("  PKG:NPM/Left-Pad@1.3.0 "),
            "pkg:npm/Left-Pad@1.3.0"
        );
        assert_eq!(canonical_purl("pkg:PyPI/Requests"), "pkg:pypi/Requests");
    }

    #[test]
    fn canonicalization_folds_legacy_spellings_onto_spec() {
        // Legacy fletch spellings fold onto the spec/common-practice form, so a
        // stored spec PURL and a scanned legacy PURL hit the same filter key.
        let pairs = [
            ("pkg:chrome/KhKimila", "pkg:chrome-extension/khkimila"),
            (
                "pkg:chrome/KhKimila@25.7.1",
                "pkg:chrome-extension/khkimila@25.7.1",
            ),
            (
                "pkg:vscode/Saoudrizwan/Claude-Dev",
                "pkg:vscode-extension/saoudrizwan/claude-dev",
            ),
            (
                "pkg:openvsx/jinryx/crontally@1.0.3",
                "pkg:vscode-extension/jinryx/crontally@1.0.3?repository_url=https://open-vsx.org",
            ),
            ("pkg:debian/curl", "pkg:deb/debian/curl"),
            ("pkg:arch/pacman@6.0", "pkg:alpm/arch/pacman@6.0"),
            ("pkg:fedora/curl", "pkg:rpm/fedora/curl"),
            ("pkg:alpine/musl", "pkg:apk/alpine/musl"),
        ];
        for (legacy, spec) in pairs {
            assert_eq!(canonical_purl(legacy), spec, "fold {legacy}");
            // Idempotent: the spec form canonicalizes to itself.
            assert_eq!(canonical_purl(spec), spec, "idempotent {spec}");
        }
    }

    #[test]
    fn false_positive_rate_is_bounded() {
        // 2000 present keys at p=1e-3; 20000 absent probes should yield well
        // under 1% hits (loose bound — deterministic keys, just guards sanity).
        let present: Vec<String> = (0..2000)
            .map(|i| format!("pkg:npm/pkg-{i}@1.0.0"))
            .collect();
        let mut b = Builder::sized_for(Kind::Purl, Tier::Good, present.len() as u64, 1e-3, 0);
        for p in &present {
            b.insert_key(p.as_bytes());
        }
        let f = b.build();
        let fp = (0..20_000)
            .filter(|i| f.contains_key(format!("pkg:npm/absent-{i}@9.9.9").as_bytes()))
            .count();
        assert!(fp < 200, "false positives {fp} exceed loose 1% bound");
    }

    #[test]
    fn round_trip_serialization() {
        let f = good_purl_filter(&["pkg:npm/a@1", "pkg:npm/b@2"], 1e-6);
        let loaded = Filter::load(&f.to_bytes()).expect("round trip");
        assert_eq!(loaded.kind(), Kind::Purl);
        assert_eq!(loaded.tier(), Tier::Good);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key(b"pkg:npm/a@1"));
        assert!(loaded.contains_key(b"pkg:npm/b@2"));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut bytes = good_purl_filter(&["pkg:npm/a@1"], 1e-6).to_bytes();
        bytes[4] = 0xFF; // corrupt the version field
        bytes[5] = 0xFF;
        assert!(matches!(
            Filter::load(&bytes),
            Err(LoadError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let bytes = good_purl_filter(&["pkg:npm/a@1"], 1e-6).to_bytes();
        let mut corrupt = bytes.clone();
        corrupt[0] = b'X';
        assert!(matches!(Filter::load(&corrupt), Err(LoadError::BadMagic)));
        assert!(matches!(
            Filter::load(&bytes[..HEADER_LEN - 1]),
            Err(LoadError::Truncated)
        ));
    }

    #[test]
    fn digest_path_matches_for_sha256_kind() {
        let digest = sha256(b"artifact-contents");
        let mut b = Builder::sized_for(Kind::Sha256, Tier::Bad, 1, 1e-6, 0);
        b.insert_digest(&digest);
        let f = b.build();
        assert!(f.contains_digest(&digest));
        assert!(!f.contains_digest(&sha256(b"other")));
    }

    #[test]
    fn parses_sha256_hex_and_rejects_garbage() {
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(parse_sha256_hex(hex), Some(sha256(b"")));
        assert_eq!(parse_sha256_hex(&hex.to_uppercase()), Some(sha256(b"")));
        assert_eq!(parse_sha256_hex("  too short  "), None);
        assert_eq!(parse_sha256_hex(&"z".repeat(64)), None);
    }

    #[test]
    fn artifact_stems_name_each_filter() {
        let stems: Vec<String> = generate(
            [Record {
                purl: Some("pkg:npm/a@1".into()),
                sha256: Some(sha256(b"a")),
            }],
            std::iter::empty(),
            1e-6,
        )
        .iter()
        .map(Filter::artifact_stem)
        .collect();
        assert_eq!(
            stems,
            ["purl-good", "purl-bad", "sha256-good", "sha256-bad"]
        );
    }

    #[test]
    fn generate_subtracts_bad_from_good() {
        // One key appears in both pools (a later-revoked bless); one is purely
        // good. Bad must win for both the PURL and the SHA-256 forms.
        let revoked = sha256(b"revoked-artifact");
        let clean = sha256(b"clean-artifact");
        let good = [
            Record {
                purl: Some("pkg:npm/evil@1".into()),
                sha256: Some(revoked),
            },
            Record {
                purl: Some("pkg:npm/good@1".into()),
                sha256: Some(clean),
            },
        ];
        let bad = [Record {
            purl: Some("pkg:npm/evil@1".into()),
            sha256: Some(revoked),
        }];

        let filters = generate(good, bad, 1e-9);
        let by = |k: Kind, t: Tier| {
            filters
                .iter()
                .find(|f| f.kind() == k && f.tier() == t)
                .expect("filter present")
        };

        // PURL: revoked excluded from good, present in bad; clean stays good.
        assert!(!by(Kind::Purl, Tier::Good).contains_key(b"pkg:npm/evil@1"));
        assert!(by(Kind::Purl, Tier::Bad).contains_key(b"pkg:npm/evil@1"));
        assert!(by(Kind::Purl, Tier::Good).contains_key(b"pkg:npm/good@1"));

        // SHA-256: same algebra on the digest path. The skip test
        // (good && !bad) must reject the revoked artifact.
        let good_sha = by(Kind::Sha256, Tier::Good);
        let bad_sha = by(Kind::Sha256, Tier::Bad);
        assert!(!good_sha.contains_digest(&revoked));
        assert!(bad_sha.contains_digest(&revoked));
        assert!(good_sha.contains_digest(&clean) && !bad_sha.contains_digest(&clean));
    }
}
