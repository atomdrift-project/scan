use super::import_tokens_from_parts;

/// The tokens a `(library, name)` pair contributes. Both forms matter:
/// LOGIC_GAPS matches bare names (`socket`, `CreateProcess`), while the
/// symbol vocab carries the `lib!name` form.
#[test]
fn yields_bare_name_and_lib_qualified() {
    let t = import_tokens_from_parts("kernel32", "CreateProcessW");
    assert!(t.contains(&"CreateProcessW".to_string()));
    assert!(t.contains(&"kernel32!CreateProcessW".to_string()));
}

/// The `len >= 2` gate is applied per token, exactly as collimator applies
/// it: a one-character *name* is dropped on its own but still contributes
/// the qualified `lib!name` token, which does pass. Matching collimator
/// here is the whole point — it defines the features the model was trained
/// on, so "more sensible" behaviour would be a mismatch.
#[test]
fn length_gate_is_per_token_not_per_entry() {
    assert_eq!(import_tokens_from_parts("x", "a"), vec!["x!a"]);
    assert!(import_tokens_from_parts("", "").is_empty());
}

/// An ordinal-only import has no name; the library alone is the token.
#[test]
fn library_only_entry_falls_back_to_library() {
    assert_eq!(import_tokens_from_parts("ws2_32", ""), vec!["ws2_32"]);
}
