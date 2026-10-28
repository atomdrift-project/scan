use super::has_qualifier;

#[test]
fn a_qualifier_is_found_by_key_only() {
    assert!(has_qualifier(
        "pkg:pypi/x?version_requirement=%3E%3D1",
        "version_requirement"
    ));
    assert!(has_qualifier(
        "pkg:cargo/x?a=1&version_requirement=2#sub",
        "version_requirement"
    ));
    assert!(!has_qualifier("pkg:pypi/x", "version_requirement"));
    assert!(!has_qualifier(
        "pkg:pypi/x?other=version_requirement=",
        "version_requirement"
    ));
    assert!(!has_qualifier(
        "pkg:pypi/x#version_requirement=1",
        "version_requirement"
    ));
}
