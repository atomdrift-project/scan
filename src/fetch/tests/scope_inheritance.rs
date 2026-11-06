use super::*;
#[test]
fn dependency_context_follows_build_dev_and_ci_edges_without_losing_hooks() {
    for scope in [
        filefacts::DependencyScope::Build,
        filefacts::DependencyScope::Development,
        filefacts::DependencyScope::Ci,
    ] {
        let mut parent = Reference::new(
            RefLocator::Purl("pkg:npm/parent".into()),
            RefKind::Dependency,
            "lock",
            "parent",
        );
        parent.context = Some(filefacts::DependencyContext {
            scope,
            optional: true,
            has_install_script: false,
            installed_path: None,
        });
        let mut child = Reference::new(
            RefLocator::Purl("pkg:npm/child".into()),
            RefKind::Dependency,
            "lock",
            "child",
        );
        child.context = Some(filefacts::DependencyContext {
            scope: filefacts::DependencyScope::Runtime,
            optional: false,
            has_install_script: true,
            installed_path: Some("node_modules/child".into()),
        });
        let mut groups = vec![("source".into(), vec![child])];
        inherit_dependency_context(&parent, &mut groups);
        let context = groups[0].1[0].context.as_ref().unwrap();
        assert_eq!(context.scope, scope);
        assert!(context.optional);
        assert!(context.has_install_script);
        assert_eq!(
            context.installed_path.as_deref(),
            Some("node_modules/child")
        );
    }
}
