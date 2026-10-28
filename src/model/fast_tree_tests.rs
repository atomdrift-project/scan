use super::{FastTreeBackend, TractOnnxBackend};
use tract_onnx::pb::{AttributeProto, GraphProto, NodeProto};
use tract_onnx::prelude::Framework as _;

const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/cli-model.onnx");

fn fixture_graph() -> GraphProto {
    tract_onnx::onnx()
        .proto_model_for_read(&mut &FIXTURE[..])
        .expect("fixture parses")
        .graph
        .expect("fixture has a graph")
}

fn tree_node(graph: &mut GraphProto) -> &mut NodeProto {
    graph
        .node
        .iter_mut()
        .find(|node| node.op_type == "TreeEnsembleClassifier")
        .expect("fixture has a tree")
}

fn set_ints(node: &mut NodeProto, name: &str, ints: Vec<i64>) {
    attribute(node, name).ints = ints;
}

fn attribute<'a>(node: &'a mut NodeProto, name: &str) -> &'a mut AttributeProto {
    node.attribute
        .iter_mut()
        .find(|attr| attr.name == name)
        .expect("fixture attribute")
}

/// The fast evaluator must agree with tract wherever it accepts a model.
#[test]
fn the_fast_path_agrees_with_tract() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.onnx");
    std::fs::write(&path, FIXTURE).expect("write");
    let fast = FastTreeBackend::load(&path).expect("the fixture takes the fast path");
    let tract = TractOnnxBackend::load(&path).expect("tract loads the fixture");
    for x in [-1.0, 0.0, 0.5, 1e9, f32::NAN] {
        let (a, b) = (
            fast.predict(&[x]).expect("fast"),
            tract.predict(&[x]).expect("tract"),
        );
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "fast {a} != tract {b} at {x}"
        );
    }
}

/// The same agreement over a real bundle, on deterministic pseudo-random
/// inputs, including values on and around each split.
#[test]
#[ignore = "needs SCAN_MODELS_DIR pointing at an installed model bundle"]
fn the_fast_path_agrees_with_tract_on_a_real_bundle() {
    let dir = crate::env::path(crate::env::SCAN_MODELS_DIR).expect("SCAN_MODELS_DIR");
    let models: Vec<_> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "onnx"))
        .collect();
    assert!(!models.is_empty(), "no .onnx under SCAN_MODELS_DIR");
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Mostly small counts and ratios, the range features take.
        (state % 10_000) as f32 / 100.0 - 5.0
    };
    for entry in models {
        let fast = FastTreeBackend::load(entry.path()).expect("fast path");
        let tract = TractOnnxBackend::load(entry.path()).expect("tract");
        for _ in 0..200 {
            let x: Vec<f32> = (0..fast.n_features).map(|_| next()).collect();
            let (a, b) = (
                fast.predict(&x).expect("fast"),
                tract.predict(&x).expect("tract"),
            );
            assert!(
                (a - b).abs() <= 1e-6,
                "{}: fast {a} != tract {b}",
                entry.path().display()
            );
        }
    }
}

/// A branch that leads back to its own root would loop forever in
/// `predict`. Such a model is refused at load and left to tract.
#[test]
fn a_cyclic_tree_is_refused() {
    let mut graph = fixture_graph();
    let node = tree_node(&mut graph);
    // Node 0 branches to node 1 (a leaf) and back to itself.
    set_ints(node, "nodes_featureids", vec![0, 0]);
    set_ints(node, "nodes_nodeids", vec![0, 1]);
    set_ints(node, "nodes_treeids", vec![0, 0]);
    set_ints(node, "nodes_truenodeids", vec![1, 0]);
    set_ints(node, "nodes_falsenodeids", vec![0, 0]);
    attribute(node, "nodes_values").floats = vec![0.5, 0.0];
    attribute(node, "nodes_modes").strings = vec![b"BRANCH_LEQ".to_vec(), b"LEAF".to_vec()];
    set_ints(node, "class_nodeids", vec![1, 1]);
    let error = FastTreeBackend::from_graph(&graph).expect_err("cycle refused");
    assert!(error.to_string().contains("is not a tree"), "{error}");
}

/// A node that transforms the probabilities changes the answer, and the
/// fast evaluator would skip it. One on the label path does not.
#[test]
fn only_a_probability_transform_is_left_to_tract() {
    let mut graph = fixture_graph();
    let probabilities = tree_node(&mut graph).output[1].clone();
    let label = tree_node(&mut graph).output[0].clone();
    graph.node.push(NodeProto {
        op_type: "Cast".into(),
        input: vec![label],
        output: vec!["label_cast".into()],
        ..NodeProto::default()
    });
    FastTreeBackend::from_graph(&graph).expect("a label cast is harmless");
    graph.node.push(NodeProto {
        op_type: "Scaler".into(),
        input: vec![probabilities],
        output: vec!["scaled".into()],
        ..NodeProto::default()
    });
    let error = FastTreeBackend::from_graph(&graph).expect_err("refused");
    assert!(error.to_string().contains("Scaler"), "{error}");
}

/// LightGBM's exporter multiplies the probabilities by a constant 1, which
/// changes nothing; a multiply by anything else does.
#[test]
fn only_a_multiply_by_one_passes() {
    use tract_onnx::pb::TensorProto;
    let with_factor = |factor: f32| {
        let mut graph = fixture_graph();
        let probabilities = tree_node(&mut graph).output[1].clone();
        graph.initializer.push(TensorProto {
            name: "factor".into(),
            float_data: vec![factor],
            ..TensorProto::default()
        });
        graph.node.push(NodeProto {
            op_type: "Mul".into(),
            input: vec![probabilities, "factor".into()],
            output: vec!["scaled".into()],
            ..NodeProto::default()
        });
        FastTreeBackend::from_graph(&graph)
    };
    with_factor(1.0).expect("a multiply by one is a pass-through");
    assert!(with_factor(2.0).is_err());
}
