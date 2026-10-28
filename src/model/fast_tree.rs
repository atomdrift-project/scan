//! A direct evaluator for ONNX tree-ensemble classifiers, accepted only for
//! graphs it provably evaluates exactly as tract would.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Class-score accumulator size kept on the stack. The route models are
/// binary; anything wider falls back to a heap vector.
pub(super) const FAST_TREE_STACK_CLASSES: usize = 8;

pub(super) struct FastTreeBackend {
    pub(super) n_features: usize,
    pub(super) n_classes: usize,
    pub(super) trees: Vec<usize>,
    pub(super) nodes: Vec<FastTreeNode>,
    pub(super) leaves: Vec<(u32, f32)>,
    pub(super) base_values: Vec<f32>,
    pub(super) post_transform: FastPostTransform,
    pub(super) binary_result_layout: bool,
    /// The logistic post-transform kernel, built once at load. The factory
    /// behind `tract_linalg::ops().sigmoid_f32` allocates a fresh boxed op on
    /// every call, and this runs once per route per embedded member — tens of
    /// thousands of times in a directory scan — to transform `n_classes`
    /// floats. `ElementWise<f32>` is `Send + Sync`, so one instance serves
    /// every thread. `None` unless `post_transform` is `Logistic`.
    pub(super) sigmoid: Option<Box<dyn tract_linalg::element_wise::ElementWise<f32>>>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FastPostTransform {
    None,
    Logistic,
    Softmax,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FastCmp {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

// Compact indices keep more of the model in cache during per-member
// inference. All narrowing is checked while loading; scoring keeps the same
// tree order, branch comparisons and floating-point accumulation order.
#[derive(Debug, Clone, Copy)]
pub(super) enum FastTreeNode {
    Branch {
        feature_id: u32,
        threshold: f32,
        true_id: u32,
        false_id: u32,
        cmp: FastCmp,
        nan_is_true: bool,
    },
    Leaf {
        start: u32,
        end: u32,
    },
}

impl std::fmt::Debug for FastTreeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastTreeBackend")
            .field("n_features", &self.n_features)
            .field("n_classes", &self.n_classes)
            .field("trees", &self.trees.len())
            .field("post_transform", &self.post_transform)
            .finish_non_exhaustive()
    }
}

/// Whether `name` is an initializer holding the single value `1.0`.
pub(super) fn is_constant_one(graph: &tract_onnx::pb::GraphProto, name: &str) -> bool {
    graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == name)
        .is_some_and(|tensor| {
            tensor.dims.iter().all(|&dim| dim == 1)
                && (tensor.float_data == [1.0] || tensor.raw_data == 1.0f32.to_le_bytes())
        })
}

/// The graph's `TreeEnsembleClassifier`, when it reads the graph input and
/// its probabilities reach the output through pass-through nodes alone. What
/// happens to the label output is irrelevant; it is never read.
fn ensemble_node(graph: &tract_onnx::pb::GraphProto) -> Result<&tract_onnx::pb::NodeProto> {
    let node = graph
        .node
        .iter()
        .find(|node| node.op_type == "TreeEnsembleClassifier")
        .ok_or_else(|| anyhow::anyhow!("ONNX graph has no TreeEnsembleClassifier node"))?;
    if node.input.first() != graph.input.first().map(|input| &input.name) {
        anyhow::bail!("TreeEnsembleClassifier does not read the graph input directly");
    }
    // ONNX nodes are topologically sorted, so one pass follows every tensor
    // derived from the probabilities.
    let mut probabilities: std::collections::HashSet<&str> =
        node.output.get(1).map(String::as_str).into_iter().collect();
    for other in graph
        .node
        .iter()
        .filter(|other| !std::ptr::eq(*other, node))
    {
        if !other
            .input
            .iter()
            .any(|input| probabilities.contains(input.as_str()))
        {
            continue;
        }
        let pass_through = match other.op_type.as_str() {
            "ZipMap" | "Identity" => true,
            // LightGBM's exporter multiplies the probabilities by a constant 1.
            "Mul" => other
                .input
                .iter()
                .find(|input| !probabilities.contains(input.as_str()))
                .is_some_and(|factor| is_constant_one(graph, factor)),
            _ => false,
        };
        if !pass_through {
            anyhow::bail!(
                "ONNX graph transforms the probabilities with {}, which the fast evaluator does not run",
                other.op_type
            );
        }
        probabilities.extend(other.output.iter().map(String::as_str));
    }
    Ok(node)
}

/// Every tree must be one: each branch points inside its own tree, and no
/// node is reached twice from the root. Together these make every walk in
/// `predict` end at a leaf.
pub(super) fn check_trees(roots: &[usize], nodes: &[FastTreeNode]) -> Result<()> {
    let mut seen = vec![false; nodes.len()];
    for (tree, &root) in roots.iter().enumerate() {
        let end = roots.get(tree + 1).copied().unwrap_or(nodes.len());
        let mut stack = vec![root];
        while let Some(idx) = stack.pop() {
            if !(root..end).contains(&idx) || std::mem::replace(&mut seen[idx], true) {
                anyhow::bail!("TreeEnsembleClassifier tree {tree} is not a tree");
            }
            if let FastTreeNode::Branch {
                true_id, false_id, ..
            } = nodes[idx]
            {
                stack.extend([true_id as usize, false_id as usize]);
            }
        }
    }
    Ok(())
}

impl FastTreeBackend {
    pub(super) fn load(path: &Path) -> Result<Self> {
        use tract_onnx::prelude::Framework;
        let proto = tract_onnx::onnx()
            .proto_model_for_path(path)
            .with_context(|| format!("parsing ONNX protobuf {}", path.display()))?;
        let graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;
        Self::from_graph(graph)
    }

    /// Accept only a graph this evaluator computes exactly; see
    /// [`ensemble_node`]. Anything else is left to tract.
    pub(super) fn from_graph(graph: &tract_onnx::pb::GraphProto) -> Result<Self> {
        let node = ensemble_node(graph)?;
        let n_features = onnx_input_feature_count(graph)?;
        let n_classes = onnx_class_count(node)?;
        let nodes_featureids = attr_usizes(node, "nodes_featureids")?;
        if nodes_featureids.is_empty() {
            anyhow::bail!("TreeEnsembleClassifier has no nodes");
        }
        let n_nodes = nodes_featureids.len();
        let node_ids = attr_usizes_exact(node, "nodes_nodeids", n_nodes)?;
        let tree_ids = attr_usizes_exact(node, "nodes_treeids", n_nodes)?;
        let true_ids = attr_usizes_exact(node, "nodes_truenodeids", n_nodes)?;
        let false_ids = attr_usizes_exact(node, "nodes_falsenodeids", n_nodes)?;
        let node_values = attr_floats_exact(node, "nodes_values", n_nodes)?;
        let node_modes = attr_strings_exact(node, "nodes_modes", n_nodes)?;
        let nan_is_true = optional_attr_bools(node, "nodes_missing_value_tracks_true", n_nodes)?;

        let leaf_node_ids = attr_usizes(node, "class_nodeids")?;
        if leaf_node_ids.is_empty() {
            anyhow::bail!("TreeEnsembleClassifier has no class leaves");
        }
        let n_leaves = leaf_node_ids.len();
        let leaf_tree_ids = attr_usizes_exact(node, "class_treeids", n_leaves)?;
        let leaf_class_ids = attr_usizes_exact(node, "class_ids", n_leaves)?;
        let leaf_weights = attr_floats_exact(node, "class_weights", n_leaves)?;

        if tree_ids.first().copied() != Some(0) || leaf_tree_ids.first().copied() != Some(0) {
            anyhow::bail!("TreeEnsembleClassifier tree ids must start at 0");
        }
        let n_trees = tree_ids
            .last()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier has no tree ids"))?
            + 1;
        if leaf_tree_ids.last().copied() != Some(n_trees - 1) {
            anyhow::bail!("TreeEnsembleClassifier node/leaf tree counts mismatch");
        }

        let mut node_order: Vec<usize> = (0..n_nodes).collect();
        node_order.sort_by_key(|&idx| (tree_ids[idx], node_ids[idx]));
        let mut leaf_order: Vec<usize> = (0..n_leaves).collect();
        leaf_order.sort_by_key(|&idx| (leaf_tree_ids[idx], leaf_node_ids[idx]));

        let mut trees = Vec::with_capacity(n_trees);
        let mut nodes = Vec::with_capacity(n_nodes);
        let mut leaves = Vec::with_capacity(n_leaves);
        let mut current_tree_id = None;
        let mut in_leaf_idx = 0usize;
        for node_idx in node_order {
            let tree_id = tree_ids[node_idx];
            if Some(tree_id) != current_tree_id {
                current_tree_id = Some(tree_id);
                trees.push(nodes.len());
            }

            let tree_offset = *trees
                .last()
                .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier missing tree root"))?;
            // Branch targets index this layout by node id, so a tree's ids
            // must be exactly 0, 1, 2, … in the sorted order above.
            if node_ids[node_idx] != nodes.len() - tree_offset {
                anyhow::bail!("TreeEnsembleClassifier tree {tree_id} node ids are not dense");
            }
            if let Some(cmp) = parse_fast_cmp(&node_modes[node_idx])? {
                nodes.push(FastTreeNode::Branch {
                    feature_id: u32::try_from(nodes_featureids[node_idx])
                        .context("ONNX feature index exceeds compact node capacity")?,
                    threshold: node_values[node_idx],
                    true_id: u32::try_from(
                        true_ids[node_idx]
                            .checked_add(tree_offset)
                            .context("ONNX true branch index overflow")?,
                    )
                    .context("ONNX true branch exceeds compact node capacity")?,
                    false_id: u32::try_from(
                        false_ids[node_idx]
                            .checked_add(tree_offset)
                            .context("ONNX false branch index overflow")?,
                    )
                    .context("ONNX false branch exceeds compact node capacity")?,
                    cmp,
                    nan_is_true: nan_is_true[node_idx],
                });
            } else {
                let start = leaves.len();
                while in_leaf_idx < leaf_order.len() {
                    let leaf_idx = leaf_order[in_leaf_idx];
                    if leaf_tree_ids[leaf_idx] != tree_id
                        || leaf_node_ids[leaf_idx] != node_ids[node_idx]
                    {
                        break;
                    }
                    leaves.push((
                        u32::try_from(leaf_class_ids[leaf_idx])
                            .context("ONNX class index exceeds compact node capacity")?,
                        leaf_weights[leaf_idx],
                    ));
                    in_leaf_idx += 1;
                }
                nodes.push(FastTreeNode::Leaf {
                    start: u32::try_from(start)
                        .context("ONNX leaf index exceeds compact node capacity")?,
                    end: u32::try_from(leaves.len())
                        .context("ONNX leaf index exceeds compact node capacity")?,
                });
            }
        }

        if in_leaf_idx != leaf_order.len() {
            anyhow::bail!("TreeEnsembleClassifier has leaves that do not match any node");
        }
        check_trees(&trees, &nodes)?;
        let max_feature = nodes_featureids.iter().copied().max().unwrap_or(0);
        if max_feature >= n_features {
            anyhow::bail!(
                "TreeEnsembleClassifier uses feature {max_feature}, input has {n_features}"
            );
        }
        if leaf_class_ids.iter().any(|&class_id| class_id >= n_classes) {
            anyhow::bail!("TreeEnsembleClassifier leaf class id exceeds class count");
        }

        let mut base_values = optional_onnx_attr(node, "base_values")
            .map(|attr| attr.floats.clone())
            .unwrap_or_else(|| vec![0.0; n_classes]);
        if base_values.is_empty() {
            base_values.resize(n_classes, 0.0);
        }
        if base_values.len() != n_classes {
            anyhow::bail!(
                "TreeEnsembleClassifier base_values length {} != class count {}",
                base_values.len(),
                n_classes
            );
        }

        let post_transform = match optional_attr_string(node, "post_transform")?.as_deref() {
            None | Some("NONE") => FastPostTransform::None,
            Some("LOGISTIC") => FastPostTransform::Logistic,
            Some("SOFTMAX") => FastPostTransform::Softmax,
            Some(other) => anyhow::bail!("unsupported post_transform {other}"),
        };
        let binary_result_layout =
            n_classes < 3 && leaves.iter().all(|(class_id, _)| *class_id == 0);

        Ok(Self {
            n_features,
            n_classes,
            trees,
            nodes,
            leaves,
            base_values,
            post_transform,
            binary_result_layout,
            sigmoid: match post_transform {
                FastPostTransform::Logistic => Some((tract_linalg::ops().sigmoid_f32)()),
                FastPostTransform::None | FastPostTransform::Softmax => None,
            },
        })
    }

    pub(super) fn predict(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.n_features {
            anyhow::bail!(
                "feature vector length {} != ONNX expected {}",
                features.len(),
                self.n_features
            );
        }

        // Class counts are tiny (2 for the binary route models). Keeping the
        // accumulator on the stack for the common case avoids a heap
        // allocation per prediction on a path that runs once per route per
        // embedded member.
        let mut score_buf = [0.0f32; FAST_TREE_STACK_CLASSES];
        let mut score_heap;
        let scores: &mut [f32] = if self.n_classes <= FAST_TREE_STACK_CLASSES {
            &mut score_buf[..self.n_classes]
        } else {
            score_heap = vec![0.0f32; self.n_classes];
            &mut score_heap[..]
        };
        for &root in &self.trees {
            let mut node_idx = root;
            loop {
                match self.nodes.get(node_idx) {
                    Some(FastTreeNode::Branch {
                        feature_id,
                        threshold,
                        true_id,
                        false_id,
                        cmp,
                        nan_is_true,
                    }) => {
                        let feature = features[*feature_id as usize];
                        node_idx = if fast_compare(*cmp, feature, *threshold, *nan_is_true) {
                            *true_id as usize
                        } else {
                            *false_id as usize
                        };
                    }
                    Some(FastTreeNode::Leaf { start, end }) => {
                        for &(class_id, weight) in &self.leaves[*start as usize..*end as usize] {
                            scores[class_id as usize] += weight;
                        }
                        break;
                    }
                    None => {
                        anyhow::bail!("TreeEnsembleClassifier branch points outside node table")
                    }
                }
            }
        }

        for (score, base) in scores.iter_mut().zip(self.base_values.iter()) {
            *score += *base;
        }
        match self.post_transform {
            FastPostTransform::None => {}
            FastPostTransform::Logistic => {
                if let Some(sigmoid) = &self.sigmoid {
                    sigmoid
                        .run(scores)
                        .context("ONNX sigmoid post-transform failed")?;
                }
            }
            FastPostTransform::Softmax => {
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                if sum != 0.0 {
                    for score in scores.iter_mut() {
                        *score /= sum;
                    }
                }
            }
        }

        if self.binary_result_layout {
            scores
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier emitted no score"))
        } else {
            scores
                .get(1)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier emitted no positive class"))
        }
    }
}

pub(super) fn onnx_attr<'a>(
    node: &'a tract_onnx::pb::NodeProto,
    name: &str,
) -> Result<&'a tract_onnx::pb::AttributeProto> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier missing attribute {name}"))
}

pub(super) fn optional_onnx_attr<'a>(
    node: &'a tract_onnx::pb::NodeProto,
    name: &str,
) -> Option<&'a tract_onnx::pb::AttributeProto> {
    node.attribute.iter().find(|attr| attr.name == name)
}

pub(super) fn attr_usizes(node: &tract_onnx::pb::NodeProto, name: &str) -> Result<Vec<usize>> {
    onnx_attr(node, name)?
        .ints
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .with_context(|| format!("attribute {name} has negative value {value}"))
        })
        .collect()
}

pub(super) fn attr_usizes_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<usize>> {
    let values = attr_usizes(node, name)?;
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

pub(super) fn attr_floats_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>> {
    let values = onnx_attr(node, name)?.floats.clone();
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

pub(super) fn attr_strings_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<String>> {
    let values: Vec<String> = onnx_attr(node, name)?
        .strings
        .iter()
        .map(|bytes| {
            std::str::from_utf8(bytes)
                .map(str::to_owned)
                .with_context(|| format!("attribute {name} contains non-UTF8 string"))
        })
        .collect::<Result<_>>()?;
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

pub(super) fn optional_attr_string(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
) -> Result<Option<String>> {
    optional_onnx_attr(node, name)
        .map(|attr| {
            std::str::from_utf8(&attr.s)
                .map(str::to_owned)
                .with_context(|| format!("attribute {name} contains non-UTF8 string"))
        })
        .transpose()
}

pub(super) fn optional_attr_bools(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<bool>> {
    let Some(attr) = optional_onnx_attr(node, name) else {
        return Ok(vec![false; expected]);
    };
    if attr.ints.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            attr.ints.len()
        );
    }
    Ok(attr.ints.iter().map(|&value| value != 0).collect())
}

pub(super) fn onnx_class_count(node: &tract_onnx::pb::NodeProto) -> Result<usize> {
    let int_count = optional_onnx_attr(node, "classlabels_int64s").map(|attr| attr.ints.len());
    let string_count =
        optional_onnx_attr(node, "classlabels_strings").map(|attr| attr.strings.len());
    match (int_count, string_count) {
        (Some(count), None) | (None, Some(count)) if count > 0 => Ok(count),
        (Some(_), Some(_)) => {
            anyhow::bail!("TreeEnsembleClassifier has both integer and string class labels")
        }
        _ => anyhow::bail!("TreeEnsembleClassifier has no class labels"),
    }
}

pub(super) fn onnx_input_feature_count(graph: &tract_onnx::pb::GraphProto) -> Result<usize> {
    let input = graph
        .input
        .first()
        .ok_or_else(|| anyhow::anyhow!("ONNX graph has no input"))?;
    let value = input
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .ok_or_else(|| anyhow::anyhow!("ONNX input has no type"))?;
    let tract_onnx::pb::type_proto::Value::TensorType(tensor_type) = value;
    let shape = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ONNX input tensor has no shape"))?;
    let feature_dim = shape
        .dim
        .get(1)
        .and_then(|dim| dim.value.as_ref())
        .ok_or_else(|| anyhow::anyhow!("ONNX input has no feature dimension"))?;
    match feature_dim {
        tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(value) => {
            usize::try_from(*value).context("ONNX feature dimension is negative")
        }
        tract_onnx::pb::tensor_shape_proto::dimension::Value::DimParam(param) => {
            anyhow::bail!("ONNX feature dimension is symbolic: {param}")
        }
    }
}

pub(super) fn parse_fast_cmp(mode: &str) -> Result<Option<FastCmp>> {
    match mode {
        "BRANCH_LEQ" => Ok(Some(FastCmp::LessEqual)),
        "BRANCH_LT" => Ok(Some(FastCmp::Less)),
        "BRANCH_GTE" => Ok(Some(FastCmp::GreaterEqual)),
        "BRANCH_GT" => Ok(Some(FastCmp::Greater)),
        "BRANCH_EQ" => Ok(Some(FastCmp::Equal)),
        "BRANCH_NEQ" => Ok(Some(FastCmp::NotEqual)),
        "LEAF" => Ok(None),
        other => anyhow::bail!("unsupported tree node mode {other}"),
    }
}

pub(super) fn fast_compare(cmp: FastCmp, feature: f32, threshold: f32, nan_is_true: bool) -> bool {
    if feature.is_nan() {
        return nan_is_true;
    }
    match cmp {
        FastCmp::Equal => feature == threshold,
        FastCmp::NotEqual => feature != threshold,
        FastCmp::Less => feature < threshold,
        FastCmp::Greater => feature > threshold,
        FastCmp::LessEqual => feature <= threshold,
        FastCmp::GreaterEqual => feature >= threshold,
    }
}
