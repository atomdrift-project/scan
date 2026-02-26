//! litmus - ML-powered malware classification using cleave static analysis.
//!
//! This library provides APIs for classifying files as benign, suspicious, or hostile
//! using a trained ML model and features extracted via cleave.

pub mod classifier;
pub mod features;
pub mod hybrid_model;
pub mod multi_model;

pub use classifier::{Classification, ClassificationResult, Classifier};
pub use features::{FeatureExtractor, FeatureVector};
pub use hybrid_model::{HybridClassifier, HybridStats};
pub use multi_model::{ModelRegistry, detect_filetype};
