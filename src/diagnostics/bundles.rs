use serde::{Deserialize, Serialize};

/// Summary of a bundle produced from near-collinear segments.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleDescriptor {
    pub center: [f32; 2],
    pub line: [f32; 3],
    pub weight: f32,
}

/// Bundles collected for a single pyramid level.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundlingLevel {
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub bundles: Vec<BundleDescriptor>,
}

/// Result of the bundling stage executed before homography refinement. The
/// detector first refines segments on progressively finer pyramid levels before
/// merging them into bundles; `segment_refine_ms` captures this pre-processing
/// cost.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundlingStage {
    pub elapsed_ms: f64,
    pub segment_refine_ms: f64,
    pub orientation_tol_deg: f32,
    pub merge_distance_px: f32,
    pub min_weight: f32,
    pub source_segments: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_applied: Option<[f32; 2]>,
    pub levels: Vec<BundlingLevel>,
}
