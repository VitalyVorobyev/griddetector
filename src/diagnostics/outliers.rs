use crate::diagnostics::segments::SegmentSample;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutlierThresholds {
    pub angle_threshold_deg: f32,
    pub angle_margin_deg: f32
}

/// Detailed report of the coarse outlier filter applied before refinement.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutlierFilterStage {
    pub elapsed_ms: f64,
    pub total: usize,
    pub kept: usize,
    pub rejected: usize,
    pub kept_u: usize,
    pub kept_v: usize,
    pub degenerate_segments: usize,
    pub thresholds: OutlierThresholds,
    pub classifications: Vec<SegmentSample>,
}
