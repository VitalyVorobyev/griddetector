use crate::segments::Segment;
use serde::Serialize;

use crate::diagnostics::{timing::TimingBreakdown, LsdStage};

/// High-level report for the gradient-based segment refinement pipeline.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentRefineStage {
    pub pyramid: crate::diagnostics::PyramidStage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsd: Option<LsdStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsd_segments: Option<Vec<Segment>>,
    pub levels: Vec<SegmentRefineLevel>,
    pub timings: TimingBreakdown,
}

/// Diagnostics for a coarse→fine refinement transition.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentRefineLevel {
    pub coarse_level: usize,
    pub finer_level: usize,
    pub elapsed_ms: f64,
    pub segments_in: usize,
    pub accepted: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_score: Option<f32>,
    pub results: Vec<SegmentRefineSample>,
}

/// Per-segment refinement outcome, recording the updated geometry and scoring.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentRefineSample {
    pub segment: Segment,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_points: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region_pixels: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_refinements: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_samples: Option<usize>,
}
