use crate::diagnostics::{
    BundlingStage, GradientRefineStage, GridIndexingStage, LsdStage, OutlierFilterStage,
    RefinementStage, TimingBreakdown,
};
use crate::segments::Segment;
use crate::types::GridResult;
use nalgebra::{Matrix3, Isometry3};
use serde::Serialize;

/// Result produced by [`GridDetector::process_with_diagnostics`](crate::GridDetector).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionReport {
    pub grid: GridResult,
    pub trace: PipelineTrace,
}

fn format_optional(val: Option<f32>) -> String {
    val.map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
}

/// End-to-end trace describing the internal execution of the detector.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineTrace {
    pub input: InputDescriptor,
    pub timings: TimingBreakdown,
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pyramid_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsd: Option<LsdStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outlier_filter: Option<OutlierFilterStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_refine: Option<GradientRefineStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundling: Option<BundlingStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_indexing: Option<GridIndexingStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refinement: Option<RefinementStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coarse_homography: Option<Matrix3<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<Isometry3<f32>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InputDescriptor {
    pub width: usize,
    pub height: usize,
    pub pyramid_levels: usize,
}

// Camera pose recovered from the refined homography is serialized via `Pose`.
