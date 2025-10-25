use crate::types::GridResult;
use nalgebra::Matrix3;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct BundleEntryDiagnostics {
    pub center: [f32; 2],
    pub line: [f32; 3],
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BundleDiagnostics {
    pub level_index: usize,
    pub bundles: Vec<BundleEntryDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SegmentFilterDiagnostics {
    pub total: usize,
    pub kept: usize,
    pub rejected: usize,
    pub kept_u: usize,
    pub kept_v: usize,
    pub skipped_degenerate: usize,
    pub angle_threshold_deg: f32,
    pub residual_threshold_px: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PyramidLevelDiagnostics {
    pub level: usize,
    pub width: usize,
    pub height: usize,
    pub mean_intensity: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LsdSegmentDiagnostics {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub len: f32,
    pub strength: f32,
    pub family: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LsdDiagnostics {
    pub segments_total: usize,
    pub dominant_angles_deg: [f32; 2],
    pub family_u_count: usize,
    pub family_v_count: usize,
    pub confidence: f32,
    pub elapsed_ms: f64,
    pub segments_sample: Vec<LsdSegmentDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefinementLevelDiagnostics {
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub segments: usize,
    pub bundles: usize,
    pub family_u_count: usize,
    pub family_v_count: usize,
    pub improvement: Option<f32>,
    pub confidence: Option<f32>,
    pub inlier_ratio: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefinementDiagnostics {
    pub levels_used: usize,
    pub aggregated_confidence: f32,
    pub final_inlier_ratio: f32,
    pub levels: Vec<RefinementLevelDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProcessingDiagnostics {
    pub input_width: usize,
    pub input_height: usize,
    pub pyramid_levels: Vec<PyramidLevelDiagnostics>,
    pub pyramid_build_ms: f64,
    pub lsd_ms: f64,
    pub segment_filter: Option<SegmentFilterDiagnostics>,
    pub outlier_filter_ms: f64,
    pub bundling_ms: f64,
    pub segment_refine_ms: f64,
    pub refine_ms: f64,
    pub refinement_passes: usize,
    pub lsd: Option<LsdDiagnostics>,
    pub refinement: Option<RefinementDiagnostics>,
    pub bundling: Option<Vec<BundleDiagnostics>>,
    pub homography: Matrix3<f32>,
    pub total_latency_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DetailedResult {
    pub result: GridResult,
    pub diagnostics: ProcessingDiagnostics,
}
