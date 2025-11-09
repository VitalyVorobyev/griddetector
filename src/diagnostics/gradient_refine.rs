use serde::Serialize;

/// Diagnostics for the coarsest-level gradient-only refinement step.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GradientRefineStage {
    pub level_index: usize,
    pub elapsed_ms: f64,
    pub segments_in: usize,
    pub accepted: usize,
    pub rejected: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_movement_px: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_region_pixels: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_normal_refinements: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_gradient_samples: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_support_points: Option<f32>,
    pub fallback_to_input: bool,
}
