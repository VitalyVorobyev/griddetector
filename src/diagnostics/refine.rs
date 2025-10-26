use serde::{Deserialize, Serialize};

/// Coarse-to-fine refinement trace capturing every attempted iteration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementStage {
    pub elapsed_ms: f64,
    pub passes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<RefinementOutcome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementOutcome {
    pub levels_used: usize,
    pub confidence: f32,
    pub inlier_ratio: f32,
    pub iterations: Vec<RefinementIteration>,
}

/// Diagnostics collected for every refinement level that was processed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementIteration {
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub segments: usize,
    pub bundles: usize,
    pub family_u_count: usize,
    pub family_v_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improvement: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inlier_ratio: Option<f32>,
}
