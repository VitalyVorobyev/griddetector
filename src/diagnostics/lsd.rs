use crate::diagnostics::SegmentId;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FamilyCounts {
    pub family_u: usize,
    pub family_v: usize,
    pub unassigned: usize,
}

/// Outcome of the coarse LSD→VP stage.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LsdStage {
    pub elapsed_ms: f64,
    pub confidence: f32,
    pub dominant_angles_deg: [f32; 2],
    pub family_counts: FamilyCounts,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sample_ids: Vec<SegmentId>,
    pub used_gradient_refinement: bool,
}
