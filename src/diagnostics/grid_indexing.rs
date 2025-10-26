use crate::lsd_vp::FamilyLabel;
use serde::Serialize;

/// Per-bundle assignment within a grid family after rectification.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GridLineIndex {
    pub bundle_index: usize,
    pub family: FamilyLabel,
    pub rectified_offset: f32,
    pub grid_index: i32,
    pub weight: f32,
    pub residual: f32,
}

/// Summary describing spacing and offsets for a single vanishing-line family.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FamilyIndexing {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_offset: Option<f32>,
    pub confidence: f32,
    pub lines: Vec<GridLineIndex>,
}

/// Result of associating bundled lines with discrete grid indices.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GridIndexingStage {
    pub elapsed_ms: f64,
    pub family_u: FamilyIndexing,
    pub family_v: FamilyIndexing,
    pub origin_uv: (i32, i32),
    pub visible_range: (i32, i32, i32, i32),
}
