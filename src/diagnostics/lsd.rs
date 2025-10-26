use crate::diagnostics::SegmentId;
use crate::lsd_vp::FamilyLabel;
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
    /// Per-segment family assignments. The indices align with the descriptors in
    /// [`PipelineTrace::segments`](crate::diagnostics::pipeline::PipelineTrace).
    pub segment_families: Vec<Option<FamilyLabel>>,
    /// Optional sample of segment identifiers highlighted by demos for JSON size
    /// control. When empty, downstream tooling should assume that all segments
    /// are part of the report.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sample_ids: Vec<SegmentId>,
}
