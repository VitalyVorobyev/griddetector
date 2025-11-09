use crate::segments::Segment;
use serde::Deserialize;

/// Parameters driving the coarse-to-fine gradient refinement.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SegmentRefineParams {
    /// Half-width of the normal search window (px).
    pub anchor_search_radius: f32,
    /// Sampling step along the normal when probing for gradients (px).
    pub normal_step: f32,
    /// Step along the tangent for region growing (px).
    pub tangent_step: f32,
    /// Maximum number of growth steps in each tangent direction.
    pub max_grow_steps: usize,
    /// Minimum gradient magnitude accepted when collecting support (Scharr units).
    pub gradient_threshold: f32,
    /// Maximum allowed displacement between predicted and refined point along the normal.
    pub max_normal_shift: f32,
    /// Maximum allowed distance between consecutive refined points (fails fast when jumping off edge).
    pub max_point_dist: f32,
    /// Minimum number of refined support points required to accept the segment.
    pub min_support_points: usize,
    /// Maximum number of seed samples evaluated along the predicted segment.
    pub max_seeds: usize,
}

impl Default for SegmentRefineParams {
    fn default() -> Self {
        Self {
            anchor_search_radius: 3.0,
            normal_step: 0.5,
            tangent_step: 0.75,
            max_grow_steps: 64,
            gradient_threshold: 0.08,
            max_normal_shift: 2.0,
            max_point_dist: 3.5,
            min_support_points: 4,
            max_seeds: 3,
        }
    }
}

/// Diagnostics collected while refining a single segment.
#[derive(Clone, Debug, Default)]
pub struct RefineDiagnostics {
    /// Number of seed anchors tested before choosing the best one.
    pub anchor_trials: usize,
    /// Total number of 1D normal refinement passes executed.
    pub normal_refinements: usize,
    /// Total number of tangent growth steps that succeeded.
    pub tangent_steps: usize,
    /// Total number of gradient samples queried from the pyramid level.
    pub gradient_samples: usize,
}

/// Outcome of refining a predicted segment on a single level.
#[derive(Clone, Debug)]
pub struct RefineResult {
    pub seg: Segment,
    pub ok: bool,
    /// Average gradient magnitude along the refined support.
    pub score: f32,
    /// Number of refined support points used to fit the carrier.
    pub support_points: usize,
    pub diagnostics: RefineDiagnostics,
}

impl RefineResult {
    pub(crate) fn failed(seg: Segment) -> Self {
        Self {
            seg,
            ok: false,
            score: 0.0,
            support_points: 0,
            diagnostics: RefineDiagnostics::default(),
        }
    }
}

/// Coordinate mapping from pyramid level `l+1` to level `l`.
pub trait ScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2];
}
