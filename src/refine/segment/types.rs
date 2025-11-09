use crate::segments::Segment;
use serde::Deserialize;

/// Parameters driving the coarse-to-fine gradient refinement.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SegmentRefineParams {
    /// Half-width of the normal search window (px).
    pub anchor_search_radius: f32,
    /// Sampling step along the normal when probing for gradients (px).
    pub anchor_step: f32,
    /// Minimum gradient magnitude accepted when collecting support (Scharr units).
    pub gradient_threshold: f32,
    /// Padding (px) added to the predicted segment AABB when defining the grow window.
    pub region_padding_px: f32,
    /// Orientation tolerance (deg) around the seed normal allowed during region growth.
    pub region_angle_tolerance_deg: f32,
    /// Minimum number of pixels collected by the LSD-style growth to accept a segment.
    pub region_min_pixels: usize,
    /// Maximum number of pixels visited during growth (capacity guard).
    pub region_max_pixels: usize,
    /// Minimum fraction of pixels whose gradients align with the seed normal.
    pub region_min_alignment: f32,
    /// Minimum refined segment length (px) required to accept the fit.
    pub min_length_px: f32,
    /// Optional cap on the normal span of the grown region.
    pub normal_span_limit_px: Option<f32>,
    /// Maximum number of seed samples evaluated along the predicted segment.
    pub max_seeds: usize,
}

impl Default for SegmentRefineParams {
    fn default() -> Self {
        Self {
            anchor_search_radius: 3.0,
            anchor_step: 0.5,
            gradient_threshold: 0.08,
            region_padding_px: 6.0,
            region_angle_tolerance_deg: 18.0,
            region_min_pixels: 16,
            region_max_pixels: 512,
            region_min_alignment: 0.55,
            min_length_px: 4.0,
            normal_span_limit_px: None,
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
    /// Total number of gradient samples queried from the pyramid level.
    pub gradient_samples: usize,
    /// Number of pixels retained in the final grown region.
    pub region_pixels: usize,
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
