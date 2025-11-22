use serde::{Deserialize, Serialize};

/// Options controlling region growth heuristics in the LSD-like extractor.
///
/// - `enforce_polarity`: Compare signed angles (no pi folding) during growth to
///   avoid fusing opposite-polarity parallel edges.
/// - `normal_span_limit`: Cap the perpendicular thickness of a grown region by
///   rejecting segments whose span along the fitted normal exceeds the value.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LsdOptions {
    /// Minimum gradient magnitude for seed pixels in the coarsest level (Sobel units).
    pub magnitude_threshold: f32,
    /// Orientation tolerance around the seed normal in degrees.
    pub angle_tolerance_deg: f32,
    /// Minimum accepted segment length in pixels at the current level.
    pub min_length_px: f32,
    /// If true, disallow merging gradients that flip polarity (180° apart).
    pub enforce_polarity: bool,
    /// Optional maximum span (in pixels) along the segment normal.
    pub normal_span_limit_px: Option<f32>,
    pub min_aligned_fraction: f32,
}

impl Default for LsdOptions {
    fn default() -> Self {
        Self {
            magnitude_threshold: 0.05,
            angle_tolerance_deg: 22.5,
            // Keep default minimum segment length permissive at coarse scales
            // to match pre-refactor engine behavior and tests.
            min_length_px: 4.0,
            enforce_polarity: false,
            normal_span_limit_px: None,
            min_aligned_fraction: 0.6,
        }
    }
}

impl LsdOptions {
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.min_length_px *= scale;
        self
    }
}
