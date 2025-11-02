use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Identifier referencing a segment recorded in the pipeline trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub u32);

/// Line segment produced by the LSD-like extractor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub id: SegmentId,
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub dir: [f32; 2],
    pub len: f32,
    pub line: Vector3<f32>, // ax + by + c = 0, with sqrt(a^2+b^2)=1
    pub avg_mag: f32,
    pub strength: f32,
}

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
        }
    }
}

impl LsdOptions {
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.min_length_px *= scale;
        self
    }
}
