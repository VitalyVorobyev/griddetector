use nalgebra::Vector3;

/// Line segment produced by the LSD-like extractor.
#[derive(Clone, Debug)]
pub struct Segment {
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
#[derive(Clone, Copy, Debug, Default)]
pub struct LsdOptions {
    /// If true, disallow merging gradients that flip polarity (180° apart).
    pub enforce_polarity: bool,
    /// Optional maximum span (in pixels) along the segment normal.
    pub normal_span_limit: Option<f32>,
}
