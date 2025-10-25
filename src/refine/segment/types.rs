//! Public types used by the gradient-driven segment refiner.

/// Single pyramid level with precomputed Sobel/Scharr gradients.
#[derive(Clone, Copy, Debug)]
pub struct PyramidLevel<'a> {
    pub width: usize,
    pub height: usize,
    pub gx: &'a [f32],
    pub gy: &'a [f32],
}

/// Seed line segment expressed via two subpixel endpoints.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

impl Segment {
    /// Returns the unit direction vector from `p0` to `p1`.
    #[inline]
    pub fn dir(&self) -> [f32; 2] {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        let len = (dx * dx + dy * dy).sqrt().max(super::EPS);
        [dx / len, dy / len]
    }

    /// Returns the Euclidean length of the segment.
    #[inline]
    pub fn length(&self) -> f32 {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// Parameters controlling the gradient-driven refinement.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Along-segment sample spacing (px) used for normal probing.
    pub delta_s: f32,
    /// Half-width (px) of the normal search corridor.
    pub w_perp: f32,
    /// Step size (px) for sweeping along the normal.
    pub delta_t: f32,
    /// Padding (px) added around the segment when forming the ROI.
    pub pad: f32,
    /// Minimum gradient magnitude accepted as support.
    pub tau_mag: f32,
    /// Orientation tolerance (degrees) applied during endpoint gating.
    pub tau_ori_deg: f32,
    /// Huber delta used to weight the normal projected gradient.
    pub huber_delta: f32,
    /// Maximum number of outer carrier update iterations.
    pub max_iters: usize,
    /// Minimum ratio between refined and seed length required for acceptance.
    pub min_inlier_frac: f32,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            delta_s: 0.75,
            w_perp: 3.0,
            delta_t: 0.5,
            pad: 8.0,
            tau_mag: 0.1,
            tau_ori_deg: 25.0,
            huber_delta: 0.25,
            max_iters: 3,
            min_inlier_frac: 0.4,
        }
    }
}

/// Outcome of a refinement attempt.
#[derive(Clone, Debug)]
pub struct RefineResult {
    /// Refined segment (or the fallback seed when `ok == false`).
    pub seg: Segment,
    /// Mean absolute normal-projected gradient across inlier samples.
    pub score: f32,
    /// Whether the refinement satisfied the length and score thresholds.
    pub ok: bool,
    /// Number of inlier samples supporting the endpoints.
    pub inliers: usize,
    /// Total number of centreline samples considered.
    pub total: usize,
}

impl RefineResult {
    pub(crate) fn failed(seg: Segment) -> Self {
        Self {
            seg,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        }
    }

    pub(crate) fn rejected(seg: Segment, inliers: usize, total: usize, score: f32) -> Self {
        Self {
            seg,
            score,
            ok: false,
            inliers,
            total,
        }
    }
}

/// Coordinate mapping from pyramid level `l+1` to level `l`.
///
/// Implementors can model pure dyadic scaling, fractional pixel offsets, or
/// any bespoke decimation geometry used to build the image pyramid.
pub trait ScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2];
}
