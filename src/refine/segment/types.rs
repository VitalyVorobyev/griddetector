//! Public types used by the gradient-driven segment refiner.
//!
//! The refiner is driven by image gradients on a single pyramid level. To
//! support both full-frame and cropped gradient tiles, [`PyramidLevel`]
//! describes gradients in a small, contiguous window of the level together with
//! enough metadata to interpret positions in full-image coordinates.

use crate::segments::Segment;
use serde::Deserialize;

/// Single pyramid level with precomputed Sobel/Scharr gradients.
///
/// A `PyramidLevel` does not own the gradient buffers; it is a lightweight
/// view over a **gradient tile** provided by the detector workspace. The tile
/// may either cover the full image or a cropped window around the segments of
/// interest.
///
/// Coordinate conventions:
/// - `(width, height)` describe the full image dimensions at this level.
/// - `(origin_x, origin_y)` give the top-left corner of the tile in full-image
///   coordinates.
/// - `(tile_width, tile_height)` describe the size of the tile in pixels.
/// - `gx`, `gy` store gradients for the tile in row-major order with stride
///   equal to `tile_width`.
///
/// Callers must convert full-image coordinates into tile-relative ones by
/// subtracting `(origin_x, origin_y)` before indexing into `gx`/`gy`. The
/// internal sampling helper takes care of this for subpixel sampling.
#[derive(Clone, Debug)]
pub struct PyramidLevel<'a> {
    /// Full-resolution width of the level (global coordinates).
    pub width: usize,
    /// Full-resolution height of the level (global coordinates).
    pub height: usize,
    /// Left/top origin of the gradient tile within the full-resolution level.
    pub origin_x: usize,
    pub origin_y: usize,
    /// Width/height of the gradient tile.
    pub tile_width: usize,
    pub tile_height: usize,
    /// Horizontal and vertical derivatives stored for the tile.
    pub gx: &'a [f32],
    pub gy: &'a [f32],
    /// Pyramid level index (0 = finest).
    pub level_index: usize,
}

/// Axis-aligned region of interest around a segment in full-resolution coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentRoi {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl SegmentRoi {
    #[inline]
    pub fn contains(&self, p: &[f32; 2]) -> bool {
        p[0] >= self.x0 && p[0] <= self.x1 && p[1] >= self.y0 && p[1] <= self.y1
    }

    #[inline]
    pub fn clamp_inside(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0].clamp(self.x0, self.x1), p[1].clamp(self.y0, self.y1)]
    }
}

/// Parameters controlling the gradient-driven refinement.
#[derive(Clone, Debug, Deserialize)]
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
