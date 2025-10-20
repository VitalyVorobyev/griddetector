//! Lightweight LSD-like segment extractor.
//!
//! This module implements a fast, edge-based line-segment extractor inspired by
//! LSD (Line Segment Detector) but tailored for grid/chessboard detection and
//! multi-scale refinement. The algorithm performs:
//!
//! - Gradient computation (via `edges::sobel_gradients`), producing per-pixel
//!   `gx`, `gy`, magnitude, and implicitly an orientation.
//! - Region growing from seeds using orientation consistency: pixels whose
//!   gradient orientation is within a tolerance of the seed normal are grown
//!   into a region, while enforcing a minimum gradient magnitude.
//! - PCA line fitting: the pixel coordinates of a grown region are summarized
//!   online and a 2x2 covariance matrix is eigendecomposed to obtain the
//!   principal direction. This yields a robust tangent direction for the line.
//! - Endpoint projection and normal form: by projecting region points onto the
//!   principal axis we obtain endpoints `p0` and `p1`. The line is stored in
//!   normalized normal form `ax + by + c = 0` with `sqrt(a^2+b^2)=1`.
//! - Significance tests: require a minimum region size, minimum length, and a
//!   minimum fraction of pixels aligned with the seed orientation.
//!
//! Output segments include auxiliary attributes used by refinement/bundling:
//! - `len`: endpoint distance along the tangent.
//! - `avg_mag`: average gradient magnitude over the region.
//! - `strength`: `len * avg_mag` (proxy for saliency used as a weight).
//!
//! Notes
//! - Orientation is taken modulo pi (180°) by default, appropriate for grid
//!   lines where directionality is ambiguous. See `angle::normalize_half_pi`.
//!   You can enable polarity-gated growth to require consistent signed
//!   gradients and prevent merging opposite-polarity parallel edges.
//! - The extractor is designed to be lightweight rather than exhaustive; it's
//!   biased toward long, coherent edges that are useful for vanishing points
//!   and later refinement.
//! - Parameters are expressed in the current pyramid level’s pixel scale; when
//!   used across scales, callers should adapt thresholds accordingly.
//! - An optional normal-span limit can reject grown regions that are too thick
//!   across the fitted line, mitigating double-ridge merges (e.g., Charuco).
//!
//! Complexity
//! - Region growing visits each pixel at most once, giving O(W*H) behavior per
//!   level; PCA fitting and endpoint estimation are linear in region size.
//!
//! See also
//! - `crate::lsd_vp` for orientation clustering and VP estimation.
//! - `crate::refine` for coarse-to-fine Huber-weighted refinement using bundles.

mod extractor;
pub mod types;

pub use types::{LsdOptions, Segment};

use crate::image::ImageF32;

/// Lightweight LSD-like extractor (region growing on gradient orientation, PCA fit, simple significance test)
pub fn lsd_extract_segments(
    l: &ImageF32,
    mag_thresh: f32, // min gradient magnitude (0..1 scale at this pyramid level)
    angle_tol: f32,  // radians, tolerance around seed normal angle
    min_len: f32,    // min length in pixels at this level
) -> Vec<Segment> {
    lsd_extract_segments_with_options(l, mag_thresh, angle_tol, min_len, LsdOptions::default())
}

/// Same as [`lsd_extract_segments`] but allows passing custom growth options.
pub fn lsd_extract_segments_with_options(
    l: &ImageF32,
    mag_thresh: f32,
    angle_tol: f32,
    min_len: f32,
    options: LsdOptions,
) -> Vec<Segment> {
    lsd_extract_segments_masked_with_options(l, mag_thresh, angle_tol, min_len, None, options)
}

/// Same as [`lsd_extract_segments`] but restricts seeds and region growth to pixels where `mask == 1`.
pub fn lsd_extract_segments_masked(
    l: &ImageF32,
    mag_thresh: f32,
    angle_tol: f32,
    min_len: f32,
    mask: Option<&[u8]>,
) -> Vec<Segment> {
    lsd_extract_segments_masked_with_options(
        l,
        mag_thresh,
        angle_tol,
        min_len,
        mask,
        LsdOptions::default(),
    )
}

/// Masked variant with explicit options.
pub fn lsd_extract_segments_masked_with_options(
    l: &ImageF32,
    mag_thresh: f32,
    angle_tol: f32,
    min_len: f32,
    mask: Option<&[u8]>,
    options: LsdOptions,
) -> Vec<Segment> {
    extractor::LsdExtractor::new(l, mag_thresh, angle_tol, min_len, mask, options).extract()
}

#[cfg(test)]
mod tests;
