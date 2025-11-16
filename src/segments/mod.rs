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
mod options;
mod region_accumulator;
mod segment;

pub use extractor::LsdResult;
pub use options::LsdOptions;
pub use segment::{Segment, SegmentId};

use crate::edges::Grad;
use crate::image::ImageF32;
use crate::pyramid::Pyramid;

/// Lightweight LSD-like extractor (region growing on gradient orientation, PCA fit, simple significance test)
pub fn lsd_extract_segments(l: &ImageF32, options: LsdOptions) -> LsdResult {
    extractor::LsdExtractor::new(l, options).extract()
}

pub fn lsd_extract_segments_coarse(pyramid: &Pyramid, options: LsdOptions) -> LsdResult {
    if let Some(coarse_level) = pyramid.levels.last() {
        let scale = pyramid.scale_for_level(coarse_level);
        extractor::LsdExtractor::new(coarse_level, options.with_scale(scale)).extract()
    } else {
        LsdResult {
            segments: Vec::new(),
            grad: Grad::default(),
            elapsed_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests;
