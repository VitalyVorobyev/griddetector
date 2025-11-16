//! Detector pipeline orchestrating end-to-end grid detection.
//!
//! The [`GridDetector`] exposes a simple API: feed a grayscale image and get a
//! coarse-to-fine homography estimate with detailed diagnostics. Internally it
//! coordinates the LSD→VP coarse hypothesis, outlier filtering, preparation
//! (per-level bundling + segment refinement), an IRLS homography update, and
//! a final grid indexing pass in the rectified frame.
//!
//! Typical usage:
//! ```no_run
//! use grid_detector::{GridDetector, GridParams};
//! use grid_detector::image::ImageU8;
//!
//! # fn example(gray: ImageU8) {
//! let mut detector = GridDetector::new(GridParams::default());
//! let report = detector.process(gray);
//! if report.grid.found {
//!     println!("confidence: {:.3}", report.grid.confidence);
//! }
//! # }
//! ```

// mod outliers;
// mod reporting;

// Stages
// - Pyramid: build multi-level grayscale pyramid (float levels with optional blur).
// - LSD→VP: detect segments on the coarsest level and estimate a coarse homography.
// - Outlier filter: reject coarse segments against the VP geometry.
// - Prepare: for each level from coarse→fine, bundle constraints and refine segments forward.
// - Refine: run IRLS (`refinement::refine_homography`) across prepared levels; update confidence.
// - Indexing: map bundles to discrete U/V grid lines (`refinement::index_grid_from_bundles`).
// - Reporting: assemble timings, diagnostics, and (optionally) derive camera pose.
//
// Submodules
// - `lsd`: lightweight LSD-like and VP inference.
// - `outliers`: coarse segment classification.
// - `bundling`: image/rectified bundling strategies + coarsest convenience.
// - `refinement::prepare`: per-level bundling + segment refinement.
// - `refinement::homography`: IRLS-driven homography update and confidence blend.
// - `refinement::indexing`: bundle-to-grid indexing in rectified space.

use super::options::GridParams;
use crate::edges::Grad;
use crate::image::ImageU8;
use crate::pyramid::build_pyramid;
use crate::refine::refine_coarse_segments;
use crate::segments::lsd_extract_segments_coarse;
use crate::types::GridResult;
use std::time::Instant;

/// Grid detector orchestrating pyramid construction, LSD→VP, outlier
/// filtering, coarse-to-fine segment refinement and homography IRLS.
pub struct GridDetector {
    params: GridParams,
}

impl GridDetector {
    /// Create a detector with the supplied parameters.
    pub fn new(params: GridParams) -> Self {
        Self { params }
    }

    /// Run the full coarse-to-fine pipeline and capture detailed diagnostics.
    pub fn process(&mut self, gray: ImageU8) -> GridResult {
        let total_start = Instant::now();

        let pyramid = build_pyramid(gray, self.params.pyramid);
        let mut lsd = lsd_extract_segments_coarse(&pyramid.pyramid, self.params.lsd);
        let refine = refine_coarse_segments(
            &pyramid.pyramid,
            &lsd.segments,
            &self.params.refine,
            Some(lsd.grad),
        );
        let elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        lsd.grad = Grad::default();

        GridResult {
            pyramid,
            lsd,
            refine,
            elapsed_ms,
            ..GridResult::default()
        }
    }
}
