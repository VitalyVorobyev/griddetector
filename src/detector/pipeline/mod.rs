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
mod reporting;

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

use super::options::{BundlingOptions, GridParams, OutlierFilterOptions};
use crate::image::ImageU8;
use crate::pyramid::{Pyramid};
use crate::refine::RefineOptions;
use crate::segments::{LsdOptions, LsdResult, lsd_extract_segments_coarse};
use crate::types::GridResult;
use nalgebra::Matrix3;
use std::time::Instant;

/// Grid detector orchestrating pyramid construction, LSD→VP, outlier
/// filtering, coarse-to-fine segment refinement and homography IRLS.
pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
}
struct PyramidBuildResult {
    pyramid: Pyramid,
    elapsed_ms: f64,
}

impl GridDetector {
    /// Create a detector with the supplied parameters.
    pub fn new(params: GridParams) -> Self {
        Self {
            params,
            last_hmtx: None,
        }
    }

    /// Run the full coarse-to-fine pipeline and capture detailed diagnostics.
    pub fn process(&mut self, gray: ImageU8) -> GridResult {
        let total_start = Instant::now();

        let PyramidBuildResult {
            pyramid,
            elapsed_ms: pyr_ms,
        } = self.build_pyramid(gray);

        let LsdResult {
            segments: coarse_segments,
            grad,
            elapsed_ms: lsd_ms,
        } = lsd_extract_segments_coarse(&pyramid, self.params.lsd_params);

        let elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let grid = GridResult::default();
        grid
    }

    fn build_pyramid(&self, gray: ImageU8) -> PyramidBuildResult {
        let pyr_start = Instant::now();
        let pyramid = Pyramid::build_u8(gray, self.params.pyramid);
        let elapsed_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
        PyramidBuildResult {
            pyramid,
            elapsed_ms,
        }
    }

    /// Update the camera intrinsics used for pose recovery.
    pub fn set_intrinsics(&mut self, k: Matrix3<f32>) {
        self.params.kmtx = k;
    }

    /// Update the assumed grid spacing (millimetres).
    pub fn set_spacing(&mut self, s_mm: f32) {
        self.params.spacing_mm = s_mm;
    }

    /// Update gradient-driven segment refinement parameters.
    pub fn set_segment_refine_params(&mut self, params: RefineOptions) {
        self.params.segment_refine_params = params;
    }

    /// Update LSD coarse stage parameters.
    pub fn set_lsd_params(&mut self, params: LsdOptions) {
        self.params.lsd_params = params;
    }

    /// Update bundling parameters (orientation/distance/scale mode).
    pub fn set_bundling_params(&mut self, params: BundlingOptions) {
        self.params.bundling_params = params;
    }

    /// Update coarse segment outlier filter thresholds.
    pub fn set_outlier_filter(&mut self, params: OutlierFilterOptions) {
        self.params.outlier_filter = params;
    }
}
