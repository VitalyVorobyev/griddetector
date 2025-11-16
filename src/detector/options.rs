//! Parameter types configuring the detector stages.
//!
//! This module groups knobs for the LSD→VP coarse hypothesis, segment
//! filtering, bundling, and the IRLS-based homography refinement.
//!
//! Defaults aim for robust, real-time behaviour at common resolutions. For
//! tuning, start with the LSD thresholds and the refinement schedule.

use crate::refine::RefineOptions;
use crate::pyramid::PyramidOptions;
use crate::segments::LsdOptions;
use nalgebra::Matrix3;
use serde::Deserialize;

/// Detector-wide parameters controlling the multi-stage pipeline.
#[derive(Clone, Debug, Deserialize)]
pub struct GridParams {
    pub pyramid: PyramidOptions,
    /// Grid spacing used by downstream pose estimation (millimetres).
    pub spacing_mm: f32,
    /// Camera intrinsics used to derive the pose from the final homography.
    pub kmtx: Matrix3<f32>,
    /// Minimum grid cells required before declaring success.
    pub min_cells: i32,
    /// Confidence gate applied to the refined homography.
    pub confidence_thresh: f32,
    /// Gradient-driven segment refinement parameters.
    pub refine: RefineOptions,
    /// Parameters exposed by the coarse LSD→VP engine.
    pub lsd: LsdOptions,
    /// Bundling configuration shared by the refinement stages.
    pub bundling_params: BundlingOptions,
    /// Segment outlier rejection prior to refinement.
    pub outlier_filter: OutlierFilterOptions,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            pyramid: PyramidOptions {
                levels: 4,
                blur_levels: 0,
                filter: Default::default(),
            },
            spacing_mm: 5.0,
            kmtx: Matrix3::identity(),
            min_cells: 6,
            confidence_thresh: 0.35,
            refine: RefineOptions::default(),
            lsd: LsdOptions::default(),
            bundling_params: BundlingOptions::default(),
            outlier_filter: OutlierFilterOptions::default(),
        }
    }
}

/// Bundling parameters shared by the detector and refiner.
///
/// - `orientation_tol_deg`: angular proximity used when aggregating lines.
/// - `merge_dist_px`: maximum |c| offset difference in the normal form
///   `ax + by + c = 0` to consider two constraints co-located.
/// - `min_weight`: minimum segment strength required to contribute.
#[derive(Clone, Debug, Deserialize)]
pub struct BundlingOptions {
    pub orientation_tol_deg: f32,
    pub merge_dist_px: f32,
    pub min_weight: f32,
}

impl Default for BundlingOptions {
    fn default() -> Self {
        Self {
            orientation_tol_deg: 22.5,
            merge_dist_px: 1.5,
            min_weight: 3.0,
        }
    }
}

/// Configuration for segment outlier rejection against the coarse homography.
///
/// Filters coarse segments prior to refinement by combining an angular
/// check (relative to the H-implied vanishing directions) and a residual
/// check that evaluates how well the segment line intersects the family VP.
#[derive(Clone, Debug, Deserialize)]
pub struct OutlierFilterOptions {
    /// Additional angular margin (degrees) beyond the LSD tolerance.
    pub angle_margin_deg: f32,
}

impl Default for OutlierFilterOptions {
    fn default() -> Self {
        Self {
            angle_margin_deg: 8.0,
        }
    }
}
