//! Parameter types configuring the detector stages.
//!
//! This module groups knobs for the LSD→VP coarse hypothesis, segment
//! filtering, bundling, and the IRLS-based homography refinement.
//!
//! Defaults aim for robust, real-time behaviour at common resolutions. For
//! tuning, start with the LSD thresholds and the refinement schedule.

use crate::refine::segment::RefineParams as SegmentRefineParams;
use crate::refine::RefineParams as HomographyRefineParams;
use crate::segments::LsdOptions;
use nalgebra::Matrix3;
use serde::Deserialize;

/// Detector-wide parameters controlling the multi-stage pipeline.
#[derive(Clone, Debug, Deserialize)]
pub struct GridParams {
    /// Number of pyramid levels (>=1).
    pub pyramid_levels: usize,
    /// Optional number of levels that apply Gaussian blur before decimation.
    /// `None` falls back to legacy behaviour (blur at every step).
    pub pyramid_blur_levels: usize,
    /// Grid spacing used by downstream pose estimation (millimetres).
    pub spacing_mm: f32,
    /// Camera intrinsics used to derive the pose from the final homography.
    pub kmtx: Matrix3<f32>,
    /// Minimum grid cells required before declaring success.
    pub min_cells: i32,
    /// Confidence gate applied to the refined homography.
    pub confidence_thresh: f32,
    /// Enables or disables coarse-to-fine refinement.
    pub enable_refine: bool,
    /// Controls how many refinement passes are attempted.
    pub refinement_schedule: RefinementSchedule,
    /// IRLS parameters for the homography refinement.
    pub refine_params: HomographyRefineParams,
    /// Gradient-driven segment refinement parameters.
    pub segment_refine_params: SegmentRefineParams,
    /// Parameters exposed by the coarse LSD→VP engine.
    pub lsd_params: LsdOptions,
    /// Bundling configuration shared by the refinement stages.
    pub bundling_params: BundlingParams,
    /// Segment outlier rejection prior to refinement.
    pub outlier_filter: OutlierFilterParams,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            pyramid_blur_levels: 0,
            spacing_mm: 5.0,
            kmtx: Matrix3::identity(),
            min_cells: 6,
            confidence_thresh: 0.35,
            enable_refine: true,
            refinement_schedule: RefinementSchedule::default(),
            refine_params: HomographyRefineParams::default(),
            segment_refine_params: SegmentRefineParams::default(),
            lsd_params: LsdOptions::default(),
            bundling_params: BundlingParams::default(),
            outlier_filter: OutlierFilterParams::default(),
        }
    }
}

/// Bundling parameters shared by the detector and refiner.
///
/// - `orientation_tol_deg`: angular proximity used when aggregating lines.
/// - `merge_dist_px`: maximum |c| offset difference in the normal form
///   `ax + by + c = 0` to consider two constraints co-located.
/// - `min_weight`: minimum segment strength required to contribute.
/// - `scale_mode`: whether thresholds are interpreted at full-resolution.
#[derive(Clone, Debug, Deserialize)]
pub struct BundlingParams {
    pub orientation_tol_deg: f32,
    pub merge_dist_px: f32,
    pub min_weight: f32,
}

impl Default for BundlingParams {
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
pub struct OutlierFilterParams {
    /// Additional angular margin (degrees) beyond the LSD tolerance.
    pub angle_margin_deg: f32,
}

impl Default for OutlierFilterParams {
    fn default() -> Self {
        Self {
            angle_margin_deg: 8.0,
        }
    }
}

/// Controls multiple refinement passes.
///
/// A value of `passes = 1` preserves legacy behaviour. Use `passes = 2`
/// to allow one more update when the homography significantly improves.
#[derive(Clone, Debug, Deserialize)]
pub struct RefinementSchedule {
    /// Maximum number of passes (>=1). `1` preserves legacy single pass.
    pub passes: usize,
    /// Minimum Frobenius improvement required to launch another pass.
    pub improvement_thresh: f32,
}

impl Default for RefinementSchedule {
    fn default() -> Self {
        Self {
            passes: 1,
            improvement_thresh: 5e-4,
        }
    }
}
