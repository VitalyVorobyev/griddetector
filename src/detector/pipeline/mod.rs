//! Detector pipeline driving the grid detection end-to-end.
//!
//! The [`GridDetector`] exposes a simple API: feed a grayscale image and get a
//! coarse-to-fine homography estimate with detailed diagnostics. Internally it
//! coordinates the LSD→VP coarse hypothesis, outlier filtering, segment
//! refinement across the pyramid, bundling, and a Huber-weighted IRLS update
//! of the vanishing-point columns.
//!
//! Typical usage:
//! ```no_run
//! use grid_detector::{GridDetector, GridParams};
//! use grid_detector::image::ImageU8;
//!
//! # fn example(gray: ImageU8) {
//! let mut detector = GridDetector::new(GridParams::default());
//! let report = detector.process_with_diagnostics(gray);
//! if report.grid.found {
//!     println!("confidence: {:.3}", report.grid.confidence);
//! }
//! # }
//! ```
mod bundling;
mod lsd;
mod outliers;
mod refinement;
mod reporting;

use super::params::{
    BundlingParams, GridParams, LsdVpParams, OutlierFilterParams, RefinementSchedule,
};
use super::workspace::DetectorWorkspace;
use crate::diagnostics::{
    BundlingStage, DetectionReport, InputDescriptor, PipelineTrace, PyramidStage, TimingBreakdown,
};
use crate::image::ImageU8;
use crate::lsd_vp::Engine as LsdVpEngine;
use crate::pyramid::{Pyramid, PyramidOptions};
use crate::refine::segment::RefineParams as SegmentRefineParams;
use crate::refine::RefineParams as HomographyRefineParams;
use crate::refine::Refiner;
use crate::segments::{Bundle, Segment};
use crate::types::GridResult;
use bundling::BundleStack;
use log::debug;
use lsd::LsdComputation;
use nalgebra::Matrix3;
use outliers::OutlierComputation;
use refinement::{PreparedLevels, RefinementComputation};
use reporting::pose_from_h;
use std::time::Instant;

/// Grid detector orchestrating pyramid construction, LSD→VP, outlier
/// filtering, coarse-to-fine segment refinement and homography IRLS.
pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
    refiner: Refiner,
    lsd_engine: LsdVpEngine,
    workspace: DetectorWorkspace,
}
struct PyramidBuildResult {
    pyramid: Pyramid,
    stage: Option<PyramidStage>,
    elapsed_ms: f64,
}

impl GridDetector {
    /// Create a detector with the supplied parameters.
    pub fn new(params: GridParams) -> Self {
        let refiner = Refiner::new(params.refine_params.clone());
        let lsd_engine = lsd::make_engine(&params.lsd_vp_params);
        Self {
            params,
            last_hmtx: None,
            refiner,
            lsd_engine,
            workspace: DetectorWorkspace::new(),
        }
    }

    /// Run the detector on a grayscale image, returning a compact result.
    pub fn process(&mut self, gray: ImageU8) -> GridResult {
        self.process_full_with_diagnostics(gray).grid
    }

    /// Run the detector and return both the result and a detailed report.
    pub fn process_with_diagnostics(&mut self, gray: ImageU8) -> DetectionReport {
        self.process_full_with_diagnostics(gray)
    }

    /// Run the full coarse-to-fine pipeline and capture detailed diagnostics.
    pub fn process_full_with_diagnostics(&mut self, gray: ImageU8) -> DetectionReport {
        let (width, height) = (gray.w, gray.h);
        debug!(
            "GridDetector::process start w={} h={} levels={}",
            width, height, self.params.pyramid_levels
        );
        let total_start = Instant::now();

        let PyramidBuildResult {
            pyramid,
            stage: pyramid_stage,
            elapsed_ms: pyr_ms,
        } = self.build_pyramid(gray);

        let bundler = BundleStack::new(&self.params.bundling_params);
        let LsdComputation {
            stage: mut lsd_stage,
            segments: coarse_segments,
            coarse_h,
            full_h,
            mut confidence,
            elapsed_ms: lsd_ms,
        } = lsd::run_on_coarsest(&self.lsd_engine, &pyramid, width, height);
        let segment_trace = coarse_segments.clone();

        let initial_h_full = full_h.clone();
        let coarse_h_matrix = full_h.clone();

        if let Some(stage) = lsd_stage.as_mut() {
            stage.elapsed_ms = lsd_ms;
        }

        let OutlierComputation {
            stage: outlier_stage,
            segments: filtered_segments,
            elapsed_ms: outlier_filter_ms,
        } = outliers::filter_segments(
            &coarse_segments,
            coarse_h.as_ref(),
            &self.params.outlier_filter,
            &self.params.lsd_vp_params,
        );

        let mut prepared_levels = PreparedLevels::empty();
        if !filtered_segments.is_empty() && !pyramid.levels.is_empty() {
            self.workspace.reset(pyramid.levels.len());
            prepared_levels = refinement::prepare_levels(
                &mut self.workspace,
                &bundler,
                &self.params.segment_refine_params,
                &pyramid,
                full_h.as_ref(),
                filtered_segments.clone(),
                width,
                height,
            );
        }

        let bundling_stage = refinement::build_bundling_stage(
            &self.params.bundling_params,
            &prepared_levels,
            filtered_segments.len(),
        );

        let RefinementComputation {
            hmtx: refined_h,
            confidence: refined_confidence,
            stage: refinement_stage,
            elapsed_ms: refine_ms,
        } = refinement::run_refinement_stage(
            &mut self.refiner,
            &prepared_levels,
            initial_h_full,
            confidence,
            &self.params.refinement_schedule,
            self.params.enable_refine,
            self.last_hmtx,
        );

        let hmtx = refined_h;
        confidence = refined_confidence;

        let mut origin_uv = (0, 0);
        let mut visible_range = (0, 0, 0, 0);
        let grid_indexing_stage = if let Some(level) = prepared_levels.levels.first() {
            let stage = refinement::run_grid_indexing_stage(
                level.bundles.as_slice(),
                Some(&hmtx),
                self.params.refine_params.orientation_tol_deg.to_radians(),
            );
            if let Some(stage_ref) = stage.as_ref() {
                origin_uv = stage_ref.origin_uv;
                visible_range = stage_ref.visible_range;
            }
            stage
        } else {
            None
        };

        if hmtx != Matrix3::identity() {
            self.last_hmtx = Some(hmtx);
        }

        let found = hmtx != Matrix3::identity() && confidence >= self.params.confidence_thresh;
        let pose = if found {
            Some(pose_from_h(self.params.kmtx, hmtx))
        } else {
            None
        };
        let latency = total_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "GridDetector::process done found={} confidence={:.3} latency_ms={:.3}",
            found, confidence, latency
        );
        let result = GridResult {
            found,
            hmtx,
            pose: pose.clone(),
            origin_uv,
            visible_range,
            coverage: 0.0,
            reproj_rmse: 0.0,
            confidence,
            latency_ms: latency,
        };

        let mut timings = TimingBreakdown::with_total(latency);
        if pyr_ms > 0.0 {
            timings.push("pyramid", pyr_ms);
        }
        if lsd_ms > 0.0 {
            timings.push("lsd_vp", lsd_ms);
        }
        if outlier_filter_ms > 0.0 {
            timings.push("outlier_filter", outlier_filter_ms);
        }
        if prepared_levels.segment_refine_ms > 0.0 {
            timings.push("segment_refine", prepared_levels.segment_refine_ms);
        }
        if prepared_levels.bundling_ms > 0.0 {
            timings.push("bundling", prepared_levels.bundling_ms);
        }
        if refine_ms > 0.0 {
            timings.push("refinement", refine_ms);
        }
        if let Some(stage) = grid_indexing_stage.as_ref() {
            if stage.elapsed_ms > 0.0 {
                timings.push("grid_indexing", stage.elapsed_ms);
            }
        }

        let trace = PipelineTrace {
            input: InputDescriptor {
                width,
                height,
                pyramid_levels: pyramid.levels.len(),
            },
            timings,
            segments: segment_trace,
            pyramid: pyramid_stage,
            lsd: lsd_stage,
            outlier_filter: outlier_stage,
            bundling: bundling_stage,
            grid_indexing: grid_indexing_stage,
            refinement: refinement_stage,
            coarse_homography: coarse_h_matrix,
            pose: pose.clone(),
        };

        DetectionReport {
            grid: result,
            trace,
        }
    }

    fn build_pyramid(&self, gray: ImageU8) -> PyramidBuildResult {
        let pyr_start = Instant::now();
        let pyramid_opts = match self.params.pyramid_blur_levels {
            Some(blur_levels) => {
                PyramidOptions::new(self.params.pyramid_levels).with_blur_levels(Some(blur_levels))
            }
            None => PyramidOptions::new(self.params.pyramid_levels),
        };
        let pyramid = Pyramid::build_u8(gray, pyramid_opts);
        let elapsed_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
        let stage = if pyramid.levels.is_empty() {
            None
        } else {
            Some(PyramidStage::from_pyramid(&pyramid, elapsed_ms))
        };
        PyramidBuildResult {
            pyramid,
            stage,
            elapsed_ms,
        }
    }

    /// Execute the coarse-only pipeline (pyramid → LSD → outlier filter → bundling) with diagnostics.
    pub fn process_coarsest_with_diagnostics(&mut self, gray: ImageU8) -> DetectionReport {
        let (width, height) = (gray.w, gray.h);
        debug!(
            "GridDetector::process_coarsest start w={} h={} levels={}",
            width, height, self.params.pyramid_levels
        );
        let total_start = Instant::now();

        let PyramidBuildResult {
            pyramid,
            stage: pyramid_stage,
            elapsed_ms: pyr_ms,
        } = self.build_pyramid(gray);

        let bundler = BundleStack::new(&self.params.bundling_params);
        let LsdComputation {
            stage: mut lsd_stage,
            segments: coarse_segments,
            coarse_h,
            full_h,
            confidence,
            elapsed_ms: lsd_ms,
        } = lsd::run_on_coarsest(&self.lsd_engine, &pyramid, width, height);
        let segment_trace = coarse_segments.clone();

        if let Some(stage) = lsd_stage.as_mut() {
            stage.elapsed_ms = lsd_ms;
        }

        let OutlierComputation {
            stage: outlier_stage,
            segments: filtered_segments,
            elapsed_ms: outlier_filter_ms,
        } = outliers::filter_segments(
            &coarse_segments,
            coarse_h.as_ref(),
            &self.params.outlier_filter,
            &self.params.lsd_vp_params,
        );

        let mut bundling_stage = None;
        let mut coarse_bundles: Vec<Bundle> = Vec::new();
        if let Some((stage, bundles)) = refinement::bundle_coarsest(
            &bundler,
            &pyramid,
            coarse_h.as_ref(),
            &filtered_segments,
            width,
            height,
        ) {
            coarse_bundles = bundles;
            bundling_stage = Some(stage);
        }
        let bundling_ms = bundling_stage
            .as_ref()
            .map(|stage| stage.elapsed_ms)
            .unwrap_or(0.0);

        let hmtx = full_h.unwrap_or_else(Matrix3::identity);
        let coarse_h_matrix = full_h;

        if hmtx != Matrix3::identity() {
            self.last_hmtx = Some(hmtx);
        }

        let mut origin_uv = (0, 0);
        let mut visible_range = (0, 0, 0, 0);
        let grid_indexing_stage = refinement::run_grid_indexing_stage(
            &coarse_bundles,
            full_h.as_ref(),
            self.params.refine_params.orientation_tol_deg.to_radians(),
        );
        if let Some(stage) = grid_indexing_stage.as_ref() {
            origin_uv = stage.origin_uv;
            visible_range = stage.visible_range;
        }

        let found = hmtx != Matrix3::identity() && confidence >= self.params.confidence_thresh;
        let pose = if found {
            Some(pose_from_h(self.params.kmtx, hmtx))
        } else {
            None
        };
        let latency = total_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "GridDetector::process_coarsest done found={} confidence={:.3} latency_ms={:.3}",
            found, confidence, latency
        );

        let result = GridResult {
            found,
            hmtx,
            pose: pose.clone(),
            origin_uv,
            visible_range,
            coverage: 0.0,
            reproj_rmse: 0.0,
            confidence,
            latency_ms: latency,
        };

        let mut timings = TimingBreakdown::with_total(latency);
        if pyr_ms > 0.0 {
            timings.push("pyramid", pyr_ms);
        }
        if lsd_ms > 0.0 {
            timings.push("lsd_vp", lsd_ms);
        }
        if outlier_filter_ms > 0.0 {
            timings.push("outlier_filter", outlier_filter_ms);
        }
        if bundling_ms > 0.0 {
            timings.push("bundling", bundling_ms);
        }
        if let Some(stage) = grid_indexing_stage.as_ref() {
            if stage.elapsed_ms > 0.0 {
                timings.push("grid_indexing", stage.elapsed_ms);
            }
        }

        let trace = PipelineTrace {
            input: InputDescriptor {
                width,
                height,
                pyramid_levels: pyramid.levels.len(),
            },
            timings,
            segments: segment_trace,
            pyramid: pyramid_stage,
            lsd: lsd_stage,
            outlier_filter: outlier_stage,
            bundling: bundling_stage,
            grid_indexing: grid_indexing_stage,
            refinement: None,
            coarse_homography: coarse_h_matrix,
            pose: pose.clone(),
        };

        DetectionReport {
            grid: result,
            trace,
        }
    }

    /// Bundle coarse segments on the coarsest pyramid level and rescale bundles to full resolution.
    pub fn bundle_coarsest(
        &self,
        pyramid: &Pyramid,
        coarse_h: Option<&Matrix3<f32>>,
        segments: &[Segment],
        full_width: usize,
        full_height: usize,
    ) -> Option<(BundlingStage, Vec<Bundle>)> {
        let bundler = BundleStack::new(&self.params.bundling_params);
        refinement::bundle_coarsest(
            &bundler,
            pyramid,
            coarse_h,
            segments,
            full_width,
            full_height,
        )
    }

    /// Update the camera intrinsics used for pose recovery.
    pub fn set_intrinsics(&mut self, k: Matrix3<f32>) {
        self.params.kmtx = k;
    }

    /// Update the assumed grid spacing (millimetres).
    pub fn set_spacing(&mut self, s_mm: f32) {
        self.params.spacing_mm = s_mm;
    }

    /// Update IRLS parameters for the homography refinement.
    pub fn set_refine_params(&mut self, params: HomographyRefineParams) {
        self.params.refine_params = params.clone();
        self.refiner = Refiner::new(params);
    }

    /// Update gradient-driven segment refinement parameters.
    pub fn set_segment_refine_params(&mut self, params: SegmentRefineParams) {
        self.params.segment_refine_params = params;
    }

    /// Update LSD→VP coarse stage parameters.
    pub fn set_lsd_vp_params(&mut self, params: LsdVpParams) {
        self.params.lsd_vp_params = params.clone();
        self.lsd_engine = lsd::make_engine(&params);
    }

    /// Update bundling parameters (orientation/distance/scale mode).
    pub fn set_bundling_params(&mut self, params: BundlingParams) {
        self.params.bundling_params = params;
    }

    /// Update coarse segment outlier filter thresholds.
    pub fn set_outlier_filter(&mut self, params: OutlierFilterParams) {
        self.params.outlier_filter = params;
    }

    /// Update the refinement schedule (number of passes and threshold).
    pub fn set_refinement_schedule(&mut self, schedule: RefinementSchedule) {
        self.params.refinement_schedule = schedule;
    }
}
