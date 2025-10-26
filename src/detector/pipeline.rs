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
use super::outliers::classify_segments_with_details;
use super::params::{
    BundlingParams, BundlingScaleMode, GridParams, LsdVpParams, OutlierFilterParams,
    RefinementSchedule,
};
use super::scaling::{rescale_bundle_to_full_res, LevelScaleMap, LevelScaling};
use super::workspace::DetectorWorkspace;
use crate::diagnostics::builders::convert_refined_segment;
use crate::diagnostics::{
    BundleDescriptor, BundlingLevel, BundlingStage, DetectionReport, FamilyCounts, FamilyIndexing,
    GridIndexingStage, GridLineIndex, InputDescriptor, LsdStage, OutlierFilterStage,
    OutlierThresholds, PipelineTrace, PyramidStage, RefinementOutcome, RefinementStage,
    SegmentDescriptor, SegmentId, SegmentSample, TimingBreakdown,
};
use crate::image::{ImageU8, ImageView};
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine, FamilyLabel};
use crate::pyramid::{Pyramid, PyramidOptions};
use crate::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, RefineParams as SegmentRefineParams,
    Segment as SegmentSeed,
};
use crate::refine::{RefineLevel, RefineParams as HomographyRefineParams, Refiner};
use crate::segments::{bundle_segments, Bundle, Segment};
use crate::types::{GridResult, Pose};
use log::debug;
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::time::Instant;

const EPS: f32 = 1e-6;

/// Grid detector orchestrating pyramid construction, LSD→VP, outlier
/// filtering, coarse-to-fine segment refinement and homography IRLS.
pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
    refiner: Refiner,
    lsd_engine: LsdVpEngine,
    workspace: DetectorWorkspace,
}

struct PreparedLevel {
    level_index: usize,
    level_width: usize,
    level_height: usize,
    segments: usize,
    bundles: Vec<Bundle>,
}

struct PreparedLevels {
    levels: Vec<PreparedLevel>,
    bundling_ms: f64,
    segment_refine_ms: f64,
}

impl PreparedLevels {
    fn empty() -> Self {
        Self {
            levels: Vec::new(),
            bundling_ms: 0.0,
            segment_refine_ms: 0.0,
        }
    }
}

struct PyramidBuildResult {
    pyramid: Pyramid,
    stage: Option<PyramidStage>,
    elapsed_ms: f64,
}

struct LsdComputation {
    stage: Option<LsdStage>,
    descriptors: Vec<SegmentDescriptor>,
    segments: Vec<Segment>,
    coarse_h: Option<Matrix3<f32>>,
    full_h: Option<Matrix3<f32>>,
    confidence: f32,
    elapsed_ms: f64,
}

struct OutlierComputation {
    stage: Option<OutlierFilterStage>,
    segments: Vec<Segment>,
    elapsed_ms: f64,
}

struct RefinementComputation {
    hmtx: Matrix3<f32>,
    confidence: f32,
    stage: Option<RefinementStage>,
    elapsed_ms: f64,
}

impl GridDetector {
    /// Create a detector with the supplied parameters.
    pub fn new(params: GridParams) -> Self {
        let refiner = Refiner::new(params.refine_params.clone());
        let lsd_engine = Self::make_lsd_engine(&params.lsd_vp_params);
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

        let LsdComputation {
            stage: mut lsd_stage,
            descriptors: segment_descriptors,
            segments: coarse_segments,
            coarse_h,
            full_h,
            mut confidence,
            elapsed_ms: lsd_ms,
        } = self.run_lsd_on_coarsest(&pyramid, width, height);

        let initial_h_full = full_h;
        let coarse_h_matrix = full_h;

        if let Some(stage) = lsd_stage.as_mut() {
            stage.elapsed_ms = lsd_ms;
        }

        let OutlierComputation {
            stage: outlier_stage,
            segments: filtered_segments,
            elapsed_ms: outlier_filter_ms,
        } = self.run_outlier_filter_stage(&coarse_segments, coarse_h.as_ref());

        let mut prepared_levels = PreparedLevels::empty();
        if !filtered_segments.is_empty() && !pyramid.levels.is_empty() {
            self.workspace.reset(pyramid.levels.len());
            prepared_levels =
                self.prepare_refinement_levels(&pyramid, filtered_segments.clone(), width, height);
        }

        let bundling_stage = self.build_bundling_stage(&prepared_levels, filtered_segments.len());

        let RefinementComputation {
            hmtx: refined_h,
            confidence: refined_confidence,
            stage: refinement_stage,
            elapsed_ms: refine_ms,
            ..
        } = self.run_refinement_stage(&prepared_levels, initial_h_full, confidence);

        let hmtx = refined_h;
        confidence = refined_confidence;

        let mut origin_uv = (0, 0);
        let mut visible_range = (0, 0, 0, 0);
        let grid_indexing_stage = if let Some(level) = prepared_levels.levels.first() {
            let stage = self.run_grid_indexing_stage(level.bundles.as_slice(), Some(&hmtx));
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
            segments: segment_descriptors,
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

    fn run_lsd_on_coarsest(
        &self,
        pyramid: &Pyramid,
        width: usize,
        height: usize,
    ) -> LsdComputation {
        let Some(coarse_level) = pyramid.levels.last() else {
            debug!("GridDetector::process pyramid has no levels");
            return LsdComputation {
                stage: None,
                descriptors: Vec::new(),
                segments: Vec::new(),
                coarse_h: None,
                full_h: None,
                confidence: 0.0,
                elapsed_ms: 0.0,
            };
        };

        let lsd_start = Instant::now();
        match self.lsd_engine.infer_detailed(coarse_level) {
            Some(DetailedInference {
                hypothesis,
                dominant_angles_rad,
                families,
                segments,
            }) => {
                let mut descriptors = Vec::with_capacity(segments.len());
                for (idx, seg) in segments.iter().enumerate() {
                    descriptors.push(SegmentDescriptor::from_segment(SegmentId(idx as u32), seg));
                }

                let scale_x = if coarse_level.w > 0 {
                    width as f32 / coarse_level.w as f32
                } else {
                    1.0
                };
                let scale_y = if coarse_level.h > 0 {
                    height as f32 / coarse_level.h as f32
                } else {
                    1.0
                };
                let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
                let h_full = if coarse_level.w > 0 && coarse_level.h > 0 {
                    scale * hypothesis.hmtx0
                } else {
                    hypothesis.hmtx0
                };

                let dominant_angles_deg = [
                    dominant_angles_rad[0].to_degrees(),
                    dominant_angles_rad[1].to_degrees(),
                ];
                let family_counts = compute_family_counts(&families);
                let stage = LsdStage {
                    elapsed_ms: 0.0,
                    confidence: hypothesis.confidence,
                    dominant_angles_deg,
                    family_counts,
                    segment_families: families,
                    sample_ids: Vec::new(),
                };

                LsdComputation {
                    stage: Some(stage),
                    descriptors,
                    segments,
                    coarse_h: Some(hypothesis.hmtx0),
                    full_h: Some(h_full),
                    confidence: hypothesis.confidence,
                    elapsed_ms: lsd_start.elapsed().as_secs_f64() * 1000.0,
                }
            }
            None => {
                debug!("GridDetector::process LSD→VP engine returned no hypothesis");
                LsdComputation {
                    stage: None,
                    descriptors: Vec::new(),
                    segments: Vec::new(),
                    coarse_h: None,
                    full_h: None,
                    confidence: 0.0,
                    elapsed_ms: lsd_start.elapsed().as_secs_f64() * 1000.0,
                }
            }
        }
    }

    fn run_outlier_filter_stage(
        &self,
        segments: &[Segment],
        coarse_h: Option<&Matrix3<f32>>,
    ) -> OutlierComputation {
        let Some(h) = coarse_h else {
            return OutlierComputation {
                stage: None,
                segments: Vec::new(),
                elapsed_ms: 0.0,
            };
        };
        if segments.is_empty() {
            return OutlierComputation {
                stage: None,
                segments: Vec::new(),
                elapsed_ms: 0.0,
            };
        }

        let filter_start = Instant::now();
        let (decisions, diag) = classify_segments_with_details(
            segments,
            h,
            &self.params.outlier_filter,
            &self.params.lsd_vp_params,
        );
        let elapsed_ms = filter_start.elapsed().as_secs_f64() * 1000.0;

        let mut kept_segments = Vec::new();
        let mut classifications = Vec::with_capacity(decisions.len());
        for decision in &decisions {
            let seg_id = SegmentId(decision.index as u32);
            if decision.inlier {
                kept_segments.push(segments[decision.index].clone());
            }
            classifications.push(SegmentSample::from_decision(seg_id, decision));
        }
        if diag.total > 0 && kept_segments.is_empty() {
            debug!(
                "GridDetector::process segment filter rejected all {} segments -> fallback to unfiltered set",
                diag.total
            );
            kept_segments = segments.to_vec();
        }

        let stage = OutlierFilterStage {
            elapsed_ms,
            total: diag.total,
            kept: kept_segments.len(),
            rejected: diag.rejected,
            kept_u: diag.kept_u,
            kept_v: diag.kept_v,
            degenerate_segments: diag.skipped_degenerate,
            thresholds: OutlierThresholds {
                angle_threshold_deg: diag.angle_threshold_deg,
                angle_margin_deg: self.params.outlier_filter.angle_margin_deg,
                residual_threshold_px: diag.residual_threshold_px,
            },
            classifications,
        };

        OutlierComputation {
            stage: Some(stage),
            segments: kept_segments,
            elapsed_ms,
        }
    }

    fn build_bundling_stage(
        &self,
        prepared: &PreparedLevels,
        source_segments: usize,
    ) -> Option<BundlingStage> {
        if prepared.levels.is_empty() {
            return None;
        }
        let levels = prepared
            .levels
            .iter()
            .map(|lvl| BundlingLevel {
                level_index: lvl.level_index,
                width: lvl.level_width,
                height: lvl.level_height,
                bundles: lvl
                    .bundles
                    .iter()
                    .map(|b| BundleDescriptor {
                        center: b.center,
                        line: b.line,
                        weight: b.weight,
                    })
                    .collect(),
            })
            .collect();
        Some(BundlingStage {
            elapsed_ms: prepared.bundling_ms,
            segment_refine_ms: prepared.segment_refine_ms,
            orientation_tol_deg: self.params.bundling_params.orientation_tol_deg,
            merge_distance_px: self.params.bundling_params.merge_dist_px,
            min_weight: self.params.bundling_params.min_weight,
            source_segments,
            scale_applied: None,
            levels,
        })
    }

    fn run_refinement_stage(
        &mut self,
        prepared_levels: &PreparedLevels,
        initial_h: Option<Matrix3<f32>>,
        base_confidence: f32,
    ) -> RefinementComputation {
        let mut confidence = base_confidence;
        let mut hmtx = initial_h.unwrap_or_else(Matrix3::identity);

        if !self.params.enable_refine
            || initial_h.is_none()
            || hmtx == Matrix3::identity()
            || prepared_levels.levels.is_empty()
        {
            return RefinementComputation {
                hmtx,
                confidence,
                stage: None,
                elapsed_ms: 0.0,
            };
        }

        let refine_levels = convert_refine_levels(&prepared_levels.levels);
        let schedule = &self.params.refinement_schedule;
        let mut current_h = hmtx;
        let mut passes = 0usize;
        let mut refine_ms = 0.0f64;
        let mut last_outcome: Option<RefinementOutcome> = None;
        let mut attempted = false;

        while passes < schedule.passes {
            attempted = true;
            let refine_start = Instant::now();
            match self.refiner.refine(current_h, &refine_levels) {
                Some(refine_res) => {
                    refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                    passes += 1;
                    let improvement = frobenius_improvement(&current_h, &refine_res.h_refined);
                    let outcome = RefinementOutcome {
                        levels_used: refine_res.levels_used,
                        confidence: refine_res.confidence,
                        inlier_ratio: refine_res.inlier_ratio,
                        iterations: refine_res.level_reports.clone(),
                    };
                    last_outcome = Some(outcome);

                    let base_conf = confidence;
                    let combined = combine_confidence(
                        base_conf,
                        refine_res.confidence,
                        refine_res.inlier_ratio,
                    );
                    confidence = combined.max(base_conf);
                    current_h = refine_res.h_refined;
                    hmtx = current_h;

                    if passes >= schedule.passes || improvement < schedule.improvement_thresh {
                        break;
                    }
                }
                None => {
                    refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(prev) = self.last_hmtx {
                        debug!("GridDetector::process refine failed -> fallback to last_hmtx");
                        hmtx = prev;
                        confidence *= 0.5;
                    } else {
                        debug!("GridDetector::process refine failed -> keeping coarse hypothesis");
                    }
                    break;
                }
            }
        }

        let stage = if attempted {
            Some(RefinementStage {
                elapsed_ms: refine_ms,
                passes,
                outcome: last_outcome,
            })
        } else {
            None
        };

        RefinementComputation {
            hmtx,
            confidence,
            stage,
            elapsed_ms: refine_ms,
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

        let LsdComputation {
            stage: mut lsd_stage,
            descriptors: segment_descriptors,
            segments: coarse_segments,
            coarse_h,
            full_h,
            confidence,
            elapsed_ms: lsd_ms,
        } = self.run_lsd_on_coarsest(&pyramid, width, height);

        if let Some(stage) = lsd_stage.as_mut() {
            stage.elapsed_ms = lsd_ms;
        }

        let OutlierComputation {
            stage: outlier_stage,
            segments: filtered_segments,
            elapsed_ms: outlier_filter_ms,
        } = self.run_outlier_filter_stage(&coarse_segments, coarse_h.as_ref());

        let mut bundling_stage = None;
        let mut coarse_bundles: Vec<Bundle> = Vec::new();
        if let Some((stage, bundles)) =
            self.bundle_coarsest(&pyramid, &filtered_segments, width, height)
        {
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
        let grid_indexing_stage = self.run_grid_indexing_stage(&coarse_bundles, full_h.as_ref());
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
            segments: segment_descriptors,
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
        segments: &[Segment],
        full_width: usize,
        full_height: usize,
    ) -> Option<(BundlingStage, Vec<Bundle>)> {
        if segments.is_empty() {
            return None;
        }
        let level_index = pyramid.levels.len().checked_sub(1)?;
        let level = &pyramid.levels[level_index];

        let bundling_params = &self.params.bundling_params;
        let orientation_tol = bundling_params.orientation_tol_deg.to_radians();
        let scaling = LevelScaling::from_dimensions(level.w, level.h, full_width, full_height);
        let (dist_tol_level, min_weight_level) =
            adapt_bundling_thresholds(bundling_params, &scaling);

        let bundling_start = Instant::now();
        let bundles_level =
            bundle_segments(segments, orientation_tol, dist_tol_level, min_weight_level);
        let mut bundles_full: Vec<Bundle> = Vec::with_capacity(bundles_level.len());
        for bundle in bundles_level {
            bundles_full.push(rescale_bundle_to_full_res(
                bundle,
                scaling.scale_x_to_full,
                scaling.scale_y_to_full,
            ));
        }
        let elapsed_ms = bundling_start.elapsed().as_secs_f64() * 1000.0;

        let level_descriptor = BundlingLevel {
            level_index,
            width: level.w,
            height: level.h,
            bundles: bundles_full
                .iter()
                .map(|b| BundleDescriptor {
                    center: b.center,
                    line: b.line,
                    weight: b.weight,
                })
                .collect(),
        };

        let stage = BundlingStage {
            elapsed_ms,
            segment_refine_ms: 0.0,
            orientation_tol_deg: bundling_params.orientation_tol_deg,
            merge_distance_px: bundling_params.merge_dist_px,
            min_weight: bundling_params.min_weight,
            source_segments: segments.len(),
            scale_applied: Some([scaling.scale_x_to_full, scaling.scale_y_to_full]),
            levels: vec![level_descriptor],
        };

        Some((stage, bundles_full))
    }

    fn run_grid_indexing_stage(
        &self,
        bundles: &[Bundle],
        hmtx: Option<&Matrix3<f32>>,
    ) -> Option<GridIndexingStage> {
        let h = hmtx?;
        if bundles.is_empty() {
            return None;
        }
        let h_inv = h.try_inverse()?;
        let h_inv_t = h_inv.transpose();
        let orientation_tol = self.params.refine_params.orientation_tol_deg.to_radians();
        let buckets = crate::refine::split_bundles(h, bundles, orientation_tol)?;

        let mut index_map: HashMap<*const Bundle, usize> = HashMap::new();
        for (idx, bundle) in bundles.iter().enumerate() {
            index_map.insert(bundle as *const Bundle, idx);
        }

        let indexing_start = Instant::now();
        let (family_u, range_u) =
            self.build_family_indexing(FamilyLabel::U, &buckets.family_u, &h_inv_t, &index_map);
        let (family_v, range_v) =
            self.build_family_indexing(FamilyLabel::V, &buckets.family_v, &h_inv_t, &index_map);
        let elapsed_ms = indexing_start.elapsed().as_secs_f64() * 1000.0;

        let origin_uv = (0, 0);
        let visible_range = (
            range_u.map(|(min, _)| min).unwrap_or(0),
            range_u.map(|(_, max)| max).unwrap_or(0),
            range_v.map(|(min, _)| min).unwrap_or(0),
            range_v.map(|(_, max)| max).unwrap_or(0),
        );

        Some(GridIndexingStage {
            elapsed_ms,
            family_u,
            family_v,
            origin_uv,
            visible_range,
        })
    }

    fn build_family_indexing(
        &self,
        label: FamilyLabel,
        family_bundles: &[&Bundle],
        h_inv_t: &Matrix3<f32>,
        index_map: &HashMap<*const Bundle, usize>,
    ) -> (FamilyIndexing, Option<(i32, i32)>) {
        struct LineEntry {
            index: usize,
            offset: f32,
            weight: f32,
        }

        let mut entries: Vec<LineEntry> = Vec::new();
        for bundle in family_bundles {
            let Some(&bundle_index) = index_map.get(&(*bundle as *const Bundle)) else {
                continue;
            };
            let line = nalgebra::Vector3::new(bundle.line[0], bundle.line[1], bundle.line[2]);
            let mut rect = h_inv_t * line;
            if !rect[0].is_finite() || !rect[1].is_finite() || !rect[2].is_finite() {
                continue;
            }
            let norm = (rect[0] * rect[0] + rect[1] * rect[1]).sqrt();
            if norm <= EPS {
                continue;
            }
            rect /= norm;

            let (primary, secondary) = (rect[0].abs(), rect[1].abs());
            let use_u_axis = primary >= secondary;
            match label {
                FamilyLabel::U if !use_u_axis => continue,
                FamilyLabel::V if use_u_axis => continue,
                _ => {}
            }

            let coeff = if label == FamilyLabel::U {
                rect[0]
            } else {
                rect[1]
            };
            if coeff.abs() <= EPS {
                continue;
            }
            let offset = -rect[2] / coeff;
            if !offset.is_finite() {
                continue;
            }

            entries.push(LineEntry {
                index: bundle_index,
                offset,
                weight: bundle.weight,
            });
        }

        if entries.is_empty() {
            return (
                FamilyIndexing {
                    spacing: None,
                    base_offset: None,
                    confidence: 0.0,
                    lines: Vec::new(),
                },
                None,
            );
        }

        let total = family_bundles.len().max(1);
        let mut offsets: Vec<f32> = entries.iter().map(|e| e.offset).collect();
        offsets.retain(|v| v.is_finite());

        let spacing = Self::estimate_spacing(&offsets);
        let mut sorted_offsets = offsets.clone();
        sorted_offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let base_offset = sorted_offsets.get(sorted_offsets.len() / 2).copied();

        let mut line_indices = Vec::with_capacity(entries.len());
        let mut min_idx = i32::MAX;
        let mut max_idx = i32::MIN;
        for entry in entries {
            let idx = if let (Some(spacing), Some(base)) = (spacing, base_offset) {
                ((entry.offset - base) / spacing)
                    .round()
                    .clamp(i32::MIN as f32, i32::MAX as f32) as i32
            } else {
                0
            };
            let expected = if let (Some(spacing), Some(base)) = (spacing, base_offset) {
                base + idx as f32 * spacing
            } else {
                base_offset.unwrap_or(0.0)
            };
            let residual = (entry.offset - expected).abs();
            min_idx = min_idx.min(idx);
            max_idx = max_idx.max(idx);
            line_indices.push(GridLineIndex {
                bundle_index: entry.index,
                family: label,
                rectified_offset: entry.offset,
                grid_index: idx,
                weight: entry.weight,
                residual,
            });
        }

        let confidence = (line_indices.len() as f32 / total as f32).clamp(0.0, 1.0);

        (
            FamilyIndexing {
                spacing,
                base_offset,
                confidence,
                lines: line_indices,
            },
            if min_idx <= max_idx {
                Some((min_idx, max_idx))
            } else {
                None
            },
        )
    }

    fn estimate_spacing(values: &[f32]) -> Option<f32> {
        if values.len() < 2 {
            return None;
        }
        let mut sorted = values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect::<Vec<f32>>();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut diffs = Vec::new();
        for pair in sorted.windows(2) {
            let diff = (pair[1] - pair[0]).abs();
            if diff > 1e-3 {
                diffs.push(diff);
            }
        }
        if diffs.is_empty() {
            return None;
        }
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = diffs[diffs.len() / 2];
        if median.is_finite() {
            Some(median)
        } else {
            None
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
        self.lsd_engine = Self::make_lsd_engine(&params);
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

    fn make_lsd_engine(params: &LsdVpParams) -> LsdVpEngine {
        LsdVpEngine {
            mag_thresh: params.mag_thresh,
            angle_tol_deg: params.angle_tol_deg,
            min_len: params.min_len,
            options: params.to_lsd_options(),
        }
    }

    fn prepare_refinement_levels(
        &mut self,
        pyramid: &Pyramid,
        initial_segments: Vec<Segment>,
        full_width: usize,
        full_height: usize,
    ) -> PreparedLevels {
        if pyramid.levels.is_empty() {
            return PreparedLevels {
                levels: Vec::new(),
                bundling_ms: 0.0,
                segment_refine_ms: 0.0,
            };
        }

        let bundling_params = &self.params.bundling_params;
        let orientation_tol = bundling_params.orientation_tol_deg.to_radians();
        let mut levels_data = Vec::new();
        let mut current_segments = initial_segments;
        let mut bundling_ms = 0.0f64;
        let mut segment_refine_ms = 0.0f64;
        let coarse_idx = pyramid.levels.len() - 1;

        for level_idx in (0..=coarse_idx).rev() {
            let lvl = &pyramid.levels[level_idx];
            let scaling = LevelScaling::from_dimensions(lvl.w, lvl.h, full_width, full_height);

            let bundling_start = Instant::now();
            let (dist_tol_level, min_weight_level) =
                adapt_bundling_thresholds(bundling_params, &scaling);
            let bundles_raw = bundle_segments(
                &current_segments,
                orientation_tol,
                dist_tol_level,
                min_weight_level,
            );
            let bundles_full: Vec<Bundle> = bundles_raw
                .into_iter()
                .map(|b| {
                    rescale_bundle_to_full_res(b, scaling.scale_x_to_full, scaling.scale_y_to_full)
                })
                .collect();
            bundling_ms += bundling_start.elapsed().as_secs_f64() * 1000.0;

            debug!(
                "GridDetector::level L{}: segments={} bundles={}",
                level_idx,
                current_segments.len(),
                bundles_full.len()
            );

            levels_data.push(PreparedLevel {
                level_index: level_idx,
                level_width: lvl.w,
                level_height: lvl.h,
                segments: current_segments.len(),
                bundles: bundles_full,
            });

            if current_segments.is_empty() {
                break;
            }
            if level_idx == 0 {
                continue;
            }
            let finer_idx = level_idx - 1;
            let finer_lvl = &pyramid.levels[finer_idx];
            let sx = if lvl.w > 0 {
                finer_lvl.w as f32 / lvl.w as f32
            } else {
                2.0
            };
            let sy = if lvl.h > 0 {
                finer_lvl.h as f32 / lvl.h as f32
            } else {
                2.0
            };
            let scale_map = LevelScaleMap::new(sx, sy);

            let grad = self.workspace.sobel_gradients(finer_idx, finer_lvl);
            let gx = grad.gx.as_slice().unwrap_or(&grad.gx.data[..]);
            let gy = grad.gy.as_slice().unwrap_or(&grad.gy.data[..]);
            let grad_level = SegmentGradientLevel {
                width: finer_lvl.w,
                height: finer_lvl.h,
                gx,
                gy,
            };

            let refine_start = Instant::now();
            let mut refined_segments = Vec::with_capacity(current_segments.len());
            let mut accepted = 0usize;
            for seg in &current_segments {
                let seed = SegmentSeed {
                    p0: seg.p0,
                    p1: seg.p1,
                };
                let result = segment::refine_segment(
                    &grad_level,
                    seed,
                    &scale_map,
                    &self.params.segment_refine_params,
                );
                if result.ok {
                    accepted += 1;
                }
                let updated = convert_refined_segment(seg, result);
                refined_segments.push(updated);
            }
            segment_refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
            debug!(
                "GridDetector::refine L{}→L{} accepted={}/{}",
                level_idx,
                finer_idx,
                accepted,
                current_segments.len()
            );
            current_segments = refined_segments;
        }

        PreparedLevels {
            levels: levels_data,
            bundling_ms,
            segment_refine_ms,
        }
    }
}

fn convert_refine_levels(levels: &[PreparedLevel]) -> Vec<RefineLevel<'_>> {
    let mut out = Vec::with_capacity(levels.len());
    for data in levels.iter().rev() {
        out.push(RefineLevel {
            level_index: data.level_index,
            width: data.level_width,
            height: data.level_height,
            segments: data.segments,
            bundles: data.bundles.as_slice(),
        });
    }
    out
}

fn adapt_bundling_thresholds(params: &BundlingParams, scaling: &LevelScaling) -> (f32, f32) {
    match params.scale_mode {
        BundlingScaleMode::FixedPixel => (params.merge_dist_px, params.min_weight),
        BundlingScaleMode::FullResInvariant => {
            let dist = params.merge_dist_px * scaling.mean_scale_from_full;
            let weight = params.min_weight * scaling.mean_scale_from_full;
            (dist.max(EPS), weight.max(EPS))
        }
    }
}

fn compute_family_counts(families: &[Option<FamilyLabel>]) -> FamilyCounts {
    let mut counts = FamilyCounts {
        family_u: 0,
        family_v: 0,
        unassigned: 0,
    };
    for fam in families {
        match fam {
            Some(FamilyLabel::U) => counts.family_u += 1,
            Some(FamilyLabel::V) => counts.family_v += 1,
            None => counts.unassigned += 1,
        }
    }
    counts
}

fn combine_confidence(base: f32, refine_conf: f32, inlier_ratio: f32) -> f32 {
    if inlier_ratio <= 1e-6 {
        return base.clamp(0.0, 1.0);
    }
    let blended = 0.5 * base + 0.5 * refine_conf;
    (blended * inlier_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn frobenius_improvement(a: &Matrix3<f32>, b: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let diff = b[(i, j)] - a[(i, j)];
            sum += diff * diff;
        }
    }
    sum.sqrt() / (frobenius_norm(a) + EPS)
}

fn frobenius_norm(m: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            sum += m[(i, j)] * m[(i, j)];
        }
    }
    sum.sqrt()
}

fn pose_from_h(k: Matrix3<f32>, h: Matrix3<f32>) -> Pose {
    // Computes R,t from planar homography: K^-1 H ~= [r1 r2 t]
    let k_inv = k.try_inverse().unwrap_or_else(Matrix3::identity);
    let normalized_h = k_inv * h;

    let mut r1 = normalized_h.column(0).into_owned();
    let mut r2 = normalized_h.column(1).into_owned();
    let t_raw = normalized_h.column(2).into_owned();

    // Use average column norm for a more stable scale estimate
    let n1 = r1.norm();
    let n2 = r2.norm();
    let average_norm = (n1 + n2).max(1e-6) * 0.5;
    let inv_scale = 1.0 / average_norm;

    r1 *= inv_scale;
    r2 *= inv_scale;

    // Initial orthonormal basis prior to projection
    let r3 = r1.cross(&r2);
    let mut rot = Matrix3::from_columns(&[r1, r2, r3]);

    let svd = rot.svd(true, true);
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        rot = u * v_t;
    }

    if rot.determinant() < 0.0 {
        let mut c2 = rot.column_mut(2);
        c2.neg_mut();
    }

    let t = t_raw * inv_scale;
    Pose { r: rot, t }
}

// matrix_to_array removed: all matrices now use nalgebra serde directly.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundling_thresholds_scale_when_requested() {
        let mut params = BundlingParams {
            orientation_tol_deg: 20.0,
            merge_dist_px: 2.0,
            min_weight: 4.0,
            scale_mode: BundlingScaleMode::FullResInvariant,
        };
        let scaling = LevelScaling::from_dimensions(100, 100, 400, 400);
        let (dist, weight) = adapt_bundling_thresholds(&params, &scaling);
        assert!((dist - 0.5).abs() < 1e-6, "dist={}", dist);
        assert!((weight - 1.0).abs() < 1e-6, "weight={}", weight);

        params.scale_mode = BundlingScaleMode::FixedPixel;
        let (dist_fixed, weight_fixed) = adapt_bundling_thresholds(&params, &scaling);
        assert!((dist_fixed - 2.0).abs() < 1e-6);
        assert!((weight_fixed - 4.0).abs() < 1e-6);
    }
}
