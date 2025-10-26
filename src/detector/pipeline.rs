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
use crate::diagnostics::{
    BundleDescriptor, BundlingLevel, BundlingStage, DetectionReport, FamilyCounts, InputDescriptor,
    LsdStage, OutlierFilterStage, OutlierThresholds, PipelineTrace, PoseStage, PyramidStage,
    RefinementOutcome, RefinementStage, SegmentDescriptor, SegmentId, SegmentSample,
    TimingBreakdown,
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
        self.process_with_diagnostics(gray).grid
    }

    /// Run the detector and return both the result and a detailed report.
    pub fn process_with_diagnostics(&mut self, gray: ImageU8) -> DetectionReport {
        let (width, height) = (gray.w, gray.h);
        debug!(
            "GridDetector::process start w={} h={} levels={}",
            width, height, self.params.pyramid_levels
        );
        let total_start = Instant::now();

        let pyr_start = Instant::now();
        let pyramid_opts = match self.params.pyramid_blur_levels {
            Some(blur_levels) => {
                PyramidOptions::new(self.params.pyramid_levels).with_blur_levels(Some(blur_levels))
            }
            None => PyramidOptions::new(self.params.pyramid_levels),
        };
        let pyramid = Pyramid::build_u8(gray, pyramid_opts);
        let pyr_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;

        let pyramid_stage = if pyramid.levels.is_empty() {
            None
        } else {
            Some(PyramidStage::from_pyramid(&pyramid, pyr_ms))
        };

        let mut segment_descriptors: Vec<SegmentDescriptor> = Vec::new();
        let mut lsd_stage: Option<LsdStage> = None;
        let mut outlier_stage: Option<OutlierFilterStage> = None;
        let mut bundling_stage: Option<BundlingStage> = None;
        let mut refinement_stage: Option<RefinementStage> = None;

        let mut lsd_ms = 0.0f64;
        let mut outlier_filter_ms = 0.0f64;
        let mut refine_ms = 0.0f64;

        let mut confidence = 0.0f32;
        let mut h_full: Option<Matrix3<f32>> = None;
        let mut hmtx = Matrix3::identity();

        let mut filtered_segments: Vec<Segment> = Vec::new();

        if let Some(coarse_level) = pyramid.levels.last() {
            let lsd_start = Instant::now();
            if let Some(detailed) = self.lsd_engine.infer_detailed(coarse_level) {
                let DetailedInference {
                    hypothesis,
                    dominant_angles_rad,
                    families,
                    segments,
                } = detailed;
                confidence = hypothesis.confidence;
                let hmtx0 = hypothesis.hmtx0;

                for (idx, seg) in segments.iter().enumerate() {
                    segment_descriptors
                        .push(SegmentDescriptor::from_segment(SegmentId(idx as u32), seg));
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
                h_full = Some(if coarse_level.w > 0 && coarse_level.h > 0 {
                    scale * hmtx0
                } else {
                    hmtx0
                });
                hmtx = h_full.unwrap_or_else(Matrix3::identity);

                let dominant_angles_deg = [
                    dominant_angles_rad[0].to_degrees(),
                    dominant_angles_rad[1].to_degrees(),
                ];
                let family_counts = compute_family_counts(&families);
                lsd_stage = Some(LsdStage {
                    elapsed_ms: 0.0,
                    confidence,
                    dominant_angles_deg,
                    family_counts,
                    segment_families: families.clone(),
                    sample_ids: Vec::new(),
                });

                let filter_start = Instant::now();
                let (decisions, diag) = classify_segments_with_details(
                    &segments,
                    &hmtx0,
                    &self.params.outlier_filter,
                    &self.params.lsd_vp_params,
                );
                outlier_filter_ms = filter_start.elapsed().as_secs_f64() * 1000.0;

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
                    kept_segments = segments.clone();
                }

                outlier_stage = Some(OutlierFilterStage {
                    elapsed_ms: outlier_filter_ms,
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
                });

                filtered_segments = kept_segments;
            } else {
                debug!("GridDetector::process LSD→VP engine returned no hypothesis");
            }
            lsd_ms = lsd_start.elapsed().as_secs_f64() * 1000.0;
            if let Some(stage) = lsd_stage.as_mut() {
                stage.elapsed_ms = lsd_ms;
            }
        } else {
            debug!("GridDetector::process pyramid has no levels");
        }

        let mut prepared_levels = PreparedLevels {
            levels: Vec::new(),
            bundling_ms: 0.0,
            segment_refine_ms: 0.0,
        };
        if !filtered_segments.is_empty() && !pyramid.levels.is_empty() {
            self.workspace.reset(pyramid.levels.len());
            prepared_levels =
                self.prepare_refinement_levels(&pyramid, filtered_segments.clone(), width, height);
        }

        if !prepared_levels.levels.is_empty() {
            let levels = prepared_levels
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
            bundling_stage = Some(BundlingStage {
                elapsed_ms: prepared_levels.bundling_ms,
                segment_refine_ms: prepared_levels.segment_refine_ms,
                orientation_tol_deg: self.params.bundling_params.orientation_tol_deg,
                merge_distance_px: self.params.bundling_params.merge_dist_px,
                min_weight: self.params.bundling_params.min_weight,
                source_segments: filtered_segments.len(),
                scale_applied: None,
                levels,
            });
        }

        let mut refinement_passes = 0usize;
        if self.params.enable_refine
            && h_full.is_some()
            && hmtx != Matrix3::identity()
            && !prepared_levels.levels.is_empty()
        {
            let refine_levels = convert_refine_levels(&prepared_levels.levels);
            let schedule = &self.params.refinement_schedule;
            let mut current_h = hmtx;
            let mut passes = 0usize;
            let mut last_outcome: Option<RefinementOutcome> = None;
            while passes < schedule.passes {
                let refine_start = Instant::now();
                match self.refiner.refine(current_h, &refine_levels) {
                    Some(refine_res) => {
                        refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                        passes += 1;
                        refinement_passes = passes;
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
                            debug!(
                                "GridDetector::process refine failed -> keeping coarse hypothesis"
                            );
                        }
                        break;
                    }
                }
            }

            refinement_stage = Some(RefinementStage {
                elapsed_ms: refine_ms,
                passes: refinement_passes,
                outcome: last_outcome,
            });
        }

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
            origin_uv: (0, 0),
            visible_range: (0, 0, 0, 0),
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
            refinement: refinement_stage,
            pose: pose.as_ref().map(PoseStage::from_pose),
        };

        DetectionReport {
            grid: result,
            trace,
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
                let updated = convert_refine_result(seg, result);
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

fn convert_refine_result(prev: &Segment, result: segment::RefineResult) -> Segment {
    let seg = result.seg;
    let mut p0 = seg.p0;
    let mut p1 = seg.p1;
    if !p0[0].is_finite() || !p0[1].is_finite() || !p1[0].is_finite() || !p1[1].is_finite() {
        p0 = prev.p0;
        p1 = prev.p1;
    }
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let mut len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() {
        len = prev.len;
    }
    let dir = if len > f32::EPSILON {
        [dx / len, dy / len]
    } else {
        prev.dir
    };

    let mut normal = [-dir[1], dir[0]];
    let norm = (normal[0] * normal[0] + normal[1] * normal[1])
        .sqrt()
        .max(EPS);
    normal[0] /= norm;
    normal[1] /= norm;
    let c = -(normal[0] * p0[0] + normal[1] * p0[1]);

    let avg_mag = if result.ok && result.score.is_finite() && result.score > 0.0 {
        result.score
    } else {
        prev.avg_mag
    }
    .max(0.0);
    let strength = len.max(1e-3) * avg_mag;

    Segment {
        p0,
        p1,
        dir,
        len,
        line: [normal[0], normal[1], c],
        avg_mag,
        strength,
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
