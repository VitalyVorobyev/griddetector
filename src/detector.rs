use crate::diagnostics::{
    BundleDiagnostics, BundleEntryDiagnostics, DetailedResult, LsdDiagnostics,
    ProcessingDiagnostics, PyramidLevelDiagnostics, RefinementDiagnostics,
};
use crate::edges::grad::{sobel_gradients, Grad as EdgeGrad};
use crate::image::{ImageU8, ImageView};
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine};
use crate::pyramid::Pyramid;
use crate::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, ScaleMap, Segment as SegmentSeed,
};
use crate::refine::{
    segment::RefineParams as SegmentRefineParams, RefineLevel,
    RefineParams as HomographyRefineParams, Refiner,
};
use crate::segments::{bundle_segments, Bundle, Segment};
use crate::types::{GridResult, Pose};
use log::debug;
use nalgebra::Matrix3;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GridParams {
    pub pyramid_levels: usize,
    pub spacing_mm: f32,
    pub kmtx: Matrix3<f32>,
    pub min_cells: i32,
    pub confidence_thresh: f32,
    pub enable_refine: bool,
    pub refine_params: HomographyRefineParams,
    pub segment_refine_params: SegmentRefineParams,
    pub lsd_vp_params: LsdVpParams,
    pub bundling_params: BundlingParams,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            spacing_mm: 5.0,
            kmtx: Matrix3::identity(),
            min_cells: 6,
            confidence_thresh: 0.35,
            enable_refine: true,
            refine_params: HomographyRefineParams::default(),
            segment_refine_params: SegmentRefineParams::default(),
            lsd_vp_params: LsdVpParams::default(),
            bundling_params: BundlingParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LsdVpParams {
    pub mag_thresh: f32,
    pub angle_tol_deg: f32,
    pub min_len: f32,
}

impl Default for LsdVpParams {
    fn default() -> Self {
        Self {
            mag_thresh: 0.05,
            angle_tol_deg: 22.5,
            min_len: 4.0,
        }
    }
}

#[derive(Clone, Debug)]
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

struct LevelScaleMap {
    sx: f32,
    sy: f32,
}

impl ScaleMap for LevelScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2] {
        [p_coarse[0] * self.sx, p_coarse[1] * self.sy]
    }
}

#[derive(Debug)]
struct RefineLevelData {
    level_index: usize,
    level_width: usize,
    level_height: usize,
    segments: usize,
    bundles: Vec<Bundle>,
}

fn rescale_lsd_segments(diag: &mut LsdDiagnostics, scale_x: f32, scale_y: f32) {
    if !scale_x.is_finite() || !scale_y.is_finite() {
        return;
    }
    for seg in &mut diag.segments_sample {
        let old_len = seg.len;
        seg.p0[0] *= scale_x;
        seg.p0[1] *= scale_y;
        seg.p1[0] *= scale_x;
        seg.p1[1] *= scale_y;
        let dx = seg.p1[0] - seg.p0[0];
        let dy = seg.p1[1] - seg.p0[1];
        let new_len = (dx * dx + dy * dy).sqrt();
        if old_len > f32::EPSILON {
            seg.strength *= new_len / old_len;
        } else if new_len <= f32::EPSILON {
            seg.strength = 0.0;
        }
        seg.len = new_len;
    }
}

fn rescale_bundle_to_full_res(mut bundle: Bundle, scale_x: f32, scale_y: f32) -> Bundle {
    bundle.center[0] *= scale_x;
    bundle.center[1] *= scale_y;

    let mut a = bundle.line[0] / scale_x;
    let mut b = bundle.line[1] / scale_y;
    let mut c = bundle.line[2];
    let norm = (a * a + b * b).sqrt().max(1e-6);
    a /= norm;
    b /= norm;
    c /= norm;

    bundle.line = [a, b, c];
    bundle.weight *= 0.5 * (scale_x + scale_y);
    bundle
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
        .max(1e-6);
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

pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
    refiner: Refiner,
    lsd_engine: LsdVpEngine,
}

impl GridDetector {
    pub fn new(params: GridParams) -> Self {
        let refiner = Refiner::new(params.refine_params.clone());
        let lsd_engine = Self::make_lsd_engine(&params.lsd_vp_params);
        Self {
            params,
            last_hmtx: None,
            refiner,
            lsd_engine,
        }
    }

    pub fn process(&mut self, gray: ImageU8) -> GridResult {
        self.process_with_diagnostics(gray).result
    }

    pub fn process_with_diagnostics(&mut self, gray: ImageU8) -> DetailedResult {
        let (width, height) = (gray.w, gray.h);
        debug!(
            "GridDetector::process start w={} h={} levels={}",
            width, height, self.params.pyramid_levels
        );
        let total_start = Instant::now();

        let pyr_start = Instant::now();
        let pyr = Pyramid::build_u8(gray, self.params.pyramid_levels);
        let pyr_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
        debug!("GridDetector::process pyramid built in {:.3} ms", pyr_ms);

        let pyramid_levels_diag = pyr
            .levels
            .iter()
            .enumerate()
            .map(|(level, lvl)| {
                let sum: f32 = if let Some(slice) = lvl.as_slice() {
                    slice.iter().copied().sum()
                } else {
                    lvl.rows().map(|r| r.iter().copied().sum::<f32>()).sum()
                };
                let denom = (lvl.w * lvl.h).max(1) as f32;
                PyramidLevelDiagnostics {
                    level,
                    width: lvl.w,
                    height: lvl.h,
                    mean_intensity: sum / denom,
                }
            })
            .collect::<Vec<_>>();

        let mut confidence = 0.0f32;
        let mut h_candidate = None;
        let mut lsd_diag = None;
        let mut refine_levels_data: Vec<RefineLevelData> = Vec::new();

        if let Some(coarse_level) = pyr.levels.last() {
            if let Some(detailed) = self.lsd_engine.infer_detailed(coarse_level) {
                let DetailedInference {
                    mut hypothesis,
                    segments,
                    ..
                } = detailed;
                confidence = hypothesis.confidence;
                let hmtx0 = hypothesis.hmtx0;
                if coarse_level.w > 0 && coarse_level.h > 0 {
                    let scale_x = width as f32 / coarse_level.w as f32;
                    let scale_y = height as f32 / coarse_level.h as f32;
                    rescale_lsd_segments(&mut hypothesis.diagnostics, scale_x, scale_y);
                    let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
                    h_candidate = Some(scale * hmtx0);
                } else {
                    debug!(
                        "GridDetector::process coarsest level has zero dimension, skipping scale"
                    );
                    h_candidate = Some(hmtx0);
                }
                lsd_diag = Some(hypothesis.diagnostics.clone());
                refine_levels_data = self.prepare_refine_levels(&pyr, segments, width, height);
                debug!(
                    "GridDetector::process prepared {} refinement levels",
                    refine_levels_data.len()
                );
            } else {
                debug!("GridDetector::process LSD→VP engine returned no hypothesis");
            }
        } else {
            debug!("GridDetector::process pyramid has no levels");
        }
        let mut hmtx = h_candidate.unwrap_or_else(Matrix3::identity);

        let mut refinement_diag = None;
        if self.params.enable_refine
            && h_candidate.is_some()
            && hmtx != Matrix3::identity()
            && !refine_levels_data.is_empty()
        {
            let mut refine_levels: Vec<RefineLevel<'_>> =
                Vec::with_capacity(refine_levels_data.len());
            for data in refine_levels_data.iter().rev() {
                refine_levels.push(RefineLevel {
                    level_index: data.level_index,
                    width: data.level_width,
                    height: data.level_height,
                    segments: data.segments,
                    bundles: data.bundles.as_slice(),
                });
            }
            match self.refiner.refine(hmtx, &refine_levels) {
                Some(refine_res) => {
                    debug!(
                        "GridDetector::process refine confidence={:.3} inlier_ratio={:.3} levels_used={}",
                        refine_res.confidence, refine_res.inlier_ratio, refine_res.levels_used
                    );
                    refinement_diag = Some(RefinementDiagnostics {
                        levels_used: refine_res.levels_used,
                        aggregated_confidence: refine_res.confidence,
                        final_inlier_ratio: refine_res.inlier_ratio,
                        levels: refine_res.level_reports.clone(),
                    });
                    hmtx = refine_res.h_refined;
                    confidence = combine_confidence(
                        confidence,
                        refine_res.confidence,
                        refine_res.inlier_ratio,
                    );
                }
                None => {
                    if let Some(prev) = self.last_hmtx {
                        debug!("GridDetector::process refine failed -> fallback to last_hmtx");
                        hmtx = prev;
                        confidence *= 0.5;
                    } else {
                        debug!("GridDetector::process refine failed -> keeping coarse hypothesis");
                    }
                }
            }
        }

        let bundling_diag = if !refine_levels_data.is_empty() {
            Some(
                refine_levels_data
                    .iter()
                    .map(|data| BundleDiagnostics {
                        level_index: data.level_index,
                        bundles: data
                            .bundles
                            .iter()
                            .map(|b| BundleEntryDiagnostics {
                                center: b.center,
                                line: b.line,
                                weight: b.weight,
                            })
                            .collect(),
                    })
                    .collect::<Vec<_>>(),
            )
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
            pose,
            origin_uv: (0, 0),
            visible_range: (0, 0, 0, 0),
            coverage: 0.0,
            reproj_rmse: 0.0,
            confidence,
            latency_ms: latency,
        };
        let diagnostics = ProcessingDiagnostics {
            input_width: width,
            input_height: height,
            pyramid_levels: pyramid_levels_diag,
            pyramid_build_ms: pyr_ms,
            lsd: lsd_diag,
            refinement: refinement_diag,
            bundling: bundling_diag,
            homography: hmtx,
            total_latency_ms: latency,
        };
        DetailedResult {
            result,
            diagnostics,
        }
    }

    pub fn set_intrinsics(&mut self, k: Matrix3<f32>) {
        self.params.kmtx = k;
    }
    pub fn set_spacing(&mut self, s_mm: f32) {
        self.params.spacing_mm = s_mm;
    }
    pub fn set_refine_params(&mut self, params: HomographyRefineParams) {
        self.params.refine_params = params.clone();
        self.refiner = Refiner::new(params);
    }

    pub fn set_segment_refine_params(&mut self, params: SegmentRefineParams) {
        self.params.segment_refine_params = params;
    }

    pub fn set_lsd_vp_params(&mut self, params: LsdVpParams) {
        self.params.lsd_vp_params = params.clone();
        self.lsd_engine = Self::make_lsd_engine(&params);
    }

    pub fn set_bundling_params(&mut self, params: BundlingParams) {
        self.params.bundling_params = params;
    }

    fn make_lsd_engine(params: &LsdVpParams) -> LsdVpEngine {
        LsdVpEngine {
            mag_thresh: params.mag_thresh,
            angle_tol_deg: params.angle_tol_deg,
            min_len: params.min_len,
        }
    }

    fn prepare_refine_levels(
        &self,
        pyramid: &Pyramid,
        initial_segments: Vec<Segment>,
        full_width: usize,
        full_height: usize,
    ) -> Vec<RefineLevelData> {
        if pyramid.levels.is_empty() {
            return Vec::new();
        }

        let mut levels_data = Vec::new();
        let mut current_segments = initial_segments;
        let mut grad_cache: Vec<Option<EdgeGrad>> = vec![None; pyramid.levels.len()];
        let bundling_params = &self.params.bundling_params;
        let orientation_tol = bundling_params.orientation_tol_deg.to_radians();
        let coarse_idx = pyramid.levels.len() - 1;

        for level_idx in (0..=coarse_idx).rev() {
            let lvl = &pyramid.levels[level_idx];
            let scale_x_full = if lvl.w > 0 {
                full_width as f32 / lvl.w as f32
            } else {
                1.0
            };
            let scale_y_full = if lvl.h > 0 {
                full_height as f32 / lvl.h as f32
            } else {
                1.0
            };

            let bundles_raw = bundle_segments(
                &current_segments,
                orientation_tol,
                bundling_params.merge_dist_px,
                bundling_params.min_weight,
            );
            let bundles_full: Vec<Bundle> = bundles_raw
                .into_iter()
                .map(|b| rescale_bundle_to_full_res(b, scale_x_full, scale_y_full))
                .collect();

            debug!(
                "GridDetector::level L{}: segments={} bundles={}",
                level_idx,
                current_segments.len(),
                bundles_full.len()
            );

            levels_data.push(RefineLevelData {
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
                break;
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
            let scale_map = LevelScaleMap { sx, sy };

            let grad = grad_cache
                .get_mut(finer_idx)
                .expect("gradient cache entry")
                .get_or_insert_with(|| sobel_gradients(finer_lvl));
            let gx = grad.gx.as_slice().unwrap_or(grad.gx.data.as_slice());
            let gy = grad.gy.as_slice().unwrap_or(grad.gy.data.as_slice());
            let grad_level = SegmentGradientLevel {
                width: finer_lvl.w,
                height: finer_lvl.h,
                gx,
                gy,
            };

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
            debug!(
                "GridDetector::refine L{}→L{} accepted={}/{}",
                level_idx,
                finer_idx,
                accepted,
                current_segments.len()
            );
            current_segments = refined_segments;
        }

        levels_data
    }
}

fn combine_confidence(base: f32, refine_conf: f32, inlier_ratio: f32) -> f32 {
    if inlier_ratio <= 1e-6 {
        return base.clamp(0.0, 1.0);
    }
    let blended = 0.5 * base + 0.5 * refine_conf;
    (blended * inlier_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
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
