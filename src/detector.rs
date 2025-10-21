use crate::diagnostics::{
    DetailedResult, LsdDiagnostics, ProcessingDiagnostics, PyramidLevelDiagnostics,
    RefinementDiagnostics,
};
use crate::image::{ImageU8, ImageView};
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine};
use crate::pyramid::Pyramid;
use crate::refine::{RefineLevel, RefineParams, Refiner};
use crate::segments::{bundle_segments, Bundle};
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
    pub refine_params: RefineParams,
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
            refine_params: RefineParams::default(),
        }
    }
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

const DEFAULT_BUNDLE_MERGE_DIST: f32 = 1.5;
const DEFAULT_MIN_BUNDLE_WEIGHT: f32 = 3.0;

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

pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
    refiner: Refiner,
}

impl GridDetector {
    pub fn new(params: GridParams) -> Self {
        let refiner = Refiner::new(params.refine_params.clone());
        Self {
            params,
            last_hmtx: None,
            refiner,
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

        let l = pyr.levels.last().unwrap();
        let engine = LsdVpEngine::default();
        let mut confidence = 0.0f32;
        let mut h_candidate = None;
        let mut lsd_diag = None;
        let mut refine_bundles: Option<(Vec<Bundle>, usize)> = None;
        if let Some(detailed) = engine.infer_detailed(l) {
            let DetailedInference {
                mut hypothesis,
                segments,
                ..
            } = detailed;
            confidence = hypothesis.confidence;
            let hmtx0 = hypothesis.hmtx0;
            if l.w > 0 && l.h > 0 {
                let scale_x = width as f32 / l.w as f32;
                let scale_y = height as f32 / l.h as f32;
                rescale_lsd_segments(&mut hypothesis.diagnostics, scale_x, scale_y);
                let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
                let scaled = scale * hmtx0;
                h_candidate = Some(scaled);
                let orientation_tol = self.params.refine_params.orientation_tol_deg.to_radians();
                let bundles = bundle_segments(
                    &segments,
                    orientation_tol,
                    DEFAULT_BUNDLE_MERGE_DIST,
                    DEFAULT_MIN_BUNDLE_WEIGHT,
                );
                let bundles_full = bundles
                    .into_iter()
                    .map(|b| rescale_bundle_to_full_res(b, scale_x, scale_y))
                    .collect::<Vec<_>>();
                refine_bundles = Some((bundles_full, segments.len()));
                debug!(
                    "GridDetector::process hypothesis confidence={:.3} scale=({:.3},{:.3}) bundles={}",
                    confidence,
                    scale_x,
                    scale_y,
                    refine_bundles.as_ref().map(|(b, _)| b.len()).unwrap_or(0)
                );
            } else {
                debug!("GridDetector::process coarsest level has zero dimension, skipping scale");
                h_candidate = Some(hmtx0);
            }
            lsd_diag = Some(hypothesis.diagnostics.clone());
        } else {
            debug!("GridDetector::process LSD→VP engine returned no hypothesis");
        }
        let mut hmtx = h_candidate.unwrap_or_else(Matrix3::identity);

        let mut refinement_diag = None;
        if self.params.enable_refine
            && h_candidate.is_some()
            && hmtx != Matrix3::identity()
            && refine_bundles.is_some()
        {
            let (bundles_full, segments_total) = refine_bundles.unwrap();
            let level_index = pyr.levels.len().saturating_sub(1);
            let level = RefineLevel {
                level_index,
                width,
                height,
                segments: segments_total,
                bundles: bundles_full.as_slice(),
            };
            let levels = [level];
            match self.refiner.refine(hmtx, &levels) {
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
    pub fn set_refine_params(&mut self, params: RefineParams) {
        self.params.refine_params = params.clone();
        self.refiner = Refiner::new(params);
    }
}

fn combine_confidence(base: f32, refine_conf: f32, inlier_ratio: f32) -> f32 {
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
