use crate::lsd_vp::Engine as LsdVpEngine;
use crate::pyramid::Pyramid;
use crate::types::{GridResult, ImageU8, Pose};
use log::debug;
use nalgebra::Matrix3;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GridParams {
    pub pyramid_levels: usize,
    pub spacing_mm: f32,
    pub kmtx: Matrix3<f32>,
    pub canny_low: f32,
    pub canny_high: f32,
    pub min_cells: i32,
    pub confidence_thresh: f32,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            spacing_mm: 5.0,
            kmtx: Matrix3::identity(),
            canny_low: 20.0,
            canny_high: 60.0,
            min_cells: 6,
            confidence_thresh: 0.35,
        }
    }
}

pub struct GridDetector {
    params: GridParams,
    last_hmtx: Option<Matrix3<f32>>,
}

impl GridDetector {
    pub fn new(params: GridParams) -> Self {
        Self {
            params,
            last_hmtx: None,
        }
    }

    pub fn process(&mut self, gray: ImageU8) -> GridResult {
        let (width, height) = (gray.w, gray.h);
        debug!(
            "GridDetector::process start w={} h={} levels={}",
            width, height, self.params.pyramid_levels
        );
        let total_start = Instant::now();

        let pyr_start = Instant::now();
        // 1) Pyramid
        let pyr = Pyramid::build_u8(gray, self.params.pyramid_levels);
        let pyr_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "GridDetector::process pyramid built in {:.3} ms",
            pyr_ms
        );
        // 2) Low-res hypothesis with LSD→VP engine
        let l = pyr.levels.last().unwrap();
        let mut engine = LsdVpEngine::default();
        let mut hmtx0 = Matrix3::<f32>::identity();
        let mut confidence = 0.0f32;
        let mut hmtx = Matrix3::<f32>::identity();
        if let Some(hyp) = engine.infer(l) {
            hmtx0 = hyp.hmtx0;
            confidence = hyp.confidence;
            if l.w > 0 && l.h > 0 {
                let scale_x = width as f32 / l.w as f32;
                let scale_y = height as f32 / l.h as f32;
                let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
                hmtx = scale * hmtx0;
                self.last_hmtx = Some(hmtx);
                debug!(
                    "GridDetector::process hypothesis confidence={:.3} scale=({:.3},{:.3})",
                    confidence, scale_x, scale_y
                );
            } else {
                debug!("GridDetector::process coarsest level has zero dimension, skipping scale");
            }
        } else {
            debug!("GridDetector::process LSD→VP engine returned no hypothesis");
        }
        // 3) Refinement (placeholder)
        let hmtx = if hmtx != Matrix3::identity() {
            hmtx
        } else {
            hmtx0
        }; // TODO: refine via edge bundles & robust fit

        // 4) Pose recovery from H and intrinsics
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
        GridResult {
            found,
            hmtx,
            pose,
            origin_uv: (0, 0),
            visible_range: (0, 0, 0, 0),
            coverage: 0.0,
            reproj_rmse: 0.0,
            confidence: confidence,
            latency_ms: latency,
        }
    }

    pub fn set_intrinsics(&mut self, k: Matrix3<f32>) {
        self.params.kmtx = k;
    }
    pub fn set_spacing(&mut self, s_mm: f32) {
        self.params.spacing_mm = s_mm;
    }
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
