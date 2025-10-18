use crate::pyramid::Pyramid;
use crate::segments::LsdVpEngine;
use crate::types::{GridResult, ImageU8, Pose};
use nalgebra::{Matrix3, Vector3};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GridParams {
    pub pyramid_levels: usize,
    pub spacing_mm: f32,
    pub kmtx: Matrix3<f32>,
    pub canny_low: f32,
    pub canny_high: f32,
    pub min_cells: i32,
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
        let t0 = Instant::now();
        // 1) Pyramid
        let pyr = Pyramid::build_u8(gray, self.params.pyramid_levels);
        // 2) Low-res hypothesis with LSD→VP engine
        let l = pyr.levels.last().unwrap();
        let mut engine = LsdVpEngine::default();
        let mut H = Matrix3::<f32>::identity();
        let mut confidence = 0.0f32;
        if let Some(hyp) = engine.infer(l) {
            H = hyp.H0;
            confidence = hyp.confidence;
            self.last_hmtx = Some(H);
        }
        // TODO: randomized Hough for two line families -> initial homography H0
        let hmtx0 = Matrix3::<f32>::identity(); // placeholder
                                                // 3) Refinement (placeholder)
        let hmtx = hmtx0; // TODO: refine via edge bundles & robust fit
                          // 4) Pose recovery from H and intrinsics
        let pose = if hmtx != Matrix3::identity() {
            Some(pose_from_h(self.params.kmtx, hmtx))
        } else {
            None
        };
        let latency = t0.elapsed().as_secs_f64() * 1000.0;
        GridResult {
            found: false, // flip to true when H is valid
            hmtx,
            pose,
            origin_uv: (0, 0),
            visible_range: (0, 0, 0, 0),
            coverage: 0.0,
            reproj_rmse: 0.0,
            confidence: 0.0,
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
    // Computes R,t from planar homography: K^-1 H = [r1 r2 t] up to scale
    let k_inv = k.try_inverse().unwrap_or_else(Matrix3::identity);
    let h_ = k_inv * h;
    let h1 = h_.column(0).clone_owned();
    let h2 = h_.column(1).clone_owned();
    let h3 = h_.column(2).clone_owned();
    let norm = 1.0 / h1.norm();
    let mut r1 = h1 * norm;
    let mut r2 = h2 * norm;
    let mut r3 = r1.cross(&r2);
    // Orthonormalize via polar decomposition (R = UV^T)
    let mut rot = nalgebra::Matrix3::from_columns(&[r1, r2, r3]);
    let svd = rot.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    rot = u * v_t;
    r1 = rot.column(0).clone_owned();
    r2 = rot.column(1).clone_owned();
    r3 = rot.column(2).clone_owned();
    let t = h3 * norm;
    Pose {
        r: rot,
        t: Vector3::new(t[0], t[1], t[2]),
    }
}
