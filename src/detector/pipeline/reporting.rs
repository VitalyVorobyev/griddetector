use crate::types::Pose;
use nalgebra::Matrix3;

pub fn pose_from_h(k: Matrix3<f32>, h: Matrix3<f32>) -> Pose {
    let k_inv = k.try_inverse().unwrap_or_else(Matrix3::identity);
    let normalized_h = k_inv * h;

    let mut r1 = normalized_h.column(0).into_owned();
    let mut r2 = normalized_h.column(1).into_owned();
    let t_raw = normalized_h.column(2).into_owned();

    let n1 = r1.norm();
    let n2 = r2.norm();
    let average_norm = (n1 + n2).max(1e-6) * 0.5;
    let inv_scale = 1.0 / average_norm;

    r1 *= inv_scale;
    r2 *= inv_scale;

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
