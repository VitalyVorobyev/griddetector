use nalgebra::{Matrix3, Vector3};

const EPS: f32 = 1e-9;

pub fn rescale_homography_image_space(
    h: &Matrix3<f32>,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Matrix3<f32> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return *h;
    }
    let sx = dst_w as f32 / src_w as f32;
    let sy = dst_h as f32 / src_h as f32;
    let scale = Matrix3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0);
    scale * h
}

pub fn apply_homography_points(h: &Matrix3<f32>, pts: &[[f32; 2]]) -> Option<Vec<[f32; 2]>> {
    let mut out = Vec::with_capacity(pts.len());
    for &p in pts {
        let v = h * Vector3::new(p[0], p[1], 1.0);
        let w = v[2];
        if !w.is_finite() || w.abs() <= EPS || !v[0].is_finite() || !v[1].is_finite() {
            return None;
        }
        out.push([v[0] / w, v[1] / w]);
    }
    Some(out)
}
