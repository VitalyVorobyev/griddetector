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

/// # Safety
/// Dereferences raw pointers
#[no_mangle]
pub unsafe extern "C" fn grid_rescale_homography(
    matrix_ptr: *const f32,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    out_ptr: *mut f32,
) -> bool {
    if matrix_ptr.is_null() || out_ptr.is_null() {
        return false;
    }
    let input = unsafe { std::slice::from_raw_parts(matrix_ptr, 9) };
    let mut m = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = input[i * 3 + j];
        }
    }
    let result = rescale_homography_image_space(
        &m,
        src_w as usize,
        src_h as usize,
        dst_w as usize,
        dst_h as usize,
    );
    let output = unsafe { std::slice::from_raw_parts_mut(out_ptr, 9) };
    for i in 0..3 {
        for j in 0..3 {
            output[i * 3 + j] = result[(i, j)];
        }
    }
    true
}

/// # Safety
/// Dereferences raw pointers
#[no_mangle]
pub unsafe extern "C" fn grid_apply_homography_points(
    matrix_ptr: *const f32,
    pts_ptr: *const f32,
    point_count: usize,
    out_ptr: *mut f32,
) -> i32 {
    if matrix_ptr.is_null() || pts_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let matrix_slice = unsafe { std::slice::from_raw_parts(matrix_ptr, 9) };
    let mut m = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = matrix_slice[i * 3 + j];
        }
    }
    let pts_slice = unsafe { std::slice::from_raw_parts(pts_ptr, point_count * 2) };
    let mut pts = Vec::with_capacity(point_count);
    for idx in 0..point_count {
        let x = pts_slice[2 * idx];
        let y = pts_slice[2 * idx + 1];
        pts.push([x, y]);
    }
    match apply_homography_points(&m, &pts) {
        Some(result) => {
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, point_count * 2) };
            for (idx, pt) in result.iter().enumerate() {
                out_slice[2 * idx] = pt[0];
                out_slice[2 * idx + 1] = pt[1];
            }
            point_count as i32
        }
        None => -1,
    }
}
