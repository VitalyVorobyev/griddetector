//! Gradient sampling along the carrier normal.

#[cfg(feature = "profile_refine")]
use super::profile;
use super::{
    fit::SupportPoint,
    types::{PyramidLevel, RefineParams, SegmentRoi},
    EPS,
};

pub(crate) fn search_along_normal(
    lvl: &PyramidLevel<'_>,
    roi: &SegmentRoi,
    center: &[f32; 2],
    normal: &[f32; 2],
    params: &RefineParams,
    w_perp: f32,
) -> Option<SupportPoint> {
    let tau_mag_sq = params.tau_mag * params.tau_mag;
    let mut best_t = 0.0f32;
    let mut best_proj = 0.0f32;
    let mut found = false;
    let mut t = -w_perp;
    while t <= w_perp + 1e-3 {
        let p = [center[0] + t * normal[0], center[1] + t * normal[1]];
        if !roi.contains(&p) {
            t += params.delta_t;
            continue;
        }
        if let Some((gx, gy)) = bilinear_grad(lvl, p[0], p[1]) {
            let mag_sq = gx * gx + gy * gy;
            if mag_sq < tau_mag_sq {
                t += params.delta_t;
                continue;
            }
            let proj = gx * normal[0] + gy * normal[1];
            let val = proj.abs();
            if !found || val > best_proj {
                best_proj = val;
                best_t = t;
                found = true;
            }
        }
        t += params.delta_t;
    }

    if !found {
        return None;
    }

    let refined_t = quadratic_refine(lvl, roi, center, normal, params.delta_t, best_t, best_proj);
    let pos = [
        center[0] + refined_t * normal[0],
        center[1] + refined_t * normal[1],
    ];
    let (gx, gy) = bilinear_grad(lvl, pos[0], pos[1])?;
    let proj = gx * normal[0] + gy * normal[1];
    let peak = proj.abs();
    let weight = huber_weight(peak, params.huber_delta);
    Some(SupportPoint {
        pos,
        weight,
        grad: [gx, gy],
    })
}

#[inline(always)]
pub(crate) fn bilinear_grad(lvl: &PyramidLevel<'_>, x: f32, y: f32) -> Option<(f32, f32)> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let x_rel = x - lvl.origin_x as f32;
    let y_rel = y - lvl.origin_y as f32;
    if x_rel < 0.0 || y_rel < 0.0 {
        return None;
    }
    let xf = x_rel.floor();
    let yf = y_rel.floor();
    let x0 = xf as isize;
    let y0 = yf as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x1 >= lvl.tile_width as isize || y1 >= lvl.tile_height as isize {
        return None;
    }
    #[cfg(feature = "profile_refine")]
    profile::record_sample(lvl.level_index);
    let tx = x_rel - xf;
    let ty = y_rel - yf;
    let stride = lvl.tile_width as isize;
    let base = y0 * stride + x0;
    let gx00 = unsafe { *lvl.gx.get_unchecked(base as usize) };
    let gx10 = unsafe { *lvl.gx.get_unchecked((base + 1) as usize) };
    let gx01 = unsafe { *lvl.gx.get_unchecked((base + stride) as usize) };
    let gx11 = unsafe { *lvl.gx.get_unchecked((base + stride + 1) as usize) };
    let gy00 = unsafe { *lvl.gy.get_unchecked(base as usize) };
    let gy10 = unsafe { *lvl.gy.get_unchecked((base + 1) as usize) };
    let gy01 = unsafe { *lvl.gy.get_unchecked((base + stride) as usize) };
    let gy11 = unsafe { *lvl.gy.get_unchecked((base + stride + 1) as usize) };

    let inv_tx = 1.0 - tx;
    let inv_ty = 1.0 - ty;
    let gx0 = gx00 * inv_tx + gx10 * tx;
    let gx1 = gx01 * inv_tx + gx11 * tx;
    let gx = gx0 * inv_ty + gx1 * ty;

    let gy0 = gy00 * inv_tx + gy10 * tx;
    let gy1 = gy01 * inv_tx + gy11 * tx;
    let gy = gy0 * inv_ty + gy1 * ty;
    Some((gx, gy))
}

fn quadratic_refine(
    lvl: &PyramidLevel<'_>,
    roi: &SegmentRoi,
    center: &[f32; 2],
    normal: &[f32; 2],
    delta_t: f32,
    t_best: f32,
    peak_best: f32,
) -> f32 {
    let offsets = [-delta_t, 0.0, delta_t];
    let mut samples = [peak_best; 3];
    for (i, off) in offsets.iter().enumerate() {
        if *off == 0.0 {
            continue;
        }
        let t = t_best + *off;
        let p = [center[0] + t * normal[0], center[1] + t * normal[1]];
        if !roi.contains(&p) {
            continue;
        }
        if let Some((gx, gy)) = bilinear_grad(lvl, p[0], p[1]) {
            let proj = gx * normal[0] + gy * normal[1];
            samples[i] = proj.abs();
        }
    }
    let f0 = samples[0];
    let f1 = samples[1];
    let f2 = samples[2];
    let denom = (f0 - 2.0 * f1 + f2).abs().max(EPS);
    let shift = 0.5 * (f0 - f2) / denom;
    (t_best + shift * delta_t).clamp(t_best - delta_t, t_best + delta_t)
}

fn huber_weight(value: f32, delta: f32) -> f32 {
    if value <= delta {
        1.0
    } else {
        (delta / value).max(1e-3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_grad_returns_none_outside() {
        let lvl = PyramidLevel {
            width: 4,
            height: 4,
            origin_x: 0,
            origin_y: 0,
            tile_width: 4,
            tile_height: 4,
            gx: &[0.0; 16],
            gy: &[0.0; 16],
            level_index: 0,
        };
        assert!(bilinear_grad(&lvl, -1.0, 0.0).is_none());
        assert!(bilinear_grad(&lvl, 3.5, 3.5).is_none());
    }

    #[test]
    fn huber_weight_basic() {
        assert!((huber_weight(0.05, 0.25) - 1.0).abs() < 1e-6);
        assert!((huber_weight(1.0, 0.25) - 0.25).abs() < 1e-6);
    }
}
