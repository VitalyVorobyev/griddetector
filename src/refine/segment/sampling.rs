//! Gradient sampling along the carrier normal.

use super::{fit::SupportPoint, types::PyramidLevel, types::RefineParams, Roi, EPS};

pub(crate) fn search_along_normal(
    lvl: &PyramidLevel<'_>,
    roi: &Roi,
    center: &[f32; 2],
    normal: &[f32; 2],
    params: &RefineParams,
    w_perp: f32,
) -> Option<SupportPoint> {
    let mut best: Option<(f32, f32, [f32; 2], f32)> = None; // (t, |g|, grad, proj)
    let mut t = -w_perp;
    while t <= w_perp + 1e-3 {
        let p = [center[0] + t * normal[0], center[1] + t * normal[1]];
        if !roi.contains(&p) {
            t += params.delta_t;
            continue;
        }
        if let Some((gx, gy)) = bilinear_grad(lvl, p[0], p[1]) {
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < params.tau_mag {
                t += params.delta_t;
                continue;
            }
            let proj = gx * normal[0] + gy * normal[1];
            let val = proj.abs();
            if let Some(ref mut current) = best {
                if val > current.1 {
                    *current = (t, val, [gx, gy], proj);
                }
            } else {
                best = Some((t, val, [gx, gy], proj));
            }
        }
        t += params.delta_t;
    }

    let (t_best, mut peak, _, _) = best?;

    let refined_t = quadratic_refine(lvl, roi, center, normal, params.delta_t, t_best, peak);
    let pos = [
        center[0] + refined_t * normal[0],
        center[1] + refined_t * normal[1],
    ];
    let (gx, gy) = bilinear_grad(lvl, pos[0], pos[1])?;
    let proj = gx * normal[0] + gy * normal[1];
    peak = proj.abs();
    let weight = huber_weight(peak, params.huber_delta);
    Some(SupportPoint {
        pos,
        weight,
        grad: [gx, gy],
    })
}

pub(crate) fn bilinear_grad(lvl: &PyramidLevel<'_>, x: f32, y: f32) -> Option<(f32, f32)> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let w = lvl.width as isize;
    let h = lvl.height as isize;
    let xf = x.floor();
    let yf = y.floor();
    let x0 = xf as isize;
    let y0 = yf as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x1 >= w || y1 >= h {
        return None;
    }
    let tx = x - xf;
    let ty = y - yf;
    let idx = |xx: isize, yy: isize| -> usize { (yy as usize) * lvl.width + (xx as usize) };
    let gx00 = lvl.gx[idx(x0, y0)];
    let gx10 = lvl.gx[idx(x1, y0)];
    let gx01 = lvl.gx[idx(x0, y1)];
    let gx11 = lvl.gx[idx(x1, y1)];
    let gy00 = lvl.gy[idx(x0, y0)];
    let gy10 = lvl.gy[idx(x1, y0)];
    let gy01 = lvl.gy[idx(x0, y1)];
    let gy11 = lvl.gy[idx(x1, y1)];

    let gx0 = gx00 * (1.0 - tx) + gx10 * tx;
    let gx1 = gx01 * (1.0 - tx) + gx11 * tx;
    let gx = gx0 * (1.0 - ty) + gx1 * ty;

    let gy0 = gy00 * (1.0 - tx) + gy10 * tx;
    let gy1 = gy01 * (1.0 - tx) + gy11 * tx;
    let gy = gy0 * (1.0 - ty) + gy1 * ty;
    Some((gx, gy))
}

fn quadratic_refine(
    lvl: &PyramidLevel<'_>,
    roi: &Roi,
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
            gx: &[0.0; 16],
            gy: &[0.0; 16],
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
