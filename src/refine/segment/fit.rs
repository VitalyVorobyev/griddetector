//! Support point representation and robust carrier fitting utilities.

use super::{types::RefineParams, EPS};

/// Weighted support sample collected along the carrier normal.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SupportPoint {
    pub(crate) pos: [f32; 2],
    pub(crate) weight: f32,
    pub(crate) grad: [f32; 2],
}

type LineFitResult = ([f32; 2], [f32; 2], f32, [f32; 2]);

pub(crate) fn weighted_line_fit(
    supports: &[SupportPoint],
    params: &RefineParams,
) -> Option<LineFitResult> {
    let mut sum_w = 0.0;
    let mut mu = [0.0f32, 0.0];
    for s in supports {
        sum_w += s.weight;
        mu[0] += s.weight * s.pos[0];
        mu[1] += s.weight * s.pos[1];
    }
    if sum_w <= EPS {
        return None;
    }
    mu[0] /= sum_w;
    mu[1] /= sum_w;

    let mut cov_xx = 0.0f32;
    let mut cov_xy = 0.0f32;
    let mut cov_yy = 0.0f32;
    for s in supports {
        let dx = s.pos[0] - mu[0];
        let dy = s.pos[1] - mu[1];
        cov_xx += s.weight * dx * dx;
        cov_xy += s.weight * dx * dy;
        cov_yy += s.weight * dy * dy;
    }
    cov_xx /= sum_w;
    cov_xy /= sum_w;
    cov_yy /= sum_w;

    let trace = cov_xx + cov_yy;
    let det_part = (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy;
    let lambda = 0.5 * (trace + det_part.max(0.0).sqrt());
    let mut dir = [cov_xy, lambda - cov_xx];
    let norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if norm <= EPS {
        dir = [1.0, 0.0];
    } else {
        dir[0] /= norm;
        dir[1] /= norm;
    }
    let mut normal = [-dir[1], dir[0]];

    if params.tau_ori_deg > 0.0 {
        let tau_rad = params.tau_ori_deg.to_radians();
        let mut weights = Vec::with_capacity(supports.len());
        let mut sum_w2 = 0.0f32;
        for s in supports {
            let grad = s.grad;
            let mag = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt().max(EPS);
            let dot = (grad[0] * normal[0] + grad[1] * normal[1]) / mag;
            let cos = dot.clamp(-1.0, 1.0);
            let angle = cos.acos();
            let ori_weight = if angle <= tau_rad { cos.max(0.0) } else { 0.0 };
            let w = (s.weight * ori_weight).max(0.0);
            weights.push(w);
            sum_w2 += w;
        }
        if sum_w2 > EPS {
            let mut mu2 = [0.0f32, 0.0];
            for (s, &w) in supports.iter().zip(weights.iter()) {
                mu2[0] += w * s.pos[0];
                mu2[1] += w * s.pos[1];
            }
            mu2[0] /= sum_w2;
            mu2[1] /= sum_w2;
            let mut cov_xx2 = 0.0f32;
            let mut cov_xy2 = 0.0f32;
            let mut cov_yy2 = 0.0f32;
            for (s, &w) in supports.iter().zip(weights.iter()) {
                let dx = s.pos[0] - mu2[0];
                let dy = s.pos[1] - mu2[1];
                cov_xx2 += w * dx * dx;
                cov_xy2 += w * dx * dy;
                cov_yy2 += w * dy * dy;
            }
            cov_xx2 /= sum_w2;
            cov_xy2 /= sum_w2;
            cov_yy2 /= sum_w2;
            let trace2 = cov_xx2 + cov_yy2;
            let det_part2 = (cov_xx2 - cov_yy2) * (cov_xx2 - cov_yy2) + 4.0 * cov_xy2 * cov_xy2;
            let lambda2 = 0.5 * (trace2 + det_part2.max(0.0).sqrt());
            let mut dir2 = [cov_xy2, lambda2 - cov_xx2];
            let norm2 = (dir2[0] * dir2[0] + dir2[1] * dir2[1]).sqrt();
            if norm2 > EPS {
                dir2[0] /= norm2;
                dir2[1] /= norm2;
                dir = dir2;
                normal = [-dir[1], dir[0]];
                mu = mu2;
            }
        }
    }

    let rho = normal[0] * mu[0] + normal[1] * mu[1];
    Some((dir, normal, rho, mu))
}

#[inline]
pub(crate) fn project_point_to_line(p: &[f32; 2], normal: &[f32; 2], rho: f32) -> [f32; 2] {
    let dist = (normal[0] * p[0] + normal[1] * p[1]) - rho;
    [p[0] - dist * normal[0], p[1] - dist * normal[1]]
}

#[inline]
pub(crate) fn distance(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

#[inline]
pub(crate) fn direction(p0: &[f32; 2], p1: &[f32; 2]) -> Option<[f32; 2]> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let norm = (dx * dx + dy * dy).sqrt();
    if norm <= EPS {
        None
    } else {
        Some([dx / norm, dy / norm])
    }
}

#[inline]
pub(crate) fn normal_from_segment(p0: &[f32; 2], p1: &[f32; 2]) -> Option<[f32; 2]> {
    direction(p0, p1).map(|d| [-d[1], d[0]])
}

#[inline]
pub(crate) fn midpoint(a: &[f32; 2], b: &[f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}
