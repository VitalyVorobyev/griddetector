use super::bundling::Bundle;
use log::warn;
use nalgebra::Vector3;

const EPS: f32 = 1e-6;

struct VpStats {
    confidence: f32,
    total_weight: f32,
    inlier_weight: f32,
}

fn normalize_vp(vp: &mut Vector3<f32>) {
    if vp[2].abs() <= 1e-3 {
        let norm = (vp[0] * vp[0] + vp[1] * vp[1]).sqrt().max(EPS);
        vp[0] /= norm;
        vp[1] /= norm;
        vp[2] = 0.0;
    } else {
        let inv = 1.0 / vp[2];
        vp[0] *= inv;
        vp[1] *= inv;
        vp[2] = 1.0;
    }
}

fn huber_weight(residual: f32, delta: f32) -> (f32, bool) {
    let abs = residual.abs();
    if abs <= delta {
        (1.0, true)
    } else {
        (delta / abs, false)
    }
}

/// Estimate a vanishing point by solving the Huber-weighted normal equations
/// over a bundle family using Iteratively Reweighted Least Squares (IRLS).
///
/// The solver monitors singular configurations and falls back to a simple
/// average of bundle tangents if the covariance collapses.
pub(crate) fn estimate_vp_huber(
    bundles: &[&Bundle],
    vp_init: &Vector3<f32>,
    delta: f32,
    max_iters: usize,
) -> Option<(Vector3<f32>, VpStats)> {
    let mut vp = *vp_init;
    normalize_vp(&mut vp);
    let mut stats = VpStats {
        confidence: 0.0,
        total_weight: 0.0,
        inlier_weight: 0.0,
    };
    for _ in 0..max_iters {
        let mut accum = NormalEquationAccum::default();
        for bundle in bundles {
            accum.accumulate(bundle, &vp, delta);
        }
        if accum.total_weight <= EPS {
            break;
        }
        let det = accum.determinant();
        if det.abs() <= EPS {
            warn!("VP estimation: normal equations are singular, falling back to tangent average");
            return fallback_vp_from_bundles(bundles);
        }
        let (x, y) = accum.solve(det);
        let new_vp = Vector3::new(x, y, 1.0);
        // Ensure the updated vanishing point is normalised for subsequent iterations.
        let mut vp_norm = new_vp;
        normalize_vp(&mut vp_norm);
        stats.total_weight = accum.total_weight;
        stats.inlier_weight = accum.inlier_weight;
        stats.confidence = (accum.inlier_weight / (bundles.len() as f32 + EPS)).clamp(0.0, 1.0);
        if (vp_norm - vp).norm() < 1e-3 {
            return Some((vp_norm, stats));
        }
        vp = vp_norm;
    }
    Some((vp, stats))
}

#[derive(Default)]
struct NormalEquationAccum {
    a11: f32,
    a12: f32,
    a22: f32,
    bx: f32,
    by: f32,
    total_weight: f32,
    inlier_weight: f32,
}

impl NormalEquationAccum {
    fn accumulate(&mut self, bundle: &Bundle, vp: &Vector3<f32>, delta: f32) {
        let line = bundle.line;
        let residual = line[0] * vp[0] + line[1] * vp[1] + line[2] * vp[2];
        let (h_weight, inlier) = huber_weight(residual, delta);
        let w = bundle.weight * h_weight;
        if w <= EPS {
            return;
        }
        self.a11 += w * line[0] * line[0];
        self.a12 += w * line[0] * line[1];
        self.a22 += w * line[1] * line[1];
        self.bx += -w * line[2] * line[0];
        self.by += -w * line[2] * line[1];
        self.total_weight += w;
        if inlier {
            self.inlier_weight += bundle.weight;
        }
    }

    fn determinant(&self) -> f32 {
        self.a11 * self.a22 - self.a12 * self.a12
    }

    fn solve(&self, det: f32) -> (f32, f32) {
        let inv11 = self.a22 / det;
        let inv12 = -self.a12 / det;
        let inv22 = self.a11 / det;
        let x = inv11 * self.bx + inv12 * self.by;
        let y = inv12 * self.bx + inv22 * self.by;
        (x, y)
    }
}

fn fallback_vp_from_bundles(bundles: &[&Bundle]) -> Option<(Vector3<f32>, VpStats)> {
    let mut sum_tx = 0.0f32;
    let mut sum_ty = 0.0f32;
    let mut total_w = 0.0f32;
    for bundle in bundles {
        let tangent = bundle.tangent();
        sum_tx += tangent[0] * bundle.weight;
        sum_ty += tangent[1] * bundle.weight;
        total_w += bundle.weight;
    }
    let norm = (sum_tx * sum_tx + sum_ty * sum_ty).sqrt();
    if norm <= EPS {
        return None;
    }
    let vp = Vector3::new(sum_tx / norm, sum_ty / norm, 0.0);
    // Assign a small positive confidence proportional to support so callers
    // can distinguish a valid (at-infinity) estimate from a hard failure.
    let support_conf = ((bundles.len().min(50) as f32) / 50.0).clamp(0.0, 1.0);
    Some((
        vp,
        VpStats {
            confidence: support_conf,
            total_weight: total_w,
            inlier_weight: total_w,
        },
    ))
}
