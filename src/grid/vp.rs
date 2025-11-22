//! Vanishing-point estimation from bundled line constraints.
//!
//! This module clusters bundle orientations to find two dominant, roughly
//! orthogonal line families and fits vanishing points via weighted least
//! squares on bundle lines. When lines are near-parallel, the estimator falls
//! back to a vanishing point at infinity along the peak orientation.

use super::bundling::{Bundle, BundleId};
use super::histogram::OrientationHistogram;
use crate::angle::{angular_difference, normalize_half_pi};
use serde::{Deserialize, Serialize};

const VP_EPS: f32 = 1e-6;

/// Label for the two dominant families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FamilyLabel {
    U,
    V,
}

/// Vanishing point estimate with support information.
#[derive(Clone, Debug, Serialize)]
pub struct VanishingPoint {
    pub pos: [f32; 3],
    pub dir: [f32; 2],
    pub support: Vec<BundleId>,
    pub score: f32,
}

impl VanishingPoint {
    pub fn at_infinity(dir: [f32; 2], support: Vec<BundleId>, score: f32) -> Self {
        Self {
            pos: [dir[0], dir[1], 0.0],
            dir,
            support,
            score,
        }
    }
}

/// Pair of vanishing points representing the orthogonal grid axes.
#[derive(Clone, Debug, Serialize)]
pub struct VanishingPair {
    pub u: VanishingPoint,
    pub v: VanishingPoint,
    pub separation_rad: f32,
    pub assignments: Vec<Option<FamilyLabel>>,
}

/// Options controlling VP estimation robustness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VpEstimationOptions {
    /// Minimum separation between dominant peaks (degrees).
    pub min_peak_sep_deg: f32,
    /// Orientation tolerance when assigning bundles to a family (degrees).
    pub assign_tol_deg: f32,
    /// Minimum supporting bundles per VP.
    pub min_support: usize,
}

impl Default for VpEstimationOptions {
    fn default() -> Self {
        Self {
            min_peak_sep_deg: 35.0,
            assign_tol_deg: 20.0,
            min_support: 4,
        }
    }
}

/// Estimate two orthogonal vanishing points from bundled lines.
pub fn estimate_vanishing_pair(
    bundles: &[Bundle],
    opts: &VpEstimationOptions,
) -> Option<VanishingPair> {
    if bundles.len() < opts.min_support * 2 {
        return None;
    }
    let mut hist = OrientationHistogram::default();
    let mut angles = Vec::with_capacity(bundles.len());
    for b in bundles {
        let t = b.tangent();
        let th = normalize_half_pi(t[1].atan2(t[0]));
        hist.accumulate(th, b.weight.max(1.0));
        angles.push(th);
    }
    hist.smooth_121();
    let min_sep = opts.min_peak_sep_deg.to_radians();
    let (peak_u, peak_v) = hist.find_two_peaks(min_sep)?;
    let theta_u = hist.refined_angle(peak_u, 1);
    let theta_v = hist.refined_angle(peak_v, 1);
    let assignments = assign_families(&angles, theta_u, theta_v, opts.assign_tol_deg.to_radians());
    let support_u: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter_map(|(i, lab)| matches!(lab, Some(FamilyLabel::U)).then_some(i))
        .collect();
    let support_v: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter_map(|(i, lab)| matches!(lab, Some(FamilyLabel::V)).then_some(i))
        .collect();
    if support_u.len() < opts.min_support || support_v.len() < opts.min_support {
        return None;
    }

    let vp_u = fit_vp(bundles, &support_u, theta_u)?;
    let vp_v = fit_vp(bundles, &support_v, theta_v)?;
    let sep = angular_difference(theta_u, theta_v);
    Some(VanishingPair {
        u: vp_u,
        v: vp_v,
        separation_rad: sep,
        assignments,
    })
}

fn assign_families(
    angles: &[f32],
    theta_u: f32,
    theta_v: f32,
    tol: f32,
) -> Vec<Option<FamilyLabel>> {
    let mut families = Vec::with_capacity(angles.len());
    for &angle in angles {
        let d1 = angular_difference(angle, theta_u);
        let d2 = angular_difference(angle, theta_v);
        if d1 <= d2 && d1 <= tol {
            families.push(Some(FamilyLabel::U));
        } else if d2 < d1 && d2 <= tol {
            families.push(Some(FamilyLabel::V));
        } else {
            families.push(None);
        }
    }
    families
}

fn fit_vp(bundles: &[Bundle], indices: &[usize], fallback_theta: f32) -> Option<VanishingPoint> {
    let mut a11 = 0.0f32;
    let mut a12 = 0.0f32;
    let mut a22 = 0.0f32;
    let mut bx = 0.0f32;
    let mut by = 0.0f32;
    let mut total_w = 0.0f32;
    for &idx in indices {
        let b = &bundles[idx];
        let line = b.line;
        let a = line[0];
        let c = line[1];
        let d = line[2];
        let w = b.weight.max(1.0);
        a11 += w * a * a;
        a12 += w * a * c;
        a22 += w * c * c;
        bx += -w * d * a;
        by += -w * d * c;
        total_w += w;
    }
    if total_w <= VP_EPS {
        return None;
    }
    let det = a11 * a22 - a12 * a12;
    let trace = a11 + a22;
    let pos = if det.abs() <= 1e-6f32.max(1e-6 * trace * trace) {
        let dir_x = fallback_theta.cos();
        let dir_y = fallback_theta.sin();
        [dir_x, dir_y, 0.0]
    } else {
        let inv11 = a22 / det;
        let inv12 = -a12 / det;
        let inv22 = a11 / det;
        let x = inv11 * bx + inv12 * by;
        let y = inv12 * bx + inv22 * by;
        [x, y, 1.0]
    };

    let dir = vp_direction(&pos, &[0.0, 0.0, 1.0])?;
    let support_ids = indices.iter().map(|&i| bundles[i].id).collect::<Vec<_>>();
    Some(VanishingPoint {
        pos,
        dir,
        support: support_ids,
        score: total_w,
    })
}

/// Computes a unit direction in image space from the translation anchor
/// towards the vanishing point. For VPs at infinity (vp.z≈0), returns the
/// normalized direction encoded by `(vp.x, vp.y, 0)`.
pub fn vp_direction(vp: &[f32; 3], anchor: &[f32; 3]) -> Option<[f32; 2]> {
    if vp[2].abs() <= VP_EPS {
        let norm = (vp[0] * vp[0] + vp[1] * vp[1]).sqrt();
        if norm <= 1e-6 {
            return None;
        }
        Some([vp[0] / norm, vp[1] / norm])
    } else {
        let vx = vp[0] / vp[2];
        let vy = vp[1] / vp[2];
        let ax = anchor[0] / anchor[2];
        let ay = anchor[1] / anchor[2];
        let dx = vx - ax;
        let dy = vy - ay;
        let norm = (dx * dx + dy * dy).sqrt();
        if norm <= 1e-6 {
            None
        } else {
            Some([dx / norm, dy / norm])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::bundling::{Bundle, BundleId};

    fn make_bundle(angle: f32, rho: f32, weight: f32, id: u32) -> Bundle {
        let n = [angle.cos(), angle.sin()];
        Bundle {
            id: BundleId(id),
            line: [n[0], n[1], -rho],
            center: [rho * -n[0], rho * -n[1]],
            weight,
            members: vec![],
        }
    }

    #[test]
    fn estimate_two_finite_vps() {
        // Lines x = 10 and y = 20 should yield finite VPs.
        let bundles = vec![
            make_bundle(0.0, 10.0, 5.0, 0),
            make_bundle(std::f32::consts::FRAC_PI_2, 20.0, 5.0, 1),
        ];
        let opts = VpEstimationOptions {
            min_support: 1,
            min_peak_sep_deg: 20.0,
            assign_tol_deg: 25.0,
        };
        let vp = estimate_vanishing_pair(&bundles, &opts).expect("vp pair");
        assert!((vp.u.pos[0] - 10.0).abs() < 1e-2 && (vp.u.pos[2] - 1.0).abs() < 1e-3);
        assert!((vp.v.pos[1] - 20.0).abs() < 1e-2 && (vp.v.pos[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn fallback_for_parallel_lines() {
        let angle = std::f32::consts::FRAC_PI_4;
        let bundles = vec![
            make_bundle(angle, 10.0, 3.0, 0),
            make_bundle(angle, 12.0, 3.0, 1),
        ];
        let opts = VpEstimationOptions {
            min_support: 1,
            ..Default::default()
        };
        let vp = estimate_vanishing_pair(&bundles, &opts).expect("vp pair");
        assert!(
            vp.u.pos[2].abs() <= 1e-6,
            "expected VP at infinity for parallel lines"
        );
    }
}
