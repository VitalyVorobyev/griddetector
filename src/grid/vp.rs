use crate::segments::Segment;
use log::{debug, warn};
use nalgebra::Vector3;

const VP_EPS: f32 = 1e-6;

/// Estimates a vanishing point from a family of line segments given in normal form.
pub(crate) fn estimate_vp(
    segs: &[Segment],
    indices: &[usize],
    fallback_theta: f32,
) -> Option<Vector3<f32>> {
    if indices.is_empty() {
        return None;
    }

    // Solve for v=(x,y) minimizing Σ w (a x + b y + c)^2 where each line is ax+by+c=0
    let mut a11 = 0.0f32;
    let mut a12 = 0.0f32;
    let mut a22 = 0.0f32;
    let mut bx = 0.0f32;
    let mut by = 0.0f32;
    for &idx in indices {
        let s = &segs[idx];
        let line = s.line();
        let a = line[0];
        let b = line[1];
        let c = line[2];
        let w = s.strength.max(1.0);
        a11 += w * a * a;
        a12 += w * a * b;
        a22 += w * b * b;
        bx += -w * c * a;
        by += -w * c * b;
    }
    let det = a11 * a22 - a12 * a12;
    let trace = a11 + a22;
    if det.abs() <= 1e-6f32.max(1e-6 * trace * trace) {
        debug!(
            "LSD-VP: normal matrix near-singular when estimating VP, falling back to histogram direction"
        );
        let dir_x = fallback_theta.cos();
        let dir_y = fallback_theta.sin();
        return Some(Vector3::new(dir_x, dir_y, 0.0));
    }
    let inv11 = a22 / det;
    let inv12 = -a12 / det;
    let inv22 = a11 / det;
    let x = inv11 * bx + inv12 * by;
    let y = inv12 * bx + inv22 * by;
    Some(Vector3::new(x, y, 1.0))
}

/// Computes a unit direction in image space from the translation anchor
/// towards the vanishing point. For VPs at infinity (vp.z≈0), returns the
/// normalized direction encoded by `(vp.x, vp.y, 0)`.
///
/// Returns `None` if the direction cannot be determined (degenerate inputs).
#[inline]
pub fn vp_direction(vp: &Vector3<f32>, anchor: &Vector3<f32>) -> Option<[f32; 2]> {
    if vp[2].abs() <= VP_EPS {
        let norm = (vp[0] * vp[0] + vp[1] * vp[1]).sqrt();
        if norm <= 1e-6 {
            warn!("Degenerate vanishing point at infinity encountered");
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
    use crate::segments::{Segment, SegmentId};
    use nalgebra::Vector3;

    fn make_segment(p0: [f32; 2], p1: [f32; 2], strength: f32, id: u32) -> Segment {
        Segment::new(SegmentId(id), p0, p1, 1.0, strength)
    }

    fn approx_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> bool {
        (a - b).norm() < 1e-3
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn finite_intersection() {
        // Lines x=10 and y=20 should intersect at (10,20,1)
        let segs = vec![
            make_segment([10.0, 0.0], [10.0, 1.0], 20.0, 0),
            make_segment([0.0, 20.0], [1.0, 20.0], 20.0, 1),
        ];
        let vp = estimate_vp(&segs, &[0, 1], 0.0).expect("vp");
        assert!(approx_vec(&vp, &Vector3::new(10.0, 20.0, 1.0)));
    }

    #[test]
    fn fallback_direction_when_parallel() {
        // Parallel lines should trigger fallback direction (point at infinity)
        let segs = vec![
            make_segment([11.0, 0.0], [11.0, 1.0], 15.0, 0),
            make_segment([10.0, 0.0], [10.0, 1.0], 15.0, 1),
        ];
        let fallback = std::f32::consts::FRAC_PI_4;
        let vp = estimate_vp(&segs, &[0, 1], fallback).expect("vp");
        assert!(approx_vec(
            &vp,
            &Vector3::new(fallback.cos(), fallback.sin(), 0.0)
        ));
    }

    #[test]
    fn vp_direction_finite_and_infinite() {
        // Finite VP to the right of anchor
        let anchor = Vector3::new(100.0f32, 100.0, 1.0);
        let vp_fin = Vector3::new(200.0f32, 100.0, 1.0);
        let dir = vp_direction(&vp_fin, &anchor).expect("finite vp direction");
        assert!(approx_eq(dir[0], 1.0) && approx_eq(dir[1], 0.0));

        // VP at infinity along +x
        let vp_inf = Vector3::new(1.0f32, 0.0, 0.0);
        let dir_inf = vp_direction(&vp_inf, &anchor).expect("infinite vp direction");
        assert!(approx_eq(dir_inf[0], 1.0) && approx_eq(dir_inf[1], 0.0));

        // Degenerate VP
        let vp_bad = Vector3::new(0.0f32, 0.0, 0.0);
        assert!(vp_direction(&vp_bad, &anchor).is_none());
    }
}
