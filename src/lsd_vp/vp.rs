use crate::segments::Segment;
use log::debug;
use nalgebra::Vector3;

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
        let a = s.line[0];
        let b = s.line[1];
        let c = s.line[2];
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

#[cfg(test)]
mod tests {
    use super::estimate_vp;
    use crate::segments::Segment;
    use nalgebra::Vector3;

    fn make_segment(line: [f32; 3], dir: [f32; 2], strength: f32) -> Segment {
        Segment {
            p0: [0.0, 0.0],
            p1: [dir[0], dir[1]],
            dir,
            len: strength,
            line: nalgebra::Vector3::new(line[0], line[1], line[2]),
            avg_mag: 1.0,
            strength,
        }
    }

    fn approx_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> bool {
        (a - b).norm() < 1e-3
    }

    #[test]
    fn finite_intersection() {
        // Lines x=10 and y=20 should intersect at (10,20,1)
        let segs = vec![
            make_segment([1.0, 0.0, -10.0], [0.0, 1.0], 20.0),
            make_segment([0.0, 1.0, -20.0], [1.0, 0.0], 20.0),
        ];
        let vp = estimate_vp(&segs, &[0, 1], 0.0).expect("vp");
        assert!(approx_vec(&vp, &Vector3::new(10.0, 20.0, 1.0)));
    }

    #[test]
    fn fallback_direction_when_parallel() {
        // Parallel lines should trigger fallback direction (point at infinity)
        let segs = vec![
            make_segment([1.0, 0.0, -5.0], [0.0, 1.0], 15.0),
            make_segment([1.0, 0.0, -15.0], [0.0, 1.0], 15.0),
        ];
        let fallback = std::f32::consts::FRAC_PI_4;
        let vp = estimate_vp(&segs, &[0, 1], fallback).expect("vp");
        assert!(approx_vec(
            &vp,
            &Vector3::new(fallback.cos(), fallback.sin(), 0.0)
        ));
    }
}
