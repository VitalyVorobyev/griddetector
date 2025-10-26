//! Angle utilities used across the detector pipeline.
use log::warn;
use nalgebra::Vector3;

/// Small epsilon used to detect vanishing points at infinity (homogeneous z≈0).
pub const VP_EPS: f32 = 1e-3;

/// Normalizes an angle into the range [0, π).
#[inline]
pub fn normalize_half_pi(angle: f32) -> f32 {
    let mut norm = angle.rem_euclid(std::f32::consts::PI);
    if norm >= std::f32::consts::PI {
        norm -= std::f32::consts::PI;
    }
    if norm >= std::f32::consts::PI - 1e-6 {
        0.0
    } else {
        norm
    }
}

/// Computes the smallest unsigned angular difference between two angles,
/// treating antipodal directions as equivalent (i.e. π apart → 0).
#[inline]
pub fn angular_difference(a: f32, b: f32) -> f32 {
    let mut diff = (a - b).abs();
    if diff > std::f32::consts::PI {
        diff = diff.rem_euclid(std::f32::consts::PI);
    }
    if diff > std::f32::consts::FRAC_PI_2 {
        std::f32::consts::PI - diff
    } else {
        diff
    }
}

/// Computes the unsigned angle between two 2D vectors in radians.
/// Returns a value in [0, π]. Zero if the vectors are parallel
/// and pointing in the same direction; π if they are opposite.
#[inline]
pub fn angle_between(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let na = (a[0] * a[0] + a[1] * a[1]).sqrt().max(1e-6);
    let nb = (b[0] * b[0] + b[1] * b[1]).sqrt().max(1e-6);
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

/// Computes the orientation difference between two 2D vectors while treating
/// antipodal directions as equivalent. Returns a value in [0, π/2].
///
/// This is appropriate when comparing line tangents or vanishing directions
/// where the sign of the direction is not meaningful.
#[inline]
pub fn angle_between_dirless(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let na = (a[0] * a[0] + a[1] * a[1]).sqrt().max(1e-6);
    let nb = (b[0] * b[0] + b[1] * b[1]).sqrt().max(1e-6);
    let cos = (dot / (na * nb)).abs().clamp(-1.0, 1.0);
    cos.acos()
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
    use nalgebra::Vector3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn normalize_half_pi_basic() {
        assert!(approx_eq(normalize_half_pi(0.5), 0.5));
        assert!(approx_eq(
            normalize_half_pi(-std::f32::consts::FRAC_PI_4),
            3.0 * std::f32::consts::FRAC_PI_4
        ));
        assert!(approx_eq(normalize_half_pi(std::f32::consts::PI), 0.0));
        assert!(approx_eq(
            normalize_half_pi(3.0 * std::f32::consts::PI),
            0.0
        ));
    }

    #[test]
    fn angular_difference_is_symmetric() {
        let a = 0.25f32;
        let b = 1.7f32;
        assert!(approx_eq(
            angular_difference(a, b),
            angular_difference(b, a)
        ));
    }

    #[test]
    fn angular_difference_handles_wrap() {
        assert!(approx_eq(
            angular_difference(0.0, std::f32::consts::PI),
            0.0
        ));
        assert!(approx_eq(
            angular_difference(0.0, std::f32::consts::FRAC_PI_2),
            std::f32::consts::FRAC_PI_2
        ));
        assert!(approx_eq(
            angular_difference(std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_4),
            std::f32::consts::FRAC_PI_2
        ));
    }

    #[test]
    fn angle_between_basic() {
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 0.0];
        assert!(approx_eq(angle_between(&a, &b), 0.0));

        let c = [-1.0f32, 0.0];
        assert!(approx_eq(angle_between(&a, &c), std::f32::consts::PI));

        let d = [0.0f32, 1.0];
        assert!(approx_eq(
            angle_between(&a, &d),
            std::f32::consts::FRAC_PI_2
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
