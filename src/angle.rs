//! Angle utilities used across the detector pipeline.

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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn normalize_half_pi_basic() {
        assert!(approx_eq(normalize_half_pi(0.5), 0.5));
        assert!(approx_eq(normalize_half_pi(-std::f32::consts::FRAC_PI_4), 3.0 * std::f32::consts::FRAC_PI_4));
        assert!(approx_eq(normalize_half_pi(std::f32::consts::PI), 0.0));
        assert!(approx_eq(normalize_half_pi(3.0 * std::f32::consts::PI), 0.0));
    }

    #[test]
    fn angular_difference_is_symmetric() {
        let a = 0.25f32;
        let b = 1.7f32;
        assert!(approx_eq(angular_difference(a, b), angular_difference(b, a)));
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
}
