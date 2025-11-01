use crate::angle::normalize_half_pi;

/// Circular histogram over [0, π) used to find dominant line orientations.
pub(crate) struct OrientationHistogram {
    bins: Vec<f32>,
    bin_width: f32,
}

impl OrientationHistogram {
    pub(crate) fn new(num_bins: usize) -> Self {
        assert!(
            num_bins > 0,
            "orientation histogram requires at least one bin"
        );
        OrientationHistogram {
            bins: vec![0.0; num_bins],
            bin_width: std::f32::consts::PI / num_bins as f32,
        }
    }

    #[cfg(test)]
    pub(crate) fn bin_width(&self) -> f32 {
        self.bin_width
    }

    #[cfg(test)]
    pub(crate) fn bins(&self) -> &[f32] {
        &self.bins
    }

    pub(crate) fn accumulate(&mut self, angle: f32, weight: f32) {
        if self.bins.is_empty() || !angle.is_finite() {
            return;
        }
        let mut idx = (angle / self.bin_width) as usize;
        if idx >= self.bins.len() {
            idx = self.bins.len() - 1;
        }
        self.bins[idx] += weight.max(0.0);
    }

    /// Applies a circular [1, 2, 1]/4 smoothing kernel to reduce bin quantization noise.
    pub(crate) fn smooth_121(&mut self) {
        let n = self.bins.len();
        if n <= 1 {
            return;
        }
        let mut smoothed = vec![0.0f32; n];
        for (i, dst) in smoothed.iter_mut().enumerate() {
            let prev = self.bins[(i + n - 1) % n];
            let curr = self.bins[i];
            let next = self.bins[(i + 1) % n];
            *dst = (prev + 2.0 * curr + next) * 0.25;
        }
        self.bins = smoothed;
    }

    /// Finds the two dominant peaks separated by at least `min_separation_rad`.
    pub(crate) fn find_two_peaks(&self, min_separation_rad: f32) -> Option<(usize, usize)> {
        if self.bins.is_empty() {
            return None;
        }
        let first = self.argmax()?;
        let first_val = self.bins[first];
        if first_val <= 0.0 {
            return None;
        }

        let n = self.bins.len();
        let mut min_sep_bins = if min_separation_rad.is_finite() {
            (min_separation_rad / self.bin_width).ceil() as isize
        } else {
            n as isize
        };
        if min_sep_bins < 0 {
            min_sep_bins = 0;
        }
        if min_sep_bins as usize >= n {
            return None;
        }

        let mut suppressed = vec![false; n];
        for di in -min_sep_bins..=min_sep_bins {
            let j = ((first as isize + di).rem_euclid(n as isize)) as usize;
            suppressed[j] = true;
        }

        let mut second_idx = None;
        let mut best_val = f32::MIN;
        for (i, &val) in self.bins.iter().enumerate() {
            if suppressed[i] || val <= 0.0 {
                continue;
            }
            if val > best_val {
                best_val = val;
                second_idx = Some(i);
            }
        }
        let second = second_idx?;
        Some((first, second))
    }

    /// Refines a peak angle via a weighted circular mean (mod π).
    ///
    /// This treats each histogram bin within a symmetric window around `index`
    /// as a weighted sample located at the bin center and computes a mean
    /// direction on the circle of orientations modulo π (i.e., θ ≡ θ+π).
    ///
    /// Implementation notes
    /// - Bin center: angle at bin `i` is `(i + 0.5) * bin_width`.
    /// - Window: spans `index - half_window .. index + half_window` with wrap-around.
    /// - Weights: per-bin magnitudes stored in the histogram (non-negative).
    /// - Mod-π mean: uses the “angle-doubling” trick — accumulate on the unit
    ///   circle using 2θ (cos, sin), then halve the resulting mean angle.
    /// - Fallbacks: if the window has zero total weight or an almost-cancelled
    ///   resultant vector, return the center angle of `index`.
    /// - Output is normalized to [0, π).
    ///
    /// Parameters
    /// - `index`: Peak bin index to refine around (0-based, wraps internally).
    /// - `half_window`: Number of bins to include on each side of `index`.
    ///
    /// Returns
    /// - A refined orientation in radians within [0, π).
    pub(crate) fn refined_angle(&self, index: usize, half_window: usize) -> f32 {
        if self.bins.is_empty() {
            return 0.0;
        }
        let n = self.bins.len();
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut total = 0.0f32;
        let half = half_window.min(n.saturating_sub(1));
        for offset in -(half as isize)..=(half as isize) {
            let idx = ((index as isize + offset).rem_euclid(n as isize)) as usize;
            let weight = self.bins[idx];
            if weight <= 0.0 {
                continue;
            }
            total += weight;
            let angle = ((idx as f32) + 0.5) * self.bin_width;
            let doubled = angle * 2.0;
            sx += weight * doubled.cos();
            sy += weight * doubled.sin();
        }
        if total <= 0.0 || (sx * sx + sy * sy) <= 1e-12 {
            return ((index as f32) + 0.5) * self.bin_width;
        }
        let mut mean = 0.5 * sy.atan2(sx);
        if mean < 0.0 {
            mean += std::f32::consts::PI;
        }
        normalize_half_pi(mean)
    }

    fn argmax(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_val = f32::MIN;
        for (i, &val) in self.bins.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::OrientationHistogram;
    use crate::angle::normalize_half_pi;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn accumulate_and_smooth() {
        let mut hist = OrientationHistogram::new(8);
        hist.accumulate(0.0, 1.0);
        hist.accumulate(FRAC_PI_2, 3.0);
        hist.smooth_121();
        let bins = hist.bins();
        assert_eq!(bins.len(), 8);
        assert!(bins.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn find_two_peaks_basic() {
        let mut hist = OrientationHistogram::new(12);
        hist.accumulate(0.0, 5.0);
        hist.accumulate(FRAC_PI_2, 4.0);
        hist.smooth_121();
        let (p0, p1) = hist.find_two_peaks((20.0f32).to_radians()).expect("peaks");
        assert_ne!(p0, p1);
    }

    #[test]
    fn refined_angle_interpolates() {
        let mut hist = OrientationHistogram::new(18);
        let bw = hist.bin_width();
        let target = bw * 4.3; // between bins 4 and 5
        hist.accumulate(target, 2.0);
        hist.accumulate(target + bw * 0.5, 1.0);
        hist.smooth_121();
        let refined = hist.refined_angle(4, 1);
        let expected = normalize_half_pi(target);
        let diff = (refined - expected).abs();
        assert!(
            diff < bw * 0.5,
            "refined={} expected={} diff={}",
            refined,
            expected,
            diff
        );
    }
}
