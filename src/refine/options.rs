//! Public types used by the gradient-driven segment refiner.
//!
//! The refiner is driven by image gradients on a single pyramid level. To
//! support both full-frame and cropped gradient tiles, [`PyramidLevel`]
//! describes gradients in a small, contiguous window of the level together with
//! enough metadata to interpret positions in full-image coordinates.

use serde::Deserialize;

/// Parameters controlling the gradient-driven refinement.
#[derive(Clone, Debug, Deserialize)]
pub struct RefineOptions {
    /// Along-segment sample spacing (px) used for normal probing.
    pub delta_s: f32,
    /// Half-width (px) of the normal search corridor.
    pub w_perp: f32,
    /// Step size (px) for sweeping along the normal.
    pub delta_t: f32,
    /// Padding (px) added around the segment when forming the ROI.
    pub pad: f32,
    /// Minimum gradient magnitude accepted as support.
    pub tau_mag: f32,
    /// Orientation tolerance (degrees) applied during endpoint gating.
    pub tau_ori_deg: f32,
    /// Huber delta used to weight the normal projected gradient.
    pub huber_delta: f32,
    /// Maximum number of outer carrier update iterations.
    pub max_iters: usize,
    /// Minimum ratio between refined and seed length required for acceptance.
    pub min_inlier_frac: f32,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            delta_s: 0.75,
            w_perp: 3.0,
            delta_t: 0.5,
            pad: 8.0,
            tau_mag: 0.1,
            tau_ori_deg: 25.0,
            huber_delta: 0.25,
            max_iters: 3,
            min_inlier_frac: 0.4,
        }
    }
}

impl RefineOptions {
    /// Scale the refinement parameters for a specific pyramid level.
    ///
    /// `full_width` is the width of the finest level (L0). `level_width` is the
    /// width of the level we are about to process. Spatial parameters scale
    /// inversely with the pixel size so that the number of samples and corridor
    /// widths remain roughly constant in physical units.
    pub fn for_level(&self, full_width: usize, level_width: usize) -> Self {
        let scale = if full_width == 0 || level_width == 0 {
            1.0f32
        } else {
            full_width as f32 / level_width as f32
        };

        let mut params = self.clone();

        let scale_spacing = |value: f32| -> f32 {
            if !value.is_finite() {
                return value;
            }
            let scaled = value / scale;
            if scaled.is_finite() {
                scaled.max(0.05)
            } else {
                value
            }
        };

        let scale_width = |value: f32| -> f32 {
            if !value.is_finite() {
                return value;
            }
            let scaled = value / scale;
            if scaled.is_finite() {
                scaled.max(0.5)
            } else {
                value
            }
        };

        params.delta_s = scale_spacing(self.delta_s);
        params.delta_t = scale_spacing(self.delta_t);
        params.w_perp = scale_width(self.w_perp);
        params.pad = scale_width(self.pad);
        params.tau_mag = scale_spacing(self.tau_mag).max(0.01);
        params
    }
}
