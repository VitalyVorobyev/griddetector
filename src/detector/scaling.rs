//! Cross-level scaling helpers used by the detector pipeline.
//!
//! - Adapts bundling thresholds to the current level when operating in a
//!   full-resolution invariant mode.
use crate::refine::segment::ScaleMap;
use crate::segments::bundling::Bundle;

/// Rescales a bundle into full-resolution coordinates.
pub fn rescale_bundle_to_full_res(mut bundle: Bundle, scale_x: f32, scale_y: f32) -> Bundle {
    bundle.center[0] *= scale_x;
    bundle.center[1] *= scale_y;

    let mut a = bundle.line[0] / scale_x;
    let mut b = bundle.line[1] / scale_y;
    let mut c = bundle.line[2];
    let norm = (a * a + b * b).sqrt().max(1e-6);
    a /= norm;
    b /= norm;
    c /= norm;

    bundle.line = [a, b, c];
    bundle.weight *= 0.5 * (scale_x + scale_y);
    bundle
}

/// Per-level scaling factors between a pyramid level and the full-resolution image.
#[derive(Clone, Copy, Debug)]
pub struct LevelScaling {
    pub scale_x_to_full: f32,
    pub scale_y_to_full: f32,
    pub mean_scale_from_full: f32,
}

impl LevelScaling {
    pub fn from_dimensions(
        level_width: usize,
        level_height: usize,
        full_width: usize,
        full_height: usize,
    ) -> Self {
        let scale_x = if level_width > 0 {
            full_width as f32 / level_width as f32
        } else {
            1.0
        };
        let scale_y = if level_height > 0 {
            full_height as f32 / level_height as f32
        } else {
            1.0
        };
        let mean_to_full = 0.5 * (scale_x + scale_y);
        let mean_from_full = if mean_to_full > 0.0 {
            1.0 / mean_to_full
        } else {
            1.0
        };
        Self {
            scale_x_to_full: scale_x,
            scale_y_to_full: scale_y,
            mean_scale_from_full: mean_from_full,
        }
    }
}

/// Scale map used by the segment refiner to lift from level `l+1` to `l`.
pub struct LevelScaleMap {
    sx: f32,
    sy: f32,
}

impl LevelScaleMap {
    pub fn new(sx: f32, sy: f32) -> Self {
        Self { sx, sy }
    }
}

impl ScaleMap for LevelScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2] {
        [p_coarse[0] * self.sx, p_coarse[1] * self.sy]
    }
}
