//! Grayscale image pyramid with configurable separable blur and 2× decimation.
//!
//! The pyramid converts level 0 from 8-bit grayscale to `ImageF32` in `[0, 1]`
//! and repeatedly downsamples by 2×. Prior to each decimation step an optional
//! separable filter (Gaussian by default) can be applied. Border samples clamp
//! to the image extents, matching the previous implementation.

pub mod filters;

use crate::image::{ImageF32, ImageU8, ImageView, ImageViewMut};
use filters::{apply as apply_filter, SeparableFilter, GAUSSIAN_5TAP};

#[derive(Clone, Debug)]
pub struct Pyramid {
    pub levels: Vec<ImageF32>,
}

/// Options controlling pyramid construction.
#[derive(Clone, Copy)]
pub struct PyramidOptions<'a> {
    /// Number of pyramid levels (>= 1).
    pub levels: usize,
    /// Number of initial downscale steps that apply the separable filter.
    ///
    /// `None` applies the filter before every decimation. `Some(0)` skips blur
    /// entirely. `Some(k)` applies the filter for the first `k` downscale
    /// operations (e.g. `k >= levels` → blur everywhere).
    pub blur_levels: Option<usize>,
    /// Filter used for the separable blur stage.
    pub filter: &'a dyn SeparableFilter,
}

impl<'a> PyramidOptions<'a> {
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            blur_levels: None,
            filter: &GAUSSIAN_5TAP,
        }
    }

    pub fn with_blur_levels(mut self, blur_levels: Option<usize>) -> Self {
        self.blur_levels = blur_levels;
        self
    }

    pub fn with_filter(mut self, filter: &'a dyn SeparableFilter) -> Self {
        self.filter = filter;
        self
    }
}

impl std::fmt::Debug for PyramidOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyramidOptions")
            .field("levels", &self.levels)
            .field("blur_levels", &self.blur_levels)
            .field("filter_taps", &self.filter.taps().len())
            .finish()
    }
}

impl Pyramid {
    /// Build a pyramid from an 8-bit grayscale input using the provided options.
    pub fn build_u8(gray: ImageU8<'_>, options: PyramidOptions<'_>) -> Self {
        assert!(options.levels >= 1, "pyramid requires at least one level");
        let mut levels = Vec::with_capacity(options.levels);
        levels.push(convert_l0(gray));

        let blur_limit = options.blur_levels.unwrap_or(usize::MAX);
        for lvl in 1..options.levels {
            let prev = levels.last().expect("previous level available");
            let use_blur = lvl <= blur_limit;
            let filtered = if use_blur {
                Some(apply_filter(options.filter, prev))
            } else {
                None
            };
            let src_img = filtered.as_ref().unwrap_or(prev);

            let (nw, nh) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            let mut down = ImageF32::new(nw, nh);
            for y in 0..nh {
                let dst_row = down.row_mut(y);
                let sy = (y * 2).min(src_img.h - 1);
                let src_row = src_img.row(sy);
                for (x, dst_px) in dst_row.iter_mut().enumerate() {
                    let sx = (x * 2).min(src_img.w - 1);
                    *dst_px = src_row[sx];
                }
            }
            levels.push(down);
        }

        Self { levels }
    }
}

fn convert_l0(gray: ImageU8<'_>) -> ImageF32 {
    let mut out = ImageF32::new(gray.w, gray.h);
    for y in 0..gray.h {
        let src = gray.row(y);
        let dst = out.row_mut(y);
        for x in 0..gray.w {
            dst[x] = src[x] as f32 / 255.0;
        }
    }
    out
}
