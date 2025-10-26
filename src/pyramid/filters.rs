use crate::image::traits::{ImageView, ImageViewMut};
use crate::image::ImageF32;

/// Trait implemented by separable 1D filters used for pyramid construction.
pub trait SeparableFilter {
    /// Return the 1D taps (in left-to-right order). The kernel is assumed to be
    /// symmetric around its centre, but the implementation does not rely on it.
    fn taps(&self) -> &[f32];
}

/// Simple wrapper around a static filter kernel.
#[derive(Clone, Copy, Debug)]
pub struct StaticSeparableFilter {
    taps: &'static [f32],
}

impl StaticSeparableFilter {
    pub const fn new(taps: &'static [f32]) -> Self {
        Self { taps }
    }
}

impl SeparableFilter for StaticSeparableFilter {
    #[inline]
    fn taps(&self) -> &[f32] {
        self.taps
    }
}

/// Normalised 5-tap Gaussian filter `[1, 4, 6, 4, 1] / 16`.
pub const GAUSSIAN_5TAP: StaticSeparableFilter =
    StaticSeparableFilter::new(&[0.0625, 0.25, 0.375, 0.25, 0.0625]);

/// Apply a separable filter to `input`, returning a new `ImageF32` with the
/// same dimensions. Border handling clamps to the valid range.
pub fn apply(filter: &dyn SeparableFilter, input: &ImageF32) -> ImageF32 {
    let w = input.w;
    let h = input.h;
    let taps = filter.taps();
    assert!(
        !taps.is_empty(),
        "separable filter requires at least one tap"
    );
    let radius = taps.len() / 2;

    let mut tmp = ImageF32::new(w, h);
    // Horizontal pass
    for y in 0..h {
        let in_row = input.row(y);
        let out_row = tmp.row_mut(y);
        for (x, dst_px) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (k, &tap) in taps.iter().enumerate() {
                let offset = k as isize - radius as isize;
                let sx = clamp_index(x as isize + offset, w);
                acc += tap * in_row[sx];
            }
            *dst_px = acc;
        }
    }

    let mut out = ImageF32::new(w, h);
    // Vertical pass
    for y in 0..h {
        let out_row = out.row_mut(y);
        for (x, dst_px) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (k, &tap) in taps.iter().enumerate() {
                let offset = k as isize - radius as isize;
                let sy = clamp_index(y as isize + offset, h);
                acc += tap * tmp.row(sy)[x];
            }
            *dst_px = acc;
        }
    }

    out
}

fn clamp_index(idx: isize, upper: usize) -> usize {
    if idx < 0 {
        0
    } else if (idx as usize) >= upper {
        upper.saturating_sub(1)
    } else {
        idx as usize
    }
}
