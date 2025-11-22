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

impl Default for StaticSeparableFilter {
    fn default() -> Self {
        GAUSSIAN_5TAP
    }
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
