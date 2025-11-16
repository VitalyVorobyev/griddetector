use super::filters::{SeparableFilter, StaticSeparableFilter, GAUSSIAN_5TAP};

use serde::Deserialize;

/// Options controlling pyramid construction.
#[derive(Clone, Copy, Deserialize)]
pub struct PyramidOptions {
    /// Number of pyramid levels (>= 1).
    pub levels: usize,
    /// Number of initial downscale steps that apply the separable filter.
    ///
    ///`Some(k)` applies the filter for the first `k` downscale
    /// operations (e.g. `k >= levels` → blur everywhere).
    pub blur_levels: usize,
    /// Filter used for the separable blur stage.
    #[serde(skip)]
    pub filter: StaticSeparableFilter,
}

impl PyramidOptions {
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            blur_levels: 0,
            filter: GAUSSIAN_5TAP,
        }
    }

    pub fn with_blur_levels(mut self, blur_levels: usize) -> Self {
        self.blur_levels = blur_levels;
        self
    }

    pub fn with_filter(mut self, filter: StaticSeparableFilter) -> Self {
        self.filter = filter;
        self
    }
}

impl std::fmt::Debug for PyramidOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyramidOptions")
            .field("levels", &self.levels)
            .field("blur_levels", &self.blur_levels)
            .field("filter_taps", &self.filter.taps().len())
            .finish()
    }
}
