//! Grayscale image pyramid with configurable separable blur and 2× decimation.
//!
//! The pyramid converts level 0 from 8-bit grayscale to `ImageF32` in `[0, 1]`
//! and repeatedly downsamples by 2×. Prior to each decimation step an optional
//! separable filter (Gaussian by default) can be applied. Border samples clamp
//! to the image extents, matching the previous implementation.

mod filters;
mod options;
mod pyramid;
mod scaling;

pub use options::PyramidOptions;
pub use pyramid::{Pyramid, build_pyramid, PyramidResult};
pub use scaling::{ScaleMap, LevelScaleMap};
pub use filters::GAUSSIAN_5TAP;
