//! Grid detector orchestrating a coarse-to-fine line-based pipeline.
//!
//! Overview
//! - Builds an image pyramid with optional blur limiting.
//! - Runs a lightweight LSD-like extractor on the coarsest level and clusters
//!   two dominant orientations to estimate vanishing points (VPs). Composes a
//!   coarse projective basis `H0` from the two VPs and an image-centre anchor.
//! - Rejects coarse segments that are inconsistent with `H0` (angular margin
//!   and homography-residual checks).
//! - Walks the pyramid from coarse → fine, refining segments on each level,
//!   bundling near-collinear constraints, and finally refining the homography
//!   via a Huber-weighted IRLS update of the vanishing-point columns.
//! - Optionally attempts an extra refinement pass when the Frobenius
//!   improvement exceeds a small threshold.
//!
//! Modules
//! - [`params`] – configuration types used by the detector and CLI.
//! - `pipeline` – the main [`GridDetector`] implementation.
//! - `scaling` – helpers for rescaling segments/bundles across pyramid levels.
//! - [`outliers`] – filters for rejecting segment outliers before refinement.
//! - `workspace` – reusable buffers that amortise allocations across frames.
//!
//! Key Ideas
//! - Orientation is ambiguous modulo π for grid lines; the LSD stage and the
//!   histogram work in `[0, π)`.
//! - Bundling merges near-collinear segments and operates either with fixed
//!   pixel thresholds or in a full-resolution invariant mode that adapts to the
//!   current level scale.
//! - The refiner performs a robust VP update with Huber weights and assigns
//!   bundles to the two vanishing families based on the current homography.
//!
//! See `README.md` for a gentle overview and `doc/*.md` for deep dives.

pub mod outliers;
pub mod params;
mod pipeline;
mod scaling;
mod workspace;

pub use params::{
    BundlingParams, BundlingScaleMode, GridParams, OutlierFilterParams, RefinementSchedule,
};
pub use pipeline::GridDetector;
pub use scaling::{rescale_bundle_to_full_res, LevelScaling};
