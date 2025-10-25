//! Detector module orchestrating the grid pipeline.
//!
//! The refactor splits the previous monolithic implementation into smaller
//! components:
//! - [`params`] – configuration structs used by the detector and CLI.
//! - [`pipeline`] – the main `GridDetector` implementation.
//! - [`scaling`] – helpers for rescaling segments/bundles across pyramid levels.
//! - [`outliers`] – filters for rejecting segment outliers before refinement.
//! - [`workspace`] – reusable buffers that amortise allocations across frames.

mod outliers;
pub mod params;
mod pipeline;
mod scaling;
mod workspace;

pub use params::{
    BundlingParams, BundlingScaleMode, GridParams, LsdVpParams, OutlierFilterParams,
    RefinementSchedule,
};
pub use pipeline::GridDetector;
