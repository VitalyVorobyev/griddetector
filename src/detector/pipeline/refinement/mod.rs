//! Refinement submodule: level preparation, homography refinement, and indexing.
//!
//! This module groups the coarse-to-fine logic previously held in a single file
//! into focused components:
//!
//! - `prepare`: bundle per level and refine segments forward (L→L-1)
//! - `homography`: IRLS-driven update for the homography across prepared levels
//! - `indexing`: associate bundles to discrete grid indices after rectification
//!
//! The top-level re-exports provide a small, easy-to-read surface used by the
//! pipeline driver (`pipeline::mod`).

pub mod homography;
pub mod indexing;
pub mod prepare;

pub use homography::{refine_homography, RefinementComputation};
pub use indexing::index_grid_from_bundles;
pub use prepare::{build_bundling_stage, prepare_levels, PreparedLevels};
