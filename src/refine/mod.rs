//! Gradient-guided refinement utilities.
//!
//! The `refine` module bundles two complementary refinement routines used by
//! the grid detector:
//!
//! - [`homography`] refines coarse homography hypotheses by re-weighting
//!   bundled line constraints via an IRLS scheme around vanishing-point
//!   directions.
//! - [`segment`] lifts a line segment detected on a coarser pyramid level and
//!   snaps it to gradient support on the next finer level, performing the
//!   coarse-to-fine carrier update used by the detector’s refinement cascade.
//!
//! Downstream code typically invokes the segment refiner first (per level) to
//! obtain cleaned-up bundles before running the homography IRLS pass.

const EPS: f32 = 1e-6;

mod anchor;
mod families;
pub mod homography;
mod irls;
pub mod segment;
mod types;

pub use homography::{RefineLevel, RefineParams, RefinementResult, Refiner};
