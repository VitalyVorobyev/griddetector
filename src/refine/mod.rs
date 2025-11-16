//! Gradient-guided refinement utilities.
//!
//! The `refine` module bundles two complementary refinement routines used by
//! the grid detector:
//!
//! - [`segment`] lifts a line segment detected on a coarser pyramid level and
//!   snaps it to gradient support on the next finer level, performing the
//!   coarse-to-fine carrier update used by the detector’s refinement cascade.
//!
//! Downstream pipeline typically:
//! - prepares per-level bundles + refined segments in the detector module,
//! - then runs the homography IRLS via [`homography::Refiner`],
//! - and finally uses `families::split_bundles` (re-exported as
//!   `split_bundles`) during grid indexing in rectified space.

mod endpoints;
mod fit;
mod iteration;
mod options;
mod refinesegment;
mod roi;
mod sampling;
mod workspace;

pub use options::RefineOptions;
pub use refinesegment::{refine_coarse_segments, SegmentsRefinementResult};

#[cfg(feature = "profile_refine")]
mod profile;
