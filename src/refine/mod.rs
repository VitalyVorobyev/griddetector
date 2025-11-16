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
//! Downstream pipeline typically:
//! - prepares per-level bundles + refined segments in the detector module,
//! - then runs the homography IRLS via [`homography::Refiner`],
//! - and finally uses `families::split_bundles` (re-exported as
//!   `split_bundles`) during grid indexing in rectified space.

const EPS: f32 = 1e-6;

mod anchor;
mod families;
mod irls;
pub mod segment;
mod types;

pub(crate) use families::split_bundles;
