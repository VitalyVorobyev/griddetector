//! Gradient-guided refinement utilities.
//!
//! The `refine` module lifts coarse line segments down the pyramid using
//! image gradients:
//!
//! - [`refine_coarse_segments`] reuses one Scharr gradient tile per level and
//!   snaps each segment to local support on the next finer level. Segments that
//!   fail to gather support are dropped instead of propagated, improving
//!   alignment quality.
//! - Parameters are scaled per level (`RefineOptions::for_level`) so sampling
//!   spacing and magnitude thresholds remain consistent in physical units.
//!
//! Downstream pipeline typically:
//! - prepares per-level bundles + refined segments in the detector module,
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
