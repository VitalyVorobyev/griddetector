//! Grid-related utilities and structures.
//!
//! This module is being rebuilt toward an orthogonal-grid detector. The
//! current surface includes:
//! - [`bundling`]: merge near-collinear segments into weighted line bundles.
//! - [`vp`]: estimate two dominant vanishing points from bundles.
//! - [`histogram`]: simple orientation histogram used by the VP stage.
//! - [`hypothesis`]: build a minimal grid hypothesis (bundles + VPs).

pub mod bundling;
pub mod hypothesis;
pub mod histogram;
pub mod vp;

pub use bundling::{bundle_segments, Bundle, BundleId, BundlingParams};
pub use hypothesis::{build_grid_hypothesis, GridHypothesis};
pub use vp::{
    estimate_vanishing_pair, FamilyLabel, VanishingPair, VanishingPoint, VpEstimationOptions,
};
