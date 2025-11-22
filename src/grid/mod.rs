//! Grid-related utilities and structures.
//!
//! This module is being rebuilt toward an orthogonal-grid detector. The
//! current surface includes:
//! - [`bundling`]: merge near-collinear segments into weighted line bundles.
//! - [`vp`]: estimate two dominant vanishing points from bundles.
//! - [`histogram`]: simple orientation histogram used by the VP stage.

pub mod bundling;
pub mod histogram;
pub mod vp;
