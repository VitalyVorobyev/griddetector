//! Minimal grid hypothesis builder: bundle segments and estimate dominant vanishing points.
//!
//! This stage produces a lightweight hypothesis consisting of:
//! - Bundled line constraints (merged from refined segments).
//! - Two orthogonal vanishing points estimated from bundle orientations.
//! - Timing and support statistics.

use super::bundling::{bundle_segments, Bundle, BundlingParams};
use super::vp::{estimate_vanishing_pair, VanishingPair, VpEstimationOptions};
use crate::segments::Segment;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct GridHypothesis {
    pub bundles: Vec<Bundle>,
    pub vp: Option<VanishingPair>,
    pub bundle_ms: f64,
    pub vp_ms: f64,
}

/// Build a grid hypothesis from refined segments.
pub fn build_grid_hypothesis(
    segments: &[Segment],
    bundling: &BundlingParams,
    vp_opts: &VpEstimationOptions,
) -> GridHypothesis {
    let t0 = Instant::now();
    let bundles = bundle_segments(segments, bundling);
    let bundle_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let vp = estimate_vanishing_pair(&bundles, vp_opts);
    let vp_ms = t1.elapsed().as_secs_f64() * 1000.0;

    GridHypothesis {
        bundles,
        vp,
        bundle_ms,
        vp_ms,
    }
}
