//! LSD→VP engine for coarse homography hypothesis.
//!
//! This module groups LSD-like segments into two dominant orientation families,
//! estimates their vanishing points (VPs), and composes a coarse projective
//! basis `H0 = [vpu | vpv | x0]` where `x0` anchors translation near the image
//! center. It trades completeness for speed and stability to seed downstream
//! coarse-to-fine refinement.
//!
//! Pipeline
//! - Segment extraction: calls `segments::lsd_extract_segments` at a single
//!   pyramid level (typically the coarsest).
//! - Orientation histogram: builds a [0,π) histogram weighted by segment
//!   strength, selects two dominant peaks separated by at least ~2× the
//!   tolerance.
//! - Family assignment: soft-assign segments to the two peaks based on angular
//!   proximity within a tolerance.
//! - VP estimation: solve a length-weighted least-squares on line constraints
//!   `ax+by+c≈0` for each family to obtain the VPs; fall back to a
//!   point-at-infinity direction if the normal matrix becomes near-singular
//!   (e.g., perfectly parallel segments).
//! - Hypothesis: compose `H0` from the two VPs and an anchor column at the
//!   image center; report a confidence from support and angular separation.
//!
//! Notes
//! - Works with orientation modulo π; directions are unsigned for grid lines.
//! - Confidence is heuristic: grows with family support and peak separation.
//! - The returned `H0` is defined in the current level’s coordinates; callers
//!   should rescale to full-resolution before use.

mod engine;
mod histogram;
mod vp;
pub mod bundling;

pub use engine::{DetailedInference, Engine, FamilyLabel, Hypothesis};
