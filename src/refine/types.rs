use crate::diagnostics::RefinementIteration;
use nalgebra::Matrix3;
use serde::Deserialize;

/// Parameters controlling homography refinement from bundled constraints.
///
/// The fields mirror the IRLS schedule used by [`homography::Refiner`](crate::refine::homography::Refiner).
#[derive(Clone, Debug, Deserialize)]
pub struct RefineParams {
    /// Orientation tolerance (degrees) for assigning bundles to the u/v families.
    pub orientation_tol_deg: f32,
    /// Huber delta (pixels) controlling the inlier region during IRLS.
    pub huber_delta: f32,
    /// Maximum IRLS iterations per refinement level.
    pub max_iterations: usize,
    /// Minimum number of bundles per family (u and v) to attempt an update.
    pub min_bundles_per_family: usize,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            orientation_tol_deg: 22.5,
            huber_delta: 1.0,
            max_iterations: 6,
            min_bundles_per_family: 4,
        }
    }
}

/// Input bundles describing a single refinement level.
#[derive(Clone, Debug)]
pub struct RefineLevel<'a> {
    /// Pyramid level index (0 = finest).
    pub level_index: usize,
    /// Level width in pixels.
    pub width: usize,
    /// Level height in pixels.
    pub height: usize,
    /// Number of raw segments contributing to the bundles, used for diagnostics.
    pub segments: usize,
    /// View of the bundles collected at this level.
    pub bundles: &'a [Bundle],
}

/// Result of the homography refinement.
#[derive(Clone, Debug)]
pub struct RefinementResult {
    /// Refined homography expressed in pixel space of the finest level.
    pub h_refined: Matrix3<f32>,
    /// Confidence score aggregated across levels.
    pub confidence: f32,
    /// Ratio between inlier and total bundle weight in the last successful level.
    pub inlier_ratio: f32,
    /// Number of pyramid levels that contributed a valid update.
    pub levels_used: usize,
    /// Per-level diagnostics useful for visualisation and logging.
    pub level_reports: Vec<RefinementIteration>,
}

pub(crate) struct LevelRefinement {
    pub h_new: Matrix3<f32>,
    pub confidence: f32,
    pub inlier_ratio: f32,
    pub family_u_count: usize,
    pub family_v_count: usize,
}

pub(crate) struct VpStats {
    /// Confidence for the vanishing point update (0..1).
    pub confidence: f32,
    /// Sum of IRLS weights processed.
    pub total_weight: f32,
    /// Sum of inlier weights falling inside the Huber loss quadratic region.
    pub inlier_weight: f32,
}
