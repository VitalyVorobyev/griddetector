use crate::diagnostics::RefinementLevelDiagnostics;
use crate::segments::bundling::Bundle;
use nalgebra::Matrix3;

/// Parameters controlling homography refinement from bundled constraints.
#[derive(Clone, Debug)]
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
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub segments: usize,
    pub bundles: &'a [Bundle],
}

/// Result of the homography refinement.
#[derive(Clone, Debug)]
pub struct RefinementResult {
    pub h_refined: Matrix3<f32>,
    pub confidence: f32,
    pub inlier_ratio: f32,
    pub levels_used: usize,
    pub level_reports: Vec<RefinementLevelDiagnostics>,
}

pub(crate) struct LevelRefinement {
    pub h_new: Matrix3<f32>,
    pub confidence: f32,
    pub inlier_ratio: f32,
    pub family_u_count: usize,
    pub family_v_count: usize,
}

pub(crate) struct VpStats {
    pub confidence: f32,
    pub total_weight: f32,
    pub inlier_weight: f32,
}
