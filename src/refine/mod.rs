//! Coarse-to-fine homography refinement using pre-bundled line constraints.
//!
//! The refiner consumes bundles provided by the detection stage, assigns them
//! to the two dominant families implied by the current homography, and performs
//! a Huber-weighted IRLS update of the vanishing points and anchor.

mod anchor;
mod families;
mod irls;
mod types;

pub use types::{RefineLevel, RefineParams, RefinementResult};

use crate::diagnostics::RefinementLevelDiagnostics;
use nalgebra::Matrix3;

use families::FamilyBuckets;
use types::LevelRefinement;

const EPS: f32 = 1e-6;

/// Coarse-to-fine refiner operating on externally bundled line constraints.
pub struct Refiner {
    params: RefineParams,
}

impl Refiner {
    pub fn new(params: RefineParams) -> Self {
        Self { params }
    }

    /// Refine an initial homography using pre-computed bundle sets ordered from fine to coarse.
    pub fn refine(
        &self,
        initial_h: Matrix3<f32>,
        levels: &[RefineLevel<'_>],
    ) -> Option<RefinementResult> {
        if levels.is_empty() {
            return None;
        }

        let mut current_h = initial_h;
        let mut last_good_h = initial_h;
        let mut last_inlier_ratio = 0.0f32;
        let mut accumulated_conf = 0.0f32;
        let mut accumulated_weight = 0.0f32;
        let mut total_levels = 0usize;
        let mut level_reports: Vec<RefinementLevelDiagnostics> = Vec::new();

        let orientation_tol = self.params.orientation_tol_deg.to_radians();
        for level in levels.iter().rev() {
            total_levels += 1;
            let mut level_diag = RefinementLevelDiagnostics {
                level_index: level.level_index,
                width: level.width,
                height: level.height,
                segments: level.segments,
                bundles: level.bundles.len(),
                family_u_count: 0,
                family_v_count: 0,
                improvement: None,
                confidence: None,
                inlier_ratio: None,
            };

            if level.bundles.len() < self.params.min_bundles_per_family * 2 {
                level_reports.push(level_diag);
                continue;
            }

            let buckets = match families::split_bundles(&current_h, level.bundles, orientation_tol)
            {
                Some(b) => b,
                None => {
                    level_reports.push(level_diag);
                    continue;
                }
            };

            if buckets.family_u.len() < self.params.min_bundles_per_family
                || buckets.family_v.len() < self.params.min_bundles_per_family
            {
                level_reports.push(level_diag);
                continue;
            }

            if let Some(level_res) = self.refine_level(&buckets) {
                let improvement = (frobenius_norm(&(level_res.h_new - current_h))
                    / (frobenius_norm(&current_h) + EPS))
                    .abs();
                let weight = (level.level_index + 1) as f32;
                accumulated_conf += level_res.confidence * weight;
                accumulated_weight += weight;
                last_inlier_ratio = level_res.inlier_ratio;
                last_good_h = level_res.h_new;
                current_h = last_good_h;

                level_diag.family_u_count = level_res.family_u_count;
                level_diag.family_v_count = level_res.family_v_count;
                level_diag.improvement = Some(improvement);
                level_diag.confidence = Some(level_res.confidence);
                level_diag.inlier_ratio = Some(level_res.inlier_ratio);
                level_reports.push(level_diag);

                if improvement < 1e-3 {
                    break;
                }
            } else {
                level_reports.push(level_diag);
            }
        }

        if accumulated_weight <= EPS {
            None
        } else {
            Some(RefinementResult {
                h_refined: last_good_h,
                confidence: (accumulated_conf / accumulated_weight).clamp(0.0, 1.0),
                inlier_ratio: last_inlier_ratio,
                levels_used: total_levels,
                level_reports,
            })
        }
    }

    fn refine_level(&self, buckets: &FamilyBuckets<'_>) -> Option<LevelRefinement> {
        let delta = self.params.huber_delta;
        let (vpu_new, stats_u) = irls::estimate_vp_huber(
            &buckets.family_u,
            &buckets.vpu,
            delta,
            self.params.max_iterations,
        )?;
        let (vpv_new, stats_v) = irls::estimate_vp_huber(
            &buckets.family_v,
            &buckets.vpv,
            delta,
            self.params.max_iterations,
        )?;
        let anchor_new = anchor::estimate_anchor(&buckets.family_u, &buckets.family_v)
            .unwrap_or_else(|| buckets.anchor.clone());

        let h_new = Matrix3::from_columns(&[vpu_new, vpv_new, anchor_new]);
        let combined_inlier = (stats_u.inlier_weight + stats_v.inlier_weight)
            / (stats_u.total_weight + stats_v.total_weight + EPS);
        let confidence = ((stats_u.confidence + stats_v.confidence) * 0.5).clamp(0.0, 1.0);
        Some(LevelRefinement {
            h_new,
            confidence,
            inlier_ratio: combined_inlier,
            family_u_count: buckets.family_u.len(),
            family_v_count: buckets.family_v.len(),
        })
    }
}

fn frobenius_norm(m: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            sum += m[(i, j)] * m[(i, j)];
        }
    }
    sum.sqrt()
}
