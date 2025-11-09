use crate::angle::{angle_between, angular_difference, vp_direction};
use crate::image::ImageF32;
use crate::segments::{lsd_extract_segments, LsdOptions, Segment};
use log::debug;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::time::Instant;

use super::families::{analyze_families, FamilyAnalysisError, FamilyAssignments};
use super::vp::estimate_vp;

/// Minimal angular separation (degrees) between the two vanishing directions.
/// Used to reject nearly colinear vanishing directions.
const MIN_VP_SEPARATION_DEG: f32 = 10.0;

/// Identifier for the two dominant line families found by the LSD→VP engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FamilyLabel {
    U,
    V,
}

/// Detailed inference outcome including per-segment assignments.
#[derive(Clone, Debug)]
pub struct DetailedInference {
    pub hypothesis: Hypothesis,
    pub dominant_angles_rad: [f32; 2],
    pub families: Vec<Option<FamilyLabel>>,
    pub segments: Vec<Segment>,
}

/// Coarse hypothesis returned by the LSD→VP engine
#[derive(Clone, Debug, Serialize)]
pub struct Hypothesis {
    pub hmtx0: Matrix3<f32>,
    pub confidence: f32,
}

impl Hypothesis {
    pub fn scaled(&self, scale_x: f32, scale_y: f32) -> Matrix3<f32> {
        let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
        scale * self.hmtx0
    }
}

/// Lightweight engine that finds two dominant line families from LSD segments,
/// estimates their vanishing points, and returns a coarse projective basis H0.
#[derive(Clone, Debug, Default)]
pub struct Engine {
    pub options: LsdOptions,
}

impl Engine {
    /// Run LSD and perform vanishing-point inference with per-segment assignments.
    pub fn infer(&self, l: &ImageF32) -> Option<DetailedInference> {
        let segments = lsd_extract_segments(l, self.options);
        self.infer_with_segments(l, segments)
    }

    /// Run inference on pre-computed segments (e.g., using custom LSD options).
    ///
    /// Pipeline overview:
    /// - Build a strength-weighted orientation histogram over [0, pi)
    /// - Find two dominant peaks separated by at least `2*angle_tolerance_deg`
    /// - Soft-assign segments to two families within `angle_tolerance_deg`
    /// - Estimate VPs for each family and validate separation
    /// - Compose a coarse projective basis H0 and compute confidence
    ///
    /// Returns `None` on: insufficient segments, peaks not found, weak family
    /// support, VP estimation failure, or near-colinear VPs.
    pub fn infer_with_segments(
        &self,
        l: &ImageF32,
        segments: Vec<Segment>,
    ) -> Option<DetailedInference> {
        let t0 = Instant::now();
        let assignments = match analyze_families(&segments, self.options.angle_tolerance_deg) {
            Ok(assignments) => assignments,
            Err(FamilyAnalysisError::InsufficientSegments { found, minimum }) => {
                debug!(
                    "LSD-VP: insufficient segments on level {}x{} ({} < {})",
                    l.w, l.h, found, minimum
                );
                return None;
            }
            Err(FamilyAnalysisError::DominantPeaksNotFound { min_separation_deg }) => {
                debug!(
                    "LSD-VP: dominant orientation peaks not found (min_sep_deg={:.1})",
                    min_separation_deg
                );
                return None;
            }
            Err(FamilyAnalysisError::WeakFamilySupport {
                family_u,
                family_v,
                minimum,
            }) => {
                debug!(
                    "LSD-VP: insufficient family support fam1={} fam2={} (need >= {})",
                    family_u, family_v, minimum
                );
                return None;
            }
        };
        let confidence = assignments.confidence();
        let FamilyAssignments {
            dominant_angles_rad,
            families,
            u_support,
            v_support,
        } = assignments;
        let theta_u = dominant_angles_rad[0];
        let theta_v = dominant_angles_rad[1];

        // 4) Estimate VPs for both families
        let (vpu, vpv) =
            Self::estimate_family_vps(&segments, &u_support, &v_support, theta_u, theta_v)?;

        // 5) Validate VP separation
        let cx = (l.w as f32) * 0.5;
        let cy = (l.h as f32) * 0.5;
        if !Self::validate_vp_separation(&vpu, &vpv, cx, cy, MIN_VP_SEPARATION_DEG) {
            return None;
        }

        // 6) Compose basis and compute confidence
        let x0 = Vector3::new(cx, cy, 1.0);
        let hmtx0 = Matrix3::from_columns(&[vpu, vpv, x0]);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "LSD-VP: segs={} fam1={} fam2={} sep_deg={:.1} confidence={:.3} elapsed_ms={:.3}",
            segments.len(),
            u_support.len(),
            v_support.len(),
            angular_difference(theta_u, theta_v).to_degrees(),
            confidence,
            elapsed_ms
        );

        let hypothesis = Hypothesis { hmtx0, confidence };

        Some(DetailedInference {
            hypothesis,
            dominant_angles_rad: [theta_u, theta_v],
            families,
            segments,
        })
    }

    /// Estimate vanishing points for both families.
    fn estimate_family_vps(
        segments: &[Segment],
        u_idx: &[usize],
        v_idx: &[usize],
        theta_u: f32,
        theta_v: f32,
    ) -> Option<(Vector3<f32>, Vector3<f32>)> {
        let vpu = estimate_vp(segments, u_idx, theta_u)?;
        let vpv = estimate_vp(segments, v_idx, theta_v)?;
        Some((vpu, vpv))
    }

    /// Validate that the two vanishing directions are sufficiently separated.
    /// Logs a debug message and returns false when invalid.
    fn validate_vp_separation(
        vpu: &Vector3<f32>,
        vpv: &Vector3<f32>,
        cx: f32,
        cy: f32,
        min_sep_deg: f32,
    ) -> bool {
        let x0 = Vector3::new(cx, cy, 1.0);
        let sep_min_rad = min_sep_deg.to_radians();
        let du = vp_direction(vpu, &x0);
        let dv = vp_direction(vpv, &x0);
        if let (Some(du), Some(dv)) = (du, dv) {
            let angle = angle_between(&du, &dv);
            let folded = angle.min(std::f32::consts::PI - angle);
            if folded < sep_min_rad {
                debug!(
                    "LSD-VP: vanishing directions too close (angle_deg={:.2})",
                    folded.to_degrees()
                );
                return false;
            }
            true
        } else {
            debug!("LSD-VP: could not derive vanishing directions from VPs");
            false
        }
    }
}
