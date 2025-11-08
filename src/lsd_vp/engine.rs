use crate::angle::{angle_between, angular_difference, normalize_half_pi, vp_direction};
use crate::image::ImageF32;
use crate::segments::{lsd_extract_segments, LsdOptions, Segment};
use log::debug;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::time::Instant;

use super::histogram::OrientationHistogram;
use super::vp::estimate_vp;

const DEFAULT_BINS: usize = 36;
const MIN_SEGS: usize = 12;
const MIN_FAMILY: usize = 6;
const MIN_VP_SEPARATION_DEG: f32 = 10.0; // reject nearly colinear vanishing directions

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
    /// Returns detailed inference with segment assignments.
    pub fn infer(&self, l: &ImageF32) -> Option<DetailedInference> {
        let segments = lsd_extract_segments(l, self.options);
        self.infer_with_segments_internal(l, segments)
    }

    /// Run inference on pre-computed segments (e.g., using custom LSD options).
    pub fn infer_with_segments(
        &self,
        l: &ImageF32,
        segments: Vec<Segment>,
    ) -> Option<DetailedInference> {
        self.infer_with_segments_internal(l, segments)
    }

    fn infer_with_segments_internal(
        &self,
        l: &ImageF32,
        segments: Vec<Segment>,
    ) -> Option<DetailedInference> {
        let t0 = Instant::now();
        if segments.len() < MIN_SEGS {
            debug!(
                "LSD-VP: insufficient segments on level {}x{} ({} < {})",
                l.w,
                l.h,
                segments.len(),
                MIN_SEGS
            );
            return None;
        }

        // Orientation histogram (0..pi) weighted by segment strength
        let mut hist = OrientationHistogram::new(DEFAULT_BINS);
        let mut angles = Vec::with_capacity(segments.len());
        for seg in &segments {
            let th = seg.dir[1].atan2(seg.dir[0]);
            let angle = normalize_half_pi(th);
            angles.push(angle);
            hist.accumulate(angle, seg.strength.max(1.0));
        }
        hist.smooth_121();

        // Select two dominant peaks separated by at least ~ angle_tol*2
        let min_sep = (self.options.angle_tolerance_deg * 2.0).to_radians();
        let (first_idx, second_idx) = match hist.find_two_peaks(min_sep) {
            Some(peaks) => peaks,
            None => {
                debug!(
                    "LSD-VP: dominant orientation peaks not found (min_sep_deg={:.1})",
                    min_sep.to_degrees()
                );
                return None;
            }
        };
        let theta_u = hist.refined_angle(first_idx, 1);
        let theta_v = hist.refined_angle(second_idx, 1);

        // Soft-assign segments to the two families
        let tol = self.options.angle_tolerance_deg.to_radians();
        let mut u_idx: Vec<usize> = Vec::new();
        let mut v_idx: Vec<usize> = Vec::new();
        let mut families: Vec<Option<FamilyLabel>> = vec![None; segments.len()];
        for (i, angle) in angles.iter().enumerate() {
            let d1 = angular_difference(*angle, theta_u);
            let d2 = angular_difference(*angle, theta_v);
            if d1 < d2 && d1 <= tol {
                u_idx.push(i);
                families[i] = Some(FamilyLabel::U);
            } else if d2 < d1 && d2 <= tol {
                v_idx.push(i);
                families[i] = Some(FamilyLabel::V);
            }
        }
        if u_idx.len() < MIN_FAMILY || v_idx.len() < MIN_FAMILY {
            debug!(
                "LSD-VP: insufficient family support fam1={} fam2={}",
                u_idx.len(),
                v_idx.len()
            );
            return None;
        }

        // Estimate vanishing points for each family
        let vpu = estimate_vp(&segments, &u_idx, theta_u)?;
        let vpv = estimate_vp(&segments, &v_idx, theta_v)?;

        // Compose coarse projective basis anchored at the image center
        let cx = (l.w as f32) * 0.5;
        let cy = (l.h as f32) * 0.5;
        let x0 = Vector3::new(cx, cy, 1.0);
        let hmtx0 = Matrix3::from_columns(&[vpu, vpv, x0]);

        // Reject nearly colinear vanishing directions
        let sep_min_rad = MIN_VP_SEPARATION_DEG.to_radians();
        let du = vp_direction(&vpu, &x0);
        let dv = vp_direction(&vpv, &x0);
        if let (Some(du), Some(dv)) = (du, dv) {
            let angle = angle_between(&du, &dv);
            let folded = angle.min(std::f32::consts::PI - angle);
            if folded < sep_min_rad {
                debug!(
                    "LSD-VP: vanishing directions too close (angle_deg={:.2})",
                    folded.to_degrees()
                );
                return None;
            }
        } else {
            debug!("LSD-VP: could not derive vanishing directions from VPs");
            return None;
        }

        // Confidence heuristic from support and angular separation
        let sep = angular_difference(theta_u, theta_v);
        let conf = ((u_idx.len().min(50) as f32 / 50.0)
            * (v_idx.len().min(50) as f32 / 50.0)
            * (sep / (0.5 * std::f32::consts::PI)).min(1.0))
        .clamp(0.0, 1.0);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "LSD-VP: segs={} fam1={} fam2={} sep_deg={:.1} confidence={:.3} elapsed_ms={:.3}",
            segments.len(),
            u_idx.len(),
            v_idx.len(),
            sep.to_degrees(),
            conf,
            elapsed_ms
        );

        let hypothesis = Hypothesis {
            hmtx0,
            confidence: conf,
        };

        Some(DetailedInference {
            hypothesis,
            dominant_angles_rad: [theta_u, theta_v],
            families,
            segments,
        })
    }
}
