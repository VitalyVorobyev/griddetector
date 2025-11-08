use crate::angle::{angle_between, angular_difference, normalize_half_pi, vp_direction};
use crate::image::ImageF32;
use crate::segments::{lsd_extract_segments, LsdOptions, Segment};
use log::debug;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::time::Instant;

use super::histogram::OrientationHistogram;
use super::vp::estimate_vp;

/// Number of bins in the 0..pi orientation histogram.
const DEFAULT_BINS: usize = 36;
/// Minimal number of LSD segments required to attempt inference.
const MIN_SEGS: usize = 12;
/// Minimal number of segments supporting each family to accept it as dominant.
const MIN_FAMILY: usize = 6;
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

        // 1) Orientation analysis
        let (angles, mut hist) = Self::build_orientation_histogram(&segments);
        hist.smooth_121();

        // 2) Select two dominant peaks
        let min_sep = (self.options.angle_tolerance_deg * 2.0).to_radians();
        let (theta_u, theta_v) = match Self::select_two_peaks(&hist, min_sep) {
            Some(thetas) => thetas,
            None => {
                debug!(
                    "LSD-VP: dominant orientation peaks not found (min_sep_deg={:.1})",
                    min_sep.to_degrees()
                );
                return None;
            }
        };

        // 3) Assign segment families
        let tol = self.options.angle_tolerance_deg.to_radians();
        let (families, u_idx, v_idx) = Self::assign_families(&angles, theta_u, theta_v, tol);
        if !Self::validate_family_support(u_idx.len(), v_idx.len()) {
            debug!(
                "LSD-VP: insufficient family support fam1={} fam2={}",
                u_idx.len(),
                v_idx.len()
            );
            return None;
        }

        // 4) Estimate VPs for both families
        let (vpu, vpv) = Self::estimate_family_vps(&segments, &u_idx, &v_idx, theta_u, theta_v)?;

        // 5) Validate VP separation
        let cx = (l.w as f32) * 0.5;
        let cy = (l.h as f32) * 0.5;
        if !Self::validate_vp_separation(&vpu, &vpv, cx, cy, MIN_VP_SEPARATION_DEG) {
            return None;
        }

        // 6) Compose basis and compute confidence
        let x0 = Vector3::new(cx, cy, 1.0);
        let hmtx0 = Matrix3::from_columns(&[vpu, vpv, x0]);
        let conf = Self::confidence(u_idx.len(), v_idx.len(), theta_u, theta_v);

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "LSD-VP: segs={} fam1={} fam2={} sep_deg={:.1} confidence={:.3} elapsed_ms={:.3}",
            segments.len(),
            u_idx.len(),
            v_idx.len(),
            angular_difference(theta_u, theta_v).to_degrees(),
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

    /// Build a strength-weighted orientation histogram over [0, pi) and the per-segment angles.
    fn build_orientation_histogram(segments: &[Segment]) -> (Vec<f32>, OrientationHistogram) {
        let mut hist = OrientationHistogram::new(DEFAULT_BINS);
        let mut angles = Vec::with_capacity(segments.len());
        for seg in segments.iter() {
            let th = seg.dir[1].atan2(seg.dir[0]);
            let angle = normalize_half_pi(th);
            angles.push(angle);
            hist.accumulate(angle, seg.strength.max(1.0));
        }
        (angles, hist)
    }

    /// Select two dominant peaks in the histogram and refine them to angles.
    fn select_two_peaks(hist: &OrientationHistogram, min_sep: f32) -> Option<(f32, f32)> {
        let (first_idx, second_idx) = hist.find_two_peaks(min_sep)?;
        let theta_u = hist.refined_angle(first_idx, 1);
        let theta_v = hist.refined_angle(second_idx, 1);
        Some((theta_u, theta_v))
    }

    /// Soft-assign each segment to the closest family within tolerance.
    fn assign_families(
        angles: &[f32],
        theta_u: f32,
        theta_v: f32,
        tol: f32,
    ) -> (Vec<Option<FamilyLabel>>, Vec<usize>, Vec<usize>) {
        let mut u_idx: Vec<usize> = Vec::new();
        let mut v_idx: Vec<usize> = Vec::new();
        let mut families: Vec<Option<FamilyLabel>> = vec![None; angles.len()];
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
        (families, u_idx, v_idx)
    }

    /// Verify that both families have sufficient support.
    #[inline]
    fn validate_family_support(u_len: usize, v_len: usize) -> bool {
        u_len >= MIN_FAMILY && v_len >= MIN_FAMILY
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

    /// Confidence heuristic from support counts and angular separation.
    fn confidence(u_len: usize, v_len: usize, theta_u: f32, theta_v: f32) -> f32 {
        let sep = angular_difference(theta_u, theta_v);
        ((u_len.min(50) as f32 / 50.0)
            * (v_len.min(50) as f32 / 50.0)
            * (sep / (0.5 * std::f32::consts::PI)).min(1.0))
        .clamp(0.0, 1.0)
    }
}
