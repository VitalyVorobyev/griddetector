use crate::angle::{angle_between, angular_difference, normalize_half_pi, vp_direction};
use crate::diagnostics::{LsdDiagnostics, LsdSegmentDiagnostics};
use crate::image::ImageF32;
use crate::segments::{lsd_extract_segments, Segment};
use log::debug;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::cmp::Ordering;
use std::time::Instant;

use super::histogram::OrientationHistogram;
use super::vp::estimate_vp;

const DEFAULT_BINS: usize = 36;
const MIN_SEGS: usize = 12;
const MIN_FAMILY: usize = 6;
const MIN_VP_SEPARATION_DEG: f32 = 10.0; // reject nearly colinear vanishing directions

/// Coarse hypothesis returned by the LSD→VP engine
#[derive(Clone, Debug, Serialize)]
pub struct Hypothesis {
    pub hmtx0: Matrix3<f32>,
    pub confidence: f32,
    pub diagnostics: LsdDiagnostics,
}

/// Lightweight engine that finds two dominant line families from LSD segments,
/// estimates their vanishing points, and returns a coarse projective basis H0.
#[derive(Clone, Debug)]
pub struct Engine {
    /// Gradient magnitude threshold at the pyramid level (0..1)
    pub mag_thresh: f32,
    /// Angular tolerance (degrees) for clustering around histogram peaks
    pub angle_tol_deg: f32,
    /// Minimum accepted segment length in pixels (at that level)
    pub min_len: f32,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            mag_thresh: 0.05,
            angle_tol_deg: 22.5,
            min_len: 4.0,
        }
    }
}

impl Engine {
    /// Run the engine on a single pyramid level image. Returns a coarse H0 if successful.
    pub fn infer(&self, l: &ImageF32) -> Option<Hypothesis> {
        let t0 = Instant::now();
        // 1) LSD-like segment extraction
        let segs = lsd_extract_segments(
            l,
            self.mag_thresh,
            self.angle_tol_deg.to_radians(),
            self.min_len,
        );
        if segs.len() < MIN_SEGS {
            debug!(
                "LSD-VP: insufficient segments on level {}x{} ({} < {})",
                l.w,
                l.h,
                segs.len(),
                MIN_SEGS
            );
            return None;
        }

        // 2) Orientation histogram (0..pi) weighted by segment strength
        let mut hist = OrientationHistogram::new(DEFAULT_BINS);
        let mut angles = Vec::with_capacity(segs.len());
        for s in &segs {
            // tangent direction
            let th = s.dir[1].atan2(s.dir[0]);
            let angle = normalize_half_pi(th);
            angles.push(angle);
            hist.accumulate(angle, s.strength.max(1.0));
        }
        hist.smooth_121();

        // 3) Take two dominant peaks separated by at least ~ angle_tol*2
        let min_sep = (self.angle_tol_deg * 2.0).to_radians();
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

        // 4) Soft-assign segments to the two families
        let tol = self.angle_tol_deg.to_radians();
        let mut u_idx: Vec<usize> = Vec::new();
        let mut v_idx: Vec<usize> = Vec::new();
        let mut families = vec![None; segs.len()];
        for (i, angle) in angles.iter().enumerate() {
            let d1 = angular_difference(*angle, theta_u);
            let d2 = angular_difference(*angle, theta_v);
            if d1 < d2 && d1 <= tol {
                u_idx.push(i);
                families[i] = Some(0);
            } else if d2 < d1 && d2 <= tol {
                v_idx.push(i);
                families[i] = Some(1);
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

        // 5) Estimate VPs from sets of normal-form lines (ax + by + c = 0)
        let vpu = estimate_vp(&segs, &u_idx, theta_u)?; // u-direction VP
        let vpv = estimate_vp(&segs, &v_idx, theta_v)?; // v-direction VP

        // 6) Compose a coarse projective basis H0. This isn't the final homography
        // yet; refinement will enforce spacing/metric later. We use the image center
        // as the third column to anchor translation.
        let cx = (l.w as f32) * 0.5;
        let cy = (l.h as f32) * 0.5;
        let x0 = Vector3::new(cx, cy, 1.0);
        let hmtx0 = Matrix3::from_columns(&[vpu, vpv, x0]);

        // Validate vanishing direction separation to avoid degenerate H0
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

        // 7) Confidence heuristic from support and angular separation
        let sep = angular_difference(theta_u, theta_v);
        let conf = ((u_idx.len().min(50) as f32 / 50.0)
            * (v_idx.len().min(50) as f32 / 50.0)
            * (sep / (0.5 * std::f32::consts::PI)).min(1.0))
        .clamp(0.0, 1.0);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "LSD-VP: segs={} fam1={} fam2={} sep_deg={:.1} confidence={:.3} elapsed_ms={:.3}",
            segs.len(),
            u_idx.len(),
            v_idx.len(),
            sep.to_degrees(),
            conf,
            elapsed_ms
        );
        let diagnostics =
            build_lsd_diagnostics(&segs, &families, theta_u, theta_v, conf, elapsed_ms);
        Some(Hypothesis {
            hmtx0,
            confidence: conf,
            diagnostics,
        })
    }
}

fn build_lsd_diagnostics(
    segs: &[Segment],
    families: &[Option<u8>],
    angle_u: f32,
    angle_v: f32,
    confidence: f32,
    elapsed_ms: f64,
) -> LsdDiagnostics {
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by(|a, b| {
        segs[*b]
            .strength
            .partial_cmp(&segs[*a].strength)
            .unwrap_or(Ordering::Equal)
    });
    let sample_cap = 512usize;
    let mut segments_sample = Vec::new();
    for idx in order.into_iter().take(sample_cap) {
        let seg = &segs[idx];
        let family = match families[idx] {
            Some(0) => Some("u"),
            Some(1) => Some("v"),
            _ => None,
        };
        segments_sample.push(LsdSegmentDiagnostics {
            p0: seg.p0,
            p1: seg.p1,
            len: seg.len,
            strength: seg.strength,
            family,
        });
    }
    let fam_u = families.iter().filter(|f| matches!(f, Some(0))).count();
    let fam_v = families.iter().filter(|f| matches!(f, Some(1))).count();

    LsdDiagnostics {
        segments_total: segs.len(),
        dominant_angles_deg: [angle_u.to_degrees(), angle_v.to_degrees()],
        family_u_count: fam_u,
        family_v_count: fam_v,
        confidence,
        elapsed_ms,
        segments_sample,
    }
}
