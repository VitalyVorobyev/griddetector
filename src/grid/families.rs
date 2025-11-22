use super::bundling::Bundle;
use super::histogram::OrientationHistogram;
use super::vp::vp_direction;

use crate::angle::{angular_difference, normalize_half_pi, angle_between_dirless};
use crate::segments::Segment;

use serde::{Serialize};
use nalgebra::{Vector3, Matrix3};

const MIN_SEGS: usize = 12;
const MIN_FAMILY: usize = 6;

/// Identifier for the two dominant line families found by the LSD→VP engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FamilyLabel {
    U,
    V,
}

/// Outcome of the orientation histogram analysis.
#[derive(Clone, Debug)]
pub struct FamilyAssignments {
    pub dominant_angles_rad: [f32; 2],
    pub families: Vec<Option<FamilyLabel>>,
    pub u_support: Vec<usize>,
    pub v_support: Vec<usize>,
}

impl FamilyAssignments {
    pub fn confidence(&self) -> f32 {
        confidence(
            self.u_support.len(),
            self.v_support.len(),
            self.dominant_angles_rad[0],
            self.dominant_angles_rad[1],
        )
    }
}

/// Reasons why orientation analysis may fail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FamilyAnalysisError {
    InsufficientSegments {
        found: usize,
        minimum: usize,
    },
    DominantPeaksNotFound {
        min_separation_deg: f32,
    },
    WeakFamilySupport {
        family_u: usize,
        family_v: usize,
        minimum: usize,
    },
}

impl std::fmt::Display for FamilyAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FamilyAnalysisError::InsufficientSegments { found, minimum } => {
                write!(f, "insufficient segments ({found} < {minimum})")
            }
            FamilyAnalysisError::DominantPeaksNotFound { min_separation_deg } => write!(
                f,
                "orientation peaks not found (min separation {:.1}°)",
                min_separation_deg
            ),
            FamilyAnalysisError::WeakFamilySupport {
                family_u,
                family_v,
                minimum,
            } => write!(
                f,
                "weak family support (u={family_u}, v={family_v}, need ≥{minimum})"
            ),
        }
    }
}

impl std::error::Error for FamilyAnalysisError {}

/// Build an orientation histogram over [0, π), find two dominant peaks, and
/// assign segments to the corresponding families.
pub fn analyze_families(
    segments: &[Segment],
    angle_tolerance_deg: f32,
) -> Result<FamilyAssignments, FamilyAnalysisError> {
    if segments.len() < MIN_SEGS {
        return Err(FamilyAnalysisError::InsufficientSegments {
            found: segments.len(),
            minimum: MIN_SEGS,
        });
    }

    let (angles, mut hist) = build_orientation_histogram(segments);
    hist.smooth_121();

    let min_sep = (angle_tolerance_deg * 2.0).to_radians();
    let (theta_u, theta_v) =
        select_two_peaks(&hist, min_sep).ok_or(FamilyAnalysisError::DominantPeaksNotFound {
            min_separation_deg: min_sep.to_degrees(),
        })?;

    let tol = angle_tolerance_deg.to_radians();
    let (families, u_support, v_support) = assign_families(&angles, theta_u, theta_v, tol);
    if !validate_family_support(u_support.len(), v_support.len()) {
        return Err(FamilyAnalysisError::WeakFamilySupport {
            family_u: u_support.len(),
            family_v: v_support.len(),
            minimum: MIN_FAMILY,
        });
    }

    Ok(FamilyAssignments {
        dominant_angles_rad: [theta_u, theta_v],
        families,
        u_support,
        v_support,
    })
}

fn build_orientation_histogram(segments: &[Segment]) -> (Vec<f32>, OrientationHistogram) {
    let mut hist = OrientationHistogram::default();
    let mut angles = Vec::with_capacity(segments.len());
    for seg in segments.iter() {
        let th = seg.theta();
        let angle = normalize_half_pi(th);
        angles.push(angle);
        hist.accumulate(angle, seg.strength.max(1.0));
    }
    (angles, hist)
}

fn select_two_peaks(hist: &OrientationHistogram, min_sep: f32) -> Option<(f32, f32)> {
    let (first_idx, second_idx) = hist.find_two_peaks(min_sep)?;
    let theta_u = hist.refined_angle(first_idx, 1);
    let theta_v = hist.refined_angle(second_idx, 1);
    Some((theta_u, theta_v))
}

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


pub(crate) struct FamilyBuckets<'a> {
    pub vpu: Vector3<f32>,
    pub vpv: Vector3<f32>,
    pub anchor: Vector3<f32>,
    pub family_u: Vec<&'a Bundle>,
    pub family_v: Vec<&'a Bundle>,
}

/// Partition bundled line constraints into two vanishing-point families whose
/// tangents align with the current homography estimate.
///
/// The assignment is performed by comparing the angular distance between the
/// bundle tangent and the directions implied by the current homography columns
/// (vanishing points). Bundles outside the orientation tolerance are still
/// assigned to the closest family to keep the optimisation well conditioned.
pub(crate) fn split_bundles<'a>(
    h_current: &Matrix3<f32>,
    bundles: &'a [Bundle],
    orientation_tol: f32,
) -> Option<FamilyBuckets<'a>> {
    let vpu = h_current.column(0).into_owned();
    let vpv = h_current.column(1).into_owned();
    let anchor = h_current.column(2).into_owned();
    let dir_u = vp_direction(&vpu, &anchor)?;
    let dir_v = vp_direction(&vpv, &anchor)?;

    let mut fam_u: Vec<&Bundle> = Vec::new();
    let mut fam_v: Vec<&Bundle> = Vec::new();
    for bundle in bundles {
        let tangent = bundle.tangent();
        let du = angle_between_dirless(&tangent, &dir_u);
        let dv = angle_between_dirless(&tangent, &dir_v);
        if du <= orientation_tol && du < dv {
            fam_u.push(bundle);
        } else if dv <= orientation_tol && dv < du {
            fam_v.push(bundle);
        } else if du < dv {
            fam_u.push(bundle);
        } else {
            fam_v.push(bundle);
        }
    }

    Some(FamilyBuckets {
        vpu,
        vpv,
        anchor,
        family_u: fam_u,
        family_v: fam_v,
    })
}

#[inline]
fn validate_family_support(u_len: usize, v_len: usize) -> bool {
    u_len >= MIN_FAMILY && v_len >= MIN_FAMILY
}

fn confidence(u_len: usize, v_len: usize, theta_u: f32, theta_v: f32) -> f32 {
    let sep = angular_difference(theta_u, theta_v);
    ((u_len.min(50) as f32 / 50.0)
        * (v_len.min(50) as f32 / 50.0)
        * (sep / (0.5 * std::f32::consts::PI)).min(1.0))
    .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::{Segment, SegmentId};

    fn make_segment(dir: [f32; 2], strength: f32, id: u32) -> Segment {
        Segment::new(
            SegmentId(id),
            [0.0, 0.0],
            [dir[0], dir[1]],
            strength,
            strength,
        )
    }

    #[test]
    fn rejects_insufficient_segments() {
        let segs = vec![make_segment([1.0, 0.0], 5.0, 0)];
        assert!(matches!(
            analyze_families(&segs, 22.5),
            Err(FamilyAnalysisError::InsufficientSegments { .. })
        ));
    }
}
