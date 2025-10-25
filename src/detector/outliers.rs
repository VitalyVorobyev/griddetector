//! Segment outlier rejection against a coarse homography.
//!
//! The filter removes coarse-level segments that are unlikely to contribute to a
//! stable refinement. A segment is kept when its tangent aligns with one of the
//! two homography-implied vanishing directions (within an angular margin) and
//! the corresponding line passes sufficiently close to the vanishing point.
//!
//! The filter surfaces useful diagnostics (per-family counts, thresholds), and
//! is lightweight enough to be applied on every frame before refinement.
use crate::angle::angle_between;
use crate::angle::vp_direction;
use crate::detector::params::{LsdVpParams, OutlierFilterParams};
use crate::lsd_vp::FamilyLabel;
use crate::segments::Segment;
use nalgebra::Matrix3;

/// Diagnostics emitted by the outlier filter.
#[derive(Clone, Debug, Default)]
pub struct OutlierFilterDiagnostics {
    pub total: usize,
    pub kept: usize,
    pub rejected: usize,
    pub kept_u: usize,
    pub kept_v: usize,
    pub skipped_degenerate: usize,
    pub angle_threshold_deg: f32,
    pub residual_threshold_px: f32,
}

impl OutlierFilterDiagnostics {
    fn new(angle_threshold_deg: f32, residual_threshold_px: f32) -> Self {
        Self {
            angle_threshold_deg,
            residual_threshold_px,
            ..Default::default()
        }
    }
}

/// Filters coarse-level segments that are inconsistent with the homography’s vanishing points.
pub fn filter_segments(
    segments: Vec<Segment>,
    h_coarse: &Matrix3<f32>,
    filter_params: &OutlierFilterParams,
    lsd_params: &LsdVpParams,
) -> (Vec<Segment>, OutlierFilterDiagnostics) {
    let angle_thresh_deg = lsd_params.angle_tol_deg + filter_params.angle_margin_deg;
    let angle_thresh_rad = angle_thresh_deg.to_radians();
    let mut diag =
        OutlierFilterDiagnostics::new(angle_thresh_deg, filter_params.line_residual_thresh_px);

    let vpu = h_coarse.column(0).into_owned();
    let vpv = h_coarse.column(1).into_owned();
    let anchor = h_coarse.column(2).into_owned();
    let dir_u = vp_direction(&vpu, &anchor);
    let dir_v = vp_direction(&vpv, &anchor);
    if dir_u.is_none() && dir_v.is_none() {
        diag.skipped_degenerate = segments.len();
        diag.kept = segments.len();
        diag.total = segments.len();
        return (segments, diag);
    }

    diag.total = segments.len();
    let mut kept = Vec::with_capacity(segments.len());
    for seg in segments {
        let tangent = seg.dir;
        let du = dir_u.map(|d| angle_between(&tangent, &d));
        let dv = dir_v.map(|d| angle_between(&tangent, &d));
        let (family, angle) = match (du, dv) {
            (Some(a), Some(b)) => {
                if a <= b {
                    (FamilyLabel::U, a)
                } else {
                    (FamilyLabel::V, b)
                }
            }
            (Some(a), None) => (FamilyLabel::U, a),
            (None, Some(b)) => (FamilyLabel::V, b),
            (None, None) => {
                diag.skipped_degenerate += 1;
                kept.push(seg);
                continue;
            }
        };

        if angle > angle_thresh_rad {
            diag.rejected += 1;
            continue;
        }

        if !line_consistent_with_vp(
            &seg,
            &family,
            &vpu,
            &vpv,
            filter_params.line_residual_thresh_px,
        ) {
            diag.rejected += 1;
            continue;
        }

        match family {
            FamilyLabel::U => diag.kept_u += 1,
            FamilyLabel::V => diag.kept_v += 1,
        }
        kept.push(seg);
    }
    diag.kept = kept.len();
    (kept, diag)
}

fn line_consistent_with_vp(
    seg: &Segment,
    family: &FamilyLabel,
    vpu: &nalgebra::Vector3<f32>,
    vpv: &nalgebra::Vector3<f32>,
    residual_thresh: f32,
) -> bool {
    let vp = match family {
        FamilyLabel::U => vpu,
        FamilyLabel::V => vpv,
    };
    if vp[2].abs() <= 1e-3 {
        // Vanishing point at infinity -> evaluate direction only.
        return true;
    }
    let x = vp[0] / vp[2];
    let y = vp[1] / vp[2];
    let residual = seg.line[0] * x + seg.line[1] * y + seg.line[2];
    residual.abs() <= residual_thresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    fn make_segment(dir: [f32; 2], line: [f32; 3]) -> Segment {
        Segment {
            p0: [0.0, 0.0],
            p1: [dir[0], dir[1]],
            dir,
            len: (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-3),
            line,
            avg_mag: 1.0,
            strength: 1.0,
        }
    }

    #[test]
    fn filter_segments_respects_angle_threshold() {
        let h = Matrix3::new(
            1.0, 0.0, 0.0, // VPu at infinity along +x
            0.0, 1.0, 0.0, // VPv at infinity along +y
            0.0, 0.0, 1.0,
        );
        let seg_ok = make_segment([1.0, 0.0], [0.0, 1.0, 0.0]); // horizontal line y=0
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        let seg_bad = make_segment([r2, r2], [r2, -r2, 0.0]);
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 0.0,
            line_residual_thresh_px: 10.0,
        };
        let lsd_params = LsdVpParams {
            mag_thresh: 0.05,
            angle_tol_deg: 12.0,
            min_len: 4.0,
        };
        let segments = vec![seg_ok.clone(), seg_bad];
        let (filtered, diag) = filter_segments(segments, &h, &filter_params, &lsd_params);
        assert_eq!(filtered.len(), 1);
        assert_eq!(diag.kept, 1);
        assert_eq!(diag.rejected, 1);
        assert_eq!(filtered[0].dir, seg_ok.dir);
    }

    #[test]
    fn filter_segments_residual_gate_rejects_outliers() {
        let h = Matrix3::from_columns(&[
            nalgebra::Vector3::new(10.0, 0.0, 1.0), // VPu at (10, 0)
            nalgebra::Vector3::new(0.0, 1.0, 0.0),  // VPv at infinity along y
            nalgebra::Vector3::new(0.0, 0.0, 1.0),  // anchor at origin
        ]);
        let seg = make_segment([1.0, 0.0], [0.0, 1.0, -2.0]); // horizontal line y=2
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 20.0,
            line_residual_thresh_px: 1.0,
        };
        let lsd_params = LsdVpParams {
            mag_thresh: 0.05,
            angle_tol_deg: 10.0,
            min_len: 4.0,
        };
        let (filtered, diag) = filter_segments(vec![seg], &h, &filter_params, &lsd_params);
        assert!(
            filtered.is_empty(),
            "expected residual gate to reject segment"
        );
        assert_eq!(diag.rejected, 1);
    }
}
