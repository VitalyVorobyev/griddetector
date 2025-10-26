//! Segment outlier rejection against a coarse homography.
//!
//! The filter removes coarse-level segments that are unlikely to contribute to a
//! stable refinement. A segment is kept when its tangent aligns with one of the
//! two homography-implied vanishing directions (within an angular margin) and
//! the corresponding line passes sufficiently close to the vanishing point.
//!
//! The filter surfaces useful diagnostics (per-family counts, thresholds), and
//! is lightweight enough to be applied on every frame before refinement.
use crate::angle::{angle_between_dirless, vp_direction, VP_EPS};
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
    let (decisions, diag) =
        classify_segments_with_details(&segments, h_coarse, filter_params, lsd_params);
    let mut kept = Vec::with_capacity(segments.len());
    for d in decisions {
        if d.inlier {
            kept.push(segments[d.index].clone());
        }
    }
    (kept, diag)
}

/// Per-segment decision details emitted by [`classify_segments_with_details`].
#[derive(Clone, Debug)]
pub struct SegmentDecision {
    pub index: usize,
    pub family: Option<FamilyLabel>,
    pub angle_diff_rad: Option<f32>,
    pub residual_px: Option<f32>,
    pub inlier: bool,
    /// Rejection reason when `inlier == false` ("angle" | "residual").
    pub rejection: Option<&'static str>,
}

/// Classifies segments against the homography-implied families and residual gate,
/// returning per-segment decisions and aggregate diagnostics.
///
/// This mirrors [`filter_segments`] but does not drop outliers; instead it provides
/// a decision per input segment so tools can build rich reports/visualizations
/// while relying on the same thresholds as the detector.
pub fn classify_segments_with_details(
    segments: &[Segment],
    h_coarse: &Matrix3<f32>,
    filter_params: &OutlierFilterParams,
    lsd_params: &LsdVpParams,
) -> (Vec<SegmentDecision>, OutlierFilterDiagnostics) {
    let angle_thresh_deg = lsd_params.angle_tol_deg + filter_params.angle_margin_deg;
    let angle_thresh_rad = angle_thresh_deg.to_radians();

    let vpu = h_coarse.column(0).into_owned();
    let vpv = h_coarse.column(1).into_owned();
    let anchor = h_coarse.column(2).into_owned();
    let dir_u = vp_direction(&vpu, &anchor);
    let dir_v = vp_direction(&vpv, &anchor);

    let mut decisions = Vec::with_capacity(segments.len());

    // Degenerate: both VP directions unavailable → keep everything, mark as degenerate.
    if dir_u.is_none() && dir_v.is_none() {
        for i in 0..segments.len() {
            decisions.push(SegmentDecision {
                index: i,
                family: None,
                angle_diff_rad: None,
                residual_px: None,
                inlier: true,
                rejection: None,
            });
        }
        let diag = aggregate_diagnostics(
            &decisions,
            angle_thresh_deg,
            filter_params.line_residual_thresh_px,
        );
        return (decisions, diag);
    }

    let ctx = ClassificationContext {
        vpu: &vpu,
        vpv: &vpv,
        dir_u,
        dir_v,
        angle_thresh_rad,
        residual_thresh_px: filter_params.line_residual_thresh_px,
    };
    for (i, seg) in segments.iter().enumerate() {
        decisions.push(classify_one(i, seg, &ctx));
    }

    let diag = aggregate_diagnostics(
        &decisions,
        angle_thresh_deg,
        filter_params.line_residual_thresh_px,
    );
    (decisions, diag)
}

struct ClassificationContext<'a> {
    vpu: &'a nalgebra::Vector3<f32>,
    vpv: &'a nalgebra::Vector3<f32>,
    dir_u: Option<[f32; 2]>,
    dir_v: Option<[f32; 2]>,
    angle_thresh_rad: f32,
    residual_thresh_px: f32,
}

fn classify_one(index: usize, seg: &Segment, ctx: &ClassificationContext<'_>) -> SegmentDecision {
    let tangent = seg.dir;

    // Compare orientation with direction-invariant angle.
    let du = ctx.dir_u.map(|d| angle_between_dirless(&tangent, &d));
    let dv = ctx.dir_v.map(|d| angle_between_dirless(&tangent, &d));

    let (family, angle_opt) = match (du, dv) {
        (Some(a), Some(b)) => {
            if a <= b {
                (Some(FamilyLabel::U), Some(a))
            } else {
                (Some(FamilyLabel::V), Some(b))
            }
        }
        (Some(a), None) => (Some(FamilyLabel::U), Some(a)),
        (None, Some(b)) => (Some(FamilyLabel::V), Some(b)),
        (None, None) => (None, None),
    };

    if family.is_none() {
        return SegmentDecision {
            index,
            family: None,
            angle_diff_rad: None,
            residual_px: None,
            inlier: true,
            rejection: None,
        };
    }

    let angle = angle_opt.unwrap();
    if angle > ctx.angle_thresh_rad {
        return SegmentDecision {
            index,
            family,
            angle_diff_rad: Some(angle),
            residual_px: None,
            inlier: false,
            rejection: Some("angle"),
        };
    }

    let vp = match family.unwrap() {
        FamilyLabel::U => ctx.vpu,
        FamilyLabel::V => ctx.vpv,
    };
    let (residual_px, ok) = residual_to_vp_px(seg, vp)
        .map(|r| (Some(r), r <= ctx.residual_thresh_px))
        .unwrap_or((None, true));

    if !ok {
        return SegmentDecision {
            index,
            family,
            angle_diff_rad: Some(angle),
            residual_px,
            inlier: false,
            rejection: Some("residual"),
        };
    }

    SegmentDecision {
        index,
        family,
        angle_diff_rad: Some(angle),
        residual_px,
        inlier: true,
        rejection: None,
    }
}

fn residual_to_vp_px(seg: &Segment, vp: &nalgebra::Vector3<f32>) -> Option<f32> {
    if vp[2].abs() <= VP_EPS {
        // VP at infinity: residual undefined; treat as direction-only gate.
        None
    } else {
        let x = vp[0] / vp[2];
        let y = vp[1] / vp[2];
        let r = seg.line[0] * x + seg.line[1] * y + seg.line[2];
        Some(r.abs())
    }
}

fn aggregate_diagnostics(
    decisions: &[SegmentDecision],
    angle_threshold_deg: f32,
    residual_threshold_px: f32,
) -> OutlierFilterDiagnostics {
    let mut diag = OutlierFilterDiagnostics::new(angle_threshold_deg, residual_threshold_px);
    diag.total = decisions.len();
    for d in decisions {
        if d.family.is_none() {
            diag.skipped_degenerate += 1;
        }
        if d.inlier {
            diag.kept += 1;
            match d.family {
                Some(FamilyLabel::U) => diag.kept_u += 1,
                Some(FamilyLabel::V) => diag.kept_v += 1,
                None => {}
            }
        } else {
            diag.rejected += 1;
        }
    }
    diag
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
            enforce_polarity: false,
            normal_span_limit: None,
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
            enforce_polarity: false,
            normal_span_limit: None,
        };
        let (filtered, diag) = filter_segments(vec![seg], &h, &filter_params, &lsd_params);
        assert!(
            filtered.is_empty(),
            "expected residual gate to reject segment"
        );
        assert_eq!(diag.rejected, 1);
    }

    #[test]
    fn orientation_is_direction_invariant() {
        // Both families at infinity along axes; angle gate only.
        let h = Matrix3::new(
            1.0, 0.0, 0.0, // VPu at infinity along +x
            0.0, 1.0, 0.0, // VPv at infinity along +y
            0.0, 0.0, 1.0,
        );
        // Two segments with opposite tangents along x-axis; both should pass.
        let seg_pos = make_segment([1.0, 0.0], [0.0, 1.0, 0.0]);
        let seg_neg = make_segment([-1.0, 0.0], [0.0, 1.0, 0.0]);
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 0.0,
            line_residual_thresh_px: 10.0,
        };
        let lsd_params = LsdVpParams {
            mag_thresh: 0.05,
            angle_tol_deg: 12.0,
            min_len: 4.0,
            enforce_polarity: false,
            normal_span_limit: None,
        };
        let segments = vec![seg_pos.clone(), seg_neg.clone()];
        let (filtered, diag) = filter_segments(segments, &h, &filter_params, &lsd_params);
        assert_eq!(filtered.len(), 2);
        assert_eq!(diag.kept, 2);
        assert_eq!(diag.rejected, 0);
    }
}
