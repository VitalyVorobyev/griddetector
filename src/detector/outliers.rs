//! Segment outlier rejection against a coarse homography.
//!
//! The filter removes coarse-level segments that are unlikely to contribute to a
//! stable refinement. A segment is kept when its tangent aligns with one of the
//! two homography-implied vanishing directions (within an angular margin) and
//! the corresponding line passes sufficiently close to the vanishing point.
//!
//! The filter surfaces useful diagnostics (per-family counts, thresholds), and
//! is lightweight enough to be applied on every frame before refinement.
use crate::angle::{angle_between_dirless, vp_direction};
use crate::detector::params::OutlierFilterParams;
use crate::lsd_vp::FamilyLabel;
use crate::segments::{LsdOptions, Segment};
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
}

impl OutlierFilterDiagnostics {
    fn new(angle_threshold_deg: f32) -> Self {
        Self {
            angle_threshold_deg,
            ..Default::default()
        }
    }
}

/// Filters coarse-level segments that are inconsistent with the homography’s vanishing points.
pub fn filter_segments(
    segments: Vec<Segment>,
    h_coarse: &Matrix3<f32>,
    filter_params: &OutlierFilterParams,
    lsd_params: &LsdOptions,
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
    lsd_params: &LsdOptions,
) -> (Vec<SegmentDecision>, OutlierFilterDiagnostics) {
    let angle_thresh_deg = lsd_params.angle_tolerance_deg + filter_params.angle_margin_deg;
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
                inlier: true,
                rejection: None,
            });
        }
        let diag = aggregate_diagnostics(&decisions, angle_thresh_deg);
        return (decisions, diag);
    }

    let ctx = ClassificationContext {
        dir_u,
        dir_v,
        angle_thresh_rad,
    };
    for (i, seg) in segments.iter().enumerate() {
        decisions.push(classify_one(i, seg, &ctx));
    }

    let diag = aggregate_diagnostics(&decisions, angle_thresh_deg);
    (decisions, diag)
}

struct ClassificationContext {
    dir_u: Option<[f32; 2]>,
    dir_v: Option<[f32; 2]>,
    angle_thresh_rad: f32,
}

fn classify_one(index: usize, seg: &Segment, ctx: &ClassificationContext) -> SegmentDecision {
    let tangent = seg.direction();

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
            inlier: false,
            rejection: Some("angle"),
        };
    }

    SegmentDecision {
        index,
        family,
        angle_diff_rad: Some(angle),
        inlier: true,
        rejection: None,
    }
}

fn aggregate_diagnostics(
    decisions: &[SegmentDecision],
    angle_threshold_deg: f32,
) -> OutlierFilterDiagnostics {
    let mut diag = OutlierFilterDiagnostics::new(angle_threshold_deg);
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
    use crate::segments::SegmentId;
    use nalgebra::Matrix3;

    fn make_segment(id: u32, dir: [f32; 2]) -> Segment {
        Segment::new(SegmentId(id), [0.0, 0.0], [dir[0], dir[1]], 1.0, 1.0)
    }

    #[test]
    fn filter_segments_respects_angle_threshold() {
        let h = Matrix3::new(
            1.0, 0.0, 0.0, // VPu at infinity along +x
            0.0, 1.0, 0.0, // VPv at infinity along +y
            0.0, 0.0, 1.0,
        );
        let seg_ok = make_segment(0, [1.0, 0.0]); // horizontal line y=0
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        let seg_bad = make_segment(1, [r2, r2]);
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 0.0,
        };
        let lsd_params = LsdOptions {
            magnitude_threshold: 0.05,
            angle_tolerance_deg: 12.0,
            min_length_px: 4.0,
            enforce_polarity: false,
            normal_span_limit_px: None,
        };
        let segments = vec![seg_ok.clone(), seg_bad];
        let (filtered, diag) = filter_segments(segments, &h, &filter_params, &lsd_params);
        assert_eq!(filtered.len(), 1);
        assert_eq!(diag.kept, 1);
        assert_eq!(diag.rejected, 1);
        assert_eq!(filtered[0].direction(), seg_ok.direction());
    }

    #[test]
    fn filter_segments_residual_gate() {
        let h = Matrix3::from_columns(&[
            nalgebra::Vector3::new(10.0, 0.0, 1.0), // VPu at (10, 0)
            nalgebra::Vector3::new(0.0, 1.0, 0.0),  // VPv at infinity along y
            nalgebra::Vector3::new(0.0, 0.0, 1.0),  // anchor at origin
        ]);
        let seg = make_segment(0, [1.0, 0.0]); // horizontal line y=2
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 20.0,
        };
        let lsd_params = LsdOptions {
            magnitude_threshold: 0.05,
            angle_tolerance_deg: 10.0,
            min_length_px: 4.0,
            enforce_polarity: false,
            normal_span_limit_px: None,
        };
        let (filtered, diag) = filter_segments(vec![seg], &h, &filter_params, &lsd_params);
        assert_eq!(filtered.len(), 1, "Segment should pass angle check");
        assert_eq!(diag.rejected, 0, "Segment should not be rejected");
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
        let seg_pos = make_segment(0, [1.0, 0.0]);
        let seg_neg = make_segment(1, [-1.0, 0.0]);
        let filter_params = OutlierFilterParams {
            angle_margin_deg: 0.0,
        };
        let lsd_params = LsdOptions {
            magnitude_threshold: 0.05,
            angle_tolerance_deg: 12.0,
            min_length_px: 4.0,
            enforce_polarity: false,
            normal_span_limit_px: None,
        };
        let segments = vec![seg_pos.clone(), seg_neg.clone()];
        let (filtered, diag) = filter_segments(segments, &h, &filter_params, &lsd_params);
        assert_eq!(filtered.len(), 2);
        assert_eq!(diag.kept, 2);
        assert_eq!(diag.rejected, 0);
    }
}
