use crate::detector::outliers::SegmentDecision;
use crate::lsd_vp::FamilyLabel;
use crate::segments::Segment;
use serde::{Deserialize, Serialize};

/// Identifier referencing a segment recorded in the pipeline trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub u32);

/// Geometry snapshot for a segment detected on the coarse pyramid level.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentDescriptor {
    pub id: SegmentId,
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub direction: [f32; 2],
    pub length: f32,
    pub line: [f32; 3],
    pub average_magnitude: f32,
    pub strength: f32,
}

impl SegmentDescriptor {
    pub fn from_segment(id: SegmentId, seg: &Segment) -> Self {
        Self {
            id,
            p0: seg.p0,
            p1: seg.p1,
            direction: seg.dir,
            length: seg.len,
            line: seg.line,
            average_magnitude: seg.avg_mag,
            strength: seg.strength,
        }
    }
}

/// Classification outcome for a segment after running the outlier filter.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSample {
    pub segment: SegmentId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<FamilyLabel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_diff_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_px: Option<f32>,
    pub class: SegmentClass,
}

impl SegmentSample {
    pub fn from_decision(segment: SegmentId, decision: &SegmentDecision) -> Self {
        let class = match (decision.inlier, decision.rejection) {
            (true, _) => SegmentClass::Kept,
            (false, Some("angle")) => SegmentClass::RejectedAngle,
            (false, Some("residual")) => SegmentClass::RejectedResidual,
            (false, _) => SegmentClass::Degenerate,
        };
        Self {
            segment,
            family: decision.family,
            angle_diff_deg: decision.angle_diff_rad.map(|a| a.to_degrees()),
            residual_px: decision.residual_px,
            class,
        }
    }
}

/// Normalised classification label for the outlier filter.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentClass {
    Kept,
    RejectedAngle,
    RejectedResidual,
    Degenerate,
}
