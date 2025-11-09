use crate::detector::outliers::SegmentDecision;
use crate::lsd_vp::FamilyLabel;
use crate::segments::SegmentId;
use serde::{Deserialize, Serialize};

/// Classification outcome for a segment after running the outlier filter.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSample {
    pub segment: SegmentId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<FamilyLabel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_diff_deg: Option<f32>,
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
