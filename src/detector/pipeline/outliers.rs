use crate::detector::outliers::classify_segments_with_details;
use crate::detector::params::OutlierFilterParams;
use crate::diagnostics::{OutlierFilterStage, OutlierThresholds, SegmentSample};
use crate::segments::{LsdOptions, Segment};
use log::debug;
use nalgebra::Matrix3;
use std::time::Instant;

#[derive(Debug)]
pub struct OutlierComputation {
    pub stage: Option<OutlierFilterStage>,
    pub segments: Vec<Segment>,
    pub elapsed_ms: f64,
}

pub fn filter_segments(
    segments: &[Segment],
    coarse_h: Option<&Matrix3<f32>>,
    params: &OutlierFilterParams,
    lsd_params: &LsdOptions,
) -> OutlierComputation {
    let Some(h) = coarse_h else {
        return OutlierComputation {
            stage: None,
            segments: Vec::new(),
            elapsed_ms: 0.0,
        };
    };
    if segments.is_empty() {
        return OutlierComputation {
            stage: None,
            segments: Vec::new(),
            elapsed_ms: 0.0,
        };
    }

    let filter_start = Instant::now();
    let (decisions, diag) = classify_segments_with_details(segments, h, params, lsd_params);
    let elapsed_ms = filter_start.elapsed().as_secs_f64() * 1000.0;

    let mut kept_segments = Vec::new();
    let mut classifications = Vec::with_capacity(decisions.len());
    for decision in &decisions {
        let seg_id = segments[decision.index].id;
        if decision.inlier {
            kept_segments.push(segments[decision.index].clone());
        }
        classifications.push(SegmentSample::from_decision(seg_id, decision));
    }
    if diag.total > 0 && kept_segments.is_empty() {
        debug!(
            "Outlier filter rejected all {} segments -> fallback to unfiltered set",
            diag.total
        );
        kept_segments = segments.to_vec();
    }

    let stage = OutlierFilterStage {
        elapsed_ms,
        total: diag.total,
        kept: kept_segments.len(),
        rejected: diag.rejected,
        kept_u: diag.kept_u,
        kept_v: diag.kept_v,
        degenerate_segments: diag.skipped_degenerate,
        thresholds: OutlierThresholds {
            angle_threshold_deg: diag.angle_threshold_deg,
            angle_margin_deg: params.angle_margin_deg,
            residual_threshold_px: diag.residual_threshold_px,
        },
        classifications,
    };

    OutlierComputation {
        stage: Some(stage),
        segments: kept_segments,
        elapsed_ms,
    }
}
