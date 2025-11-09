//! Builder helpers that expose the same diagnostics structures used by the
//! detector pipeline to standalone demos.

use crate::detector::outliers::{classify_segments_with_details, OutlierFilterDiagnostics};
use crate::detector::params::OutlierFilterParams;
use crate::diagnostics::{
    FamilyCounts, LsdStage, OutlierFilterStage, OutlierThresholds, SegmentSample,
};
use crate::image::ImageF32;
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine, FamilyLabel};
use crate::segments::{LsdOptions, Segment};
use nalgebra::Matrix3;

/// Result of running the LSD→VP stage in isolation.
pub struct LsdStageOutput {
    pub stage: LsdStage,
    pub segments: Vec<Segment>,
    pub coarse_h: Matrix3<f32>,
    pub full_h: Matrix3<f32>,
}

/// Result of applying the outlier filter on the coarse segments.
pub struct OutlierStageOutput {
    pub stage: OutlierFilterStage,
    pub inlier_segments: Vec<Segment>,
}

/// Execute the LSD→VP stage on a pyramid level, returning both the diagnostics
/// and the rescaled homography.
pub fn run_lsd_stage(
    engine: &LsdVpEngine,
    level: &ImageF32,
    segments_override: Option<Vec<Segment>>,
    full_width: usize,
    full_height: usize,
) -> Option<LsdStageOutput> {
    let used_override = segments_override.is_some();
    let DetailedInference {
        hypothesis,
        dominant_angles_rad,
        families,
        segments,
    } = if let Some(segs) = segments_override {
        engine.infer_with_segments(level, segs)?
    } else {
        engine.infer(level)?
    };

    let scale_x = if level.w > 0 {
        full_width as f32 / level.w as f32
    } else {
        1.0
    };
    let scale_y = if level.h > 0 {
        full_height as f32 / level.h as f32
    } else {
        1.0
    };
    let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    let full_h = if level.w > 0 && level.h > 0 {
        scale * hypothesis.hmtx0
    } else {
        hypothesis.hmtx0
    };

    let family_counts = compute_family_counts(&families);
    let dominant_angles_deg = [
        dominant_angles_rad[0].to_degrees(),
        dominant_angles_rad[1].to_degrees(),
    ];
    let stage = LsdStage {
        elapsed_ms: 0.0,
        confidence: hypothesis.confidence,
        dominant_angles_deg,
        family_counts,
        segment_families: families.clone(),
        sample_ids: Vec::new(),
        used_gradient_refinement: used_override,
    };

    Some(LsdStageOutput {
        stage,
        segments,
        coarse_h: hypothesis.hmtx0,
        full_h,
    })
}

/// Apply the outlier filter using the same thresholds as the detector.
pub fn run_outlier_stage(
    segments: &[Segment],
    coarse_h: &Matrix3<f32>,
    filter_params: &OutlierFilterParams,
    lsd_params: &LsdOptions,
) -> OutlierStageOutput {
    let (decisions, diag) =
        classify_segments_with_details(segments, coarse_h, filter_params, lsd_params);
    build_outlier_stage(decisions, diag, segments, filter_params)
}

fn build_outlier_stage(
    decisions: Vec<crate::detector::outliers::SegmentDecision>,
    diag: OutlierFilterDiagnostics,
    segments: &[Segment],
    filter_params: &OutlierFilterParams,
) -> OutlierStageOutput {
    let mut kept_segments = Vec::new();
    let mut classifications = Vec::with_capacity(decisions.len());
    for decision in &decisions {
        let seg_id = segments[decision.index].id;
        if decision.inlier {
            kept_segments.push(segments[decision.index].clone());
        }
        classifications.push(SegmentSample::from_decision(seg_id, decision));
    }

    let stage = OutlierFilterStage {
        elapsed_ms: 0.0,
        total: diag.total,
        kept: kept_segments.len(),
        rejected: diag.rejected,
        kept_u: diag.kept_u,
        kept_v: diag.kept_v,
        degenerate_segments: diag.skipped_degenerate,
        thresholds: OutlierThresholds {
            angle_threshold_deg: diag.angle_threshold_deg,
            angle_margin_deg: filter_params.angle_margin_deg,
        },
        classifications,
    };

    OutlierStageOutput {
        stage,
        inlier_segments: kept_segments,
    }
}

/// Count how many segments fall into each family (U/V) or remain unassigned.
pub fn compute_family_counts(families: &[Option<FamilyLabel>]) -> FamilyCounts {
    let mut counts = FamilyCounts {
        family_u: 0,
        family_v: 0,
        unassigned: 0,
    };
    for fam in families {
        match fam {
            Some(FamilyLabel::U) => counts.family_u += 1,
            Some(FamilyLabel::V) => counts.family_v += 1,
            None => counts.unassigned += 1,
        }
    }
    counts
}
