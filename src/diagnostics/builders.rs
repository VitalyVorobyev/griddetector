//! Builder helpers that expose the same diagnostics structures used by the
//! detector pipeline to standalone demos.

use crate::detector::outliers::{classify_segments_with_details, OutlierFilterDiagnostics};
use crate::detector::params::OutlierFilterParams;
use crate::detector::{run_segment_refine_pass, DetectorWorkspace, SegmentRefinePass};
use crate::diagnostics::{
    FamilyCounts, LsdStage, OutlierFilterStage, OutlierThresholds, SegmentRefineLevel,
    SegmentRefineSample, SegmentSample,
};
use crate::image::ImageF32;
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine, FamilyLabel};
use crate::pyramid::Pyramid;
use crate::refine::segment::RefineParams as SegmentRefineParams;
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

/// Result of running the coarse→fine gradient refinement outside the pipeline.
pub struct SegmentRefineLevels {
    pub levels: Vec<SegmentRefineLevel>,
    pub refined_segments: Vec<Segment>,
    pub elapsed_ms: f64,
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

/// Execute the refinement cascade and capture per-level diagnostics.
pub fn run_segment_refine_levels(
    pyramid: &Pyramid,
    workspace: &mut DetectorWorkspace,
    source_segments: &[Segment],
    refine_params: &SegmentRefineParams,
) -> SegmentRefineLevels {
    if pyramid.levels.len() < 2 || source_segments.is_empty() {
        return SegmentRefineLevels {
            levels: Vec::new(),
            refined_segments: Vec::new(),
            elapsed_ms: 0.0,
        };
    }

    let mut current_segments = source_segments.to_vec();
    let mut levels = Vec::new();
    let mut elapsed_ms = 0.0;
    let full_width = pyramid.levels.first().map(|lvl| lvl.w).unwrap_or(0);

    for coarse_idx in (1..pyramid.levels.len()).rev() {
        if current_segments.is_empty() {
            break;
        }
        let pass = run_segment_refine_pass(
            workspace,
            pyramid,
            coarse_idx,
            &current_segments,
            refine_params,
            full_width,
        );
        elapsed_ms += pass.elapsed_ms;
        levels.push(build_segment_level(&pass));
        current_segments = pass.into_accepted_segments();
    }

    SegmentRefineLevels {
        levels,
        refined_segments: current_segments,
        elapsed_ms,
    }
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

fn build_segment_level(pass: &SegmentRefinePass) -> SegmentRefineLevel {
    let mut accepted = 0usize;
    let mut score_sum = 0.0f32;
    let mut samples = Vec::with_capacity(pass.results.len());

    for result in &pass.results {
        if result.ok {
            accepted += 1;
            if result.score.is_finite() {
                score_sum += result.score;
            }
        }
        let score = result.score.is_finite().then_some(result.score);
        let inliers = (result.inliers > 0).then_some(result.inliers);
        let total = (result.total > 0).then_some(result.total);
        samples.push(SegmentRefineSample {
            segment: result.seg.clone(),
            score,
            ok: Some(result.ok),
            inliers,
            total,
        });
    }

    let acceptance_ratio =
        (!pass.results.is_empty()).then_some(accepted as f32 / pass.results.len() as f32);
    let avg_score = (accepted > 0).then_some(score_sum / accepted as f32);

    SegmentRefineLevel {
        coarse_level: pass.coarse_idx,
        finer_level: pass.finer_idx,
        elapsed_ms: pass.elapsed_ms,
        segments_in: pass.results.len(),
        accepted,
        acceptance_ratio,
        avg_score,
        results: samples,
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
