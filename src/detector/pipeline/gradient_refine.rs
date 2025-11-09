use crate::detector::scaling::LevelScaleMap;
use crate::detector::workspace::DetectorWorkspace;
use crate::diagnostics::GradientRefineStage;
use crate::pyramid::Pyramid;
use crate::refine::segment::{self, SegmentRefineParams};
use crate::segments::Segment;
use std::time::Instant;

#[derive(Debug)]
pub struct GradientRefineComputation {
    pub stage: Option<GradientRefineStage>,
    pub segments: Vec<Segment>,
    pub elapsed_ms: f64,
}

pub fn refine_coarsest_with_gradients(
    workspace: &mut DetectorWorkspace,
    pyramid: &Pyramid,
    segments: &[Segment],
    params: &SegmentRefineParams,
) -> GradientRefineComputation {
    let Some(coarse_idx) = pyramid.levels.len().checked_sub(1) else {
        return GradientRefineComputation {
            stage: None,
            segments: Vec::new(),
            elapsed_ms: 0.0,
        };
    };
    if segments.is_empty() {
        return GradientRefineComputation {
            stage: None,
            segments: Vec::new(),
            elapsed_ms: 0.0,
        };
    }

    let level = &pyramid.levels[coarse_idx];
    let grad_level = workspace.gradient_level(coarse_idx, level);
    let scale_map = LevelScaleMap::new(1.0, 1.0);

    let refine_start = Instant::now();
    let mut refined = Vec::with_capacity(segments.len());
    let mut accepted = 0usize;
    let mut score_acc = 0.0f32;
    let mut movement_acc = 0.0f32;
    let mut tangent_steps_acc = 0usize;
    let mut normal_refine_acc = 0usize;
    let mut gradient_samples_acc = 0usize;
    let mut support_acc = 0usize;

    for seg in segments {
        let result = segment::refine_segment(&grad_level, seg, &scale_map, params);
        let movement = average_endpoint_movement(seg, &result.seg);
        if result.ok {
            accepted += 1;
            score_acc += result.score;
            movement_acc += movement;
            tangent_steps_acc += result.diagnostics.tangent_steps;
            normal_refine_acc += result.diagnostics.normal_refinements;
            gradient_samples_acc += result.diagnostics.gradient_samples;
            support_acc += result.support_points;
            refined.push(result.seg);
        }
    }
    let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;

    let fallback_to_input = refined.is_empty();
    let segments_out = if fallback_to_input {
        segments.to_vec()
    } else {
        refined
    };

    let stage = GradientRefineStage {
        level_index: coarse_idx,
        elapsed_ms,
        segments_in: segments.len(),
        accepted,
        rejected: segments.len().saturating_sub(accepted),
        avg_score: if accepted > 0 {
            Some(score_acc / accepted as f32)
        } else {
            None
        },
        avg_movement_px: if accepted > 0 {
            Some(movement_acc / accepted as f32)
        } else {
            None
        },
        avg_tangent_steps: if accepted > 0 {
            Some(tangent_steps_acc as f32 / accepted as f32)
        } else {
            None
        },
        avg_normal_refinements: if accepted > 0 {
            Some(normal_refine_acc as f32 / accepted as f32)
        } else {
            None
        },
        avg_gradient_samples: if accepted > 0 {
            Some(gradient_samples_acc as f32 / accepted as f32)
        } else {
            None
        },
        avg_support_points: if accepted > 0 {
            Some(support_acc as f32 / accepted as f32)
        } else {
            None
        },
        fallback_to_input,
    };

    GradientRefineComputation {
        stage: Some(stage),
        segments: segments_out,
        elapsed_ms,
    }
}

fn average_endpoint_movement(a: &Segment, b: &Segment) -> f32 {
    let da0 = distance_squared(a.p0, b.p0).sqrt();
    let da1 = distance_squared(a.p1, b.p1).sqrt();
    0.5 * (da0 + da1)
}

fn distance_squared(p0: [f32; 2], p1: [f32; 2]) -> f32 {
    let dx = p0[0] - p1[0];
    let dy = p0[1] - p1[1];
    dx * dx + dy * dy
}
