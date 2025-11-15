use crate::detector::scaling::LevelScaleMap;
use crate::detector::workspace::DetectorWorkspace;
use crate::diagnostics::GradientRefineStage;
use crate::pyramid::Pyramid;
#[cfg(feature = "profile_refine")]
use crate::refine::segment::take_profile;
use crate::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, RefineParams as SegmentRefineParams,
};
use crate::segments::Segment;
#[cfg(feature = "profile_refine")]
use log::info;
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
    let grad_view = workspace.scharr_gradients_full(coarse_idx, level);
    let grad_level = SegmentGradientLevel {
        width: level.w,
        height: level.h,
        origin_x: grad_view.origin_x,
        origin_y: grad_view.origin_y,
        tile_width: grad_view.tile_width,
        tile_height: grad_view.tile_height,
        gx: grad_view.gx,
        gy: grad_view.gy,
        level_index: coarse_idx,
    };
    let scale_map = LevelScaleMap::new(1.0, 1.0);
    let full_width = pyramid.levels.first().map(|lvl| lvl.w).unwrap_or(level.w);
    let level_params = params.for_level(full_width, level.w);

    let refine_start = Instant::now();
    let mut refined = Vec::with_capacity(segments.len());
    let mut accepted = 0usize;
    let mut score_acc = 0.0f32;
    let mut movement_acc = 0.0f32;

    for seg in segments {
        let result = segment::refine_segment(&grad_level, seg, &scale_map, &level_params);
        let movement = average_endpoint_movement(seg, &result.seg);
        if result.ok {
            accepted += 1;
            score_acc += result.score;
            movement_acc += movement;
            refined.push(result.seg);
        }
    }
    let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(feature = "profile_refine")]
    log_segment_profile("gradient_refine", workspace);

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
        fallback_to_input,
    };

    GradientRefineComputation {
        stage: Some(stage),
        segments: segments_out,
        elapsed_ms,
    }
}

#[cfg(feature = "profile_refine")]
fn log_segment_profile(stage: &str, workspace: &DetectorWorkspace) {
    let profile = take_profile();
    if profile.is_empty() {
        return;
    }
    info!("Segment refine profile for stage '{}':", stage);
    for entry in profile {
        if entry.roi_count == 0 && entry.bilinear_samples == 0 {
            continue;
        }
        let avg_roi = if entry.roi_count > 0 {
            entry.roi_area_px / entry.roi_count as f64
        } else {
            0.0
        };
        let grad_ms = workspace.gradient_time_ms(entry.level_index).unwrap_or(0.0);
        info!(
            "  L{}: grad_ms={:.2} roi_count={} avg_roi_px={:.1} bilinear_samples={}",
            entry.level_index, grad_ms, entry.roi_count, avg_roi, entry.bilinear_samples
        );
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
