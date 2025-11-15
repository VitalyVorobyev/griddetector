use crate::detector::scaling::LevelScaleMap;
use crate::detector::workspace::DetectorWorkspace;
use crate::image::ImageF32;
use crate::pyramid::Pyramid;
use crate::refine::segment::roi::roi_to_int_bounds;
use crate::refine::segment::{
    self, IntBounds, PyramidLevel as SegmentGradientLevel, RefineParams as SegmentRefineParams,
    ScaleMap, SegmentRoi,
};
use crate::segments::Segment;
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Outcomes for a single coarse→fine refinement pass.
pub(crate) struct SegmentRefinePass {
    pub coarse_idx: usize,
    pub finer_idx: usize,
    pub elapsed_ms: f64,
    pub results: Vec<segment::RefineResult>,
}

impl SegmentRefinePass {
    pub fn into_accepted_segments(self) -> Vec<Segment> {
        self.results
            .into_iter()
            .filter_map(|result| result.ok.then_some(result.seg))
            .collect()
    }
}

/// Execute a single coarse→fine refinement pass using a shared gradient tile.
pub(crate) fn run_segment_refine_pass(
    workspace: &mut DetectorWorkspace,
    pyramid: &Pyramid,
    coarse_idx: usize,
    segments: &[Segment],
    refine_params: &SegmentRefineParams,
    full_width: usize,
) -> SegmentRefinePass {
    debug_assert!(coarse_idx > 0);
    let finer_idx = coarse_idx.saturating_sub(1);

    if segments.is_empty() {
        return SegmentRefinePass {
            coarse_idx,
            finer_idx,
            elapsed_ms: 0.0,
            results: Vec::new(),
        };
    }

    let coarse_level = &pyramid.levels[coarse_idx];
    let finer_level = &pyramid.levels[finer_idx];
    let scale_map = level_scale_map(coarse_level, finer_level);
    let level_params = refine_params.for_level(full_width, finer_level.w);

    let refine_start = Instant::now();
    let grad_level = build_gradient_level(
        workspace,
        finer_idx,
        finer_level,
        segments,
        &scale_map,
        level_params.pad,
    );
    let results = refine_segments_for_level(&grad_level, segments, &scale_map, &level_params);
    let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;

    SegmentRefinePass {
        coarse_idx,
        finer_idx,
        elapsed_ms,
        results,
    }
}

fn build_gradient_level<'a>(
    workspace: &'a mut DetectorWorkspace,
    level_idx: usize,
    level: &'a ImageF32,
    segments: &[Segment],
    scale_map: &LevelScaleMap,
    pad: f32,
) -> SegmentGradientLevel<'a> {
    let tile_bounds = merged_segment_bounds(segments, scale_map, pad, level.w, level.h)
        .and_then(|roi| roi_to_int_bounds(&roi, level.w, level.h))
        .and_then(|bounds| select_cropped_bounds(bounds, level.w, level.h));
    let grad_view = match tile_bounds {
        Some(bounds) => workspace.scharr_gradients_window(level_idx, level, &bounds),
        None => workspace.scharr_gradients_full(level_idx, level),
    };

    SegmentGradientLevel {
        width: level.w,
        height: level.h,
        origin_x: grad_view.origin_x,
        origin_y: grad_view.origin_y,
        tile_width: grad_view.tile_width,
        tile_height: grad_view.tile_height,
        gx: grad_view.gx,
        gy: grad_view.gy,
        level_index: level_idx,
    }
}

fn merged_segment_bounds(
    segments: &[Segment],
    scale_map: &LevelScaleMap,
    pad: f32,
    width: usize,
    height: usize,
) -> Option<SegmentRoi> {
    let mut union: Option<SegmentRoi> = None;
    for seg in segments {
        let roi = segment::segment_roi_from_points(
            scale_map.up(seg.p0),
            scale_map.up(seg.p1),
            pad,
            width,
            height,
        );
        let roi = match roi {
            Some(roi) => roi,
            None => continue,
        };
        union = Some(match union {
            Some(mut current) => {
                current.x0 = current.x0.min(roi.x0);
                current.y0 = current.y0.min(roi.y0);
                current.x1 = current.x1.max(roi.x1);
                current.y1 = current.y1.max(roi.y1);
                current
            }
            None => roi,
        });
    }
    union
}

fn select_cropped_bounds(bounds: IntBounds, width: usize, height: usize) -> Option<IntBounds> {
    let tile_area = bounds.width() * bounds.height();
    let full_area = width.saturating_mul(height);
    if full_area == 0 {
        return None;
    }
    let covered = tile_area as f32 / full_area as f32;
    if covered >= 0.8 {
        None
    } else {
        Some(bounds)
    }
}

fn refine_segments_for_level(
    grad_level: &SegmentGradientLevel<'_>,
    segments: &[Segment],
    scale_map: &LevelScaleMap,
    level_params: &segment::RefineParams,
) -> Vec<segment::RefineResult> {
    #[cfg(feature = "parallel")]
    {
        segments
            .par_iter()
            .map(|seg| segment::refine_segment(grad_level, seg, scale_map, level_params))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        segments
            .iter()
            .map(|seg| segment::refine_segment(grad_level, seg, scale_map, level_params))
            .collect()
    }
}

fn level_scale_map(coarse: &ImageF32, fine: &ImageF32) -> LevelScaleMap {
    let sx = if coarse.w > 0 {
        fine.w as f32 / coarse.w as f32
    } else {
        2.0
    };
    let sy = if coarse.h > 0 {
        fine.h as f32 / coarse.h as f32
    } else {
        2.0
    };
    LevelScaleMap::new(sx, sy)
}
