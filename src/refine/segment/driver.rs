use crate::detector::{DetectorWorkspace, LevelScaleMap};
use crate::image::ImageF32;
use crate::refine::segment::roi::{roi_to_int_bounds, IntBounds};
use crate::refine::segment::types::ScaleMap;
use crate::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, RefineParams, RefineResult,
};
use crate::segments::Segment;

/// Controls whether per-level refinement runs sequentially or with Rayon.
#[derive(Clone, Copy, Debug)]
pub struct ParallelRefineOptions {
    enabled: bool,
    min_segments_for_parallel: usize,
}

impl ParallelRefineOptions {
    /// Construct explicit options.
    pub fn new(enabled: bool, min_segments_for_parallel: usize) -> Self {
        Self {
            enabled,
            min_segments_for_parallel: min_segments_for_parallel.max(1),
        }
    }

    /// Disable parallel refinement regardless of segment count.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            min_segments_for_parallel: usize::MAX,
        }
    }

    /// Returns true when parallel refinement should be used for `segment_count`.
    pub fn should_parallelize(&self, segment_count: usize) -> bool {
        self.enabled && segment_count >= self.min_segments_for_parallel
    }

    /// Update the minimum segment threshold for parallel refinement.
    pub fn with_min_segments(mut self, min_segments: usize) -> Self {
        self.min_segments_for_parallel = min_segments.max(1);
        self
    }
}

impl Default for ParallelRefineOptions {
    fn default() -> Self {
        Self {
            enabled: cfg!(feature = "parallel"),
            min_segments_for_parallel: 96,
        }
    }
}

/// Refine a batch of coarse segments against a finer pyramid level.
///
/// The sequential path uses the detector workspace to reuse gradient buffers.
/// When parallelism is enabled (feature + runtime toggle), gradient tiles are
/// computed per segment and refinement is executed via Rayon.
pub fn refine_segments_between_levels(
    workspace: &mut DetectorWorkspace,
    finer_level: &ImageF32,
    finer_idx: usize,
    scale_map: &LevelScaleMap,
    params: &RefineParams,
    segments: Vec<Segment>,
    parallel: ParallelRefineOptions,
) -> Vec<RefineResult> {
    if segments.is_empty() {
        return Vec::new();
    }

    if parallel.should_parallelize(segments.len()) {
        #[cfg(feature = "parallel")]
        {
            return refine_segments_parallel(
                workspace,
                finer_level,
                finer_idx,
                scale_map,
                params,
                segments,
            );
        }
    }

    refine_segments_sequential(
        workspace,
        finer_level,
        finer_idx,
        scale_map,
        params,
        &segments,
    )
}

fn refine_segments_sequential(
    workspace: &mut DetectorWorkspace,
    finer_level: &ImageF32,
    finer_idx: usize,
    scale_map: &LevelScaleMap,
    params: &RefineParams,
    segments: &[Segment],
) -> Vec<RefineResult> {
    let mut refined = Vec::with_capacity(segments.len());
    for seg in segments {
        let grad_view = match scaled_bounds(seg, scale_map, params.pad, finer_level) {
            Some(bounds) => workspace.scharr_gradients_window(finer_idx, finer_level, &bounds),
            None => workspace.scharr_gradients_full(finer_idx, finer_level),
        };
        let grad_level = SegmentGradientLevel {
            width: finer_level.w,
            height: finer_level.h,
            origin_x: grad_view.origin_x,
            origin_y: grad_view.origin_y,
            tile_width: grad_view.tile_width,
            tile_height: grad_view.tile_height,
            gx: grad_view.gx,
            gy: grad_view.gy,
            level_index: finer_idx,
        };
        refined.push(segment::refine_segment(&grad_level, seg, scale_map, params));
    }
    refined
}

#[cfg(feature = "parallel")]
fn refine_segments_parallel(
    workspace: &mut DetectorWorkspace,
    finer_level: &ImageF32,
    finer_idx: usize,
    scale_map: &LevelScaleMap,
    params: &RefineParams,
    segments: Vec<Segment>,
) -> Vec<RefineResult> {
    use rayon::prelude::*;

    let grad_view = workspace.scharr_gradients_full(finer_idx, finer_level);
    let full_level = SegmentGradientLevel {
        width: finer_level.w,
        height: finer_level.h,
        origin_x: grad_view.origin_x,
        origin_y: grad_view.origin_y,
        tile_width: grad_view.tile_width,
        tile_height: grad_view.tile_height,
        gx: grad_view.gx,
        gy: grad_view.gy,
        level_index: finer_idx,
    };

    segments
        .into_par_iter()
        .map(|seg| segment::refine_segment(&full_level, &seg, scale_map, params))
        .collect()
}

fn scaled_bounds(
    seg: &Segment,
    scale_map: &LevelScaleMap,
    pad: f32,
    level: &ImageF32,
) -> Option<IntBounds> {
    let p0 = scale_map.up(seg.p0);
    let p1 = scale_map.up(seg.p1);
    let roi = segment::segment_roi_from_points(p0, p1, pad, level.w, level.h)?;
    roi_to_int_bounds(&roi, level.w, level.h)
}
