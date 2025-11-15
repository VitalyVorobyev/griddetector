use crate::detector::params::BundlingParams;
use crate::detector::pipeline::bundling::BundleStack;
use crate::detector::scaling::{LevelScaleMap, LevelScaling};
use crate::detector::workspace::DetectorWorkspace;
use crate::diagnostics::{BundleDescriptor, BundlingLevel, BundlingStage};
use crate::pyramid::Pyramid;
#[cfg(feature = "profile_refine")]
use crate::refine::segment::take_profile;
use crate::refine::segment::{
    self, roi::roi_to_int_bounds, PyramidLevel as SegmentGradientLevel,
    RefineParams as SegmentRefineParams, ScaleMap,
};
use crate::refine::RefineLevel;
use crate::segments::{Bundle, Segment};
use log::debug;
#[cfg(feature = "profile_refine")]
use log::info;
use nalgebra::Matrix3;
use std::time::Instant;

/// Bundled and refined data prepared at each pyramid level.
#[derive(Debug)]
pub struct PreparedLevel {
    /// Pyramid level index (0 is the finest/full-res).
    pub level_index: usize,
    /// Level width in pixels.
    pub level_width: usize,
    /// Level height in pixels.
    pub level_height: usize,
    /// Number of input segments considered at this level.
    pub segments: usize,
    /// Bundles aggregated at this level (rescaled to full resolution).
    pub bundles: Vec<Bundle>,
}

/// Prepared data across all levels with timing.
#[derive(Debug)]
pub struct PreparedLevels {
    /// Coarse-to-fine stack of levels, ordered from coarse→fine in `levels`.
    pub levels: Vec<PreparedLevel>,
    /// Time spent in bundling during preparation (ms).
    pub bundling_ms: f64,
    /// Time spent refining segments across levels (ms).
    pub segment_refine_ms: f64,
}

impl PreparedLevels {
    /// Construct an empty container.
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            bundling_ms: 0.0,
            segment_refine_ms: 0.0,
        }
    }

    /// True if no levels were prepared.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Convert into the shape expected by the homography refiner.
    pub fn as_refine_levels(&self) -> Vec<RefineLevel<'_>> {
        convert_refine_levels(&self.levels)
    }
}

/// Prepare per-level bundles and perform gradient-driven segment refinement.
///
/// For each pyramid level L from coarse→fine:
/// - Bundle the current set of segments (rescaled to full resolution)
/// - If there is a finer level (L-1), refine segments using the finer gradients,
///   lifting geometry forward for the next iteration
///
/// Inputs assume `coarse_h` and all geometric data are in full-resolution coordinates.
#[allow(clippy::too_many_arguments)]
pub fn prepare_levels(
    workspace: &mut DetectorWorkspace,
    bundler: &BundleStack<'_>,
    segment_params: &SegmentRefineParams,
    pyramid: &Pyramid,
    coarse_h: Option<&Matrix3<f32>>,
    initial_segments: Vec<Segment>,
    full_width: usize,
    full_height: usize,
) -> PreparedLevels {
    if pyramid.levels.is_empty() {
        return PreparedLevels::empty();
    }

    let mut levels_data = Vec::new();
    let mut current_segments = initial_segments;
    let mut bundling_ms = 0.0f64;
    let mut segment_refine_ms = 0.0f64;
    let coarse_idx = pyramid.levels.len() - 1;

    for level_idx in (0..=coarse_idx).rev() {
        let lvl = &pyramid.levels[level_idx];
        let scaling = LevelScaling::from_dimensions(lvl.w, lvl.h, full_width, full_height);

        let outcome = bundler.bundle_level(&current_segments, &scaling, coarse_h);
        bundling_ms += outcome.elapsed_ms;
        let bundles_full = outcome.bundles;

        debug!(
            "GridDetector::level L{}: segments={} bundles={} (frame={:?})",
            level_idx,
            current_segments.len(),
            bundles_full.len(),
            outcome.frame
        );

        levels_data.push(PreparedLevel {
            level_index: level_idx,
            level_width: lvl.w,
            level_height: lvl.h,
            segments: current_segments.len(),
            bundles: bundles_full,
        });

        if current_segments.is_empty() {
            break;
        }
        if level_idx == 0 {
            continue;
        }

        // Lift segments forward to the next finer level (L-1).
        let finer_idx = level_idx - 1;
        let finer_lvl = &pyramid.levels[finer_idx];
        let sx = if lvl.w > 0 {
            finer_lvl.w as f32 / lvl.w as f32
        } else {
            2.0
        };
        let sy = if lvl.h > 0 {
            finer_lvl.h as f32 / lvl.h as f32
        } else {
            2.0
        };
        let scale_map = LevelScaleMap::new(sx, sy);

        let refine_start = Instant::now();
        let mut refined_segments = Vec::with_capacity(current_segments.len());
        for seg in &current_segments {
            let grad_view = match segment::segment_roi_from_points(
                scale_map.up(seg.p0),
                scale_map.up(seg.p1),
                segment_params.pad,
                finer_lvl.w,
                finer_lvl.h,
            )
            .and_then(|roi| roi_to_int_bounds(&roi, finer_lvl.w, finer_lvl.h))
            {
                Some(bounds) => workspace.scharr_gradients_window(finer_idx, finer_lvl, &bounds),
                None => workspace.scharr_gradients_full(finer_idx, finer_lvl),
            };
            let grad_level = SegmentGradientLevel {
                width: finer_lvl.w,
                height: finer_lvl.h,
                origin_x: grad_view.origin_x,
                origin_y: grad_view.origin_y,
                tile_width: grad_view.tile_width,
                tile_height: grad_view.tile_height,
                gx: grad_view.gx,
                gy: grad_view.gy,
                level_index: finer_idx,
            };
            let result = segment::refine_segment(&grad_level, seg, &scale_map, segment_params);
            if result.ok {
                refined_segments.push(result.seg);
            }
        }
        segment_refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
        current_segments = refined_segments;
        #[cfg(feature = "profile_refine")]
        {
            let stage_label = format!("prepare_levels L{}", finer_idx);
            log_segment_profile(&stage_label, workspace);
        }
    }

    PreparedLevels {
        levels: levels_data,
        bundling_ms,
        segment_refine_ms,
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

/// Convert prepared levels (coarse→fine) into refine levels (fine→coarse) for the IRLS.
fn convert_refine_levels(levels: &[PreparedLevel]) -> Vec<RefineLevel<'_>> {
    let mut out = Vec::with_capacity(levels.len());
    for data in levels.iter().rev() {
        out.push(RefineLevel {
            level_index: data.level_index,
            width: data.level_width,
            height: data.level_height,
            segments: data.segments,
            bundles: data.bundles.as_slice(),
        });
    }
    out
}

/// Build a BundlingStage diagnostic snapshot from prepared levels.
///
/// This is a formatting helper for reporting; it does not affect the algorithm.
pub fn build_bundling_stage(
    params: &BundlingParams,
    prepared: &PreparedLevels,
    source_segments: usize,
) -> Option<BundlingStage> {
    if prepared.levels.is_empty() {
        return None;
    }
    let levels = prepared
        .levels
        .iter()
        .map(|lvl| BundlingLevel {
            level_index: lvl.level_index,
            width: lvl.level_width,
            height: lvl.level_height,
            bundles: lvl
                .bundles
                .iter()
                .map(|b| BundleDescriptor {
                    center: b.center,
                    line: b.line,
                    weight: b.weight,
                })
                .collect(),
        })
        .collect();
    Some(BundlingStage {
        elapsed_ms: prepared.bundling_ms,
        segment_refine_ms: prepared.segment_refine_ms,
        orientation_tol_deg: params.orientation_tol_deg,
        merge_distance_px: params.merge_dist_px,
        min_weight: params.min_weight,
        source_segments,
        scale_applied: None,
        levels,
    })
}
