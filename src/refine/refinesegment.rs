//! Line segment refinement driven by local gradient support.
//!
//! Starting from a seed segment detected on pyramid level `l+1`, the routine
//! lifts the endpoints to level `l`, searches for strong gradient responses
//! along the segment normal, re-estimates the carrier via a robust orthogonal
//! fit, and finally adjusts the segment extent to the longest contiguous run of
//! inliers. The resulting subpixel segment feeds into bundling ahead of
//! [`homography::Refiner`](super::homography::Refiner) to tighten the
//! coarse-to-fine optimisation loop.
//!
//! # Gradient handling for refinement
//!
//! Refinement operates entirely on image gradients from a single pyramid level.
//! The detector centralises gradient computation in its workspace
//! (`detector::workspace`), which exposes Scharr
//! derivatives as **gradient tiles**:
//!
//! - A full-frame tile (`origin_x = origin_y = 0`, `tile_width = width`,
//!   `tile_height = height`) is used when refining all segments on a level or
//!   when local cropping would not be cheaper.
//! - Cropped tiles are built for axis-aligned windows that tightly cover the
//!   union of per-segment ROIs. This saves work on large images when only a
//!   small fraction of pixels participate in refinement.
//!
//! The tile is wrapped in a [`types::PyramidLevel`] and passed to
//! [`refine_segment`]. `PyramidLevel` carries:
//!
//! - `width`, `height`: full image dimensions at this pyramid level.
//! - `origin_x`, `origin_y`: top-left corner of the gradient tile in the
//!   full image.
//! - `tile_width`, `tile_height`: dimensions of the tile.
//! - `gx`, `gy`: contiguous slices storing gradients for the tile.
//! - `level_index`: the pyramid level index (0 = finest).
//!
//! All sampling routines work in **full-image coordinates**. The helper
//! `sampling::bilinear_grad` converts a subpixel `(x, y)` into tile-relative
//! coordinates by subtracting `(origin_x, origin_y)`, checks bounds against
//! `(tile_width, tile_height)`, and performs bilinear interpolation inside the
//! tile. This keeps the refinement logic independent of how large the tile is
//! or whether it covers the full frame.
//!
//! Typical flow per level:
//! - The caller builds a [`SegmentRoi`] for each segment in full-image
//!   coordinates (padding is controlled by [`RefineOptions::pad`]).
//! - ROIs are optionally merged into a coarse window, or used individually, to
//!   request a cropped gradient tile from `DetectorWorkspace`.
//! - `refine_segment` consumes the resulting `PyramidLevel`, sampling
//!   gradients via `sampling::search_along_normal` and `endpoints::refine_endpoints`
//!   without concern for how the tile was produced.

use super::workspace::{RefinementWorkspace, PyramidLevel};
use super::options::RefineOptions;
use super::iteration::run_iterations;

use crate::image::ImageF32;
use crate::edges::Grad;
use crate::segments::Segment;
use crate::pyramid::{ScaleMap, Pyramid, LevelScaleMap};

use super::endpoints::refine_endpoints;
#[cfg(feature = "profile_refine")]
use super::profile::{take_profile, LevelProfile};
use super::roi::{SegmentRoi, segment_roi_from_points, roi_to_int_bounds};

use serde::Serialize;
use std::time::Instant;

const EPS: f32 = 1e-6;

#[derive(Default, Debug, Serialize)]
pub struct SegmentsRefinementLevelResult {
    pub elapsed_ms: f64,
    pub accepted: usize,
}

#[derive(Default, Debug, Serialize)]
pub struct SegmentsRefinementResult {
    pub levels: Vec<SegmentsRefinementLevelResult>,
    pub elapsed_ms: f64,
    #[cfg(feature = "profile_refine")]
    pub profile: LevelProfile
}

pub fn refine_coarse_segments(
    pyramid: &Pyramid,
    source_segments: &[Segment],
    refine_params: &RefineOptions,
    coarse_gradient: Option<Grad>,
) -> Result<SegmentsRefinementResult, String> {
    let mut current_segments: Vec<Segment> = source_segments.to_vec();
    let full_width = pyramid.levels.first().map(|lvl| lvl.w).unwrap_or(0);

    let mut workspace = match(coarse_gradient) {
        Some(grad) => RefinementWorkspace::from_coarsest_gradient(grad, pyramid.levels.len()),
        None => RefinementWorkspace::new(pyramid.levels.len())
    };

    let mut result = SegmentsRefinementResult::default();
    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let coarse_level = &pyramid.levels[coarse_idx];
        let scale_map = level_scale_map(coarse_level, finer_level);
        let level_params = refine_params.for_level(full_width, finer_level.w);
        let mut refined_segments = Vec::with_capacity(current_segments.len());
        let mut accepted = 0usize;
        let mut score_sum = 0.0f32;

        for seg in &current_segments {
            let grad_view = match segment_roi_from_points(
                scale_map.up(seg.p0),
                scale_map.up(seg.p1),
                level_params.pad,
                finer_level.w,
                finer_level.h,
            )
            .and_then(|roi| roi_to_int_bounds(&roi, finer_level.w, finer_level.h))
            {
                Some(bounds) => workspace.scharr_gradients_window(finer_idx, finer_level, &bounds),
                None => workspace.scharr_gradients_full(finer_idx, finer_level),
            };
            let grad_level = PyramidLevel {
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
            let result = refine_segment(&grad_level, seg, &scale_map, &level_params);
            let ok = result.ok;
            if ok {
                accepted += 1;
                if result.score.is_finite() {
                    score_sum += result.score;
                }
            }
            let updated = result.seg;

            refined_segments.push(updated);
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Refinement for level {} took {:.2} ms",
            finer_idx, elapsed_ms
        );
        let acceptance_ratio = (!current_segments.is_empty())
            .then_some(accepted as f32 / current_segments.len() as f32);
        let avg_score = (accepted > 0).then_some(score_sum / accepted as f32);

        current_segments = refined_segments;

        #[cfg(feature = "profile_refine")]
        dump_refine_profile(workspace);

        result.levels.push(SegmentsRefinementLevelResult {
            elapsed_ms,
            accepted: current_segments.len()
        });
    }

    Ok(result)
}

/// Outcome of a refinement attempt.
#[derive(Clone, Debug)]
pub struct RefineResult {
    /// Refined segment (or the fallback seed when `ok == false`).
    pub seg: Segment,
    /// Mean absolute normal-projected gradient across inlier samples.
    pub score: f32,
    /// Whether the refinement satisfied the length and score thresholds.
    pub ok: bool,
    /// Number of inlier samples supporting the endpoints.
    pub inliers: usize,
    /// Total number of centreline samples considered.
    pub total: usize,
}

impl RefineResult {
    pub(crate) fn failed(seg: Segment) -> Self {
        Self {
            seg,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        }
    }

    pub(crate) fn rejected(seg: Segment, inliers: usize, total: usize, score: f32) -> Self {
        Self {
            seg,
            score,
            ok: false,
            inliers,
            total,
        }
    }
}

/// Refine a coarse-level segment using gradient support on the finer level.
///
/// The routine upsamples the coarse endpoints, iteratively fits a robust line
/// to the collected support samples, and finally resizes the segment by looking
/// for the strongest contiguous inlier run along the carrier.
pub fn refine_segment(
    lvl: &PyramidLevel<'_>,
    seg_coarse: &Segment,
    scale: &dyn ScaleMap,
    params: &RefineOptions,
) -> RefineResult {
    if lvl.width == 0 || lvl.height == 0 {
        return RefineResult::failed(seg_coarse.clone());
    }

    let seg = Segment::new(
        seg_coarse.id,
        scale.up(seg_coarse.p0),
        scale.up(seg_coarse.p1),
        seg_coarse.avg_mag,
        seg_coarse.strength,
    );
    let fallback = seg.clone();

    if seg.length_sq() < 1e-3 {
        return RefineResult::failed(fallback);
    }

    let Some(roi) = compute_roi(&seg, params.pad, lvl.width, lvl.height, lvl.level_index) else {
        return RefineResult::failed(fallback);
    };

    let mut snapshot = None;
    for attempt in 0..2 {
        let w_perp = if attempt == 0 {
            params.w_perp
        } else {
            (params.w_perp * 0.5).max(1.0)
        };
        if let Some(iter) = run_iterations(lvl, &seg, &roi, params, w_perp) {
            snapshot = Some(iter);
            break;
        }
    }

    let Some(snapshot) = snapshot else {
        return RefineResult::failed(fallback);
    };

    let (p0_f, p1_f, support_count, score) = refine_endpoints(&snapshot, lvl, params);
    let refined_segment = Segment::new(
        snapshot.seg.id,
        p0_f,
        p1_f,
        snapshot.seg.avg_mag,
        snapshot.seg.strength,
    );
    let total_centers = snapshot.total_centers;
    let seed_len = fallback.length().max(EPS);
    let refined_len = refined_segment.length();
    let ok = refined_len >= params.min_inlier_frac * seed_len && score >= params.tau_mag;

    if !ok {
        return RefineResult::rejected(fallback, support_count, total_centers, score);
    }

    RefineResult {
        seg: refined_segment,
        score,
        ok,
        inliers: support_count,
        total: total_centers,
    }
}

#[cfg_attr(not(feature = "profile_refine"), allow(unused_variables))]
pub(crate) fn compute_roi(
    seg: &Segment,
    pad: f32,
    width: usize,
    height: usize,
    level_index: usize,
) -> Option<SegmentRoi> {
    let roi = segment_roi_from_points(seg.p0, seg.p1, pad, width, height)?;
    #[cfg(feature = "profile_refine")]
    profile::record_roi(level_index, &roi);
    Some(roi)
}

fn level_scale_map(coarse: &ImageF32, fine: &ImageF32) -> LevelScaleMap {
    let sx = if coarse.w == 0 {
        2.0
    } else {
        fine.w as f32 / coarse.w as f32
    };
    let sy = if coarse.h == 0 {
        2.0
    } else {
        fine.h as f32 / coarse.h as f32
    };
    LevelScaleMap::new(sx, sy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::SegmentId;

    #[test]
    fn roi_rejects_degenerate_segments() {
        let seg = Segment::new(SegmentId(1), [10.0, 10.0], [10.1, 10.1], 1.0, 1.0);
        assert!(compute_roi(&seg, 0.0, 32, 32, 0).is_some());
        assert!(compute_roi(&seg, 0.0, 1, 1, 0).is_none());
    }
}

#[cfg(feature = "profile_refine")]
fn dump_refine_profile(workspace: &DetectorWorkspace) {
    let profile = take_profile();
    if profile.is_empty() {
        return;
    }
    println!("Segment refine profile:");
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
        println!(
            "  L{}: grad_ms={:.2} roi_count={} avg_roi_px={:.1} bilinear_samples={}",
            entry.level_index, grad_ms, entry.roi_count, avg_roi, entry.bilinear_samples
        );
    }
}
