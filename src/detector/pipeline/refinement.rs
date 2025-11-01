use super::bundling::BundleStack;
use crate::detector::params::{BundlingParams, RefinementSchedule};
use crate::detector::scaling::{LevelScaleMap, LevelScaling};
use crate::detector::workspace::DetectorWorkspace;
use crate::diagnostics::builders::convert_refined_segment;
use crate::diagnostics::{
    BundleDescriptor, BundlingLevel, BundlingStage, FamilyIndexing, GridIndexingStage,
    GridLineIndex, RefinementOutcome, RefinementStage,
};
use crate::image::traits::ImageView;
use crate::lsd_vp::FamilyLabel;
use crate::pyramid::Pyramid;
use crate::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, RefineParams as SegmentRefineParams,
    Segment as SegmentSeed,
};
use crate::refine::{RefineLevel, Refiner};
use crate::segments::{Bundle, Segment};
use log::debug;
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::time::Instant;

const EPS: f32 = 1e-6;

#[derive(Debug)]
pub struct PreparedLevel {
    pub level_index: usize,
    pub level_width: usize,
    pub level_height: usize,
    pub segments: usize,
    pub bundles: Vec<Bundle>,
}

#[derive(Debug)]
pub struct PreparedLevels {
    pub levels: Vec<PreparedLevel>,
    pub bundling_ms: f64,
    pub segment_refine_ms: f64,
}

impl PreparedLevels {
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            bundling_ms: 0.0,
            segment_refine_ms: 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

#[derive(Debug)]
pub struct RefinementComputation {
    pub hmtx: Matrix3<f32>,
    pub confidence: f32,
    pub stage: Option<RefinementStage>,
    pub elapsed_ms: f64,
}

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

        let grad = workspace.sobel_gradients(finer_idx, finer_lvl);
        let gx = grad.gx.as_slice().unwrap_or(&grad.gx.data[..]);
        let gy = grad.gy.as_slice().unwrap_or(&grad.gy.data[..]);
        let grad_level = SegmentGradientLevel {
            width: finer_lvl.w,
            height: finer_lvl.h,
            gx,
            gy,
        };

        let refine_start = Instant::now();
        let mut refined_segments = Vec::with_capacity(current_segments.len());
        for seg in &current_segments {
            let seed = SegmentSeed {
                p0: seg.p0,
                p1: seg.p1,
            };
            let result = segment::refine_segment(&grad_level, seed, &scale_map, segment_params);
            let updated = convert_refined_segment(seg, result);
            refined_segments.push(updated);
        }
        segment_refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
        current_segments = refined_segments;
    }

    PreparedLevels {
        levels: levels_data,
        bundling_ms,
        segment_refine_ms,
    }
}

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

pub fn run_refinement_stage(
    refiner: &mut Refiner,
    prepared_levels: &PreparedLevels,
    initial_h: Option<Matrix3<f32>>,
    base_confidence: f32,
    schedule: &RefinementSchedule,
    enable_refine: bool,
    last_hypothesis: Option<Matrix3<f32>>,
) -> RefinementComputation {
    let mut confidence = base_confidence;
    let mut hmtx = initial_h.unwrap_or_else(Matrix3::identity);

    if !enable_refine
        || initial_h.is_none()
        || hmtx == Matrix3::identity()
        || prepared_levels.is_empty()
    {
        return RefinementComputation {
            hmtx,
            confidence,
            stage: None,
            elapsed_ms: 0.0,
        };
    }

    let refine_levels = convert_refine_levels(&prepared_levels.levels);
    let mut current_h = hmtx;
    let mut passes = 0usize;
    let mut refine_ms = 0.0f64;
    let mut last_outcome: Option<RefinementOutcome> = None;
    let mut attempted = false;

    while passes < schedule.passes {
        attempted = true;
        let refine_start = Instant::now();
        match refiner.refine(current_h, &refine_levels) {
            Some(refine_res) => {
                refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                passes += 1;
                let improvement = frobenius_improvement(&current_h, &refine_res.h_refined);
                let outcome = RefinementOutcome {
                    levels_used: refine_res.levels_used,
                    confidence: refine_res.confidence,
                    inlier_ratio: refine_res.inlier_ratio,
                    iterations: refine_res.level_reports.clone(),
                };
                last_outcome = Some(outcome);

                let base_conf = confidence;
                let combined =
                    combine_confidence(base_conf, refine_res.confidence, refine_res.inlier_ratio);
                confidence = combined.max(base_conf);
                current_h = refine_res.h_refined;
                hmtx = current_h;

                if passes >= schedule.passes || improvement < schedule.improvement_thresh {
                    break;
                }
            }
            None => {
                refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(prev) = last_hypothesis {
                    debug!("GridDetector::process refine failed -> fallback to last_hmtx");
                    hmtx = prev;
                    confidence *= 0.5;
                } else {
                    debug!("GridDetector::process refine failed -> keeping coarse hypothesis");
                }
                break;
            }
        }
    }

    let stage = if attempted {
        Some(RefinementStage {
            elapsed_ms: refine_ms,
            passes,
            outcome: last_outcome,
        })
    } else {
        None
    };

    RefinementComputation {
        hmtx,
        confidence,
        stage,
        elapsed_ms: refine_ms,
    }
}

pub fn run_grid_indexing_stage(
    bundles: &[Bundle],
    hmtx: Option<&Matrix3<f32>>,
    orientation_tol_rad: f32,
) -> Option<GridIndexingStage> {
    let h = hmtx?;
    if bundles.is_empty() {
        return None;
    }
    let h_inv = h.try_inverse()?;
    let h_inv_t = h_inv.transpose();
    let buckets = crate::refine::split_bundles(h, bundles, orientation_tol_rad)?;

    let mut index_map: HashMap<*const Bundle, usize> = HashMap::new();
    for (idx, bundle) in bundles.iter().enumerate() {
        index_map.insert(bundle as *const Bundle, idx);
    }

    let indexing_start = Instant::now();
    let (family_u, range_u) =
        build_family_indexing(FamilyLabel::U, &buckets.family_u, &h_inv_t, &index_map);
    let (family_v, range_v) =
        build_family_indexing(FamilyLabel::V, &buckets.family_v, &h_inv_t, &index_map);
    let elapsed_ms = indexing_start.elapsed().as_secs_f64() * 1000.0;

    let origin_uv = (0, 0);
    let visible_range = (
        range_u.map(|(min, _)| min).unwrap_or(0),
        range_u.map(|(_, max)| max).unwrap_or(0),
        range_v.map(|(min, _)| min).unwrap_or(0),
        range_v.map(|(_, max)| max).unwrap_or(0),
    );

    Some(GridIndexingStage {
        elapsed_ms,
        family_u,
        family_v,
        origin_uv,
        visible_range,
    })
}

fn build_family_indexing(
    label: FamilyLabel,
    family_bundles: &[&Bundle],
    h_inv_t: &Matrix3<f32>,
    index_map: &HashMap<*const Bundle, usize>,
) -> (FamilyIndexing, Option<(i32, i32)>) {
    struct LineEntry {
        index: usize,
        offset: f32,
        weight: f32,
    }

    let mut entries: Vec<LineEntry> = Vec::new();
    for bundle in family_bundles {
        let Some(&bundle_index) = index_map.get(&(*bundle as *const Bundle)) else {
            continue;
        };
        let line = nalgebra::Vector3::new(bundle.line[0], bundle.line[1], bundle.line[2]);
        let mut rect = h_inv_t * line;
        if !rect[0].is_finite() || !rect[1].is_finite() || !rect[2].is_finite() {
            continue;
        }
        let norm = (rect[0] * rect[0] + rect[1] * rect[1]).sqrt();
        if norm <= EPS {
            continue;
        }
        rect /= norm;

        let (primary, secondary) = (rect[0].abs(), rect[1].abs());
        let use_u_axis = primary >= secondary;
        match label {
            FamilyLabel::U if !use_u_axis => continue,
            FamilyLabel::V if use_u_axis => continue,
            _ => {}
        }

        let coeff = if label == FamilyLabel::U {
            rect[0]
        } else {
            rect[1]
        };
        if coeff.abs() <= EPS {
            continue;
        }
        let offset = -rect[2] / coeff;
        if !offset.is_finite() {
            continue;
        }

        entries.push(LineEntry {
            index: bundle_index,
            offset,
            weight: bundle.weight,
        });
    }

    if entries.is_empty() {
        return (
            FamilyIndexing {
                spacing: None,
                base_offset: None,
                confidence: 0.0,
                lines: Vec::new(),
            },
            None,
        );
    }

    let total = family_bundles.len().max(1);
    let mut offsets: Vec<f32> = entries.iter().map(|e| e.offset).collect();
    offsets.retain(|v| v.is_finite());

    let spacing = estimate_spacing(&offsets);
    let mut sorted_offsets = offsets.clone();
    sorted_offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let base_offset = sorted_offsets.get(sorted_offsets.len() / 2).copied();

    let mut line_indices = Vec::with_capacity(entries.len());
    let mut min_idx = i32::MAX;
    let mut max_idx = i32::MIN;
    for entry in entries {
        let idx = if let (Some(spacing), Some(base)) = (spacing, base_offset) {
            ((entry.offset - base) / spacing)
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32
        } else {
            0
        };
        let expected = if let (Some(spacing), Some(base)) = (spacing, base_offset) {
            base + idx as f32 * spacing
        } else {
            base_offset.unwrap_or(0.0)
        };
        let residual = (entry.offset - expected).abs();
        min_idx = min_idx.min(idx);
        max_idx = max_idx.max(idx);
        line_indices.push(GridLineIndex {
            bundle_index: entry.index,
            family: label,
            rectified_offset: entry.offset,
            grid_index: idx,
            weight: entry.weight,
            residual,
        });
    }

    let confidence = (line_indices.len() as f32 / total as f32).clamp(0.0, 1.0);

    (
        FamilyIndexing {
            spacing,
            base_offset,
            confidence,
            lines: line_indices,
        },
        if min_idx <= max_idx {
            Some((min_idx, max_idx))
        } else {
            None
        },
    )
}

fn estimate_spacing(values: &[f32]) -> Option<f32> {
    if values.len() < 2 {
        return None;
    }
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<f32>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut diffs = Vec::new();
    for pair in sorted.windows(2) {
        let diff = (pair[1] - pair[0]).abs();
        if diff > 1e-3 {
            diffs.push(diff);
        }
    }
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2];
    if median.is_finite() {
        Some(median)
    } else {
        None
    }
}

fn combine_confidence(base: f32, refine_conf: f32, inlier_ratio: f32) -> f32 {
    if inlier_ratio <= 1e-6 {
        return base.clamp(0.0, 1.0);
    }
    let blended = 0.5 * base + 0.5 * refine_conf;
    (blended * inlier_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn frobenius_improvement(a: &Matrix3<f32>, b: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let diff = b[(i, j)] - a[(i, j)];
            sum += diff * diff;
        }
    }
    sum.sqrt() / (frobenius_norm(a) + EPS)
}

fn frobenius_norm(m: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            sum += m[(i, j)] * m[(i, j)];
        }
    }
    sum.sqrt()
}

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

pub fn bundle_coarsest(
    bundler: &BundleStack<'_>,
    pyramid: &Pyramid,
    coarse_h: Option<&Matrix3<f32>>,
    segments: &[Segment],
    full_width: usize,
    full_height: usize,
) -> Option<(BundlingStage, Vec<Bundle>)> {
    if segments.is_empty() {
        return None;
    }
    let h = coarse_h?;
    let level_index = pyramid.levels.len().checked_sub(1)?;
    let level = &pyramid.levels[level_index];
    let scaling = LevelScaling::from_dimensions(level.w, level.h, full_width, full_height);

    let outcome = bundler.bundle_level(segments, &scaling, Some(h));
    let elapsed_ms = outcome.elapsed_ms;
    let bundles_full = outcome.bundles;
    let params = bundler.params();

    let level_descriptor = crate::diagnostics::BundlingLevel {
        level_index,
        width: level.w,
        height: level.h,
        bundles: bundles_full
            .iter()
            .map(|b| crate::diagnostics::BundleDescriptor {
                center: b.center,
                line: b.line,
                weight: b.weight,
            })
            .collect(),
    };

    let stage = BundlingStage {
        elapsed_ms,
        segment_refine_ms: 0.0,
        orientation_tol_deg: params.orientation_tol_deg,
        merge_distance_px: params.merge_dist_px,
        min_weight: params.min_weight,
        source_segments: segments.len(),
        scale_applied: outcome.applied_scale,
        levels: vec![level_descriptor],
    };

    Some((stage, bundles_full))
}
