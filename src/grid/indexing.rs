use super::families::split_bundles;
use super::bundling::Bundle;

use crate::diagnostics::{FamilyIndexing, GridIndexingStage, GridLineIndex};
use crate::grid::FamilyLabel;

use nalgebra::Matrix3;
use std::collections::HashMap;
use std::time::Instant;

const EPS: f32 = 1e-6;

/// Build per-family discrete grid indices from bundles and a homography.
///
/// - `bundles` are expected in full-resolution coordinates.
/// - `hmtx` is the homography used for rectification.
/// - `orientation_tol_rad` controls U/V family split around the dominant axis.
pub fn index_grid_from_bundles(
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
    let buckets = split_bundles(h, bundles, orientation_tol_rad)?;

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

    // By default keep origin=(0,0). It can be improved by choosing the most central.
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
