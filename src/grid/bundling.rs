//! Bundling of near-collinear segments into weighted line constraints.
//!
//! The bundler groups refined segments that are close in orientation and
//! location into a single `Bundle` with aggregated weight. This reduces the
//! number of constraints passed to vanishing-point estimation and grid
//! indexing while preserving spatial support.

use crate::angle::normalize_half_pi;
use crate::segments::Segment;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const EPS: f32 = 1e-6;

/// Identifier referencing a bundle in the current set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BundleId(pub u32);

/// Tunable parameters controlling bundling sensitivity.
#[derive(Clone, Debug)]
pub struct BundlingParams {
    /// Orientation tolerance (radians) between a segment and a bundle tangent.
    pub orientation_tol_rad: f32,
    /// Maximum endpoint-to-line distance (pixels) for merging.
    pub dist_tol_px: f32,
    /// Minimum segment strength to participate.
    pub min_strength: f32,
}

impl BundlingParams {
    pub fn tightened(&self, factor: f32) -> Self {
        let f = factor.max(0.25);
        Self {
            orientation_tol_rad: (self.orientation_tol_rad * f).max(1e-4),
            dist_tol_px: (self.dist_tol_px * f).max(0.1),
            min_strength: self.min_strength,
        }
    }
}

impl Default for BundlingParams {
    fn default() -> Self {
        Self {
            orientation_tol_rad: (22.5_f32).to_radians(),
            dist_tol_px: 1.5,
            min_strength: 0.0,
        }
    }
}

/// Aggregated, weighted line constraint derived from multiple segments.
#[derive(Clone, Debug, Serialize)]
pub struct Bundle {
    pub id: BundleId,
    /// Normalized line in ax + by + c = 0 form.
    pub line: [f32; 3],
    /// Center of mass of member segments.
    pub center: [f32; 2],
    /// Sum of member strengths.
    pub weight: f32,
    /// Indices of contributing segments (relative to the input slice).
    pub members: Vec<usize>,
}

impl Bundle {
    /// Unit tangent vector of the bundle.
    pub fn tangent(&self) -> [f32; 2] {
        [-self.line[1], self.line[0]]
    }

    /// Unit normal of the bundle (a, b).
    pub fn normal(&self) -> [f32; 2] {
        [self.line[0], self.line[1]]
    }

    /// Signed distance from the origin (rho = -c for normalized lines).
    pub fn rho(&self) -> f32 {
        -self.line[2]
    }
}

/// Group segments into bundles using orientation bins and offset hashing to
/// keep the search near O(N).
pub fn bundle_segments(segs: &[Segment], params: &BundlingParams) -> Vec<Bundle> {
    if segs.is_empty() {
        return Vec::new();
    }

    let orientation_tol = params.orientation_tol_rad.max(1e-5);
    let dist_tol = params.dist_tol_px.max(EPS);
    let cos_tol = orientation_tol.cos();

    // Bin count so that bin width ≈ orientation_tol, clamped.
    let mut nbins = (std::f32::consts::PI / orientation_tol).ceil() as usize;
    nbins = nbins.clamp(8, 90);
    let bin_width = std::f32::consts::PI / nbins as f32;

    let orient_bin = |tx: f32, ty: f32| -> usize {
        let th = normalize_half_pi(ty.atan2(tx));
        let mut idx = (th / bin_width).floor() as isize;
        if idx < 0 {
            idx += nbins as isize;
        }
        (idx as usize).min(nbins - 1)
    };

    // Canonical normals per orientation bin.
    let mut bin_normals: Vec<[f32; 2]> = Vec::with_capacity(nbins);
    for i in 0..nbins {
        let angle = (i as f32 + 0.5) * bin_width;
        let t = [angle.cos(), angle.sin()];
        bin_normals.push([-t[1], t[0]]);
    }

    // Per orientation bin, collect segment indices.
    let mut seg_indices_per_bin: Vec<Vec<usize>> = vec![Vec::new(); nbins];
    for (i, s) in segs.iter().enumerate() {
        if s.strength < params.min_strength {
            continue;
        }
        let dir = s.direction();
        let idx = orient_bin(dir[0], dir[1]);
        seg_indices_per_bin[idx].push(i);
    }

    // Output bundles and indexing structures.
    let mut bundles: Vec<Bundle> = Vec::new();
    let mut bundle_keys: Vec<(usize, i32)> = Vec::new(); // (orient_bin, offset_bin)
    let mut bin_maps: Vec<HashMap<i32, Vec<usize>>> = vec![HashMap::new(); nbins];

    for b in 0..nbins {
        if seg_indices_per_bin[b].is_empty() {
            continue;
        }
        let nbin = bin_normals[b];

        // Sort segments by projected offset along bin normal.
        let mut items: Vec<(usize, f32)> = seg_indices_per_bin[b]
            .iter()
            .map(|&idx| {
                (
                    idx,
                    nbin[0] * segment_center(&segs[idx])[0]
                        + nbin[1] * segment_center(&segs[idx])[1],
                )
            })
            .collect();
        items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (seg_idx, proj) in items.into_iter() {
            let s = &segs[seg_idx];
            let weight = s.strength;
            if weight < params.min_strength {
                continue;
            }
            let seg_center = segment_center(s);

            // Candidate search over neighboring offset buckets across adjacent orientation bins.
            let key = (proj / dist_tol).floor() as i32;
            let mut best_id: Option<usize> = None;
            let mut best_cost = f32::INFINITY;

            let mut bins_to_search: Vec<usize> = Vec::with_capacity(3);
            let th0 = (b as f32 + 0.5) * bin_width;
            for delta in [-1isize, 0, 1] {
                let angle = th0 + delta as f32 * bin_width;
                let idx = orient_bin(angle.cos(), angle.sin());
                if !bins_to_search.contains(&idx) {
                    bins_to_search.push(idx);
                }
            }

            for &bin_idx in &bins_to_search {
                let nbin_adj = bin_normals[bin_idx];
                let proj_adj = nbin_adj[0] * seg_center[0] + nbin_adj[1] * seg_center[1];
                let key_adj = (proj_adj / dist_tol).floor() as i32;
                for dk in -1..=1 {
                    if let Some(list) = bin_maps[bin_idx].get(&(key_adj + dk)) {
                        for &bid in list.iter() {
                            let cb = &bundles[bid];
                            let bt = cb.tangent();
                            let s_dir = s.direction();
                            let dot = s_dir[0] * bt[0] + s_dir[1] * bt[1];
                            if dot.abs() < cos_tol {
                                continue;
                            }
                            let cost = max_endpoint_distance(&s.p0, &s.p1, &cb.line);
                            if cost <= dist_tol && cost < best_cost {
                                best_cost = cost;
                                best_id = Some(bid);
                            }
                        }
                    }
                }
            }

            if let Some(bid) = best_id {
                let old_key = bundle_keys[bid];
                let line = s.line();
                let line_arr = [line[0], line[1], line[2]];
                merge_bundle(&mut bundles[bid], &line_arr, s, weight, seg_idx);

                // Reindex bundle if its center moved across offset buckets.
                let new_center = bundles[bid].center;
                let target_bin = old_key.0;
                let nbin_target = bin_normals[target_bin];
                let new_proj = nbin_target[0] * new_center[0] + nbin_target[1] * new_center[1];
                let new_k = (new_proj / dist_tol).floor() as i32;
                if new_k != old_key.1 {
                    if let Some(vec) = bin_maps[old_key.0].get_mut(&old_key.1) {
                        if let Some(pos) = vec.iter().position(|&x| x == bid) {
                            vec.swap_remove(pos);
                        }
                    }
                    bin_maps[target_bin].entry(new_k).or_default().push(bid);
                    bundle_keys[bid] = (target_bin, new_k);
                }
            } else {
                // Start a new bundle.
                let bundle_id = bundles.len();
                let line = s.line();
                bundles.push(Bundle {
                    id: BundleId(bundle_id as u32),
                    line: [line[0], line[1], line[2]],
                    center: seg_center,
                    weight,
                    members: vec![seg_idx],
                });
                bin_maps[b].entry(key).or_default().push(bundle_id);
                bundle_keys.push((b, key));
            }
        }
    }

    bundles
}

fn merge_bundle(
    target: &mut Bundle,
    line: &[f32; 3],
    seg: &Segment,
    weight: f32,
    seg_idx: usize,
) {
    let total = target.weight + weight;
    if total <= EPS {
        return;
    }
    // Align sign of the incoming line normal with the target bundle before averaging.
    let mut line_adj = *line;
    let dot = target.line[0] * line_adj[0] + target.line[1] * line_adj[1];
    if dot < 0.0 {
        line_adj[0] = -line_adj[0];
        line_adj[1] = -line_adj[1];
        line_adj[2] = -line_adj[2];
    }

    target.line[0] = (target.line[0] * target.weight + line_adj[0] * weight) / total;
    target.line[1] = (target.line[1] * target.weight + line_adj[1] * weight) / total;
    target.line[2] = (target.line[2] * target.weight + line_adj[2] * weight) / total;
    let norm = (target.line[0] * target.line[0] + target.line[1] * target.line[1])
        .sqrt()
        .max(EPS);
    target.line[0] /= norm;
    target.line[1] /= norm;
    target.line[2] /= norm;

    let center = segment_center(seg);
    target.center[0] = (target.center[0] * target.weight + center[0] * weight) / total;
    target.center[1] = (target.center[1] * target.weight + center[1] * weight) / total;
    target.weight = total;
    target.members.push(seg_idx);
}

fn segment_center(seg: &Segment) -> [f32; 2] {
    [0.5 * (seg.p0[0] + seg.p1[0]), 0.5 * (seg.p0[1] + seg.p1[1])]
}

#[inline]
fn max_endpoint_distance(p0: &[f32; 2], p1: &[f32; 2], line: &[f32; 3]) -> f32 {
    let d0 = (line[0] * p0[0] + line[1] * p0[1] + line[2]).abs();
    let d1 = (line[0] * p1[0] + line[1] * p1[1] + line[2]).abs();
    if d0 > d1 {
        d0
    } else {
        d1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::{Segment, SegmentId};

    fn make_test_segment(angle: f32, id: u32) -> Segment {
        let dir = [angle.cos(), angle.sin()];
        let center = [0.0f32, 0.0f32];
        let half_len = 5.0f32;
        let p0 = [center[0] - dir[0] * half_len, center[1] - dir[1] * half_len];
        let p1 = [center[0] + dir[0] * half_len, center[1] + dir[1] * half_len];
        Segment::new(SegmentId(id), p0, p1, 1.0, 1.0)
    }

    #[test]
    fn bundle_segments_merges_across_orientation_bins() {
        let params = BundlingParams {
            orientation_tol_rad: 0.2,
            dist_tol_px: 0.5,
            min_strength: 0.0,
        };

        let mut nbins =
            (std::f32::consts::PI / params.orientation_tol_rad).ceil() as usize;
        nbins = nbins.clamp(8, 90);
        let bin_width = std::f32::consts::PI / nbins as f32;

        let boundary = bin_width;
        let delta = 0.05 * bin_width;
        let angle_a = boundary - delta;
        let angle_b = boundary + delta;

        let seg_a = make_test_segment(angle_a, 0);
        let seg_b = make_test_segment(angle_b, 1);
        let segments = vec![seg_a, seg_b];

        let bundles = bundle_segments(&segments, &params);
        assert_eq!(bundles.len(), 1, "segments near bin boundary should merge");
        assert!(
            (bundles[0].weight - 2.0).abs() < 1e-3,
            "expected merged bundle weight to reflect both segments"
        );
        assert_eq!(bundles[0].members.len(), 2);
    }
}
