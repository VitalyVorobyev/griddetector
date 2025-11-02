use crate::segments::Segment;
use std::collections::HashMap;

const EPS: f32 = 1e-6;

/// Aggregated constraint built by merging near-collinear segments.
#[derive(Clone, Debug)]
pub struct Bundle {
    pub line: [f32; 3],
    pub center: [f32; 2],
    pub weight: f32,
}

impl Bundle {
    /// Returns a unit tangent vector corresponding to the bundle line.
    pub fn tangent(&self) -> [f32; 2] {
        [-self.line[1], self.line[0]]
    }
}

/// Group segments into bundled line constraints using orientation and endpoint distance thresholds.
///
/// The algorithm avoids an O(N^2) naive scan by:
/// - Binning segments by orientation (width ≈ `orientation_tol`).
/// - Within each orientation bin, hashing bundles by their projected offset along the bin normal
///   into buckets of size `dist_tol` and probing only neighboring buckets.
/// - Selecting the best candidate bundle by minimizing the max endpoint-to-line distance,
///   while also enforcing the orientation tolerance against the current bundle line.
pub fn bundle_segments(
    segs: &[Segment],
    orientation_tol: f32,
    dist_tol: f32,
    min_weight: f32,
) -> Vec<Bundle> {
    if segs.is_empty() {
        return Vec::new();
    }

    let orientation_tol = orientation_tol.max(1e-6);
    let dist_tol = dist_tol.max(EPS);
    let cos_tol = orientation_tol.cos();

    // Choose bin count so that bin width ≈ orientation_tol, clamped to [8, 90].
    let mut nbins = (std::f32::consts::PI / orientation_tol).ceil() as usize;
    nbins = nbins.clamp(8, 90);
    let bin_width = std::f32::consts::PI / nbins as f32;

    // Orientation bin helper.
    let orient_bin = |tx: f32, ty: f32| -> usize {
        // Angle of tangent vector [tx, ty] in [0, π).
        let th = crate::angle::normalize_half_pi(ty.atan2(tx));
        let mut idx = (th / bin_width).floor() as isize;
        if idx < 0 {
            idx += nbins as isize;
        }
        (idx as usize).min(nbins - 1)
    };

    // Precompute canonical tangents/normals per orientation bin.
    let mut bin_normals: Vec<[f32; 2]> = Vec::with_capacity(nbins);
    for i in 0..nbins {
        let angle = (i as f32 + 0.5) * bin_width;
        let t = [angle.cos(), angle.sin()];
        let n = [-t[1], t[0]];
        bin_normals.push(n);
    }

    // Per orientation bin, collect segment indices and precompute their projection along bin normal.
    let mut seg_indices_per_bin: Vec<Vec<usize>> = vec![Vec::new(); nbins];
    for (i, s) in segs.iter().enumerate() {
        if s.strength < min_weight {
            continue;
        }
        let idx = orient_bin(s.dir[0], s.dir[1]);
        seg_indices_per_bin[idx].push(i);
    }

    // Output bundles and indexing structures: for each orientation bin,
    // map offset bucket -> bundle indices.
    let mut bundles: Vec<Bundle> = Vec::new();
    let mut bundle_keys: Vec<(usize, i32)> = Vec::new(); // (orient_bin, offset_bin)
    let mut bin_maps: Vec<HashMap<i32, Vec<usize>>> = vec![HashMap::new(); nbins];

    for b in 0..nbins {
        if seg_indices_per_bin[b].is_empty() {
            continue;
        }
        // Canonical tangent/normal for this bin.
        let nbin = bin_normals[b];

        // Sort segments by projected offset along bin normal to visit nearby ones first.
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
            if weight < min_weight {
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
                            // Orientation gate: compare segment tangent with bundle tangent, dirless.
                            let bt = cb.tangent();
                            let dot = s.dir[0] * bt[0] + s.dir[1] * bt[1];
                            if dot.abs() < cos_tol {
                                continue;
                            }
                            // Endpoint-to-line distance cost (max of endpoints).
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
                // Merge into the best bundle and update its offset bucket if needed.
                let old_key = bundle_keys[bid];
                let line_arr = [s.line[0], s.line[1], s.line[2]];
                merge_bundle(&mut bundles[bid], &line_arr, s, weight);

                // Reindex bundle if its center moved across offset buckets.
                let new_center = bundles[bid].center;
                let target_bin = old_key.0;
                let nbin_target = bin_normals[target_bin];
                let new_proj = nbin_target[0] * new_center[0] + nbin_target[1] * new_center[1];
                let new_k = (new_proj / dist_tol).floor() as i32;
                if new_k != old_key.1 {
                    // Remove from old bucket.
                    if let Some(vec) = bin_maps[old_key.0].get_mut(&old_key.1) {
                        if let Some(pos) = vec.iter().position(|&x| x == bid) {
                            vec.swap_remove(pos);
                        }
                    }
                    // Insert into new bucket.
                    bin_maps[target_bin].entry(new_k).or_default().push(bid);
                    bundle_keys[bid] = (target_bin, new_k);
                }
            } else {
                // Start a new bundle seeded by this segment.
                let bundle_id = bundles.len();
                bundles.push(Bundle {
                    line: [s.line[0], s.line[1], s.line[2]],
                    center: seg_center,
                    weight,
                });
                bin_maps[b].entry(key).or_default().push(bundle_id);
                bundle_keys.push((b, key));
            }
        }
    }

    bundles
}

fn merge_bundle(target: &mut Bundle, line: &[f32; 3], seg: &Segment, weight: f32) {
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
