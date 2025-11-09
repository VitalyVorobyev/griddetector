//! Line segment refinement driven by local gradient support.
//!
//! Starting from a seed segment detected on pyramid level `l+1`, the routine
//! lifts the endpoints to level `l`, searches for strong gradient responses
//! along the segment normal, re-estimates the carrier via a robust orthogonal
//! fit, and finally adjusts the segment extent to the longest contiguous run of
//! inliers. The resulting subpixel segment feeds into bundling ahead of
//! [`homography::Refiner`](super::homography::Refiner) to tighten the
//! coarse-to-fine optimisation loop.

mod endpoints;
mod fit;
mod sampling;
pub mod types;

use crate::angle::angle_between;

use crate::segments::Segment;
use endpoints::refine_endpoints;
use fit::{distance, project_point_to_line, weighted_line_fit};
use sampling::search_along_normal;
pub use types::{PyramidLevel, RefineParams, RefineResult, ScaleMap};

const EPS: f32 = 1e-6;

/// Axis-aligned region of interest around the current segment.
#[derive(Clone, Copy, Debug)]
struct Roi {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Roi {
    #[inline]
    fn contains(&self, p: &[f32; 2]) -> bool {
        p[0] >= self.x0 && p[0] <= self.x1 && p[1] >= self.y0 && p[1] <= self.y1
    }

    #[inline]
    fn clamp_inside(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0].clamp(self.x0, self.x1), p[1].clamp(self.y0, self.y1)]
    }
}

/// Snapshot produced by the outer carrier update loop.
#[derive(Clone, Debug)]
struct IterationSnapshot {
    seg: Segment,
    mu: [f32; 2],
    normal: [f32; 2],
    total_centers: usize,
    roi: Roi,
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
    params: &RefineParams,
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

    let Some(roi) = compute_roi(&seg, params.pad, lvl.width, lvl.height) else {
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

fn compute_roi(seg: &Segment, pad: f32, width: usize, height: usize) -> Option<Roi> {
    let (mut min_x, mut max_x) = (seg.p0[0].min(seg.p1[0]), seg.p0[0].max(seg.p1[0]));
    let (mut min_y, mut max_y) = (seg.p0[1].min(seg.p1[1]), seg.p0[1].max(seg.p1[1]));
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    let w = width as f32;
    let h = height as f32;
    min_x = min_x.clamp(0.0, w - 1.0);
    max_x = max_x.clamp(0.0, w - 1.0);
    min_y = min_y.clamp(0.0, h - 1.0);
    max_y = max_y.clamp(0.0, h - 1.0);
    if min_x >= max_x || min_y >= max_y {
        None
    } else {
        Some(Roi {
            x0: min_x,
            y0: min_y,
            x1: max_x,
            y1: max_y,
        })
    }
}

fn run_iterations(
    lvl: &PyramidLevel<'_>,
    seg0: &Segment,
    roi: &Roi,
    params: &RefineParams,
    w_perp: f32,
) -> Option<IterationSnapshot> {
    let mut p0 = seg0.p0;
    let mut p1 = seg0.p1;
    let mut last_mu = seg0.midpoint();
    let mut last_normal = seg0.normal();
    let mut total_centers = 0usize;

    for _ in 0..params.max_iters.max(1) {
        let dir = seg0.direction();
        let normal = seg0.normal();
        let length = distance(&p0, &p1);
        let samples = (length / params.delta_s).floor() as usize;
        let mut n_centers = samples.max(4) + 1;
        if length < 3.0 {
            n_centers = 1;
        }
        total_centers = n_centers;
        let mut supports = Vec::with_capacity(n_centers);
        for i in 0..n_centers {
            let t = if n_centers <= 1 {
                0.0
            } else {
                (i as f32) / ((n_centers - 1) as f32)
            };
            let center = [p0[0] + dir[0] * t * length, p0[1] + dir[1] * t * length];
            if let Some(support) = search_along_normal(lvl, roi, &center, &normal, params, w_perp) {
                supports.push(support);
            }
        }
        let supports_needed = usize::min(3, n_centers);
        if supports.len() < supports_needed {
            return None;
        }

        let (d_final, n_final, rho, mu) = weighted_line_fit(&supports, params)?;
        last_mu = mu;
        last_normal = n_final;

        let new_p0 = project_point_to_line(&p0, &n_final, rho);
        let new_p1 = project_point_to_line(&p1, &n_final, rho);
        let moved = distance(&new_p0, &p0).max(distance(&new_p1, &p1));
        let angle_diff = angle_between(&dir, &d_final).to_degrees();
        p0 = new_p0;
        p1 = new_p1;

        if moved < 0.05 && angle_diff < 0.25 {
            break;
        }
    }

    Some(IterationSnapshot {
        seg: Segment::new(seg0.id, p0, p1, seg0.avg_mag, seg0.strength),
        mu: last_mu,
        normal: last_normal,
        total_centers,
        roi: *roi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::SegmentId;

    #[test]
    fn roi_rejects_degenerate_segments() {
        let seg = Segment::new(SegmentId(1), [10.0, 10.0], [10.1, 10.1], 1.0, 1.0);
        assert!(compute_roi(&seg, 0.0, 32, 32).is_some());
        assert!(compute_roi(&seg, 0.0, 1, 1).is_none());
    }
}
