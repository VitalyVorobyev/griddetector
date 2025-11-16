use super::roi::SegmentRoi;
use super::workspace::PyramidLevel;
use super::fit::{distance, project_point_to_line, weighted_line_fit};
use super::sampling::search_along_normal;
use super::options::RefineOptions;

use crate::segments::Segment;
use crate::angle::angle_between;

/// Snapshot produced by the outer carrier update loop.
#[derive(Clone, Debug)]
pub(crate) struct IterationSnapshot {
    pub seg: Segment,
    pub mu: [f32; 2],
    pub normal: [f32; 2],
    pub total_centers: usize,
    pub roi: SegmentRoi,
}

pub(crate) fn run_iterations(
    lvl: &PyramidLevel<'_>,
    seg0: &Segment,
    roi: &SegmentRoi,
    params: &RefineOptions,
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
