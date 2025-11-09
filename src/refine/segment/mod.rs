//! Fast coarse-to-fine segment refinement using gradient-driven region growing.
//!
//! Each segment predicted from a coarser pyramid level seeds a local search on
//! the finer level. The routine identifies a strong anchor along the segment
//! normal, then grows forward/backward along the tangent with tiny normal
//! refinements reminiscent of LSD region growing. This replaces the previous
//! dense sampling approach with a lightweight procedure that reuses cached
//! gradient fields and requires only a handful of gradient evaluations.

mod types;

use crate::angle::{angular_difference, normalize_half_pi};
use crate::edges::grad::{GradientLevel, GradientSample};
use crate::segments::{RegionAccumulator, Segment};
use nalgebra::{Matrix2, SymmetricEigen};

pub use types::{RefineDiagnostics, RefineResult, ScaleMap, SegmentRefineParams};

const EPS: f32 = 1e-6;
const NEIGH_OFFSETS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

#[derive(Clone, Debug)]
struct SamplePoint {
    pos: [f32; 2],
    normal: [f32; 2],
    mag: f32,
    offset: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct RefinedSegment {
    seg: Segment,
    score: f32,
    support: usize,
    diagnostics: RefineDiagnostics,
}

/// Refine a segment predicted at a coarser pyramid level on the current level.
pub fn refine_segment(
    grad: &GradientLevel<'_>,
    seg_coarse: &Segment,
    scale: &dyn ScaleMap,
    params: &SegmentRefineParams,
) -> RefineResult {
    if grad.width == 0 || grad.height == 0 {
        return RefineResult::failed(seg_coarse.clone());
    }

    let seg_pred = Segment::new(
        seg_coarse.id,
        scale.up(seg_coarse.p0),
        scale.up(seg_coarse.p1),
        seg_coarse.avg_mag,
        seg_coarse.strength,
    );

    let fallback = seg_pred.clone();

    if seg_pred.length_sq() < 1e-3 {
        return RefineResult::failed(fallback);
    }

    let Some(result) = refine_segment_coarse_to_fine(grad, &seg_pred, params) else {
        return RefineResult::failed(fallback);
    };

    let accept = result.support >= params.region_min_pixels
        && result.score >= params.gradient_threshold
        && result.seg.length() >= params.min_length_px;

    if !accept {
        return RefineResult {
            seg: fallback,
            ok: false,
            score: result.score,
            support_points: result.support,
            diagnostics: result.diagnostics,
        };
    }

    RefineResult {
        seg: result.seg,
        ok: true,
        score: result.score,
        support_points: result.support,
        diagnostics: result.diagnostics,
    }
}

/// Refine a predicted segment within a single pyramid level using local gradients.
pub(crate) fn refine_segment_coarse_to_fine(
    grad: &GradientLevel<'_>,
    pred_segment: &Segment,
    params: &SegmentRefineParams,
) -> Option<RefinedSegment> {
    let mut diagnostics = RefineDiagnostics::default();
    let mut best_anchor: Option<SamplePoint> = None;
    let mut best_mag = params.gradient_threshold;

    let seed_count = params.max_seeds.max(1).min(8);
    let seeds = seed_offsets(seed_count);
    for &t in &seeds {
        diagnostics.anchor_trials += 1;
        let center = lerp_points(pred_segment.p0, pred_segment.p1, t);
        let normal_guess = initial_normal(grad, center, pred_segment, params, &mut diagnostics);
        let Some(normal_guess) = normal_guess else {
            continue;
        };
        let Some(anchor) = refine_along_normal(
            grad,
            center,
            normal_guess,
            params.anchor_search_radius,
            params,
            &mut diagnostics,
        ) else {
            continue;
        };
        if anchor.mag > best_mag {
            best_mag = anchor.mag;
            best_anchor = Some(anchor);
        }
    }

    let Some(anchor) = best_anchor else {
        return None;
    };

    let mut anchor_point = anchor.clone();
    anchor_point.offset = 0.0;

    let region =
        grow_region_from_anchor(grad, pred_segment, &anchor_point, params, &mut diagnostics)?;
    let support = region.len();
    diagnostics.region_pixels = support;

    if support < params.region_min_pixels {
        return None;
    }
    if region.aligned_fraction() < params.region_min_alignment {
        return None;
    }

    let (segment, avg_mag) = fit_segment_from_region(pred_segment, &region, grad.width, params)?;

    Some(RefinedSegment {
        seg: segment,
        score: avg_mag,
        support,
        diagnostics,
    })
}

fn seed_offsets(count: usize) -> Vec<f32> {
    if count <= 1 {
        return vec![0.5];
    }
    let denom = (count + 1) as f32;
    (0..count).map(|i| (i as f32 + 1.0) / denom).collect()
}

fn initial_normal(
    grad: &GradientLevel<'_>,
    pos: [f32; 2],
    pred_segment: &Segment,
    params: &SegmentRefineParams,
    diagnostics: &mut RefineDiagnostics,
) -> Option<[f32; 2]> {
    if let Some(mut n) = normalize(pred_segment.normal()) {
        if let Some(sample) = sample_gradient(grad, pos, diagnostics) {
            if sample.mag >= 0.5 * params.gradient_threshold {
                let grad_vec = [sample.gx, sample.gy];
                if let Some(ng) = normalize(grad_vec) {
                    n = ng;
                }
            }
        }
        Some(n)
    } else {
        None
    }
}

fn grow_region_from_anchor(
    grad: &GradientLevel<'_>,
    pred_segment: &Segment,
    anchor: &SamplePoint,
    params: &SegmentRefineParams,
    diagnostics: &mut RefineDiagnostics,
) -> Option<RegionAccumulator> {
    if grad.width == 0 || grad.height == 0 {
        return None;
    }
    let pad = params
        .region_padding_px
        .max(params.anchor_search_radius)
        .max(params.anchor_step);
    let bounds = compute_region_bounds(grad, pred_segment, anchor.pos, pad)?;
    let area = bounds.area();
    if area == 0 {
        return None;
    }

    let mut used = vec![0u8; area];
    let mut stack = Vec::with_capacity(params.region_max_pixels.min(512).max(32));
    let capacity = params.region_max_pixels.max(8);
    let mut region = RegionAccumulator::with_capacity(capacity);

    let anchor_x = clamp_coord(anchor.pos[0], grad.width);
    let anchor_y = clamp_coord(anchor.pos[1], grad.height);
    if !bounds.contains(anchor_x, anchor_y) {
        return None;
    }
    let seed_angle = normalize_half_pi(anchor.normal[1].atan2(anchor.normal[0]));
    let idx = pixel_index(grad.width, anchor_x, anchor_y);
    stack.push(idx);
    used[bounds.offset(anchor_x, anchor_y)] = 1;

    let angle_tol = params.region_angle_tolerance_deg.to_radians();
    while let Some(pixel_idx) = stack.pop() {
        let x = pixel_idx % grad.width;
        let y = pixel_idx / grad.width;
        let sample = pixel_sample(grad, x, y, diagnostics);
        if sample.mag < params.gradient_threshold {
            continue;
        }
        let angle = normalize_half_pi(sample.gy.atan2(sample.gx));
        let aligned = angular_difference(angle, seed_angle) <= angle_tol;
        region.push(pixel_idx, x, y, sample.mag, aligned);
        if region.len() >= params.region_max_pixels {
            break;
        }

        for (dx, dy) in NEIGH_OFFSETS {
            let xn = match x as isize + dx {
                v if v < 0 || v >= grad.width as isize => continue,
                v => v as usize,
            };
            let yn = match y as isize + dy {
                v if v < 0 || v >= grad.height as isize => continue,
                v => v as usize,
            };
            if !bounds.contains(xn, yn) {
                continue;
            }
            let local_idx = bounds.offset(xn, yn);
            if used[local_idx] != 0 {
                continue;
            }
            let neighbor = pixel_sample(grad, xn, yn, diagnostics);
            if neighbor.mag < params.gradient_threshold {
                continue;
            }
            let neighbor_angle = normalize_half_pi(neighbor.gy.atan2(neighbor.gx));
            if angular_difference(neighbor_angle, seed_angle) > angle_tol {
                continue;
            }
            used[local_idx] = 1;
            stack.push(pixel_index(grad.width, xn, yn));
        }
    }

    if region.len() < 2 {
        None
    } else {
        Some(region)
    }
}

#[derive(Clone, Copy, Debug)]
struct RegionBounds {
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
}

impl RegionBounds {
    fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x0 && x < self.x0 + self.width && y >= self.y0 && y < self.y0 + self.height
    }

    fn offset(&self, x: usize, y: usize) -> usize {
        (y - self.y0) * self.width + (x - self.x0)
    }

    fn area(&self) -> usize {
        self.width * self.height
    }
}

fn compute_region_bounds(
    grad: &GradientLevel<'_>,
    pred_segment: &Segment,
    anchor: [f32; 2],
    padding: f32,
) -> Option<RegionBounds> {
    if grad.width == 0 || grad.height == 0 {
        return None;
    }
    let min_x = pred_segment.p0[0].min(pred_segment.p1[0]).min(anchor[0]) - padding;
    let max_x = pred_segment.p0[0].max(pred_segment.p1[0]).max(anchor[0]) + padding;
    let min_y = pred_segment.p0[1].min(pred_segment.p1[1]).min(anchor[1]) - padding;
    let max_y = pred_segment.p0[1].max(pred_segment.p1[1]).max(anchor[1]) + padding;

    let clamp = |v: f32, upper: usize| -> usize {
        let max_v = (upper.saturating_sub(1)) as f32;
        v.floor().clamp(0.0, max_v) as usize
    };

    let x0 = clamp(min_x, grad.width);
    let y0 = clamp(min_y, grad.height);
    let x1 = clamp(max_x, grad.width);
    let y1 = clamp(max_y, grad.height);
    if x0 > x1 || y0 > y1 {
        return None;
    }

    Some(RegionBounds {
        x0,
        y0,
        width: x1 - x0 + 1,
        height: y1 - y0 + 1,
    })
}

fn pixel_sample(
    grad: &GradientLevel<'_>,
    x: usize,
    y: usize,
    diagnostics: &mut RefineDiagnostics,
) -> GradientSample {
    let idx = pixel_index(grad.width, x, y);
    diagnostics.gradient_samples += 1;
    GradientSample {
        gx: grad.gx[idx],
        gy: grad.gy[idx],
        mag: grad.mag[idx],
    }
}

fn pixel_index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn clamp_coord(value: f32, upper: usize) -> usize {
    if upper == 0 {
        return 0;
    }
    let max_v = (upper - 1) as f32;
    value.round().clamp(0.0, max_v) as usize
}

fn refine_along_normal(
    grad: &GradientLevel<'_>,
    center: [f32; 2],
    normal: [f32; 2],
    radius: f32,
    params: &SegmentRefineParams,
    diagnostics: &mut RefineDiagnostics,
) -> Option<SamplePoint> {
    diagnostics.normal_refinements += 1;
    if radius <= 0.0 {
        return None;
    }
    let step = params.anchor_step.max(0.1);
    let steps = (radius / step).ceil() as i32;
    let mut best: Option<(GradientSample, [f32; 2], f32)> = None;

    for i in -steps..=steps {
        let t = i as f32 * step;
        let pos = [center[0] + normal[0] * t, center[1] + normal[1] * t];
        let Some(sample) = sample_gradient(grad, pos, diagnostics) else {
            continue;
        };
        if sample.mag < params.gradient_threshold {
            continue;
        }
        match &best {
            Some((current, _, _)) if current.mag >= sample.mag => {}
            _ => best = Some((sample, pos, t)),
        }
    }

    let (sample, pos, offset) = best?;
    let mut normal_vec = normalize([sample.gx, sample.gy])?;
    if dot(normal_vec, normal) < 0.0 {
        normal_vec = [-normal_vec[0], -normal_vec[1]];
    }

    Some(SamplePoint {
        pos,
        normal: normal_vec,
        mag: sample.mag,
        offset,
    })
}

fn fit_segment_from_region(
    pred: &Segment,
    region: &RegionAccumulator,
    width: usize,
    params: &SegmentRefineParams,
) -> Option<(Segment, f32)> {
    let count = region.len();
    if count < 2 {
        return None;
    }
    let inv = 1.0 / count as f32;
    let cx = region.sum_x * inv;
    let cy = region.sum_y * inv;
    if !cx.is_finite() || !cy.is_finite() {
        return None;
    }

    let cxx = region.sum_xx * inv - cx * cx;
    let cyy = region.sum_yy * inv - cy * cy;
    let cxy = region.sum_xy * inv - cx * cy;
    let cov = Matrix2::new(cxx, cxy, cxy, cyy);
    let eig = SymmetricEigen::new(cov);
    let (vmax, lambda_max) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
        (eig.eigenvectors.column(0), eig.eigenvalues[0])
    } else {
        (eig.eigenvectors.column(1), eig.eigenvalues[1])
    };
    if !lambda_max.is_finite() || lambda_max <= 0.0 {
        return None;
    }

    let mut tx = vmax[0];
    let mut ty = vmax[1];
    let mut norm = (tx * tx + ty * ty).sqrt();
    if !norm.is_finite() || norm < EPS {
        let fallback = pred.direction();
        tx = fallback[0];
        ty = fallback[1];
        norm = (tx * tx + ty * ty).sqrt();
        if norm < EPS {
            return None;
        }
    }
    tx /= norm;
    ty /= norm;

    let pred_dir = normalize(pred.direction()).unwrap_or([tx, ty]);
    if dot([tx, ty], pred_dir) < 0.0 {
        tx = -tx;
        ty = -ty;
    }

    let mut smin = f32::INFINITY;
    let mut smax = f32::NEG_INFINITY;
    let nx = -ty;
    let ny = tx;
    let mut nmin = f32::INFINITY;
    let mut nmax = f32::NEG_INFINITY;
    for &idx in &region.indices {
        let x = (idx % width) as f32;
        let y = (idx / width) as f32;
        let dx = x - cx;
        let dy = y - cy;
        let s = dx * tx + dy * ty;
        if s < smin {
            smin = s;
        }
        if s > smax {
            smax = s;
        }
        let n = dx * nx + dy * ny;
        if n < nmin {
            nmin = n;
        }
        if n > nmax {
            nmax = n;
        }
    }

    if !smin.is_finite() || !smax.is_finite() {
        return None;
    }
    let len = smax - smin;
    if !len.is_finite() || len <= 0.0 || len < params.min_length_px {
        return None;
    }

    if let Some(limit) = params.normal_span_limit_px {
        if nmin.is_finite() && nmax.is_finite() {
            let span = nmax - nmin;
            if !span.is_finite() || span > limit {
                return None;
            }
        }
    }

    let p0 = [cx + smin * tx, cy + smin * ty];
    let p1 = [cx + smax * tx, cy + smax * ty];
    let avg_mag = region.avg_mag();
    let strength = len * avg_mag.max(1e-3);
    let segment = Segment::new(pred.id, p0, p1, avg_mag, strength);

    Some((segment, avg_mag))
}

fn sample_gradient(
    grad: &GradientLevel<'_>,
    pos: [f32; 2],
    diagnostics: &mut RefineDiagnostics,
) -> Option<GradientSample> {
    let sample = grad.sample(pos[0], pos[1]);
    if sample.is_some() {
        diagnostics.gradient_samples += 1;
    }
    sample
}

fn normalize(v: [f32; 2]) -> Option<[f32; 2]> {
    let len = length(v);
    if len <= EPS {
        None
    } else {
        Some([v[0] / len, v[1] / len])
    }
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn lerp_points(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edges::grad::Grad;
    use crate::image::ImageF32;
    use crate::segments::{Segment, SegmentId};

    fn dummy_grad() -> Grad {
        let mut gx = ImageF32::new(8, 8);
        let mut gy = ImageF32::new(8, 8);
        let mut mag = ImageF32::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let idx = y * 8 + x;
                gx.data[idx] = 0.0;
                gy.data[idx] = 1.0;
                mag.data[idx] = 1.0;
            }
        }
        Grad {
            gx,
            gy,
            mag,
            ori_q8: vec![0; 64],
        }
    }

    #[test]
    fn seed_offsets_spacing() {
        let s = seed_offsets(3);
        assert_eq!(s.len(), 3);
        assert!(s[0] > 0.0 && s[2] < 1.0);
    }

    #[test]
    fn normalize_rejects_zero() {
        assert!(normalize([0.0, 0.0]).is_none());
        let n = normalize([3.0, 0.0]).unwrap();
        assert!((n[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fit_segment_handles_simple_case() {
        let grad_data = dummy_grad();
        let grad = GradientLevel::from_grad(&grad_data);
        let mut params = SegmentRefineParams::default();
        params.region_min_pixels = 1;
        params.region_max_pixels = 16;
        params.region_min_alignment = 0.0;
        params.min_length_px = 1.0;
        let seg = Segment::new(SegmentId(1), [2.0, 3.0], [6.0, 3.0], 1.0, 4.0);
        let result = refine_segment_coarse_to_fine(&grad, &seg, &params);
        assert!(result.is_some());
    }
}
