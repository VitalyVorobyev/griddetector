//! Fast coarse-to-fine segment refinement using gradient-driven region growing.
//!
//! Each segment predicted from a coarser pyramid level seeds a local search on
//! the finer level. The routine identifies a strong anchor along the segment
//! normal, then grows forward/backward along the tangent with tiny normal
//! refinements reminiscent of LSD region growing. This replaces the previous
//! dense sampling approach with a lightweight procedure that reuses cached
//! gradient fields and requires only a handful of gradient evaluations.

mod types;

use crate::edges::grad::{GradientLevel, GradientSample};
use crate::segments::Segment;

pub use types::{RefineDiagnostics, RefineResult, ScaleMap, SegmentRefineParams};

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
struct SamplePoint {
    pos: [f32; 2],
    grad: [f32; 2],
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

    let accept = result.support >= params.min_support_points
        && result.score >= params.gradient_threshold
        && result.seg.length_sq() > EPS;

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
    let direction = pred_segment.direction();
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

    let mut points = Vec::new();
    let mut tangent = normalize([-anchor.normal[1], anchor.normal[0]]).unwrap_or(direction);
    if dot(tangent, direction) < 0.0 {
        tangent = [-tangent[0], -tangent[1]];
    }

    let mut anchor_point = anchor.clone();
    anchor_point.offset = 0.0;

    let forward = grow_from_anchor(grad, &anchor_point, tangent, params, &mut diagnostics);
    let backward = grow_from_anchor(
        grad,
        &anchor_point,
        [-tangent[0], -tangent[1]],
        params,
        &mut diagnostics,
    );

    for p in backward.into_iter().rev() {
        points.push(p);
    }
    points.push(anchor_point);
    points.extend(forward.into_iter());

    if points.len() < params.min_support_points {
        return None;
    }

    let (segment, avg_mag) = fit_segment(pred_segment, &points)?;
    let support = points.len();

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

fn grow_from_anchor(
    grad: &GradientLevel<'_>,
    anchor: &SamplePoint,
    tangent_seed: [f32; 2],
    params: &SegmentRefineParams,
    diagnostics: &mut RefineDiagnostics,
) -> Vec<SamplePoint> {
    let mut current = anchor.clone();
    let mut tangent = normalize(tangent_seed).unwrap_or([1.0, 0.0]);
    let mut normal = current.normal;
    let mut points = Vec::new();

    for _ in 0..params.max_grow_steps {
        let predicted = [
            current.pos[0] + tangent[0] * params.tangent_step,
            current.pos[1] + tangent[1] * params.tangent_step,
        ];
        let Some(mut refined) = refine_along_normal(
            grad,
            predicted,
            normal,
            params.anchor_search_radius,
            params,
            diagnostics,
        ) else {
            break;
        };
        if refined.mag < params.gradient_threshold {
            break;
        }
        if refined.offset.abs() > params.max_normal_shift {
            break;
        }
        let dist = distance(refined.pos, current.pos);
        if dist > params.max_point_dist {
            break;
        }
        if dot(refined.normal, normal) < 0.0 {
            refined.normal = [-refined.normal[0], -refined.normal[1]];
            refined.grad = [-refined.grad[0], -refined.grad[1]];
        }
        diagnostics.tangent_steps += 1;
        tangent = normalize([-refined.normal[1], refined.normal[0]]).unwrap_or(tangent);
        if dot(tangent, [-normal[1], normal[0]]) < 0.0 {
            tangent = [-tangent[0], -tangent[1]];
        }
        normal = refined.normal;
        current = refined.clone();
        points.push(refined);
    }

    points
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
    let step = params.normal_step.max(0.1);
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
        grad: [sample.gx, sample.gy],
        normal: normal_vec,
        mag: sample.mag,
        offset,
    })
}

fn fit_segment(pred: &Segment, points: &[SamplePoint]) -> Option<(Segment, f32)> {
    let count = points.len();
    if count < 2 {
        return None;
    }
    let inv = 1.0 / count as f32;
    let mut mean = [0.0f32, 0.0f32];
    let mut mag_sum = 0.0f32;
    for p in points {
        mean[0] += p.pos[0];
        mean[1] += p.pos[1];
        mag_sum += p.mag;
    }
    mean[0] *= inv;
    mean[1] *= inv;

    let mut cov_xx = 0.0f32;
    let mut cov_xy = 0.0f32;
    let mut cov_yy = 0.0f32;
    for p in points {
        let dx = p.pos[0] - mean[0];
        let dy = p.pos[1] - mean[1];
        cov_xx += dx * dx;
        cov_xy += dx * dy;
        cov_yy += dy * dy;
    }
    cov_xx *= inv;
    cov_xy *= inv;
    cov_yy *= inv;

    let trace = cov_xx + cov_yy;
    let det = (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy;
    let lambda = 0.5 * (trace + det.max(0.0).sqrt());
    let mut dir = [cov_xy, lambda - cov_xx];
    if length(dir) <= EPS {
        dir = pred.direction();
    }
    dir = normalize(dir).unwrap_or([1.0, 0.0]);
    let pred_dir = normalize(pred.direction()).unwrap_or(dir);
    if dot(dir, pred_dir) < 0.0 {
        dir = [-dir[0], -dir[1]];
    }

    let mut min_proj = f32::MAX;
    let mut max_proj = f32::MIN;
    for p in points {
        let rel = [p.pos[0] - mean[0], p.pos[1] - mean[1]];
        let proj = rel[0] * dir[0] + rel[1] * dir[1];
        if proj < min_proj {
            min_proj = proj;
        }
        if proj > max_proj {
            max_proj = proj;
        }
    }

    if !min_proj.is_finite() || !max_proj.is_finite() {
        return None;
    }

    let p0 = [mean[0] + dir[0] * min_proj, mean[1] + dir[1] * min_proj];
    let p1 = [mean[0] + dir[0] * max_proj, mean[1] + dir[1] * max_proj];
    let len_sq = (p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2);
    if len_sq <= EPS {
        return None;
    }
    let len = len_sq.sqrt();
    let avg_mag = mag_sum * inv;
    let segment = Segment::new(pred.id, p0, p1, avg_mag, avg_mag * len);

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

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
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
        params.max_grow_steps = 1;
        params.min_support_points = 1;
        let seg = Segment::new(SegmentId(1), [2.0, 3.0], [6.0, 3.0], 1.0, 4.0);
        let result = refine_segment_coarse_to_fine(&grad, &seg, &params);
        assert!(result.is_some());
    }
}
