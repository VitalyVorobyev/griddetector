//! Line segment refinement driven by local gradient support.
//!
//! # Overview
//!
//! Starting from a seed segment detected on pyramid level `l+1`, the routine
//! lifts the endpoints to level `l`, searches for strong gradient responses
//! along the segment normal, re-estimates the carrier via a robust orthogonal
//! fit, and finally adjusts the segment extent to the longest contiguous run of
//! inliers. The resulting subpixel segment can be fed into the bundling stage
//! ahead of [`homography::Refiner`](super::homography::Refiner) to tighten the
//! coarse-to-fine optimisation loop.

use crate::angle::angle_between;

const EPS: f32 = 1e-6;

/// Single pyramid level with precomputed Sobel/Scharr gradients.
#[derive(Clone, Copy, Debug)]
pub struct PyramidLevel<'a> {
    pub width: usize,
    pub height: usize,
    pub gx: &'a [f32],
    pub gy: &'a [f32],
}

/// Seed line segment expressed via two subpixel endpoints.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

impl Segment {
    #[inline]
    pub fn dir(&self) -> [f32; 2] {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        let len = (dx * dx + dy * dy).sqrt().max(EPS);
        [dx / len, dy / len]
    }

    #[inline]
    pub fn length(&self) -> f32 {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// Parameters controlling the gradient-driven refinement.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Along-segment sample spacing (px) used for normal probing.
    pub delta_s: f32,
    /// Half-width (px) of the normal search corridor.
    pub w_perp: f32,
    /// Step size (px) for sweeping along the normal.
    pub delta_t: f32,
    /// Padding (px) added around the segment when forming the ROI.
    pub pad: f32,
    /// Minimum gradient magnitude accepted as support.
    pub tau_mag: f32,
    /// Orientation tolerance (degrees) applied during endpoint gating.
    pub tau_ori_deg: f32,
    /// Huber delta used to weight the normal projected gradient.
    pub huber_delta: f32,
    /// Maximum number of outer carrier update iterations.
    pub max_iters: usize,
    /// Minimum inlier fraction required to accept the refinement.
    pub min_inlier_frac: f32,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            delta_s: 0.75,
            w_perp: 3.0,
            delta_t: 0.5,
            pad: 8.0,
            tau_mag: 10.0,
            tau_ori_deg: 25.0,
            huber_delta: 3.0,
            max_iters: 3,
            min_inlier_frac: 0.4,
        }
    }
}

/// Outcome of a refinement attempt.
#[derive(Clone, Debug)]
pub struct RefineResult {
    /// Refined segment (or the fallback seed when `ok == false`).
    pub seg: Segment,
    /// Mean absolute normal-projected gradient across inlier samples.
    pub score: f32,
    /// Whether the refinement satisfied the inlier and score thresholds.
    pub ok: bool,
    /// Number of inlier samples supporting the endpoints.
    pub inliers: usize,
    /// Total number of centreline samples considered.
    pub total: usize,
}

/// Coordinate mapping from pyramid level `l+1` to level `l`.
///
/// Implementors can model pure dyadic scaling, fractional pixel offsets, or
/// any bespoke decimation geometry used to build the image pyramid.
pub trait ScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2];
}

struct Roi {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Roi {
    fn contains(&self, p: &[f32; 2]) -> bool {
        p[0] >= self.x0 && p[0] <= self.x1 && p[1] >= self.y0 && p[1] <= self.y1
    }

    fn clamp_inside(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0].clamp(self.x0, self.x1), p[1].clamp(self.y0, self.y1)]
    }
}

/// Result of the coarse-to-fine carrier update loop.
struct IterationSnapshot {
    seg: Segment,
    mu: [f32; 2],
    normal: [f32; 2],
    total_centers: usize,
    roi: Roi,
}

#[derive(Clone)]
struct SupportPoint {
    pos: [f32; 2],
    weight: f32,
    grad: [f32; 2],
}

type LineFitResult = ([f32; 2], [f32; 2], f32, [f32; 2]);

/// Refine a coarse-level segment using gradient support on the finer level.
///
/// # Arguments
/// * `lvl` - Gradient buffers sampled with bilinear interpolation.
/// * `seg_coarse` - Seed segment expressed in the coordinates of level `l+1`.
/// * `scale` - [`ScaleMap`] that lifts coordinates from level `l+1` to `l`.
/// * `params` - [`RefineParams`] controlling sampling density and thresholds.
///
/// # Returns
/// A [`RefineResult`] containing the refined segment, aggregated support score
/// and an acceptance flag. When `ok == false` the returned segment equals the
/// upscaled seed, allowing callers to fall back gracefully.
pub fn refine_segment(
    lvl: &PyramidLevel<'_>,
    seg_coarse: Segment,
    scale: &dyn ScaleMap,
    params: &RefineParams,
) -> RefineResult {
    if lvl.width == 0 || lvl.height == 0 {
        return RefineResult {
            seg: seg_coarse,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        };
    }

    let seg = Segment {
        p0: scale.up(seg_coarse.p0),
        p1: scale.up(seg_coarse.p1),
    };
    let fallback = seg;

    if seg.length() < 1e-3 {
        return RefineResult {
            seg: fallback,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        };
    }

    let Some(roi) = compute_roi(&seg, params.pad, lvl.width, lvl.height) else {
        return RefineResult {
            seg: fallback,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        };
    };
    let mut snapshot = None;
    for attempt in 0..2 {
        let w_perp = if attempt == 0 {
            params.w_perp
        } else {
            (params.w_perp * 0.5).max(1.0)
        };
        if let Some(iter) = run_iterations(lvl, seg, &roi, params, w_perp) {
            snapshot = Some(iter);
            break;
        }
    }

    let Some(snapshot) = snapshot else {
        return RefineResult {
            seg: fallback,
            score: 0.0,
            ok: false,
            inliers: 0,
            total: 0,
        };
    };

    let (p0_f, p1_f, support_count, score) =
        refine_endpoints(&snapshot, lvl, params, params.tau_mag);
    let refined_segment = Segment { p0: p0_f, p1: p1_f };
    let total_centers = snapshot.total_centers;
    let denom = total_centers.max(1);
    let ok =
        (support_count as f32) >= params.min_inlier_frac * denom as f32 && score >= params.tau_mag;
    if !ok {
        return RefineResult {
            seg: fallback,
            score: 0.0,
            ok: false,
            inliers: support_count,
            total: total_centers,
        };
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
    seg0: Segment,
    roi: &Roi,
    params: &RefineParams,
    w_perp: f32,
) -> Option<IterationSnapshot> {
    let mut p0 = seg0.p0;
    let mut p1 = seg0.p1;
    let mut last_mu = midpoint(&p0, &p1);
    let mut last_normal = normal_from_segment(&p0, &p1)?;
    let mut total_centers = 0usize;

    for _ in 0..params.max_iters.max(1) {
        let dir = direction(&p0, &p1)?;
        let normal = [-dir[1], dir[0]];
        let length = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
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
        if supports.len() < 3 {
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
        seg: Segment { p0, p1 },
        mu: last_mu,
        normal: last_normal,
        total_centers,
        roi: Roi {
            x0: roi.x0,
            y0: roi.y0,
            x1: roi.x1,
            y1: roi.y1,
        },
    })
}

fn search_along_normal(
    lvl: &PyramidLevel<'_>,
    roi: &Roi,
    center: &[f32; 2],
    normal: &[f32; 2],
    params: &RefineParams,
    w_perp: f32,
) -> Option<SupportPoint> {
    let mut best: Option<(f32, f32, [f32; 2], f32)> = None; // (t, |g|, grad, proj)
    let mut t = -w_perp;
    while t <= w_perp + 1e-3 {
        let p = [center[0] + t * normal[0], center[1] + t * normal[1]];
        if !roi.contains(&p) {
            t += params.delta_t;
            continue;
        }
        if let Some((gx, gy)) = bilinear_grad(lvl, p[0], p[1]) {
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < params.tau_mag {
                t += params.delta_t;
                continue;
            }
            let proj = gx * normal[0] + gy * normal[1];
            let val = proj.abs();
            if let Some(ref mut current) = best {
                if val > current.1 {
                    *current = (t, val, [gx, gy], proj);
                }
            } else {
                best = Some((t, val, [gx, gy], proj));
            }
        }
        t += params.delta_t;
    }

    let (t_best, mut peak, _, _) = best?;

    let refined_t = quadratic_refine(lvl, roi, center, normal, params.delta_t, t_best, peak);
    let pos = [
        center[0] + refined_t * normal[0],
        center[1] + refined_t * normal[1],
    ];
    let (gx, gy) = bilinear_grad(lvl, pos[0], pos[1])?;
    let proj = gx * normal[0] + gy * normal[1];
    peak = proj.abs();
    let weight = huber_weight(peak, params.huber_delta);
    Some(SupportPoint {
        pos,
        weight,
        grad: [gx, gy],
    })
}

fn huber_weight(value: f32, delta: f32) -> f32 {
    if value <= delta {
        1.0
    } else {
        (delta / value).max(1e-3)
    }
}

fn bilinear_grad(lvl: &PyramidLevel<'_>, x: f32, y: f32) -> Option<(f32, f32)> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let w = lvl.width as isize;
    let h = lvl.height as isize;
    let xf = x.floor();
    let yf = y.floor();
    let x0 = xf as isize;
    let y0 = yf as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x1 >= w || y1 >= h {
        return None;
    }
    let tx = x - xf;
    let ty = y - yf;
    let idx = |xx: isize, yy: isize| -> usize { (yy as usize) * lvl.width + (xx as usize) };
    let gx00 = lvl.gx[idx(x0, y0)];
    let gx10 = lvl.gx[idx(x1, y0)];
    let gx01 = lvl.gx[idx(x0, y1)];
    let gx11 = lvl.gx[idx(x1, y1)];
    let gy00 = lvl.gy[idx(x0, y0)];
    let gy10 = lvl.gy[idx(x1, y0)];
    let gy01 = lvl.gy[idx(x0, y1)];
    let gy11 = lvl.gy[idx(x1, y1)];

    let gx0 = gx00 * (1.0 - tx) + gx10 * tx;
    let gx1 = gx01 * (1.0 - tx) + gx11 * tx;
    let gx = gx0 * (1.0 - ty) + gx1 * ty;

    let gy0 = gy00 * (1.0 - tx) + gy10 * tx;
    let gy1 = gy01 * (1.0 - tx) + gy11 * tx;
    let gy = gy0 * (1.0 - ty) + gy1 * ty;
    Some((gx, gy))
}

fn quadratic_refine(
    lvl: &PyramidLevel<'_>,
    roi: &Roi,
    center: &[f32; 2],
    normal: &[f32; 2],
    delta_t: f32,
    t_best: f32,
    peak_best: f32,
) -> f32 {
    let offsets = [-delta_t, 0.0, delta_t];
    let mut samples = [peak_best; 3];
    for (i, off) in offsets.iter().enumerate() {
        if *off == 0.0 {
            continue;
        }
        let t = t_best + *off;
        let p = [center[0] + t * normal[0], center[1] + t * normal[1]];
        if !roi.contains(&p) {
            continue;
        }
        if let Some((gx, gy)) = bilinear_grad(lvl, p[0], p[1]) {
            let proj = gx * normal[0] + gy * normal[1];
            samples[i] = proj.abs();
        }
    }
    let f0 = samples[0];
    let f1 = samples[1];
    let f2 = samples[2];
    let denom = (f0 - 2.0 * f1 + f2).abs().max(EPS);
    let shift = 0.5 * (f0 - f2) / denom;
    (t_best + shift * delta_t).clamp(t_best - delta_t, t_best + delta_t)
}

fn weighted_line_fit(supports: &[SupportPoint], params: &RefineParams) -> Option<LineFitResult> {
    let mut sum_w = 0.0;
    let mut mu = [0.0f32, 0.0];
    for s in supports {
        sum_w += s.weight;
        mu[0] += s.weight * s.pos[0];
        mu[1] += s.weight * s.pos[1];
    }
    if sum_w <= EPS {
        return None;
    }
    mu[0] /= sum_w;
    mu[1] /= sum_w;

    let mut cov_xx = 0.0f32;
    let mut cov_xy = 0.0f32;
    let mut cov_yy = 0.0f32;
    for s in supports {
        let dx = s.pos[0] - mu[0];
        let dy = s.pos[1] - mu[1];
        cov_xx += s.weight * dx * dx;
        cov_xy += s.weight * dx * dy;
        cov_yy += s.weight * dy * dy;
    }
    cov_xx /= sum_w;
    cov_xy /= sum_w;
    cov_yy /= sum_w;

    let trace = cov_xx + cov_yy;
    let det_part = (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy;
    let lambda = 0.5 * (trace + det_part.max(0.0).sqrt());
    let mut dir = [cov_xy, lambda - cov_xx];
    let norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if norm <= EPS {
        dir = [1.0, 0.0];
    } else {
        dir[0] /= norm;
        dir[1] /= norm;
    }
    let mut normal = [-dir[1], dir[0]];

    // orientation agreement reweighting
    if params.tau_ori_deg > 0.0 {
        let tau_rad = params.tau_ori_deg.to_radians();
        let mut weights = Vec::with_capacity(supports.len());
        let mut sum_w2 = 0.0f32;
        for s in supports {
            let grad = s.grad;
            let mag = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt().max(EPS);
            let dot = (grad[0] * normal[0] + grad[1] * normal[1]) / mag;
            let cos = dot.clamp(-1.0, 1.0);
            let angle = cos.acos();
            let ori_weight = if angle <= tau_rad { cos.max(0.0) } else { 0.0 };
            let w = (s.weight * ori_weight).max(0.0);
            weights.push(w);
            sum_w2 += w;
        }
        if sum_w2 > EPS {
            let mut mu2 = [0.0f32, 0.0];
            for (s, &w) in supports.iter().zip(weights.iter()) {
                mu2[0] += w * s.pos[0];
                mu2[1] += w * s.pos[1];
            }
            mu2[0] /= sum_w2;
            mu2[1] /= sum_w2;
            let mut cov_xx2 = 0.0f32;
            let mut cov_xy2 = 0.0f32;
            let mut cov_yy2 = 0.0f32;
            for (s, &w) in supports.iter().zip(weights.iter()) {
                let dx = s.pos[0] - mu2[0];
                let dy = s.pos[1] - mu2[1];
                cov_xx2 += w * dx * dx;
                cov_xy2 += w * dx * dy;
                cov_yy2 += w * dy * dy;
            }
            cov_xx2 /= sum_w2;
            cov_xy2 /= sum_w2;
            cov_yy2 /= sum_w2;
            let trace2 = cov_xx2 + cov_yy2;
            let det_part2 = (cov_xx2 - cov_yy2) * (cov_xx2 - cov_yy2) + 4.0 * cov_xy2 * cov_xy2;
            let lambda2 = 0.5 * (trace2 + det_part2.max(0.0).sqrt());
            let mut dir2 = [cov_xy2, lambda2 - cov_xx2];
            let norm2 = (dir2[0] * dir2[0] + dir2[1] * dir2[1]).sqrt();
            if norm2 > EPS {
                dir2[0] /= norm2;
                dir2[1] /= norm2;
                dir = dir2;
                normal = [-dir[1], dir[0]];
                mu = mu2;
            }
        }
    }

    let rho = normal[0] * mu[0] + normal[1] * mu[1];
    Some((dir, normal, rho, mu))
}

fn project_point_to_line(p: &[f32; 2], normal: &[f32; 2], rho: f32) -> [f32; 2] {
    let dist = (normal[0] * p[0] + normal[1] * p[1]) - rho;
    [p[0] - dist * normal[0], p[1] - dist * normal[1]]
}

fn distance(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn direction(p0: &[f32; 2], p1: &[f32; 2]) -> Option<[f32; 2]> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let norm = (dx * dx + dy * dy).sqrt();
    if norm <= EPS {
        None
    } else {
        Some([dx / norm, dy / norm])
    }
}

fn normal_from_segment(p0: &[f32; 2], p1: &[f32; 2]) -> Option<[f32; 2]> {
    direction(p0, p1).map(|d| [-d[1], d[0]])
}

fn midpoint(a: &[f32; 2], b: &[f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

struct RunResult {
    r0: f32,
    r1: f32,
    count: usize,
    score: f32,
}

struct RunContext {
    tau_mag: f32,
    delta_r: f32,
    seg_bounds: (f32, f32),
}

fn refine_endpoints(
    snapshot: &IterationSnapshot,
    lvl: &PyramidLevel<'_>,
    params: &RefineParams,
    tau_mag: f32,
) -> ([f32; 2], [f32; 2], usize, f32) {
    let seg = &snapshot.seg;
    let mut p0 = seg.p0;
    let mut p1 = seg.p1;
    let normal = snapshot.normal;
    let d = match direction(&p0, &p1) {
        Some(v) => v,
        None => return (p0, p1, 0, 0.0),
    };
    let mu = snapshot.mu;
    let seg_r0 = (p0[0] - mu[0]) * d[0] + (p0[1] - mu[1]) * d[1];
    let seg_r1 = (p1[0] - mu[0]) * d[0] + (p1[1] - mu[1]) * d[1];
    let seg_min = seg_r0.min(seg_r1);
    let seg_max = seg_r0.max(seg_r1);

    let Some((rmin, rmax)) = line_rect_intersection(&mu, &d, &snapshot.roi) else {
        return (p0, p1, 0, 0.0);
    };
    let delta_r = 0.5f32;
    let run_params = RunContext {
        tau_mag,
        delta_r,
        seg_bounds: (seg_min, seg_max),
    };
    let mut samples = Vec::new();
    let mut r = rmin;
    while r <= rmax + 1e-3 {
        let x = [mu[0] + r * d[0], mu[1] + r * d[1]];
        if let Some((gx, gy)) = bilinear_grad(lvl, x[0], x[1]) {
            let mag = (gx * gx + gy * gy).sqrt();
            if mag >= tau_mag {
                let dot = gx * normal[0] + gy * normal[1];
                let grad_unit = mag.max(EPS);
                let angle = (dot / grad_unit).clamp(-1.0, 1.0).acos();
                samples.push((r, dot, mag, angle));
            }
        }
        r += delta_r;
    }

    if samples.is_empty() {
        return (p0, p1, 0, 0.0);
    }

    let mut pos = 0usize;
    let mut neg = 0usize;
    for (_, dot, _, _) in &samples {
        if *dot >= 0.0 {
            pos += 1;
        } else {
            neg += 1;
        }
    }
    let dominant_sign = if pos >= neg { 1.0 } else { -1.0 };
    let tau_ori = params.tau_ori_deg.to_radians();
    let mut inlier_flags = Vec::with_capacity(samples.len());
    for (_, dot, _, angle) in &samples {
        let ori_ok = *angle <= tau_ori;
        let polarity_ok =
            (*dot >= 0.0 && dominant_sign > 0.0) || (*dot < 0.0 && dominant_sign < 0.0);
        inlier_flags.push(ori_ok && polarity_ok);
    }

    let mut best = RunResult {
        r0: 0.0,
        r1: 0.0,
        count: 0,
        score: 0.0,
    };
    let mut current_start = None::<usize>;
    for (idx, flag) in inlier_flags.iter().enumerate() {
        if *flag {
            if current_start.is_none() {
                current_start = Some(idx);
            }
        } else if let Some(start) = current_start.take() {
            update_best_run(start, idx - 1, &samples, &run_params, &mut best);
        }
    }
    if let Some(start) = current_start {
        let end = samples.len() - 1;
        update_best_run(start, end, &samples, &run_params, &mut best);
    }

    if best.count >= 2 {
        p0 = [mu[0] + best.r0 * d[0], mu[1] + best.r0 * d[1]];
        p1 = [mu[0] + best.r1 * d[0], mu[1] + best.r1 * d[1]];
        let p0 = snapshot.roi.clamp_inside(p0);
        let p1 = snapshot.roi.clamp_inside(p1);
        (p0, p1, best.count, best.score / best.count as f32)
    } else {
        (p0, p1, 0, 0.0)
    }
}

fn update_best_run(
    start: usize,
    end: usize,
    samples: &[(f32, f32, f32, f32)],
    params: &RunContext,
    best: &mut RunResult,
) {
    if end < start {
        return;
    }
    let count = end - start + 1;
    let mut score_sum = 0.0f32;

    for item in samples.iter().skip(start).take(count) {
        score_sum += item.1.abs();
    }
    if count < best.count || (count == best.count && score_sum <= best.score) {
        return;
    }
    let refined_start = refine_endpoint(samples, start, 1, params.tau_mag, params.delta_r);
    let refined_end = refine_endpoint(samples, end, -1, params.tau_mag, params.delta_r);
    let mut r0 = refined_start;
    let mut r1 = refined_end;
    if r0 > r1 {
        std::mem::swap(&mut r0, &mut r1);
    }
    let (seg_min, seg_max) = params.seg_bounds;
    let overlap = (r1.min(seg_max) - r0.max(seg_min)).max(0.0);
    if overlap <= 0.0 {
        return;
    }
    *best = RunResult {
        r0,
        r1,
        count,
        score: score_sum,
    };
}

fn refine_endpoint(
    samples: &[(f32, f32, f32, f32)],
    idx: usize,
    dir: i32,
    tau_mag: f32,
    delta_r: f32,
) -> f32 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    xs.push(samples[idx].0);
    ys.push(samples[idx].2);
    let mut next = idx as isize + dir as isize;
    while xs.len() < 3 && next >= 0 && (next as usize) < samples.len() {
        xs.push(samples[next as usize].0);
        ys.push(samples[next as usize].2);
        next += dir as isize;
    }
    if xs.len() < 2 {
        return samples[idx].0;
    }
    let n = xs.len() as f32;
    let sum_x: f32 = xs.iter().sum();
    let sum_y: f32 = ys.iter().sum();
    let sum_xx: f32 = xs.iter().map(|v| v * v).sum();
    let sum_xy: f32 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= EPS {
        return samples[idx].0;
    }
    let a = (n * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_y - a * sum_x) / n;
    if a.abs() <= EPS {
        samples[idx].0
    } else {
        ((tau_mag - b) / a).clamp(samples[idx].0 - delta_r, samples[idx].0 + delta_r)
    }
}

fn line_rect_intersection(mu: &[f32; 2], d: &[f32; 2], roi: &Roi) -> Option<(f32, f32)> {
    let mut rs = Vec::new();
    let eps = 1e-6f32;
    if d[0].abs() > eps {
        for &x in [roi.x0, roi.x1].iter() {
            let r = (x - mu[0]) / d[0];
            let y = mu[1] + r * d[1];
            if y >= roi.y0 - eps && y <= roi.y1 + eps {
                rs.push(r);
            }
        }
    }
    if d[1].abs() > eps {
        for &y in [roi.y0, roi.y1].iter() {
            let r = (y - mu[1]) / d[1];
            let x = mu[0] + r * d[0];
            if x >= roi.x0 - eps && x <= roi.x1 + eps {
                rs.push(r);
            }
        }
    }
    if rs.len() < 2 {
        return None;
    }
    let min_r = rs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_r = rs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    Some((min_r.min(max_r), max_r.max(min_r)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_grad_returns_none_outside() {
        let lvl = PyramidLevel {
            width: 4,
            height: 4,
            gx: &[0.0; 16],
            gy: &[0.0; 16],
        };
        assert!(bilinear_grad(&lvl, -1.0, 0.0).is_none());
        assert!(bilinear_grad(&lvl, 3.5, 3.5).is_none());
    }

    #[test]
    fn huber_weight_basic() {
        assert!((huber_weight(1.0, 3.0) - 1.0).abs() < 1e-6);
        assert!((huber_weight(10.0, 2.0) - 0.2).abs() < 1e-6);
    }
}
