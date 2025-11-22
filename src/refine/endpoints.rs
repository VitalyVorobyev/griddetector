//! Endpoint refinement along the carrier direction.

use super::{
    iteration::IterationSnapshot, options::RefineOptions, roi::SegmentRoi, sampling::bilinear_grad,
    workspace::PyramidLevel,
};

const EPS: f32 = 1e-6;

pub(super) fn refine_endpoints(
    snapshot: &IterationSnapshot,
    lvl: &PyramidLevel<'_>,
    params: &RefineOptions,
) -> ([f32; 2], [f32; 2], usize, f32) {
    let seg = &snapshot.seg;
    let mut p0 = seg.p0;
    let mut p1 = seg.p1;
    let normal = snapshot.normal;
    let d = seg.direction();
    let mu = snapshot.mu;
    let seg_r0 = (p0[0] - mu[0]) * d[0] + (p0[1] - mu[1]) * d[1];
    let seg_r1 = (p1[0] - mu[0]) * d[0] + (p1[1] - mu[1]) * d[1];
    let seg_min = seg_r0.min(seg_r1);
    let seg_max = seg_r0.max(seg_r1);

    let Some((rmin, rmax)) = line_rect_intersection(&mu, &d, &snapshot.roi) else {
        return (p0, p1, 0, 0.0);
    };
    let delta_r = 0.5f32;
    let pad_r = (params.delta_s.max(0.5)) * 2.0;
    let scan_min = (seg_min - pad_r).max(rmin);
    let scan_max = (seg_max + pad_r).min(rmax);
    if scan_max <= scan_min {
        return (p0, p1, 0, 0.0);
    }

    let run_params = RunContext {
        tau_mag: params.tau_mag,
        delta_r,
        seg_bounds: (seg_min, seg_max),
        search_bounds: (scan_min, scan_max),
    };
    let mut samples = Vec::new();
    let mut r = scan_min;
    while r <= scan_max + 1e-3 {
        let x = [mu[0] + r * d[0], mu[1] + r * d[1]];
        if let Some((gx, gy)) = bilinear_grad(lvl, x[0], x[1]) {
            let mag = (gx * gx + gy * gy).sqrt();
            if mag >= params.tau_mag {
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
    let mixed_polarity = pos > 0 && neg > 0 && (pos.min(neg) as f32 / (pos + neg) as f32) >= 0.25;
    let tau_ori = params.tau_ori_deg.to_radians();
    let mut inlier_flags = Vec::with_capacity(samples.len());
    for (_, dot, _, angle) in &samples {
        let ori_ok = *angle <= tau_ori;
        let polarity_ok = mixed_polarity
            || ((*dot >= 0.0 && dominant_sign > 0.0) || (*dot < 0.0 && dominant_sign < 0.0));
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
            current_start.get_or_insert(idx);
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
    search_bounds: (f32, f32),
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

    let refined_start = refine_endpoint(
        samples,
        start,
        1,
        params.tau_mag,
        params.delta_r,
        params.search_bounds,
    );
    let refined_end = refine_endpoint(
        samples,
        end,
        -1,
        params.tau_mag,
        params.delta_r,
        params.search_bounds,
    );
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
    search_bounds: (f32, f32),
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
        let raw = (tau_mag - b) / a;
        raw.clamp(
            (samples[idx].0 - delta_r).max(search_bounds.0),
            (samples[idx].0 + delta_r).min(search_bounds.1),
        )
    }
}

fn line_rect_intersection(mu: &[f32; 2], d: &[f32; 2], roi: &SegmentRoi) -> Option<(f32, f32)> {
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
    fn line_rect_intersection_basic() {
        let roi = SegmentRoi {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
        };
        let mu = [5.0, 5.0];
        let d = [1.0, 0.0];
        let (r0, r1) = line_rect_intersection(&mu, &d, &roi).expect("intersection");
        assert!(r0 < r1);
    }
}
