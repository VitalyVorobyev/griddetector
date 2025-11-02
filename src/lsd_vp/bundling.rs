use crate::segments::Segment;
use log::{debug, info};
use nalgebra::{Matrix3, Vector3};

/// Internal observation used for 1D clustering in rectified space.
struct Obs {
    index: usize,
    param: f32,
    strength: f32,
}

fn is_finite_vec3(v: &Vector3<f32>) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

fn cluster_1d(mut obs: Vec<Obs>, eps: f32, min_strength: f32) -> Vec<Vec<Obs>> {
    if obs.is_empty() {
        return Vec::new();
    }
    obs.sort_by(|a, b| {
        a.param
            .partial_cmp(&b.param)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut clusters: Vec<Vec<Obs>> = Vec::new();
    let mut cur: Vec<Obs> = Vec::new();
    for o in obs.into_iter() {
        if cur.is_empty() || (o.param - cur.last().unwrap().param).abs() <= eps {
            cur.push(o);
        } else {
            let sum_w: f32 = cur.iter().map(|x| x.strength).sum();
            if sum_w >= min_strength {
                clusters.push(cur);
            }
            cur = vec![o];
        }
    }
    if !cur.is_empty() {
        let sum_w: f32 = cur.iter().map(|x| x.strength).sum();
        if sum_w >= min_strength {
            clusters.push(cur);
        }
    }
    clusters
}

fn compute_auto_eps(obs: &[Obs], default_eps: f32) -> f32 {
    let eps_cap = default_eps.max(1e-6);
    if obs.len() < 2 {
        return eps_cap;
    }
    let mut params: Vec<f32> = obs.iter().map(|o| o.param).collect();
    params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut diffs: Vec<f32> = Vec::with_capacity(params.len().saturating_sub(1));
    for i in 1..params.len() {
        let d = (params[i] - params[i - 1]).abs();
        if d.is_finite() && d > 0.0 {
            diffs.push(d);
        }
    }
    if diffs.is_empty() {
        return eps_cap;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = diffs.len();
    let median = if m % 2 == 1 {
        diffs[m / 2]
    } else {
        0.5 * (diffs[m / 2 - 1] + diffs[m / 2])
    };
    let jitter_idx = ((m as f32) * 0.25).floor() as usize;
    let jitter = diffs[jitter_idx.min(m - 1)];
    let spacing_guard = 0.5 * median;
    let eps_auto = jitter
        .min(if spacing_guard.is_finite() {
            spacing_guard.max(1e-6)
        } else {
            f32::INFINITY
        })
        .max(1e-6);
    if eps_auto.is_finite() {
        eps_auto.min(eps_cap)
    } else {
        eps_cap
    }
}

fn segment_center(seg: &Segment) -> [f32; 2] {
    [0.5 * (seg.p0[0] + seg.p1[0]), 0.5 * (seg.p0[1] + seg.p1[1])]
}

fn normalize_line(mut l: Vector3<f32>) -> Option<[f32; 3]> {
    let n = (l[0] * l[0] + l[1] * l[1]).sqrt();
    if n <= 1e-6 || !n.is_finite() {
        return None;
    }
    l /= n;
    Some([l[0], l[1], l[2]])
}

/// Bundle segments using the rectifying homography: lines are mapped to the
/// rectified plane, grouped along x=const (U family) and y=const (V family),
/// then mapped back to image space as aggregated bundles.
///
/// - `hmtx` is expected in the same coordinate system as the segments.
/// - `eps` is the maximum 1D clustering tolerance on rectified offsets (approx. pixels);
///   the actual tolerance is estimated from the distribution of offsets by taking a
///   low quantile of inter-line spacing and clamping it to half the median spacing.
/// - `min_strength` is the minimum total weight for a cluster to form a bundle.
pub fn bundle_rectified(
    segs: &[Segment],
    hmtx: &Matrix3<f32>,
    eps: f32,
    min_strength: f32,
) -> Vec<crate::segments::bundling::Bundle> {
    if segs.is_empty() {
        return Vec::new();
    }

    // Map image-space lines to rectified coordinates: l' ∝ H^{-T} l
    let h_inv_t = hmtx
        .try_inverse()
        .unwrap_or_else(|| {
            debug!("LSD-VP: H matrix non-invertible during rectified bundling, using identity");
            Matrix3::identity()
        })
        .transpose();

    let mut u_obs: Vec<Obs> = Vec::new();
    let mut v_obs: Vec<Obs> = Vec::new();

    for (i, seg) in segs.iter().enumerate() {
        let rect = h_inv_t * seg.line;
        if !is_finite_vec3(&rect) {
            continue;
        }
        // Decide the dominant axis in rectified space and compute the offset.
        let ax = rect[0].abs();
        let ay = rect[1].abs();
        if ax >= ay {
            // x = const → offset = -c/a
            if rect[0].abs() <= 1e-6 {
                continue;
            }
            let x0 = -rect[2] / rect[0];
            if x0.is_finite() {
                u_obs.push(Obs {
                    index: i,
                    param: x0,
                    strength: seg.strength.max(1.0),
                });
            }
        } else {
            // y = const → offset = -c/b
            if rect[1].abs() <= 1e-6 {
                continue;
            }
            let y0 = -rect[2] / rect[1];
            if y0.is_finite() {
                v_obs.push(Obs {
                    index: i,
                    param: y0,
                    strength: seg.strength.max(1.0),
                });
            }
        }
    }

    let mut bundles_out: Vec<crate::segments::bundling::Bundle> = Vec::new();

    // Compute adaptive eps from inter-line spacing statistics in rectified space.
    let eps_u = compute_auto_eps(&u_obs, eps);
    let eps_v = compute_auto_eps(&v_obs, eps);

    info!(
        "LSD-VP rectified bundling: eps_u={:.2}, eps_v={:.2}",
        eps_u, eps_v
    );

    let to_bundle_with = |cluster: Vec<Obs>,
                          axis: char,
                          h: &Matrix3<f32>,
                          eps_used: f32|
     -> Option<crate::segments::bundling::Bundle> {
        // Robust center: start with weighted mean, then trim outliers by param residual ≤ 2*eps.
        let mut sum_w: f32 = cluster.iter().map(|o| o.strength).sum();
        if sum_w <= 1e-6 {
            return None;
        }
        let mut avg_param = cluster
            .iter()
            .fold(0.0, |acc, o| acc + o.param * o.strength)
            / sum_w;
        let mut kept: Vec<&Obs> = cluster
            .iter()
            .filter(|o| (o.param - avg_param).abs() <= 2.0 * eps_used)
            .collect();
        if kept.is_empty() {
            // Fall back to 3*eps to avoid dropping valid singletons in noisy cases
            kept = cluster
                .iter()
                .filter(|o| (o.param - avg_param).abs() <= 3.0 * eps_used)
                .collect();
        }
        if kept.is_empty() {
            return None;
        }
        sum_w = kept.iter().map(|o| o.strength).sum();
        if sum_w < min_strength {
            return None;
        }
        avg_param = kept.iter().fold(0.0, |acc, o| acc + o.param * o.strength) / sum_w;
        // Build rectified line for this cluster and map back to image: l ≈ H^T l'
        let l_rect = match axis {
            'x' => Vector3::new(1.0, 0.0, -avg_param), // x = const
            'y' => Vector3::new(0.0, 1.0, -avg_param), // y = const
            _ => return None,
        };
        let l_img = h.transpose() * l_rect;
        let line = normalize_line(l_img)?;

        // Weighted center in image space from member segment centers.
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        for o in &kept {
            let c = segment_center(&segs[o.index]);
            cx += c[0] * o.strength;
            cy += c[1] * o.strength;
        }
        cx /= sum_w;
        cy /= sum_w;

        Some(crate::segments::bundling::Bundle {
            line,
            center: [cx, cy],
            weight: sum_w,
        })
    };

    for cluster in cluster_1d(u_obs, eps_u, min_strength) {
        if let Some(b) = to_bundle_with(cluster, 'x', hmtx, eps_u) {
            bundles_out.push(b);
        }
    }
    for cluster in cluster_1d(v_obs, eps_v, min_strength) {
        if let Some(b) = to_bundle_with(cluster, 'y', hmtx, eps_v) {
            bundles_out.push(b);
        }
    }

    bundles_out
}

#[cfg(test)]
mod tests {
    use crate::SegmentId;

    use super::*;

    fn make_vertical_segment(x: f32, y0: f32, y1: f32) -> Segment {
        let len = (y1 - y0).abs();
        Segment {
            id: SegmentId(0),
            p0: [x, y0],
            p1: [x, y1],
            dir: [0.0, 1.0],
            len,
            line: Vector3::new(1.0, 0.0, -x),
            avg_mag: 1.0,
            strength: len.max(1.0),
        }
    }

    #[test]
    fn bundle_rectified_keeps_single_observations_separate() {
        let segs = vec![
            make_vertical_segment(10.0, 0.0, 20.0),
            make_vertical_segment(23.0, -5.0, 18.0),
            make_vertical_segment(39.0, 5.0, 22.0),
        ];

        // A large configured eps would normally merge the three lines, but the adaptive
        // rule should shrink it enough to keep them separated.
        let bundles = bundle_rectified(&segs, &Matrix3::identity(), 15.0, 0.5);
        assert_eq!(bundles.len(), segs.len());

        let mut sorted_centers: Vec<f32> = bundles.iter().map(|b| b.center[0]).collect();
        sorted_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = [10.0, 23.0, 39.0];
        for (observed, exp) in sorted_centers.iter().zip(expected.iter()) {
            assert!((observed - exp).abs() < 1.0);
        }
    }
}
