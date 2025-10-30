use crate::segments::Segment;
use log::debug;
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
    obs.sort_by(|a, b| a.param.partial_cmp(&b.param).unwrap_or(std::cmp::Ordering::Equal));
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
/// - `eps` is the 1D clustering tolerance on rectified offsets (approx. pixels).
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

    let to_bundle = |cluster: Vec<Obs>, axis: char, h: &Matrix3<f32>| -> Option<crate::segments::bundling::Bundle> {
        // Robust center: start with weighted mean, then trim outliers by param residual ≤ 2*eps.
        let mut sum_w: f32 = cluster.iter().map(|o| o.strength).sum();
        if sum_w <= 1e-6 {
            return None;
        }
        let mut avg_param = cluster.iter().fold(0.0, |acc, o| acc + o.param * o.strength) / sum_w;
        let mut kept: Vec<&Obs> = cluster
            .iter()
            .filter(|o| (o.param - avg_param).abs() <= 2.0 * eps)
            .collect();
        if kept.is_empty() {
            // Fall back to 3*eps to avoid dropping valid singletons in noisy cases
            kept = cluster
                .iter()
                .filter(|o| (o.param - avg_param).abs() <= 3.0 * eps)
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

    for cluster in cluster_1d(u_obs, eps, min_strength) {
        if let Some(b) = to_bundle(cluster, 'x', hmtx) {
            bundles_out.push(b);
        }
    }
    for cluster in cluster_1d(v_obs, eps, min_strength) {
        if let Some(b) = to_bundle(cluster, 'y', hmtx) {
            bundles_out.push(b);
        }
    }

    bundles_out
}
