

/// Internal observation used for 1D clustering in rectified space.
struct Obs {
    index: usize,
    param: f32,
    strength: f32,
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
