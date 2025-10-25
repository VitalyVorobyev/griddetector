use crate::segments::Segment;

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

/// Group segments into bundled line constraints using orientation and offset thresholds.
pub fn bundle_segments(
    segs: &[Segment],
    orientation_tol: f32,
    dist_tol: f32,
    min_weight: f32,
) -> Vec<Bundle> {
    let mut bundles: Vec<Bundle> = Vec::new();
    for seg in segs {
        let weight = seg.strength;
        if weight < min_weight {
            continue;
        }
        let line = seg.line;
        let mut placed = false;
        for existing in bundles.iter_mut() {
            let dot = existing.line[0] * line[0] + existing.line[1] * line[1];
            let mut adj_line = line;
            if dot < 0.0 {
                adj_line[0] = -adj_line[0];
                adj_line[1] = -adj_line[1];
                adj_line[2] = -adj_line[2];
            }
            let dot_norm =
                (existing.line[0] * adj_line[0] + existing.line[1] * adj_line[1]).clamp(-1.0, 1.0);
            let angle = dot_norm.acos();
            let dist = (existing.line[2] - adj_line[2]).abs();
            if angle <= orientation_tol && dist <= dist_tol {
                merge_bundle(existing, &adj_line, seg, weight);
                placed = true;
                break;
            }
        }
        if !placed {
            bundles.push(Bundle {
                line,
                center: segment_center(seg),
                weight,
            });
        }
    }
    bundles
}

fn merge_bundle(target: &mut Bundle, line: &[f32; 3], seg: &Segment, weight: f32) {
    let total = target.weight + weight;
    if total <= EPS {
        return;
    }

    target.line[0] = (target.line[0] * target.weight + line[0] * weight) / total;
    target.line[1] = (target.line[1] * target.weight + line[1] * weight) / total;
    target.line[2] = (target.line[2] * target.weight + line[2] * weight) / total;
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
