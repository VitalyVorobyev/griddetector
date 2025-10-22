use crate::segments::bundling::Bundle;
use nalgebra::Vector3;

use super::EPS;

/// Estimate the homography anchor column by intersecting the strongest bundle
/// pair from the two vanishing-point families.
///
/// The anchor corresponds to the third column of the homography and acts as
/// the translation term. Choosing the highest-weight pair stabilises the
/// update against noisy or weakly supported bundles.
pub(crate) fn estimate_anchor(fam_u: &[&Bundle], fam_v: &[&Bundle]) -> Option<Vector3<f32>> {
    let best_u = fam_u
        .iter()
        .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())?;
    let best_v = fam_v
        .iter()
        .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())?;
    let line_u = Vector3::new(best_u.line[0], best_u.line[1], best_u.line[2]);
    let line_v = Vector3::new(best_v.line[0], best_v.line[1], best_v.line[2]);
    let cross = line_u.cross(&line_v);
    if cross[2].abs() <= EPS {
        None
    } else {
        Some(cross / cross[2])
    }
}
