use crate::angle::{angle_between_dirless, vp_direction};
use crate::segments::bundling::Bundle;
use nalgebra::{Matrix3, Vector3};

pub(crate) struct FamilyBuckets<'a> {
    pub vpu: Vector3<f32>,
    pub vpv: Vector3<f32>,
    pub anchor: Vector3<f32>,
    pub family_u: Vec<&'a Bundle>,
    pub family_v: Vec<&'a Bundle>,
}

/// Partition bundled line constraints into two vanishing-point families whose
/// tangents align with the current homography estimate.
///
/// The assignment is performed by comparing the angular distance between the
/// bundle tangent and the directions implied by the current homography columns
/// (vanishing points). Bundles outside the orientation tolerance are still
/// assigned to the closest family to keep the optimisation well conditioned.
pub(crate) fn split_bundles<'a>(
    h_current: &Matrix3<f32>,
    bundles: &'a [Bundle],
    orientation_tol: f32,
) -> Option<FamilyBuckets<'a>> {
    let vpu = h_current.column(0).into_owned();
    let vpv = h_current.column(1).into_owned();
    let anchor = h_current.column(2).into_owned();
    let dir_u = vp_direction(&vpu, &anchor)?;
    let dir_v = vp_direction(&vpv, &anchor)?;

    let mut fam_u: Vec<&Bundle> = Vec::new();
    let mut fam_v: Vec<&Bundle> = Vec::new();
    for bundle in bundles {
        let tangent = bundle.tangent();
        let du = angle_between_dirless(&tangent, &dir_u);
        let dv = angle_between_dirless(&tangent, &dir_v);
        if du <= orientation_tol && du < dv {
            fam_u.push(bundle);
        } else if dv <= orientation_tol && dv < du {
            fam_v.push(bundle);
        } else if du < dv {
            fam_u.push(bundle);
        } else {
            fam_v.push(bundle);
        }
    }

    Some(FamilyBuckets {
        vpu,
        vpv,
        anchor,
        family_u: fam_u,
        family_v: fam_v,
    })
}
