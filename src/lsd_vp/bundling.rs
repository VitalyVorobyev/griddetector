use crate::lsd_vp::segment::Segment;
use crate::lsd_vp::FamilyLabel;

use log::{debug, info};
use nalgebra::{Matrix3, Vector3};

struct Bundle {
    family: FamilyLabel,
    line: [f32; 3],
    p0: [f32; 2],
    p1: [f32; 2],
    members: Vec<u32>,
}

fn classify_family(line: [f32; 3], vp_u: &Vector3<f32>, vp_v: &Vector3<f32>) -> FamilyLabel {
    let normal = Vector3::new(line[0], line[1], line[2]);
    let s_u = normal.dot(vp_u).abs();
    let s_v = normal.dot(vp_v).abs();
    if s_u <= s_v { FamilyLabel::U } else { FamilyLabel::V }
}

pub fn bundle_segments(
    segs: &[Segment],
    hmtx: &Matrix3<f32>,
    tolerance: f32,
) -> Result<Vec<Bundle>, String> {
    let vp_u = hmtx.column(0);
    let vp_v = hmtx.column(1);
    let hmtx_invt = hmtx.try_inverse().unwrap_or_else(|| {
        debug!("LSD-VP: H matrix non-invertible during bundling, using identity");
        Matrix3::identity()
    }).transpose();
    let mut lines = Vec::new();
    
}