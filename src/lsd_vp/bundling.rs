use crate::segments::Segment;
use crate::lsd_vp::FamilyLabel;

use log::{debug, info};
use nalgebra::{Matrix3, Vector3};

struct Bundle {
    family: FamilyLabel,
    line: Vector3<f32>,
    p0: [f32; 2],
    p1: [f32; 2],
    members: Vec<u32>,
}

fn classify_family(line: &Vector3<f32>, vp_u: &Vector3<f32>, vp_v: &Vector3<f32>) -> FamilyLabel {
    let s_u = line.dot(vp_u).abs();
    let s_v = line.dot(vp_v).abs();
    if s_u <= s_v { FamilyLabel::U } else { FamilyLabel::V }
}

fn get_u_parameter(rect_line: &Vector3<f32>) -> f32 {
    let rh = Vector3::new(0.0, 1.0, 0.0);
    let pprime = rect_line.cross(&rh);
    if pprime.z.abs() < 1e-6 {
        info!("LSD-VP: u-parameter line at infinity");
    }
    pprime.x / pprime.z
}

fn get_v_parameter(rect_line: &Vector3<f32>) -> f32 {
    let rv = Vector3::new(1.0, 0.0, 0.0);
    let pprime = rect_line.cross(&rv);
    if pprime.z.abs() < 1e-6 {
        info!("LSD-VP: v-parameter line at infinity");
    }
    pprime.y / pprime.z
}

struct TObs {
    index: u32,
    parameter: f32,
    strength: f32,
}

fn cluster_1d(mut obs: Vec<TObs>, eps: f32, min_strength: f32) -> Vec<Vec<TObs>> {
    obs.sort_by(|a, b| a.parameter.partial_cmp(&b.parameter).unwrap());
    let mut clusters = Vec::new();
    let mut current_cluster = Vec::new();

    for ob in obs {
        if current_cluster.is_empty() || (ob.parameter - current_cluster.last().unwrap().parameter).abs() <= eps {
            current_cluster.push(ob);
        } else {
            if current_cluster.iter().map(|x| x.strength).sum() >= min_strength {
                clusters.push(current_cluster);
            }
            current_cluster = vec![ob];
        }
    }
    if !current_cluster.is_empty() && current_cluster.iter().map(|x| x.strength).sum() >= min_strength {
        clusters.push(current_cluster);
    }

    clusters
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
    let mut h_lines = Vec::new();
    let mut v_lines = Vec::new();
    
    for (i, seg) in segs.iter().enumerate() {
        let rect_line = hmtx_invt * seg.line;
        match classify_family(&seg.line, &vp_u, &vp_v) {
            FamilyLabel::U => {
                h_lines.push((i as u32, get_u_parameter(&rect_line), seg.strength));
            }
            FamilyLabel::V => {
                v_lines.push((i as u32, get_v_parameter(&rect_line), seg.strength));
            }
        }
    }



    Ok(h_lines.into_iter().chain(v_lines.into_iter()).collect())
}
