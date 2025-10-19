use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Pose {
    pub r: Matrix3<f32>,
    pub t: Vector3<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GridResult {
    pub found: bool,
    pub hmtx: Matrix3<f32>, // world->image homography
    pub pose: Option<Pose>,
    pub origin_uv: (i32, i32),
    pub visible_range: (i32, i32, i32, i32), // umin, umax, vmin, vmax
    pub coverage: f32,
    pub reproj_rmse: f32,
    pub confidence: f32,
    pub latency_ms: f64,
}
