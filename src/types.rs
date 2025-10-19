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

#[derive(Clone, Debug)]
pub struct ImageU8<'a> {
    pub w: usize,
    pub h: usize,
    pub stride: usize, // bytes between rows
    pub data: &'a [u8],
}

impl<'a> ImageU8<'a> {
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }
}

#[derive(Clone, Debug)]
pub struct ImageF32 {
    pub w: usize,
    pub h: usize,
    pub stride: usize, // number of f32 elements between rows
    pub data: Vec<f32>,
}

impl ImageF32 {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            stride: w,
            data: vec![0.0; w * h],
        }
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.idx(x, y)]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        let i = self.idx(x, y);
        self.data[i] = v;
    }
}
